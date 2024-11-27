# coding=utf-8
# INSTITUTION: Institute of Engineering Mechanics, CEA
# Author: YANG
# Time: 2024/4/12 20:04
import os
import numpy as np
import math
import csv
from collections import OrderedDict

from abaqus import *
from driverUtils import *
from odbAccess import *

executeOnCaeStartup()


def matrix(i, frame, variate, sets, in_matrix, label):
    subset = frame.fieldOutputs[variate].getSubset(region=sets)
    values = np.array([getattr(sub_value, label) for sub_value in subset.values])
    if in_matrix.ndim == 3:
        in_matrix[i, :, :] = values
    elif in_matrix.ndim == 2:
        in_matrix[i, :] = values
    return in_matrix


def calculate_disp_m(disp_all, disp_input):
    disp_all_rel = disp_all - disp_input
    disp_all_rel_mag = np.sqrt(np.sum(disp_all_rel ** 2, axis=-1))
    disp_max_rel_mag = np.max(disp_all_rel_mag, axis=-1)
    disp_max_rel = np.max(disp_all_rel_mag)
    return disp_all_rel, disp_max_rel_mag, disp_max_rel


def calculate_disp_e(stress, threshold, disp_max_rel_mag, disp_m):
    disp_e = disp_m
    for i in range(stress.shape[0]):
        if np.any(stress[i, :] > threshold):
            disp_e = disp_max_rel_mag[i]
            break
    return disp_e


def calculate_epsilon_a(peeq, num_intp):
    peeq_last = peeq[-1, :]
    sum_peeq = np.sum(peeq_last)
    epsilon_a = sum_peeq / num_intp
    return epsilon_a


def calculate_ratio(stress, threshold, num_unit, one_element_intp):
    stress = stress.reshape(stress.shape[0], num_unit, one_element_intp)
    greater_than_threshold = stress > threshold
    counts = np.sum(greater_than_threshold, axis=2)
    max_counts = np.max(counts, axis=0)
    ratio_1 = np.sum(max_counts == 1) / float(stress.shape[1])
    ratio_8 = np.sum(max_counts == 8) / float(stress.shape[1])
    return ratio_1, ratio_8


def calculate_damage_factor(epsilon_a, epsilon_u, disp_m, disp_e, rise, span, ratio_1, ratio_8):
    term1 = (disp_m - disp_e) / span
    term2 = epsilon_a / epsilon_u
    damage = 1.4 * math.sqrt((rise / span) * 100 * (term1 ** 2 + term2 ** 2) + ratio_1 ** 2 + ratio_8 ** 2)
    return damage


def save_demand_data(output_data, odb_filename, output_dir):
    save_path = os.path.join(output_dir, '{}.txt'.format(odb_filename))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.savetxt(save_path, output_data, fmt='%.10f')


class OdbAnalyzer:
    def __init__(self, rise, span, num_node, num_output_node, num_member, one_member_element):
        self.rise = rise
        self.span = span
        self.epsilon_u = 0.2
        self.threshold = 235000000

        self.num_node = num_node
        self.num_output_node = num_output_node
        self.num_member = num_member
        self.one_member_element = one_member_element
        self.one_element_intp = 8
        self.num_unit = self.num_member * self.one_member_element
        self.num_intp = self.num_unit * self.one_element_intp

    def calculate_damage(self, odb_filename, output_dir):
        odb = openOdb(odb_filename)

        '''自定义模型输入、输出元素集'''
        all_node = odb.rootAssembly.nodeSets['NODE']
        input_node = odb.rootAssembly.nodeSets['INPUT']
        output_node = odb.rootAssembly.nodeSets['OUTPUT']
        element_set = odb.rootAssembly.elementSets['WQ']
        ana_step = odb.steps['Dynamic']

        disp_all = np.zeros((len(ana_step.frames), self.num_node, 3))
        disp_input = np.zeros((len(ana_step.frames), 1, 3))
        acce_output = np.zeros((len(ana_step.frames), self.num_output_node, 3))
        peeq = np.zeros((len(ana_step.frames), self.num_intp))
        stress = np.zeros((len(ana_step.frames), self.num_intp))

        for i, frame in enumerate(ana_step.frames):
            '''自定义ODB后处理数据内容，及提取子集'''
            disp_all = matrix(i, frame, variate='UT', sets=all_node, in_matrix=disp_all, label='data')
            disp_input = matrix(i, frame, variate='UT', sets=input_node, in_matrix=disp_input, label='data')
            acce_output = matrix(i, frame, variate='AT', sets=output_node, in_matrix=acce_output, label='data')
            peeq = matrix(i, frame, variate='PEEQ', sets=element_set, in_matrix=peeq, label='data')
            stress = matrix(i, frame, variate='S', sets=element_set, in_matrix=stress, label='mises')

        disp_all_rel, disp_max_rel_mag, disp_m = calculate_disp_m(disp_all, disp_input)
        epsilon_a = calculate_epsilon_a(peeq, self.num_intp)
        disp_e = calculate_disp_e(stress, self.threshold, disp_max_rel_mag, disp_m)
        ratio_1, ratio_8 = calculate_ratio(stress, self.threshold, self.num_unit, self.one_element_intp)

        acce_output = acce_output.reshape(len(ana_step.frames), 3)
        save_demand_data(acce_output, odb_filename, output_dir)

        damage = calculate_damage_factor(epsilon_a, self.epsilon_u, disp_m, disp_e, self.rise, self.span,
                                         ratio_1, ratio_8)
        return damage


# 获取模型
'''根据自定义模型名称定义'''
model_name = 'SLRS-Shell'
model = mdb.models[model_name]

# 获取加速度时程 TXT 文件夹路径
'''根据自定义加速度时程文件路径定义'''
txt_folder = 'E:\\ABAQUS\\Time'
txt_files = [os.path.join(txt_folder, f) for f in os.listdir(txt_folder) if f.endswith('.txt')]

# 设置调幅峰值加速度
PGA = OrderedDict([('100', '0.981'),
                   ('200', '1.962'),
                   ('300', '2.943'),
                   ('400', '3.924'),
                   ('500', '4.905'),
                   ('600', '5.886'),
                   ('700', '6.867'),
                   ('800', '7.848'),
                   ('900', '8.829'),
                   ('1000', '9.81')])

d_values_csv = 'Output/D_values.csv'
damage_states_csv = 'Output/Damage_states.csv'

with open(d_values_csv, 'a') as d_file, open(damage_states_csv, 'a') as state_file:
    d_writer = csv.writer(d_file)
    state_writer = csv.writer(state_file)

    d_writer.writerow(['File_Name', 'D_Value'])
    state_writer.writerow(['File_Name', 'Intact', 'Slight', 'Medium', 'Serious', 'Collapse'])

    for txt_file in txt_files:
        file_name = os.path.splitext(os.path.basename(txt_file))[0]

        data = np.loadtxt(txt_file)
        time_data = data[:, 0]
        accel_data_x = data[:, 1]
        accel_data_y = data[:, 2]
        accel_data_z = data[:, 3]

        total_time = time_data[-1]

        step = model.steps['Dynamic']
        step.setValues(timePeriod=total_time)

        amplitude_name_x = '{}_x'.format(file_name)
        amplitude_name_y = '{}_y'.format(file_name)
        amplitude_name_z = '{}_z'.format(file_name)

        if amplitude_name_x not in model.amplitudes.keys():
            model.TabularAmplitude(name=amplitude_name_x, data=zip(time_data, accel_data_x))
        if amplitude_name_y not in model.amplitudes.keys():
            model.TabularAmplitude(name=amplitude_name_y, data=zip(time_data, accel_data_y))
        if amplitude_name_z not in model.amplitudes.keys():
            model.TabularAmplitude(name=amplitude_name_z, data=zip(time_data, accel_data_z))

        Intact, Slight, Medium, Serious, Collapse = 0, 0, 0, 0, 0

        for key, value in PGA.items():
            '''根据自定义动力分析边界条件定义'''
            bc1 = model.boundaryConditions['Earthquake_X']
            bc2 = model.boundaryConditions['Earthquake_Y']
            bc3 = model.boundaryConditions['Earthquake_Z']
            '''根据自定义动力分析步名称定义'''
            bc1.setValuesInStep(stepName='Dynamic', a1=float(value), amplitude=amplitude_name_x)
            bc2.setValuesInStep(stepName='Dynamic', a2=float(value), amplitude=amplitude_name_x)
            bc3.setValuesInStep(stepName='Dynamic', a3=float(value), amplitude=amplitude_name_x)

            job_name = '{}_{}'.format(file_name, key)
            '''根据自定义子程序文件、临时文件、CPU数量、GPU数量等进行定义'''
            mdb.Job(name=job_name, model=model_name, userSubroutine='E:\\ABAQUS\\Subroutine\\STEEL-UMAT.for',
                    scratch='E:\\ABAQUS\\Temp', numCpus=8, numDomains=8, numGPUs=0)
            mdb.jobs[job_name].submit(consistencyChecking=OFF)
            mdb.jobs[job_name].waitForCompletion()

            '''根据自定义输出文件路径定义'''
            analyzer = OdbAnalyzer(rise=5, span=30, num_node=81, num_output_node=1,
                                   num_member=176, one_member_element=3)
            damage_factor = analyzer.calculate_damage(odb_filename='{}.odb'.format(job_name), output_dir='Output/')

            d_writer.writerow(['{}_{}'.format(file_name, key), damage_factor])

            if damage_factor == 0.0:
                Intact += 1
            elif 0.3 >= damage_factor > 0.0:
                Slight += 1
            elif 0.7 >= damage_factor > 0.3:
                Medium += 1
            elif 1.0 > damage_factor > 0.7:
                Serious += 1
            elif damage_factor >= 1:
                Collapse += 1

        state_writer.writerow([file_name, Intact, Slight, Medium, Serious, Collapse])
