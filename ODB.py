# INSTITUTION: Institute of Engineering Mechanics, CEA
# Author: YANG
# Time: 2024/11/24 下午4:36
import os
import numpy as np
import math

from driverUtils import *
from odbAccess import *

executeOnCaeStartup()


def matrix(i, frame, variate, sets, in_matrix, label):
    subset = frame.fieldOutputs[variate].getSubset(region=sets)
    values = np.array([getattr(value, label) for value in subset.values])
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
        step = odb.steps['Dynamic']

        disp_all = np.zeros((len(step.frames), self.num_node, 3))
        disp_input = np.zeros((len(step.frames), 1, 3))
        acce_output = np.zeros((len(step.frames), self.num_output_node, 3))
        peeq = np.zeros((len(step.frames), self.num_intp))
        stress = np.zeros((len(step.frames), self.num_intp))

        for i, frame in enumerate(step.frames):
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

        acce_output = acce_output.reshape(len(step.frames), 3)
        save_demand_data(acce_output, odb_filename, output_dir)

        damage = calculate_damage_factor(epsilon_a, self.epsilon_u, disp_m, disp_e, self.rise, self.span,
                                         ratio_1, ratio_8)
        return damage


'''根据自定义模型参数实例化以及调用OdbAnalyzer类'''
analyzer = OdbAnalyzer(rise=5, span=30, num_node=81, num_output_node=1, num_member=176, one_member_element=3)
damage_factor = analyzer.calculate_damage(odb_filename='RSN20945_51171759_1770HNE_850E-3g.odb', output_dir='Output/')
