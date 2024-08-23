# coding=utf-8
# INSTITUTION: Institute of Engineering Mechanics, CEA
# Author: YANG
# Time: 2024/4/12 20:04
import os
import numpy as np
import math
from collections import OrderedDict

from abaqus import *
from driverUtils import *
from odbAccess import *


executeOnCaeStartup()


class OdbAnalyzer:
    def __init__(self, odb_filename, output_dir):
        self.odb_filename = odb_filename
        self.output_dir = output_dir
        self.odb = openOdb(odb_filename)
        self.node_set_A = self.odb.rootAssembly.nodeSets['NODE']
        self.node_set_B = self.odb.rootAssembly.nodeSets['INPUT']
        self.node_set_C = self.odb.rootAssembly.nodeSets['OUTPUT']
        self.element_set = self.odb.rootAssembly.elementSets['WQ']
        self.step = self.odb.steps['Dynamic']

        self.max_disp_A = []
        self.disp_B = []
        self.time_points = []
        self.elpd_sum = []
        self.acc_A = []
        self.relative_disp_at_threshold = None
        self.Ea = []

        '''根据自定义模型杆件及网格划分定义'''
        self.num_member = 176  # 杆件数
        self.one_member_element = 3  # 单根杆件划分网格数
        self.one_element_intp = 8  # 积分点数

    def analyze(self, save_disp=False, save_acc=True, save_peeq=False):
        for frame in self.step.frames:
            displacement = frame.fieldOutputs['UT']
            acceleration = frame.fieldOutputs['AT']
            peeq = frame.fieldOutputs['PEEQ']
            stress = frame.fieldOutputs['S']

            self.time_points.append(round(frame.frameValue, 2))  # 四舍五入到小数点后两位

            # 提取节点集 A 的位移幅值
            disp_A = []
            disp_A_subset = displacement.getSubset(region=self.node_set_A)
            for value in disp_A_subset.values:
                disp_A.append(value.magnitude)

            # 提取节点集 B 唯一节点的位移幅值
            disp_B_subset = displacement.getSubset(region=self.node_set_B)
            for value in disp_B_subset.values:
                self.disp_B.append(value.magnitude)
                break

            # 计算节点集 A 的最大位移幅值
            self.max_disp_A.append(max(disp_A) if disp_A else 0)

            # 提取节点集 C 的加速度分量 AT2
            acc_subset = acceleration.getSubset(region=self.node_set_C)
            for value in acc_subset.values:
                self.acc_A.append(value.data[0])  # 假设 AT2 是第二个分量

            # 提取 MEMBER 集合的 PEEQ 总和除以 8
            peeq_subset = peeq.getSubset(region=self.element_set)
            peeq_sum = sum([value.data for value in peeq_subset.values])
            self.Ea.append(peeq_sum / 8)  # PEEQ 总和除以 8，命名为 Ea

            # 检查应力是否有大于阈值的情况
            if self.relative_disp_at_threshold is None:
                stress_subset = stress.getSubset(region=self.element_set)
                stress_values = [value.mises for value in stress_subset.values]
                if np.any(np.array(stress_values) > 235000000):
                    self.relative_disp_at_threshold = self.max_disp_A[-1] - self.disp_B[-1]

        # 计算相对位移曲线
        relative_displacement = np.array(self.max_disp_A) - np.array(self.disp_B)

        # 取相对位移曲线的峰值
        dm = np.max(relative_displacement)  # 相对位移曲线的峰值，命名为 dm

        # 大于阈值的对应位移
        de = self.relative_disp_at_threshold  # 大于阈值的对应位移，命名为 de

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        if save_disp:
            self.save_disp_data(relative_displacement)

        if save_acc:
            self.save_acc_data()

        if save_peeq:
            self.save_peeq_data()

        # 计算其它参数
        r1, r8 = self.calculate_unit_statistics()

        Ea_last = self.Ea[-1]
        Damage = self.calculate_D(5.0, 30.0, dm, de, Ea_last, 0.2, r1, r8)

        # 关闭 ODB 文件
        self.odb.close()

        return Damage

    def save_disp_data(self, relative_displacement):
        disp_txt_filename = os.path.join(self.output_dir, self.odb_filename.replace('.odb', '-UT-Rela.txt'))
        with open(disp_txt_filename, 'w') as f:
            for time, rel_disp in zip(self.time_points, relative_displacement):
                f.write("{:.2f}\t{:.14f}\n".format(time, rel_disp))
        print("Relative displacement data saved to: {}".format(disp_txt_filename))

    def save_acc_data(self):
        acc_txt_filename = os.path.join(self.output_dir, self.odb_filename.replace('.odb', '-AT2.txt'))
        with open(acc_txt_filename, 'w') as f:
            for time, acc in zip(self.time_points, self.acc_A):
                f.write("{:.2f}\t{:.14f}\n".format(time, acc))
        print("Acceleration data saved to: {}".format(acc_txt_filename))

    def save_peeq_data(self):
        peeq_txt_filename = os.path.join(self.output_dir, self.odb_filename.replace('.odb', '-PEEQ.txt'))
        with open(peeq_txt_filename, 'w') as f:
            f.write("{:.2f}\t{:.14f}\n".format(self.time_points[-1], self.Ea[-1]))
        print("PEEQ (Ea) data saved to: {}".format(peeq_txt_filename))

    def calculate_unit_statistics(self):
        num_values = len(self.step.frames[0].fieldOutputs['S'].getSubset(region=self.element_set).values)
        mises_stress_data = [[] for _ in range(num_values)]
        total_elements = self.num_member * self.one_member_element
        one_member_intp = self.one_member_element * self.one_element_intp

        for frame in self.step.frames:
            field = frame.fieldOutputs['S'].getSubset(region=self.element_set)
            values = field.values

            reordered_values = []
            num_groups = num_values // total_elements

            for i in range(total_elements):
                for j in range(num_groups):
                    index = i + j * total_elements
                    reordered_values.append(values[index])

            for i, value in enumerate(reordered_values):
                mises_stress_data[i].append(value.mises)

        threshold = 235000000.0
        true_counts_per_unit = []

        num_units = num_values // one_member_intp
        for unit_index in range(num_units):
            unit_true_counts = []
            for part_index in range(self.one_member_element):
                part_true_count = 0
                for i in range(self.one_element_intp):
                    index = unit_index * one_member_intp + part_index * self.one_element_intp + i
                    if any(abs(value) >= threshold for value in mises_stress_data[index]):
                        part_true_count += 1
                unit_true_counts.append(part_true_count)
            true_counts_per_unit.append(unit_true_counts)

        eight_p_units, one_p_units = 0, 0

        for unit_true_counts in true_counts_per_unit:
            if any(count == 8 for count in unit_true_counts):
                eight_p_units += 1
            elif any(count >= 1 for count in unit_true_counts):
                one_p_units += 1

        total_units = len(true_counts_per_unit)
        eight_p_percentage = float(eight_p_units) / total_units
        one_p_percentage = float(one_p_units) / total_units

        return one_p_percentage, eight_p_percentage

    def calculate_D(self, f, L, dm, de, Ea, Eu, r1, r8):
        if Ea == 0:
            Damage = 0
        else:
            term1 = (dm - de) / L
            term2 = (Ea / (self.num_member * self.one_member_element)) / Eu

            Damage = 1.4 * math.sqrt((f / L) * 100 * (term1 ** 2 + term2 ** 2) + r1 ** 2 + r8 ** 2)

        return Damage


# 获取模型
'''根据自定义模型名称定义'''
model_name = 'Model-2-HL'
model = mdb.models[model_name]

# 获取加速度时程 TXT 文件夹路径
'''根据自定义加速度时程文件路径定义'''
txt_folder = 'E:\\ABAQUS\\Time'
txt_files = [os.path.join(txt_folder, f) for f in os.listdir(txt_folder) if f.endswith('.txt')]

# 设置调幅峰值加速度
PGA = OrderedDict([('150E-3g', '1.4715'),
                   ('300E-3g', '2.943'),
                   ('500E-3g', '4.905'),
                   ('700E-3g', '6.867'),
                   ('850E-3g', '8.3385'),
                   ('1000E-3g', '9.81')])

with open('record.txt', 'a') as file:
    # 循环处理每个 TXT 文件
    for txt_file in txt_files:
        # 创建保存当前 TXT 文件输出的目录
        file_name = os.path.splitext(os.path.basename(txt_file))[0]
        file.write('---------------------------------------------------\n')
        file.write('{}\n'.format(file_name))
        file.write('---------------------------------------------------\n')

        # 读取 TXT 文件的数据
        data = np.loadtxt(txt_file)
        time_data = data[:, 0]
        accel_data = data[:, 1]

        # 获取总时间
        # total_time = time_data[-1]

        # 修改动力学分析步的时间长度
        # step = model.steps['Dynamics']
        # step.setValues(timePeriod=total_time)

        # 创建表类型的幅值，以amplitude_name命名，时间/频率列用time_data填充，幅值用accel_data填充。
        amplitude_name = 'Amp_{}'.format(file_name)
        model.TabularAmplitude(name=amplitude_name, data=zip(time_data, accel_data))

        Intact, Slight, Medium = 0, 0, 0

        for key, value in PGA.items():
            # 修改边界条件的幅值曲线
            '''根据自定义动力分析边界条件定义'''
            bc = model.boundaryConditions['Earthquake']
            '''根据自定义动力分析步名称定义'''
            bc.setValuesInStep(stepName='Dynamic', a1=float(value), amplitude=amplitude_name)

            # 创建新的作业名称
            job_name = '{}_{}'.format(file_name, key)
            '''根据自定义子程序文件、临时文件、CPU数量、GPU数量等进行定义'''
            mdb.Job(name=job_name, model=model_name, userSubroutine='E:\\ABAQUS\\Subroutine\\STEEL-UMAT.for',
                    scratch='E:\\ABAQUS\\Temp', numCpus=5, numDomains=5, numGPUs=0)
            mdb.jobs[job_name].submit(consistencyChecking=OFF)
            mdb.jobs[job_name].waitForCompletion()

            '''根据自定义输出文件路径定义'''
            Analyzer = OdbAnalyzer(odb_filename='{}.odb'.format(job_name),
                                   output_dir='E:\\ABAQUS\\Output')
            D = Analyzer.analyze()

            file.write('{}: D={}\n'.format(key, D))
            file.flush()

            if D == 0.0:
                Intact += 1
            if 0.3 >= D > 0.0:
                Slight += 1
            if 0.7 >= D > 0.3:
                Medium += 1
            if D > 0.7:
                print('模型已在峰值加速度到达{}时，出现严重损坏，{}调幅已停止'.format(key, file_name))
                break

        file.write('完好: {}; 轻微损坏: {}; 中等损坏: {}\n'.format(Intact, Slight, Medium))
