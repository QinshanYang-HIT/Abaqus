# coding=utf-8
# INSTITUTION: Institute of Engineering Mechanics, CEA
# Author: YANG
# Time: 2024/8/24 上午9:40
import os
import numpy as np
import math
import csv
from collections import OrderedDict

from abaqus import *
from driverUtils import *
from odbAccess import *


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

    def analyze(self, save_disp=False, save_acc=False, save_peeq=False):
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


folder_path = 'E:\\ABAQUS\\ODB\\Data'

with open('Result\\record_add.txt', 'a') as txt_file:
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        Intact, Slight, Medium = 0, 0, 0
        for odb_file in os.listdir(item_path):
            txt_file.write('---------------------------------------------------\n')
            txt_file.write('{}\n'.format(item))
            txt_file.write('---------------------------------------------------\n')
            txt_file.flush()

            odb_file_path = os.path.join(item_path, odb_file)
            Analyzer = OdbAnalyzer(odb_filename='{}'.format(odb_file_path), output_dir='E:\\ABAQUS\\Output')
            D = Analyzer.analyze()

            job_name = os.path.splitext(odb_file)[0]
            txt_file.write('{}: D={}\n'.format(job_name, D))
            txt_file.flush()

            if D == 0.0:
                Intact += 1
            elif 0.3 >= D > 0.0:
                Slight += 1
            elif 0.7 >= D > 0.3:
                Medium += 1
            elif D > 0.7:
                break

        txt_file.write('完好: {}; 轻微损坏: {}; 中等损坏: {}\n'.format(Intact, Slight, Medium))
        txt_file.flush()
