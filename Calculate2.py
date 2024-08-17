# coding=UTF-8

import csv
import glob
import numpy as np
from odbAccess import *

# 定义待处理ODB文件
odb_file = glob.glob('*.odb')[0]
odb = openOdb(odb_file)

# 定义分析步及结果提取节点集
step = odb.steps['Step-3-dynamic']
node_set = odb.rootAssembly.nodeSets['SET-BC']

# 初始化结果保存数组
rt_output = np.empty((len(step.frames), 11, 3))

# 按照时间帧，节点编号，方向依次读取数据
for i, frame in enumerate(step.frames):
    for j, value in enumerate(frame.fieldOutputs['RT'].values):
        for k in range(3):
            rt_output[i, j, k] = value.data[k]

# 将数组转化为11个节点，在三个方向上，时程数据
rt_output_re = rt_output.transpose(1, 2, 0)

# 提取各组时程数据最大值，即11个支座各个方向最大节点反力
max_index = np.abs(rt_output_re).argmax(axis=2)
max_values = np.take_along_axis(rt_output_re, max_index[:, :, np.newaxis], axis=2)
max_values_2d = np.squeeze(max_values)

# 保存CSV文件至工作路径
file_name = "max_values.csv"
with open(file_name, mode='w') as file:
    writer = csv.writer(file)
    writer.writerows(max_values_2d)
