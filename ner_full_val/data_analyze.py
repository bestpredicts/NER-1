# /usr/bin/env python
# coding=utf-8
"""查看数据长度分布"""
import pandas as pd
len_list = []
file_list = ['test', 'train', 'val']
for file in file_list:
    with open(f'./data/{file}/sentences.txt', 'r', encoding='utf-8') as f:
        len_list += [len(line.strip().split(' ')) for line in f]
print(pd.Series(len_list).describe(percentiles=[.25, .5, .75, .85, .95]))


"""
count    500.000000
mean     150.672000
std       44.065815
min       36.000000
25%      119.750000
50%      147.500000
75%      176.250000
85%      191.150000
95%      224.000000
max      358.000000
"""