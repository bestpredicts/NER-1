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
count    2160.000000
mean      429.292130
std       166.207746
min       172.000000
25%       320.750000
50%       391.500000
75%       501.000000
85%       572.000000
95%       742.100000
max      1664.000000
"""