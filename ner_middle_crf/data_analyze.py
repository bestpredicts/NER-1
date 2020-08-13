# /usr/bin/env python
# coding=utf-8
"""utils"""
import pandas as pd
"""查看数据长度分布"""
len_list = []
file_list = ['test.bio', 'train.bio', 'val.bio']
for file in file_list:
    with open('./data/' + file, 'r', encoding='utf-8') as f:
        len_list += [len(line.strip().split(' ')) for line in f]
print(pd.Series(len_list).describe(percentiles=[.25, .5, .75, .85, .95]))
"""
count    46696.000000
mean        90.841314
std         98.551987
min          5.000000
25%         50.000000
50%         70.000000
75%         98.000000
85%        126.000000
95%        193.000000
max       1220.000000
max_len = 128
"""