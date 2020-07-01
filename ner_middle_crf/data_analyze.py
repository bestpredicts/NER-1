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
print(pd.Series(len_list).describe())
