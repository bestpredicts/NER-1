#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ensemble.py"""
from collections import Counter
from utils import Params


def vote(params, files_name):
    # crf models
    # init
    crf_result = []
    # get crf result
    for file in files_name:
        with open(params.root_path / 'ensemble/ex4' / file, 'r', encoding='utf-8') as f:
            crf_result += [tuple(line.strip().split('\t')) for line in f]
    # 融合策略
    crf_set = [list(en_count[0]) for en_count in Counter(crf_result).most_common()
               if en_count[1] >= 5]

    final_result = crf_set

    # 写submit
    with open(params.root_path / 'ensemble/ex4' / 'final_submit.csv', 'w', encoding='utf-8') as f_sub:
        # 获取每条文本的预测结果
        for re in final_result:
            f_sub.write(f'{re[0]}\t{re[1]}\t{re[2]}\n')


if __name__ == '__main__':
    params = Params()
    files = ['multi_ex1_nezha.csv', 'multi_ex3_ro.csv', 'multi_ex4_ro.csv',
             'multi_ex5_nezha.csv', 'nezha_rtrans_ex1.csv', 'nezha_tener_ex1.csv',
             'nezha_lstm_ex1.csv']
    vote(params, files)
