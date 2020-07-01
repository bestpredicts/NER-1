#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""postprocess"""
import json
import argparse

from utils import IO2STR
from utils import Params
from metrics import get_entities

# 设定参数
parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='test', help="mode")
args = parser.parse_args()


def get_type_entity(f, sentences):
    """获取实体类别和文本
    :param f: 标签文件
    :param sentences (List[List[str]]): 文本
    :return: result: 实体类别和文本
    """
    result = []
    sample_list = []
    for idx, line in enumerate(f):
        # get BIO-tag
        entities = get_entities(line.strip().split(' '))
        for entity in entities:
            label_type = IO2STR[entity[0]]
            start_ind = entity[1]
            end_ind = entity[2]
            # get en from sentence
            en = sentences[idx // len(IO2STR)][start_ind:end_ind + 1]
            sample_list.append((label_type, ''.join(en)))
        # one sample
        if (idx + 1) % len(IO2STR) == 0:
            result.append(sample_list)
            sample_list = []
    return result


def analyze_result(params):
    """分析文本形式结果
    :param mode: 'test' or 'val'
    """
    # get text
    with open(params.data_dir / f'{args.mode}.data', 'r', encoding='utf-8') as f:
        text_data = json.load(f)
        sentences = [list(sample['context'].strip()) for sample in text_data]

    # 预测标签
    with open(params.data_dir / f'{args.mode}_tags_pre.txt', 'r') as f:
        re_pre = get_type_entity(f, sentences)

    return re_pre


def get_submit(params):
    # 获取测试集结果
    re_pre = analyze_result(params)
    # 获取文本id
    with open(params.data_dir / f'{args.mode}.data', 'r', encoding='utf-8') as f:
        text_data = json.load(f)
        id_list = [sample['id'] for sample in text_data]
    # 写submit
    with open(params.params_path / 'submit.csv', 'w', encoding='utf-8') as f_sub:
        # 获取每条文本的预测结果
        for re_sample, idx in zip(re_pre, id_list):
            # 获取单个预测结果
            for re in set(re_sample):
                re = list(re)
                if len(re[1]) != 1 and len(re[1]) <= 20:
                    if re[1][-1] == '公':
                        re[1] += '司'
                    f_sub.write(f'{idx}\t{re[0]}\t{re[1]}\n')


if __name__ == '__main__':
    params = Params()
    get_submit(params)
