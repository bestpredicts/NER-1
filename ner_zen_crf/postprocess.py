# /usr/bin/env python
# coding=utf-8
"""Postprocess"""
import json

from utils import IO2STR
from metrics import get_entities
from utils import Params


def postprocess(params):
    """分析文本形式结果
    """
    # get text
    with open(params.data_dir / f'test.bio', 'r', encoding='utf-8') as f:
        sentences = [line.strip().split(' ') for line in f]

    # 预测标签
    with open(params.data_dir / f'test_tags_pre.bio', 'r') as f:
        result = []
        for idx, line in enumerate(f):
            # get BIO-tag
            entities = get_entities(line.strip().split(' '))
            sample_list = []
            for entity in entities:
                label_type = IO2STR[entity[0]]
                start_ind = entity[1]
                end_ind = entity[2]
                en = sentences[idx][start_ind:end_ind + 1]
                sample_list.append((label_type, ''.join(en)))
            result.append(sample_list)

    return result


def get_submit(params):
    # 获取测试集结果
    re_pre = postprocess(params)
    # 获取文本id
    with open(params.data_dir / f'test.data', 'r', encoding='utf-8') as f:
        text_data = json.load(f)
        id_list = [sample['id'] for sample in text_data]
    # 写submit
    with open(params.params_path / 'submit.csv', 'w', encoding='utf-8') as f_sub:
        # 获取每条文本的预测结果
        for re_sample, idx in zip(re_pre, id_list):
            # 获取单个预测结果
            for re in set(re_sample):
                f_sub.write(f'{idx}\t{re[0]}\t{re[1]}\n')


if __name__ == '__main__':
    params = Params()
    get_submit(params)
