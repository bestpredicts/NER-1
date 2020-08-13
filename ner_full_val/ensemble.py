# /usr/bin/env python
# coding=utf-8
# /usr/bin/env python
# coding=utf-8
"""Ensemble the result"""
import json
import argparse
from collections import Counter

from utils import Params

# 参数解析器
parser = argparse.ArgumentParser()
# 设定参数
parser.add_argument('--num_samples', type=int, default=100, help="测试集样本数")
parser.add_argument('--threshold', type=int, default=2, help="投票阈值")


def vote(result_file, num_samples, threshold, params):
    """融合模型结果
    :param result_file: list of file name
    :param num_samples: 结果样本数
    """
    # ensemble(based ner)
    # init list
    ensemble_entity = [[] for _ in range(num_samples)]
    # get all entities from every results
    for file_name in result_file:
        with open(params.root_path / 'ensemble' / file_name, 'r', encoding='utf-8') as f:
            pre = json.load(f)
            for idx in range(num_samples):
                entities = [(en['label_type'], en['start_pos'], en['end_pos'])
                            for en in pre[f'validate_V2_{idx+1}.json']]
                ensemble_entity[idx].extend(entities)
    # counter
    counters = [Counter(s).most_common() for s in ensemble_entity]
    # 融合策略(ner)
    ner_result = [[c[0] for c in c_list if c[1] >= threshold] for c_list in counters]

    # write to json file
    submit = {}
    for idx, entities in enumerate(ner_result):
        sample_list = []
        for entity in set(entities):
            enti_dict = {}
            enti_dict["label_type"] = entity[0]
            enti_dict["start_pos"] = entity[1]
            enti_dict["end_pos"] = entity[2]
            sample_list.append(enti_dict)
        submit[f"validate_V2_{idx + 1}.json"] = sample_list

    with open(params.root_path / 'ensemble/fusion_submit.txt', 'w', encoding='utf-8') as f_sub:
        # convert dict to json
        json_data = json.dumps(submit, indent=4, ensure_ascii=False)
        f_sub.write(json_data)


if __name__ == '__main__':
    args = parser.parse_args()
    params = Params()
    result_file = ['submit_test.txt', 'submit_test1.txt']
    vote(result_file, args.num_samples, args.threshold, params)
