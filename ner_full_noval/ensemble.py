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
parser.add_argument('--num_samples', type=int, default=450, help="测试集样本数")
parser.add_argument('--threshold', type=int, default=3, help="投票阈值")


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
            data_src_li = [dict(eval(line.strip())) for line in f]
            for idx in range(num_samples):
                sample = data_src_li[idx]
                entities = [(en['label_type'], en['start_pos'], en['end_pos']) for en in sample['entities']]
                ensemble_entity[idx].extend(entities)
    # counter
    counters = [Counter(s).most_common() for s in ensemble_entity]
    # 融合策略(ner)
    ner_result = [[c[0] for c in c_list if c[1] >= threshold] for c_list in counters]

    with open(params.root_path / 'ensemble/fusion_submit.txt', 'w', encoding='utf-8') as w:
        # get text
        data_text = [line["originalText"] for line in data_src_li]
        # write to json file
        for entities, text in zip(ner_result, data_text):
            sample_list = []
            for entity in entities:
                enti_dict = {}
                enti_dict["label_type"] = entity[0].strip()
                enti_dict["start_pos"] = entity[1]
                enti_dict["end_pos"] = entity[2]
                sample_list.append(enti_dict)
            json.dump({
                "originalText": text,
                "entities": sample_list
            }, w, ensure_ascii=False)
            w.write('\n')


if __name__ == '__main__':
    args = parser.parse_args()
    params = Params()
    # result_file = ['ps_roberta_lstm_ex1.txt', 'ps_roberta_lstm_ex2.txt', 'ps_nezha_lstm_ex1.txt',
    #                'ps_roberta_mrc_ex1.txt', 'ps_roberta_mrc_ex2.txt']
    result_file = ['nezha_lstm_ex2.txt', 'nezha_lstm_ex3.txt', 'roberta_lstm_ex1.txt', 'roberta_lstm_ex2.txt',
                   'roberta_mrc_ex1.txt', 'roberta_mrc_ex3.txt', 'roberta_mrc_ex4.txt']
    vote(result_file, args.num_samples, args.threshold, params)
    # print(get_crf_csw_entity(params))
