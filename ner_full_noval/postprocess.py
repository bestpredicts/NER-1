# /usr/bin/env python
# coding=utf-8

import json
import argparse

import pandas as pd

from utils import Params, IO2STR
from metrics import get_entities

# 参数解析器
parser = argparse.ArgumentParser()
parser.add_argument('--ex_index', type=int, default=2, help="实验名称索引")
parser.add_argument('--mode', type=str, default='test', help="后处理结果类型")


def apply_fn(group):
    result = []
    # 获取该组的所有实体
    for tags, s2o in zip(group.tags, group.split_to_ori):
        entities = get_entities(eval(tags))
        for entity in entities:
            result.append((entity[0], eval(s2o)[entity[1]], eval(s2o)[entity[2]]))
    return result


def postprocess(params, mode):
    # get text
    with open('./ccks2_task1_val/task1_no_val.txt', 'r', encoding='gbk') as f_scr_test, \
            open(params.data_dir / f'{mode}/sentences.txt', 'r', encoding='utf-8') as f_src_val:
        if mode == 'test':
            data_text = [dict(eval(line.strip()))["originalText"] for line in f_scr_test]
        else:
            data_text = [''.join(line.strip().split(' ')) for line in f_src_val]

    with open(params.params_path / f'submit_{mode}.txt', 'w', encoding='utf-8') as f_sub:
        # get df
        pre_df = pd.read_csv(params.params_path / f'{mode}_tags_pre.csv', encoding='utf-8')
        pre_df = pd.DataFrame(pre_df.groupby('example_id').apply(apply_fn), columns=['entities']).reset_index()

        # write to json file
        for entities, text in zip(pre_df['entities'], data_text):
            sample_list = []
            for entity in set(entities):
                enti_dict = {}
                enti_dict["label_type"] = IO2STR[entity[0].strip()]
                enti_dict["start_pos"] = entity[1]
                enti_dict["end_pos"] = entity[2] + 1
                sample_list.append(enti_dict)
            # json to str
            json_str = json.dumps({
                "originalText": text,
                "entities": sample_list
            }, ensure_ascii=False)
            f_sub.write(f'{json_str}\n')


if __name__ == '__main__':
    args = parser.parse_args()
    params = Params(ex_index=args.ex_index)
    postprocess(params, mode=args.mode)
