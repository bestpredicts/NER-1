# /usr/bin/env python
# coding=utf-8
"""preprocess"""
import copy
from pathlib import Path
import json
import re
import random
import logging

from utils import ENTI_DICT, set_logger

SRC_DATA_DIR = Path('../ccks 4_1 Data')
DATA_DIR = Path('./')


def findall(p, s):
    """Yields all the positions of the pattern p in the string s.
    :param p: sub str
    :param s: father str
    :return (start position, end position)
    """
    i = s.find(p)
    while i != -1:
        yield (i, i + len(p) - 1)
        i = s.find(p, i + 1)


def filter_chars(text):
    """过滤无用字符
    :param text: 文本
    """
    # 找出文本中所有非中，英和数字的字符
    add_chars = set(re.findall(r'[^\u4e00-\u9fa5a-zA-Z0-9]', text))
    extra_chars = set(r"""!！￥$%*（）()-——【】:：“”";；'‘’，。？,.?、""")
    add_chars = add_chars.difference(extra_chars)

    # 替换特殊字符组合
    text = re.sub('{IMG:.?.?.?}', '', text)
    text = re.sub(r'<!--IMG_\d+-->', '', text)
    text = re.sub('(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]', '', text)  # 过滤网址
    text = re.sub('<a[^>]*>', '', text).replace("</a>", "")  # 过滤a标签
    text = re.sub('<P[^>]*>', '', text).replace("</P>", "")  # 过滤P标签
    text = re.sub('<strong[^>]*>', ',', text).replace("</strong>", "")  # 过滤strong标签
    text = re.sub('<br>', ',', text)  # 过滤br标签
    text = re.sub('www.[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]', '', text).replace("()", "")  # 过滤www开头的网址
    text = re.sub(r'\s', '', text)  # 过滤不可见字符
    text = re.sub('Ⅴ', 'V', text)

    # 清洗
    for c in add_chars:
        text = text.replace(c, '')
    return text


def merge_label(content):
    """将文本相同的数据标签合并
    """
    result = []
    for s in content:
        # init
        tmp = [s[0], s[1], [s[2]], [[s[3]]]]
        # 文本相同则合并标签
        for compare in content:
            # 如果文本相同，且非自身，且类别不存在
            if s[1] == compare[1] and s[0] != compare[0] and compare[2] not in tmp[2]:
                tmp[2].append(compare[2])
                tmp[3].append([compare[3]])
            # 如果文本相同，且非自身，且类别已存在
            elif s[1] == compare[1] and s[0] != compare[0] and compare[2] in tmp[2]:
                # 获取对应位置
                idx = tmp[2].index(compare[2])
                # 将实体加入已有位置
                tmp[3][idx].append(compare[3])
        result.append(tmp)
    return result


def src2json():
    """scr data to json file
    """
    # get train data
    result = []
    with open(SRC_DATA_DIR / 'event_entity_train_data_label.csv', 'r', encoding='utf-8') as f, \
            open(DATA_DIR / 'train.data', 'w', encoding='utf-8') as f_json:
        # get src data
        content = [line.strip().split('\t') for line in f]
        # 去掉空标签
        content = [c for c in content if c[2] != 'NaN' and c[3] != 'NaN']

        logging.info('Merge label...')
        # 合并文本相同的数据
        content = merge_label(content)
        logging.info('-done')

        logging.info('Write train set to json file...')
        # write to json
        for idx, c in enumerate(content):
            # 清洗数据
            text = filter_chars(c[1])

            if c[2] != 'NaN' and c[3] != 'NaN':
                result.append({
                    'id': c[0],
                    'context': text,
                    'type': [tp for tp in c[2]],
                    # 每个类别对应一个实体列表，每个实体对应一个位置列表
                    'entity': [[loc for entity in entity_type for loc in list(findall(p=entity, s=text))]
                               for entity_type in c[3]]
                })
        print(f'get {len(result)} train samples.')
        json.dump(result, f_json, indent=4, ensure_ascii=False)

    # get test data
    result = []
    with open(SRC_DATA_DIR / 'event_entity_dev_data.csv', 'r', encoding='utf-8') as f, \
            open(DATA_DIR / 'test.data', 'w', encoding='utf-8') as f_json:
        # get src data
        content = [line.strip().split('\t') for line in f]
        # write to json
        for idx, c in enumerate(content):
            # 清洗数据
            text = filter_chars(c[1])
            result.append({
                'id': c[0],
                'context': text,
            })
        print(f'get {len(result)} test samples.')
        json.dump(result, f_json, indent=4, ensure_ascii=False)


def convert2bio():
    """json file to bio
    """
    with open(DATA_DIR / 'train.data', 'r', encoding='utf-8') as f_json:
        data_list = json.load(f_json)

    all_sentences = []
    all_tags = []

    logging.info('Convert to BIO...')
    for sample in data_list:
        # 取文本和类别
        s_text = list(sample['context'])
        s_tag = copy.deepcopy(s_text)
        # 替换多个实体
        s_type = sample['type']
        # shuffle
        random.shuffle(s_type)

        for idx, s_tp in enumerate(s_type):
            # 获取B-tag
            en_tag = ENTI_DICT[s_tp]
            # 取主体
            for entity in sample['entity'][idx]:
                # 替换实体
                s_tag[entity[0]] = en_tag[0]
                s_tag[entity[0] + 1:entity[1] + 1] = [en_tag[1] for _ in range(entity[1] - entity[0])]
        # 打O标
        for idx, item in enumerate(s_tag):
            if len(item) == 1:
                s_tag[idx] = ENTI_DICT['Others']

        # sanity check
        assert len(s_text) == len(s_tag), '标签与原文本长度不一致！'
        all_sentences.append(s_text)
        all_tags.append(s_tag)
    # sanity check
    assert len(all_sentences) == len(all_tags), '样本数不一致！'
    logging.info('-done')

    # shuffle
    random.seed(2020)
    shuffle_tmp = list(zip(all_sentences, all_tags))
    random.shuffle(shuffle_tmp)
    all_sentences, all_tags = zip(*shuffle_tmp)

    # 写入训练集
    with open(DATA_DIR / 'train.bio', 'w', encoding='utf-8') as f_sen, \
            open(DATA_DIR / 'train_tags.bio', 'w', encoding='utf-8') as f_tag:
        # 逐行写入
        for sentence, tag in zip(all_sentences[:-900], all_tags[:-900]):
            f_sen.write('{}\n'.format(' '.join(sentence)))
            f_tag.write('{}\n'.format(' '.join(tag)))

    # 写入验证集
    with open(DATA_DIR / 'val.bio', 'w', encoding='utf-8') as f_sen, \
            open(DATA_DIR / 'val_tags.bio', 'w', encoding='utf-8') as f_tag:
        # 逐行写入
        for sentence, tag in zip(all_sentences[-900:], all_tags[-900:]):
            f_sen.write('{}\n'.format(' '.join(sentence)))
            f_tag.write('{}\n'.format(' '.join(tag)))


def get_testset():
    """获取测试集
    """
    with open(DATA_DIR / 'test.data', encoding='utf-8') as f:
        data_list = json.load(f)

    # init
    all_sentences = []
    all_tags = []
    for sample in data_list:
        # 取文本和类别
        s_text = list(sample['context'])
        s_tag = [ENTI_DICT['Others'] for _ in range(len(s_text))]

        all_sentences.append(s_text)
        all_tags.append(s_tag)

    # 写入测试集
    with open(DATA_DIR / 'test.bio', 'w', encoding='utf-8') as f_sen, \
            open(DATA_DIR / 'test_tags.bio', 'w', encoding='utf-8') as f_tag:
        # 逐行写入
        for sentence, tag in zip(all_sentences, all_tags):
            f_sen.write('{}\n'.format(' '.join(sentence)))
            f_tag.write('{}\n'.format(' '.join(tag)))


if __name__ == '__main__':
    set_logger(save=False)
    # src data to json
    # src2json()
    # convert json to bio
    # convert2bio()
    get_testset()
