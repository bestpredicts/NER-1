#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dataloader.py utils"""

import json
import numpy as np


def whitespace_tokenize(text):
    """
    Desc:
        runs basic whitespace cleaning and splitting on a piece of text.
    """
    text = text.strip()
    # 内容为空则返回空列表
    if not text:
        return []
    tokens = list(text)
    return tokens


class InputExample(object):
    """a single set of samples of data_src
    """

    def __init__(self,
                 idx,
                 text,
                 enti_type=None,
                 start_position=None,
                 end_position=None,
                 ):
        self.idx = idx
        self.text = text
        self.enti_type = enti_type
        self.start_position = start_position
        self.end_position = end_position


class InputFeatures(object):
    """
    Desc:
        a single set of features of data_src
    Args:
        start_pos: start position is a list of symbol
        end_pos: end position is a list of symbol
    """

    def __init__(self,
                 input_ids,
                 input_mask,
                 start_position=None,
                 end_position=None,
                 ):
        self.input_mask = input_mask
        self.input_ids = input_ids
        self.start_position = start_position
        self.end_position = end_position


def read_pointer_ner_examples(input_file):
    """read MRC-NER data_src to InputExamples
    :return examples (List[InputExample]):
    """
    # read json file
    with open(input_file, "r", encoding='utf-8') as f:
        input_data = json.load(f)
    # get InputExample class
    examples = []
    for entry in input_data:
        idx = entry["id"]
        text = entry["context"]
        enti_type = entry["type"]
        start_position = [[pos[0] for pos in type_pos] for type_pos in entry["entity"]]
        end_position = [[pos[1] for pos in type_pos] for type_pos in entry["entity"]]

        example = InputExample(idx=idx,
                               text=text,
                               enti_type=enti_type,
                               start_position=start_position,
                               end_position=end_position)
        examples.append(example)
    print("InputExamples:", len(examples))
    return examples


def convert_examples_to_features(params, examples, tokenizer, pad_sign=True):
    """convert src data_src to features.
    :param examples (List[InputExamples]): data_src examples.
    :param pad_sign: 是否补零
    :return: features (List[InputFeatures])
    """
    # tag to id
    tag2idx = {tag: idx for idx, tag in enumerate(params.label_list)}
    features = []
    max_len = params.max_seq_length

    for (example_idx, example) in enumerate(examples):
        # List[str]
        context_doc = whitespace_tokenize(example.text)

        # init
        start_label = np.zeros((len(params.label_list), max_len))
        end_label = np.zeros((len(params.label_list), max_len))

        # 获取文本tokens
        # 标签为空的样本
        if len(example.start_position) != 0 and len(example.end_position) != 0:
            # get gold label
            for idx, enti_label in enumerate(example.enti_type):
                label_id = tag2idx[enti_label]
                for start_item, end_item in zip(example.start_position[idx], example.end_position[idx]):
                    if start_item < max_len and end_item < max_len:
                        start_label[label_id][start_item] = 1
                        end_label[label_id][end_item] = 1

        # get context_tokens
        context_doc_tokens = []
        for token in context_doc:
            # tokenize
            tmp_subword_lst = tokenizer.tokenize(token)
            if len(tmp_subword_lst) == 1:
                context_doc_tokens.extend(tmp_subword_lst)  # context len
            else:
                raise ValueError("Please check the result of tokenizer!!!")

        # cut off
        if len(context_doc_tokens) > max_len:
            context_doc_tokens = context_doc_tokens[:max_len]

        # input_mask:
        #   the mask has 1 for real tokens and 0 for padding tokens.
        #   only real tokens are attended to.
        input_mask = []
        # context
        input_mask.extend([1] * len(context_doc_tokens))
        # token to id
        input_ids = tokenizer.convert_tokens_to_ids(context_doc_tokens)

        # zero-padding up to the sequence length
        if len(input_ids) < max_len and pad_sign:
            # 补零
            padding = [0] * (max_len - len(input_ids))
            input_ids += padding
            input_mask += padding

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                start_position=start_label.T.astype(np.int32).tolist(),
                end_position=end_label.T.astype(np.int32).tolist(),
            ))

    return features
