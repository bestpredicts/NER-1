#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dataloader.py utils"""
import random, math


class InputExample(object):
    """a single set of samples of data_src
    """

    def __init__(self, sentence, tags):
        """
        Desc:
            is_impossible: bool, [True, False]
        """
        self.sentence = sentence
        self.tags = tags


class InputFeatures(object):
    """
    Desc:
        a single set of features of data_src
    Args:
        start_pos: start position is a list of symbol
        end_pos: end position is a list of symbol
    """

    def __init__(self,
                 tokens,
                 input_ids,
                 input_mask,
                 tag,
                 ngram_ids, ngram_positions, ngram_masks
                 ):
        self.tokens = tokens
        self.input_mask = input_mask
        self.input_ids = input_ids
        self.tag = tag

        # ngram
        self.ngram_ids = ngram_ids
        self.ngram_positions = ngram_positions
        # self.ngram_lengths = ngram_lengths
        self.ngram_masks = ngram_masks


def read_examples(data_dir, data_sign):
    """read BIO-NER data_src to InputExamples
    :return examples (List[InputExample]):
    """
    examples = []
    # read src data
    with open(data_dir / f'{data_sign}.bio', "r", encoding='utf-8') as f_sen, \
            open(data_dir / f'{data_sign}_tags.bio', 'r', encoding='utf-8') as f_tag:
        for sen, tag in zip(f_sen, f_tag):
            example = InputExample(sentence=sen.strip().split(' '), tags=tag.strip().split(' '))
            examples.append(example)
    print("InputExamples:", len(examples))
    return examples


def convert_examples_to_features(params, examples, tokenizer, ngram_dict, pad_sign=True):
    """convert examples to features.
    :param examples (List[InputExamples]): data_src examples.
    :param pad_sign: 是否补零
    """
    # tag to id
    tag2idx = {tag: idx for idx, tag in enumerate(params.tags)}
    features = []

    # context max len
    max_len = params.max_seq_length

    for (example_idx, example) in enumerate(examples):
        # tokenize test
        tokens = [tokenizer.tokenize(w)[0] for w in example.sentence]
        # label id
        tag_idx = [tag2idx[tag] for tag in example.tags]

        # cut off
        if len(tokens) >= max_len:
            tokens = tokens[:max_len]
            tag_idx = tag_idx[:max_len]
        # token to id
        text_ids = tokenizer.convert_tokens_to_ids(tokens)

        # zero-padding up to the sequence length
        if len(tokens) < max_len and pad_sign:
            # 补零
            pad_len = max_len - len(text_ids)
            # token_pad_id=0
            text_ids += [0] * pad_len
            tag_idx += [tag2idx['O']] * pad_len
        # mask
        input_mask = [1 if idx > 0 else 0 for idx in text_ids]

        # ----------- code for ngram BEGIN-----------
        ngram_matches = []
        #  Filter the word segment from 2 to 7 to check whether there is a word
        for p in range(2, 8):
            for q in range(0, len(tokens) - p + 1):
                character_segment = tokens[q:q + p]
                # j is the starting position of the word
                # i is the length of the current word
                character_segment = tuple(character_segment)
                if character_segment in ngram_dict.ngram_to_id_dict:
                    ngram_index = ngram_dict.ngram_to_id_dict[character_segment]
                    ngram_matches.append([ngram_index, q, p, character_segment])

        random.shuffle(ngram_matches)
        # max_word_in_seq_proportion = max_word_in_seq
        max_word_in_seq_proportion = math.ceil((len(tokens) / max_len) * ngram_dict.max_ngram_in_seq)
        if len(ngram_matches) > max_word_in_seq_proportion:
            ngram_matches = ngram_matches[:max_word_in_seq_proportion]
        ngram_ids = [ngram[0] for ngram in ngram_matches]
        ngram_positions = [ngram[1] for ngram in ngram_matches]
        ngram_lengths = [ngram[2] for ngram in ngram_matches]
        # ngram_tuples = [ngram[3] for ngram in ngram_matches]
        # ngram_seg_ids = [0 if position < (len(tokens) + 2) else 1 for position in ngram_positions]

        import numpy as np
        ngram_mask_array = np.zeros(ngram_dict.max_ngram_in_seq, dtype=np.bool)
        ngram_mask_array[:len(ngram_ids)] = 1

        # record the masked positions
        ngram_positions_matrix = np.zeros(shape=(max_len, ngram_dict.max_ngram_in_seq), dtype=np.int32)
        for i in range(len(ngram_ids)):
            ngram_positions_matrix[ngram_positions[i]:ngram_positions[i] + ngram_lengths[i], i] = 1.0

        # Zero-pad up to the max word in seq length.
        padding = [0] * (ngram_dict.max_ngram_in_seq - len(ngram_ids))
        ngram_ids += padding
        # ngram_lengths += padding
        # ngram_seg_ids += padding

        # ----------- code for ngram END-----------

        # get features
        features.append(
            InputFeatures(
                tokens=tokens,
                input_ids=text_ids,
                input_mask=input_mask,
                tag=tag_idx,
                ngram_ids=ngram_ids,
                ngram_positions=ngram_positions_matrix,
                # ngram_lengths=ngram_lengths,
                ngram_masks=ngram_mask_array
            ))

    return features
