#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""evaluate"""
import argparse
import logging
from tqdm import tqdm

import torch

import utils
from utils import EN2QUERY
from metrics_utils import pointer2bio
from metrics import classification_report, f1_score, accuracy_score

# 参数解析器
parser = argparse.ArgumentParser()
# 设定参数
parser.add_argument('--mode', default='val', help="'val' or 'test'")


def evaluate(args, model, eval_dataloader, params):
    model.eval()
    # 记录平均损失
    loss_avg = utils.RunningAverage()
    # init
    pre_result = []
    gold_result = []

    # get data
    for batch in tqdm(eval_dataloader, unit='Batch', ascii=True):
        # to device
        batch = tuple(t.to(params.device) for t in batch)
        input_ids, input_mask, segment_ids, start_pos, end_pos, en_cate, _, _ = batch

        with torch.no_grad():
            # get loss
            loss = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                         start_positions=start_pos, end_positions=end_pos)
            if params.n_gpu > 1 and args.multi_gpu:
                loss = loss.mean()  # mean() to average on multi-gpu.
            # update the average loss
            loss_avg.update(loss.item())

            # inference
            start_pre, end_pre = model(input_ids=input_ids,
                                       token_type_ids=segment_ids, attention_mask=input_mask)

        # gold label
        start_pos = start_pos.to("cpu").numpy().tolist()
        end_pos = end_pos.to("cpu").numpy().tolist()

        input_mask = input_mask.to('cpu').numpy().tolist()
        en_cate = en_cate.to("cpu").numpy().tolist()

        # predict label
        start_pre = start_pre.detach().cpu().numpy().tolist()
        end_pre = end_pre.detach().cpu().numpy().tolist()

        # idx to label
        cate_idx2label = {idx: value for idx, value in enumerate(params.tag_list)}

        # get bio result
        for start_p, end_p, start_g, end_g, input_mask_s, en_cate_s in zip(start_pre, end_pre,
                                                                           start_pos, end_pos,
                                                                           input_mask, en_cate):
            en_cate_str = cate_idx2label[en_cate_s]
            # 问题长度
            q_len = len(EN2QUERY[en_cate_str])
            # 有效长度
            act_len = sum(input_mask_s[q_len + 2:-1])
            # 转换为BIO标注
            pre_bio_labels = pointer2bio(start_p[q_len + 2:q_len + 2 + act_len],
                                         end_p[q_len + 2:q_len + 2 + act_len],
                                         en_cate=en_cate_str)
            gold_bio_labels = pointer2bio(start_g[q_len + 2:q_len + 2 + act_len],
                                          end_g[q_len + 2:q_len + 2 + act_len],
                                          en_cate=en_cate_str)
            pre_result.append(pre_bio_labels)
            gold_result.append(gold_bio_labels)

    # metrics
    f1 = f1_score(y_true=gold_result, y_pred=pre_result)
    acc = accuracy_score(y_true=gold_result, y_pred=pre_result)

    # f1, acc
    metrics = {'loss': loss_avg(), 'f1': f1, 'acc': acc}
    metrics_str = "; ".join("{}: {:05.2f}".format(k, v) for k, v in metrics.items())
    logging.info("- {} metrics: ".format('Val') + metrics_str)
    # f1 classification report
    report = classification_report(y_true=gold_result, y_pred=pre_result)
    logging.info(report)

    return metrics
