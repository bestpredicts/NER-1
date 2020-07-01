#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""evaluate"""
import argparse
import logging

import torch

import utils
from metrics_utils import mrc2bio
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
    for input_ids, input_mask, segment_ids, start_pos, end_pos, ner_cate in eval_dataloader:
        # to device
        input_ids = input_ids.to(params.device)
        input_mask = input_mask.to(params.device)
        segment_ids = segment_ids.to(params.device)
        start_pos = start_pos.to(params.device)
        end_pos = end_pos.to(params.device)

        with torch.no_grad():
            # get loss
            loss = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                         start_positions=start_pos, end_positions=end_pos)
            if params.n_gpu > 1 and args.multi_gpu:
                loss = loss.mean()  # mean() to average on multi-gpu.
            # update the average loss
            loss_avg.update(loss.item())

            # inference
            start_logits, end_logits = model(input_ids=input_ids,
                                             token_type_ids=segment_ids, attention_mask=input_mask)

        # gold label
        start_pos = start_pos.to("cpu").numpy().tolist()
        end_pos = end_pos.to("cpu").numpy().tolist()
        ner_cate = ner_cate.to("cpu").numpy().tolist()

        # predict label
        start_label = start_logits.detach().cpu().numpy().tolist()
        end_label = end_logits.detach().cpu().numpy().tolist()

        # idx to label
        cate_idx2label = {idx: value for idx, value in enumerate(params.label_list)}

        # get bio result
        for start_p, end_p, start_g, end_g, ner_cate_s in zip(start_label, end_label,
                                                              start_pos, end_pos,
                                                              ner_cate):
            ner_cate_str = cate_idx2label[ner_cate_s]
            pre_bio_labels = mrc2bio(start_p, end_p, ner_cate_str)
            gold_bio_labels = mrc2bio(start_g, end_g, ner_cate_str)
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

    return f1
