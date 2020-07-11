#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""evaluate"""
import logging

import torch

import utils
from metrics_utils import pointer2bio
from metrics import classification_report, f1_score, accuracy_score


def evaluate(args, model, eval_dataloader, params):
    model.eval()
    # 记录平均损失
    loss_avg = utils.RunningAverage()
    # init
    pre_result = []
    gold_result = []

    # get data
    for batch in eval_dataloader:
        # fetch the next training batch
        batch = tuple(t.to(params.device) for t in batch)
        input_ids, input_mask, start_pos, end_pos = batch

        with torch.no_grad():
            # get loss
            loss = model(input_ids, attention_mask=input_mask,
                         start_positions=start_pos, end_positions=end_pos)
            if params.n_gpu > 1 and args.multi_gpu:
                loss = loss.mean()  # mean() to average on multi-gpu.
            # update the average loss
            loss_avg.update(loss.item())

            # inference
            start_pre, end_pre = model(input_ids=input_ids
                                       , attention_mask=input_mask)

        # gold label
        start_pos = start_pos.to("cpu").numpy().transpose((0, 2, 1)).tolist()  # (batch_size, tag_size, seq_len)
        end_pos = end_pos.to("cpu").numpy().transpose((0, 2, 1)).tolist()

        # predict label
        start_label = start_pre.detach().cpu().numpy().transpose((0, 2, 1)).tolist()
        end_label = end_pre.detach().cpu().numpy().transpose((0, 2, 1)).tolist()

        # idx to label
        cate_idx2label = {idx: value for idx, value in enumerate(params.label_list)}

        # get bio result
        for start_p_b, end_p_b, start_g_b, end_g_b in zip(start_label, end_label,
                                                          start_pos, end_pos):
            for idx, (start_p, end_p, start_g, end_g) in enumerate(zip(start_p_b,
                                                                       end_p_b, start_g_b, end_g_b)):
                pre_bio_labels = pointer2bio(start_p, end_p, cate_idx2label[idx])
                gold_bio_labels = pointer2bio(start_g, end_g, cate_idx2label[idx])
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


if __name__ == '__main__':
    from transformers import RobertaConfig
    from model import BertMultiPointer
    from utils import Params
    from dataloader import MRCNERDataLoader

    # 设置模型使用的gpu
    torch.cuda.set_device(2)
    params = Params()
    dataloader = MRCNERDataLoader(params)
    config = RobertaConfig.from_pretrained(str(params.bert_model_dir / 'config.json'), output_hidden_states=True)
    model = BertMultiPointer.from_pretrained(str(params.bert_model_dir),
                                             config=config, params=params)
    model.to(params.device)
    dev_loader, _ = dataloader.load_data(mode='test')
    val_f1 = evaluate(model, dev_loader, params)
