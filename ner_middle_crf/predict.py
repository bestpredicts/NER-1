# /usr/bin/env python
# coding=utf-8
"""Predict"""

import argparse
import random
import logging
import os
from tqdm import tqdm

import torch

import utils
from NEZHA.model_nezha import BertConfig
from model import BertForTokenClassification
from dataloader import NERDataLoader

# 参数解析器
parser = argparse.ArgumentParser()
# 设定参数
parser.add_argument('--seed', type=int, default=2020, help="random seed for initialization")
parser.add_argument('--restore_file', default=None, required=False,
                    help="Optional, name of the file containing weights to reload before training")
parser.add_argument('--mode', default='test', help="'val' or 'test'")


def predict(model, data_iterator, params, mode):
    """Predict entities
    """
    # set model to evaluation mode
    model.eval()

    # id2tag dict
    idx2tag = {idx: tag for idx, tag in enumerate(params.tags)}

    pred_tags = []

    for batch in tqdm(data_iterator, unit='Batch'):
        # to device
        batch = tuple(t.to(params.device) for t in batch)
        input_ids, input_mask, _ = batch
        # inference
        with torch.no_grad():
            batch_output = model(input_ids, attention_mask=input_mask)
        # List[List[str]]
        pred_tags.extend([[idx2tag.get(idx) for idx in indices] for indices in batch_output])

    # write to file
    with open(params.data_dir / f'{mode}_tags_pre.bio', 'w', encoding='utf-8') as file_tags:
        for tag in pred_tags:
            file_tags.write('{}\n'.format(' '.join(tag)))


if __name__ == '__main__':
    args = parser.parse_args()
    # 设置模型使用的gpu
    torch.cuda.set_device(7)
    # 查看现在使用的设备
    print('current device:', torch.cuda.current_device())
    # 预测验证集还是测试集
    mode = args.mode
    params = utils.Params()
    # Set the random seed for reproducible experiments
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    params.seed = args.seed

    # Set the logger
    utils.set_logger()

    # get dataloader
    dataloader = NERDataLoader(params)

    # Define the model
    logging.info('Loading the model...')
    bert_config = BertConfig.from_json_file(os.path.join(params.bert_model_dir, 'bert_config.json'))
    model = BertForTokenClassification(bert_config, params=params)
    model.to(params.device)
    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(params.model_dir, args.restore_file + '.pth.tar'), model)
    logging.info('- done.')

    logging.info("Loading the dataset...")
    loader = dataloader.get_dataloader(data_sign=mode)

    logging.info("Starting prediction...")
    # Create the input data pipeline
    predict(model, loader, params, mode)
    logging.info('-done')
