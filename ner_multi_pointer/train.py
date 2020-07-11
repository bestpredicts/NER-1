#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train with valid"""

import torch
from transformers import BertConfig, RobertaConfig

# 参数解析器
import random
import argparse
import os
import logging
from tqdm import trange

from optimization import BertAdam
import utils
from dataloader import MRCNERDataLoader
from model import BertMultiPointer
from evaluate import evaluate

# 设定参数
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=2333, help="random seed for initialization")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file containing weights to reload before training")
parser.add_argument('--epoch_num', required=True, type=int,
                    help="指定epoch_num")
parser.add_argument('--multi_gpu', action='store_true', help="是否多GPU")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"


def train(model, data_iterator, optimizer, params):
    """Train the model one epoch
    """
    # set model to training mode
    model.train()

    # 记录平均损失
    loss_avg = utils.RunningAverage()

    # Use tqdm for progress bar
    # one epoch
    t = trange(params.train_steps)
    for step, _ in enumerate(t):
        # fetch the next training batch
        batch = next(iter(data_iterator))
        batch = tuple(t.to(params.device) for t in batch)
        input_ids, input_mask, start_pos, end_pos = batch

        # get loss
        loss = model(input_ids, attention_mask=input_mask,
                     start_positions=start_pos, end_positions=end_pos)
        if params.n_gpu > 1 and args.multi_gpu:
            loss = loss.mean()  # mean() to average on multi-gpu.
        # 梯度累加
        if params.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        # back-prop
        loss.backward()

        if (step + 1) % params.gradient_accumulation_steps == 0:
            # performs updates using calculated gradients
            optimizer.step()
            model.zero_grad()

        # update the average loss
        loss_avg.update(loss.item())
        # 右边第一个0为填充数，第二个5为数字个数为5位，第三个3为小数点有效数为3，最后一个f为数据类型为float类型。
        t.set_postfix(loss='{:05.3f}'.format(loss_avg()))


def train_and_evaluate(model, params, restore_file=None):
    """Train the model and evaluate every epoch."""
    # load args
    args = parser.parse_args()

    # Load training data and val data
    dataloader = MRCNERDataLoader(params)
    train_loader = dataloader.get_dataloader(data_sign='train')
    val_loader = dataloader.get_dataloader(data_sign='val')
    # 一个epoch的步数
    params.train_steps = len(train_loader)

    # Prepare optimizer
    # fine-tuning
    # 取模型权重
    param_optimizer = list(model.named_parameters())
    # pretrain model param
    param_pre = [(n, p) for n, p in param_optimizer if 'bert' in n]
    # middle model param
    param_middle = [(n, p) for n, p in param_optimizer if 'bilstm' in n]
    # last param
    param_last = [p for n, p in param_optimizer if 'start_outputs' in n or 'end_outputs' in n]
    # 不进行衰减的权重
    no_decay = ['bias', 'LayerNorm', 'layer_norm', 'dym_weight']
    # 将权重分组
    optimizer_grouped_parameters = [
        # pretrain model param
        # 衰减
        {'params': [p for n, p in param_pre if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': params.weight_decay_rate, 'lr': params.fin_tuning_lr
         },
        # 不衰减
        {'params': [p for n, p in param_pre if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0, 'lr': params.fin_tuning_lr
         },
        # middle model
        # 衰减
        {'params': [p for n, p in param_middle if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': params.weight_decay_rate, 'lr': params.middle_lr
         },
        # 不衰减
        {'params': [p for n, p in param_middle if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0, 'lr': params.middle_lr
         },
        # 单独设置学习率
        {'params': param_last,
         'weight_decay_rate': 0.0, 'lr': params.last_lr}
    ]
    num_train_optimization_steps = len(train_loader) // params.gradient_accumulation_steps * args.epoch_num
    optimizer = BertAdam(optimizer_grouped_parameters, warmup=params.warmup_prop, schedule="warmup_cosine",
                         t_total=num_train_optimization_steps, max_grad_norm=params.clip_grad)

    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(params.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        # 读取checkpoint
        utils.load_checkpoint(restore_path, model, optimizer)

    # patience stage
    best_val_f1 = 0.0
    patience_counter = 0

    for epoch in range(1, args.epoch_num + 1):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch, args.epoch_num))

        # Train for one epoch on training set
        train(model, train_loader, optimizer, params)

        # Evaluate for one epoch on training set and validation set
        # train_metrics = evaluate(model, train_loader, params, mark='Train',
        #                          verbose=True)  # Dict['loss', 'f1']
        val_metrics = evaluate(args, model, val_loader, params)  # Dict['loss', 'f1']
        # 验证集f1-score
        val_f1 = val_metrics['f1']
        # 提升的f1-score
        improve_f1 = val_f1 - best_val_f1

        # Save weights of the network
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        optimizer_to_save = optimizer
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model_to_save.state_dict(),
                               'optim_dict': optimizer_to_save.state_dict()},
                              is_best=improve_f1 > 0,
                              checkpoint=params.model_dir)
        params.save(params.params_path / 'params.json')

        # stop training based params.patience
        if improve_f1 > 0:
            logging.info("- Found new best F1")
            best_val_f1 = val_f1
            if improve_f1 < params.patience:
                patience_counter += 1
            else:
                patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping and logging best f1
        if (patience_counter > params.patience_num and epoch > params.min_epoch_num) or epoch == args.epoch_num:
            logging.info("Best val f1: {:05.2f}".format(best_val_f1))
            break


if __name__ == '__main__':
    args = parser.parse_args()
    params = utils.Params()

    if args.multi_gpu:
        params.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
        params.n_gpu = n_gpu
    else:
        # 设置模型使用的gpu
        torch.cuda.set_device(1)
        # 查看现在使用的设备
        print('current device:', torch.cuda.current_device())
        n_gpu = 1
        params.n_gpu = n_gpu

    # Set the random seed for reproducible experiments
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    params.seed = args.seed
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # Set the logger
    utils.set_logger(log_path=os.path.join(params.params_path, 'train.log'), save=True)
    logging.info("Model type: Robert-Pointer-NER")
    logging.info("device: {}".format(params.device))

    # Prepare model
    config = RobertaConfig.from_pretrained(params.bert_model_dir / 'config.json', output_hidden_states=True)
    model = BertMultiPointer.from_pretrained(params.bert_model_dir,
                                             config=config, params=params)
    model.to(params.device)

    # Train and evaluate the model
    logging.info("Starting training for {} epoch(s)".format(args.epoch_num))
    train_and_evaluate(model, params, args.restore_file)
