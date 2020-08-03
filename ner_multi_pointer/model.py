#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
"""model"""

import torch
import torch.nn as nn

from transformers import BertPreTrainedModel, RobertaModel
from utils import initial_parameter


class BiLSTM(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers, dropout):
        super(BiLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)

        # set multi-lstm dropout
        self.multi_dropout = 0. if num_layers == 1 else dropout
        self.bilstm = nn.LSTM(input_size=embedding_size,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              dropout=self.multi_dropout,
                              bidirectional=True)

        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.hidden2tag = nn.Linear(hidden_size * 2, embedding_size)

        self.reset_params()

    def reset_params(self):
        nn.init.xavier_normal_(self.hidden2tag.weight)

    def get_lstm_features(self, embed, mask):
        """
        :param seq: (seq_len, batch_size, embedding_size)
        :param mask: (seq_len, batch_size)
        :return lstm_features: (seq_len, batch_size, tag_size)
        """
        embed = self.dropout(embed)
        max_len, batch_size, embed_size = embed.size()
        embed = nn.utils.rnn.pack_padded_sequence(embed, mask.sum(0).long(), enforce_sorted=False)
        lstm_output, _ = self.bilstm(embed)  # (seq_len, batch_size, hidden_size*2)
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(lstm_output, total_length=max_len)
        lstm_output = lstm_output * mask.unsqueeze(-1)
        lstm_output = self.layer_norm(lstm_output)
        lstm_features = self.hidden2tag(lstm_output) * mask.unsqueeze(-1)  # (seq_len, batch_size, tag_size)
        return lstm_features


class BertMultiPointer(BertPreTrainedModel):
    def __init__(self, config, params):
        super(BertMultiPointer, self).__init__(config)
        # pretrain model layer
        self.bert = RobertaModel(config)
        self.tag_size = len(params.label_list)

        # lstm
        # self.bilstm = BiLSTM(embedding_size=config.hidden_size, hidden_size=params.lstm_hid,
        #                      num_layers=params.lstm_layers, dropout=params.dropout)

        # start and end position layer
        self.start_outputs = nn.Linear(config.hidden_size, self.tag_size)
        self.end_outputs = nn.Linear(config.hidden_size, self.tag_size)

        # 动态权重
        self.dym_weight = nn.Parameter(torch.ones((config.num_hidden_layers, 1, 1, 1)),
                                       requires_grad=True)

        # loss weight
        self.loss_wb = params.weight_start
        self.loss_we = params.weight_end
        self.threshold = params.multi_threshold

        self.init_weights()
        self.init_param()

    def init_param(self):
        initial_parameter(self.start_outputs)
        initial_parameter(self.end_outputs)
        nn.init.xavier_normal_(self.dym_weight)

    def get_dym_layer(self, outputs):
        """
        获取动态权重融合后的bert output(num_layer维度)
        :param outputs: origin bert output
        :return: sequence_output: 融合后的bert encoder output. (batch_size, seq_len, hidden_size[embedding_dim])
        """
        hidden_stack = torch.stack(outputs[2][1:], dim=0)  # (bert_layer, batch_size, sequence_length, hidden_size)
        sequence_output = torch.sum(hidden_stack * self.dym_weight,
                                    dim=0)  # (batch_size, seq_len, hidden_size[embedding_dim])
        return sequence_output

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, position_ids=None,
                start_positions=None, end_positions=None):
        """
        Args:
            start_positions: (batch x max_len x tag_size)
            end_positions: (batch x max_len x tag_size)
        """
        # pretrain model
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids
        )  # sequence_output, pooled_output, (hidden_states), (attentions)

        # BERT融合
        sequence_output = self.get_dym_layer(outputs)  # (batch_size, seq_len, hidden_size[embedding_dim])

        # lstm
        # sequence_output = self.bilstm.get_lstm_features(sequence_output.transpose(1, 0),
        #                                                 attention_mask.transpose(1, 0)).transpose(1, 0)
        batch_size, seq_len, hid_size = sequence_output.size()

        # get logits
        start_logits = self.start_outputs(sequence_output)  # batch x seq_len x tag_size
        end_logits = self.end_outputs(sequence_output)  # batch x seq_len x tag_size

        # expand mask
        expand_mask = attention_mask.unsqueeze(-1).expand(-1, -1, self.tag_size)  # (batch_size, seq_len , tag_size)
        # mask
        start_logits = torch.where(expand_mask == 0,
                                   torch.full(size=expand_mask.size(), fill_value=-10000),
                                   start_logits
                                   )
        end_logits = torch.where(expand_mask == 0,
                                 torch.full(size=expand_mask.size(), fill_value=-10000),
                                 end_logits
                                 )

        # train
        if start_positions is not None and end_positions is not None:
            # s & e loss
            # weight = torch.tensor([2e3]*(self.tag_size*seq_len), device=self.device)
            loss_fct = nn.BCEWithLogitsLoss()
            start_loss = loss_fct(start_logits.view(batch_size, -1), start_positions.view(batch_size, -1).float())
            end_loss = loss_fct(end_logits.view(batch_size, -1), end_positions.view(batch_size, -1).float())

            # total loss
            total_loss = self.loss_wb * start_loss + self.loss_we * end_loss
            return total_loss
        # inference
        else:
            start_pre = torch.sigmoid(start_logits)  # batch x seq_len x tag_size
            end_pre = torch.sigmoid(end_logits)
            # get output
            start_pre = torch.where(start_pre > self.threshold,
                                    torch.ones(start_pre.size(), device=start_pre.device),
                                    torch.zeros(start_pre.size(), device=start_pre.device))
            end_pre = torch.where(end_pre > self.threshold,
                                  torch.ones(start_pre.size(), device=start_pre.device),
                                  torch.zeros(start_pre.size(), device=start_pre.device))
            return start_pre, end_pre


if __name__ == '__main__':
    from transformers import RobertaConfig
    import utils

    params = utils.Params()
    # Prepare model
    config = RobertaConfig.from_pretrained(str(params.bert_model_dir / 'config.json'), output_hidden_states=True)
    model = BertMultiPointer.from_pretrained(str(params.bert_model_dir),
                                             config=config, params=params)

    for n, _ in model.named_parameters():
        print(n)
