# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/19 10:43
# @File    : QsCNNs.py
import torch
import torch.nn as nn
from model.SubAttConv import SubAttConv
from model.TransAttConv import TransAttConv
import time
import torch.nn.functional as F
from config import opt


class AttTSCN(nn.Module):
    def __init__(self, opt):
        super(AttTSCN, self).__init__()
        self.opt = opt

        self.sub_conv = nn.ModuleList(
            [
                SubAttConv(opt, opt.input_features, opt.conv_channel)
            ]
            +
            [
                SubAttConv(opt, opt.conv_channel, opt.conv_channel) for _ in range(opt.conv_num - 1)
            ]
        )

        self.trans_conv = nn.ModuleList(
            [
                TransAttConv(opt, opt.conv_channel, opt.conv_channel) for _ in range(opt.conv_num - 1)
            ]
            +
            [
                TransAttConv(opt, opt.conv_channel, opt.input_features)
            ]
        )

        self.loss1 = torch.nn.MSELoss()

    def forward(self, data):
        batch_trees = data.x_w_trees  # [batch中节点总数,15,5]
        encoder_trees = batch_trees
        atts = []

        for now in range(self.opt.conv_num):  # K:subtree层数，now 0 1 2 3
            encoder_trees, attentions = self.sub_conv[now](encoder_trees)
            atts.append(attentions)

        decoder_trees = encoder_trees
        for now in range(self.opt.conv_num):
            decoder_trees = self.trans_conv[now](decoder_trees, atts.pop())

        A_pred = torch.matmul(encoder_trees.squeeze(1), encoder_trees.squeeze(1).t())
        A_pred = torch.sigmoid(A_pred)

        loss1 = self.loss1(batch_trees, decoder_trees)

        return A_pred, loss1
