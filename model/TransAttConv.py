import torch
import torch.nn as nn
from model.DecoderLayer import DecoderDCN
from config import opt


def generate_trans_graph(output):
    """ output shape: (batch x m+1 x tree_nodes x p') """
    new_output = [output[:, 0, 0, :, None]]

    for j in range(output.shape[2]):
        for i in range(1, output.shape[1]):
            new_output.append(output[:, i, j, :, None])

    new_output = torch.cat(new_output, dim=-1)  # shape of (batch x p' x new_tree_nodes)
    new_output = new_output.permute((0, 2, 1))

    return new_output


class TransAttConv(nn.Module):

    def __init__(self, opt, in_channel, out_channel):
        super(TransAttConv, self).__init__()
        self.now = 0  # 当前子树根节点
        self.opt = opt
        self.conv = DecoderDCN(opt, opt.nheads, in_channel, out_channel)

    def forward(self, h, attentions):
        # h 为前一层的树，大小为(batch x n x p) 其中n为前一层的结点数

        m = self.opt.m
        output = self.conv(h, attentions)

        output = generate_trans_graph(output)

        return output

