import torch as t
import torch.nn as nn
from model.AttentionLayer import AttentionSCN


def generate_graph(m, num_nodes):
    now = 0
    graph = []
    while now * m + 1 < num_nodes:
        col = [now]
        col = col + [node for node in range(now * m + 1, now * m + 1 + m)]
        graph.append(t.tensor(col, dtype=t.long)[:, None])
        now += 1
    return t.cat(graph, dim=-1)


class SubAttConv(nn.Module):
    """implement attention-based subtree convolution"""
    def __init__(self, opt, in_channel, out_channel):
        super(SubAttConv, self).__init__()
        self.now = 0  # 当前子树根节点
        self.opt = opt
        self.conv = AttentionSCN(opt, opt.nheads, in_channel, out_channel)

    def forward(self, h):
        # h 为前一层的树，大小为(batch x n x p) 其中n为前一层的结点数

        m = self.opt.m
        conv_graph = generate_graph(m, h.shape[1])
        sub_graph = h[:, conv_graph, :]  # batch x (m+1) x new_tree_nodes x p
        output, attentions = self.conv(sub_graph)  # batch x new_tree_nodes x p

        return output, attentions
