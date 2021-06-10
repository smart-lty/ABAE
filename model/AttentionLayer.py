import torch as t
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    """simple layer for attention-based convolution"""
    def __init__(self, opt, in_channel, out_channel, alpha, concat=True):
        super(AttentionLayer, self).__init__()
        self.drop_out = opt.drop_out
        self.in_dim = in_channel
        self.out_dim = out_channel
        self.concat = concat
        self.alpha = alpha

        self.W = nn.Parameter(t.zeros(size=(in_channel, out_channel)))
        # nn.init.kaiming_uniform(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # 可尝试用kaiming_uniform_替代
        self.a = nn.Parameter(t.zeros(size=(2 * out_channel, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, sub_graph):
        """
        apply attention-based convolution
        :param sub_graph: previous sub_graph, shape of (batch x (m+1) x new_tree_nodes x p)
        :return: [new sub_graph, shape of (batch x new_tree_nodes x p),
                    attention cofficients, shape of batch x m x new_tree_nodes]
        """
        h = sub_graph.matmul(self.W)  # shape of (batch x (m+1) x new_tree_nodes x p')
        batch, m, tree_nodes, features = h.shape
        m -= 1

        roots = h[:, 0, :, :]  # shape of (batch  x new_tree_nodes x p')
        childs = t.cat([h[:, :, i, :] for i in range(tree_nodes)], dim=1)  # batch x new_tree_nodes*(m+1) x p'

        a_input = t.cat([roots.repeat(1, 1, m+1).view(batch, -1, features), childs], dim=-1)  # map the roots to childs

        e = self.leakyrelu(t.matmul(a_input, self.a).squeeze(-1))  # shape of (batch x ((m+1)*new_tree_nodes))

        e = t.cat([e[:, i*(m+1):(i+1)*(m+1), None] for i in range(tree_nodes)], dim=-1)  # batchx (m+1) x new_tree_nodes

        attention = F.softmax(e, dim=-2)
        attention = F.dropout(attention, self.drop_out, training=self.training)
        attention = attention.permute((0, 2, 1))
        attention = attention.unsqueeze(2)  # shape of (batch x new_tree_nodes x 1 x (m+1))

        h = h.permute((0, 2, 1, 3))  # shape of (batch x new_tree_nodes x (m+1) x p')

        new_sub_graph = t.matmul(attention, h).squeeze(2)  # shape of (batch x new_tree_nodes x p')

        if self.concat:
            new_sub_graph = F.elu(new_sub_graph)
        return new_sub_graph, attention


class AttentionSCN(nn.Module):
    """implement attention-based convolutional neural network"""
    def __init__(self, opt, nheads, in_channel, out_channel, alpha=0.2, concat=True):
        super(AttentionSCN, self).__init__()
        self.drop_out = opt.drop_out

        self.attentions = [AttentionLayer(opt, in_channel, out_channel, alpha, concat) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.W = nn.Parameter(t.zeros(size=(out_channel*nheads, out_channel)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

    def forward(self, sub_graph):
        """
        apply multihead attention-based convolution to sub_graph
        :param sub_graph: previous sub_graph, shape of (batch x (m+1) x new_tree_nodes x p)
        :return: [new sub_graph, shape of (batch x new_tree_nodes x p),
                    attention cofficients, shape of batch x m x new_tree_nodes]
        """
        # sub_graph = F.dropout(sub_graph, self.drop_out, training=self.training)  # 在调试代码时，可考虑注释本行，观察是否能提高效率！
        new_sub_graph, attentions = [], []

        for att in self.attentions:
            h, attention = att(sub_graph)
            new_sub_graph.append(h)
            attentions.append(attention)

        new_sub_graph = t.cat(new_sub_graph, dim=-1)  # shape of (batch x new_tree_nodes x (nheads*p'))
        attentions = t.cat(attentions, dim=-2)  # shape of (batch x new_tree_nodes x nheads x (m+1))

        new_sub_graph = F.dropout(new_sub_graph, self.drop_out, training=self.training)

        new_sub_graph = t.matmul(new_sub_graph, self.W)  # shape of (batch x new_tree_nodes x p')

        new_sub_graph = F.elu(new_sub_graph)

        return new_sub_graph, attentions
