import torch as t
import torch.nn as nn
import torch.nn.functional as F


class DecoderDCN(nn.Module):
    def __init__(self, opt, nheads, in_channel, out_channel, concat=True):
        super(DecoderDCN, self).__init__()
        self.nheads = nheads
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.concat = concat

        self.W1 = nn.Parameter(t.zeros(size=(in_channel, in_channel * nheads)))
        nn.init.xavier_uniform_(self.W1.data, gain=1.414)

        self.smooth = nn.Parameter(t.zeros(size=(nheads, 1)))
        nn.init.xavier_uniform_(self.smooth.data, gain=1.414)

        self.out = nn.Parameter(t.zeros(size=(in_channel, out_channel)))
        nn.init.xavier_uniform_(self.out.data, gain=1.414)

    def forward(self, sub_graph, attentions):
        """

        :param sub_graph: previous layers' output, shape of (batch x tree_nodes x p)
        :param attentions: attention weights used to produce previous layers' output, we need this to decode
                            shape of (batch x tree_nodes x nheads x (m+1))
        :return: new_sub_graph: the root tree will generate new nodes, subsequently batch x m+1 x tree_nodes x p
        """
        h = sub_graph.matmul(self.W1)  # shape of (batch x tree_nodes x nheads*p)

        h_hid = [h[:, :, i*self.in_channel:(i+1)*self.in_channel] for i in range(self.nheads)]
        att_hid = [attentions[:, :, i, :] for i in range(self.nheads)]
        new_sub_graph = []
        for i in range(self.nheads):
            sub = h_hid[i].unsqueeze(2)  # shape of (batch x tree_nodes x 1 x p)
            # att = t.pinverse(att_hid[i].unsqueeze(2))  # shape of (batch x tree_nodes x m+1 x 1)
            att = att_hid[i].unsqueeze(2).permute((0, 1, 3, 2))  # shape of (batch x tree_nodes x m+1 x 1)
            new_sub_graph.append(t.matmul(att, sub).unsqueeze(-1))  # shape of (batch x tree_nodes x m+1 x p x 1)

        new_sub_graph = t.cat(new_sub_graph, dim=-1)  # shape of (batch x tree_nodes x m+1 x p x nheads)

        new_sub_graph = t.matmul(new_sub_graph, self.smooth).squeeze(-1)  # shape of (batch x tree_nodes x m+1 x p)

        new_sub_graph = t.matmul(new_sub_graph, self.out)  # shape of (batch x tree_nodes x m+1 x p')

        new_sub_graph = new_sub_graph.permute((0, 2, 1, 3))  # shape of (batch x m+1 x tree_nodes x p')

        if self.concat:
            new_sub_graph = F.elu(new_sub_graph)

        return new_sub_graph
