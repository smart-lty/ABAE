"""
****************NOTE*****************
CREDITS : Thomas Kipf
since datasets are the same as those in kipf's implementation,
Their preprocessing source was used as-is.
*************************************
"""
import numpy as np
import scipy.sparse as sp
from config import opt
import os
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
import torch as t
from load_data.mapping import qw_score, pre_processing, construct_node_tree


def construct(data):
    # 预生成所有待测节点的m叉k层树
    trees = {i: [] for i in range(data.num_nodes)}
    score = qw_score(data)
    x = data.x.clone()
    data.x = t.arange(0, data.num_nodes)[:, None]
    indices = reversed(score.indices)
    trees = pre_processing(data, opt.m, score, trees)
    graph_trees = []
    for node in indices:
        tree = construct_node_tree(data, node.item(), trees, opt)
        graph_trees.append(tree[None, :])
    graph_trees = t.cat(graph_trees, dim=0)
    data.x_w_trees = graph_trees
    w = opt.W
    if opt.W == 'all' or opt.W > x.shape[0]:  # 大于无意义
        w = data.x.shape[0]
    tmp = t.zeros(data.x_w_trees.shape)
    ori_idx = indices[:w].clone().detach()
    tmp[ori_idx] = data.x_w_trees[:w].clone().detach()
    data.x_w_trees = tmp.clone().detach()
    data.x_w_trees = x[data.x_w_trees.squeeze().long()]
    data.x = x
    rt = ".\dataset\\" + str(opt.K) + "_" + str(opt.m) + opt.dataset + ".pth"
    t.save(data, rt)
    return data


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)


def mask_test_edges(adj, features):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    rt = "..\dataset\\" + str(opt.K) + "_" + str(opt.m) + opt.dataset + ".pth"
    if not os.path.exists(rt):
        data = Data(x=t.FloatTensor(features), edge_index=to_undirected(t.LongTensor(edges_all).t()))
        data = construct(data)
    else:
        data = t.load(rt)

    print("load dataset done!")
    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false, data
