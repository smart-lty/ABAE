import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, average_precision_score
import os
import time

from load_data.input_data import load_data
from load_data.preprocessing import *
from config import opt
from model.AttTSCN import AttTSCN


def train():
    # Train on CPU (hide GPU) due to memory constraints
    os.environ['CUDA_VISIBLE_DEVICES'] = ""

    global features, adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false, data, adj_label, weight_tensor

    adj, features = load_data(opt.dataset)
    features = sparse_to_tuple(features.tocoo())
    features = torch.sparse.FloatTensor(torch.LongTensor(features[0].T),
                                        torch.FloatTensor(features[1]),
                                        torch.Size(features[2]))

    features = torch.sparse.FloatTensor.to_dense(features)
    opt.input_features = features.shape[1]
    # adj邻接矩阵
    # x特征矩阵

    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()
    # print(adj_orig.shape)

    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false, data = mask_test_edges(adj,
                                                                                                             features)

    adj = adj_train

    # Some preprocessing
    adj_norm = preprocess_graph(adj)

    num_nodes = adj.shape[0]

    num_features = features[2][1]
    features_nonzero = features[1].shape[0]

    # Create Model
    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)

    adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T),
                                        torch.FloatTensor(adj_norm[1]),
                                        torch.Size(adj_norm[2]))
    adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T),  # 2708*2708
                                         torch.FloatTensor(adj_label[1]),
                                         torch.Size(adj_label[2]))

    weight_mask = adj_label.to_dense().view(-1) == 1  # [(2708*2708)]
    weight_tensor = torch.ones(weight_mask.size(0))  # [(2708*2708)]
    weight_tensor[weight_mask] = pos_weight  # [(2708*2708)]

    # print('weight_mask', weight_mask.shape)
    # print('adj_label', adj_label.to_dense().view(-1).shape)
    # print('weight_tensor', weight_tensor.shape)
    # print('pos_weight', pos_weight)

    # init model and optimizer
    model = AttTSCN(opt)

    model.to(opt.device)
    data.to(opt.device)

    optimizer = Adam(model.parameters(), lr=opt.lr)

    def get_scores(edges_pos, edges_neg, adj_rec):
        # print(adj_rec)
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        # Predict on test set of edges
        preds = []
        pos = []  # 全1 标签
        # print(edges_pos)
        for e in edges_pos:
            # print(e)
            # print(adj_rec[e[0], e[1]])
            preds.append(adj_rec[e[0], e[1]].item())
            pos.append(adj_orig[e[0], e[1]])

        preds_neg = []
        neg = []
        for e in edges_neg:
            preds_neg.append(adj_rec[e[0], e[1]].item())
            neg.append(adj_orig[e[0], e[1]])

        # print("preds")
        # print(len(preds))
        # print("preds_neg")
        # print(len(preds_neg))

        preds_all = np.hstack([preds, preds_neg])
        labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
        # print('labels_all', labels_all)
        # print('preds_all', preds_all)

        roc_score = roc_auc_score(labels_all, preds_all)
        ap_score = average_precision_score(labels_all, preds_all)

        return roc_score, ap_score

    def get_acc(adj_rec, adj_label):
        labels_all = adj_label.to_dense().view(-1).long()
        preds_all = (adj_rec > 0.65).view(-1).long()
        accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
        return accuracy

    # train model
    for epoch in range(opt.max_epoch):
        t = time.time()
        A_pred, loss1 = model(data)
        optimizer.zero_grad()
        loss2 = norm * F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1).to(opt.device),
                                              weight=weight_tensor.to(opt.device))
        loss = opt.loss1_weight * loss1 + opt.loss2_weight * loss2
        loss.backward()
        optimizer.step()

        train_acc = get_acc(A_pred, adj_label.to(opt.device))

        val_roc, val_ap = get_scores(val_edges, val_edges_false, A_pred.cpu())

        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss.item()),
              "train_acc=", "{:.5f}".format(train_acc), "val_roc=", "{:.5f}".format(val_roc),
              "val_ap=", "{:.5f}".format(val_ap),
              "time=", "{:.5f}".format(time.time() - t))

    test_roc, test_ap = get_scores(test_edges, test_edges_false, A_pred.cpu())
    test_acc = get_acc(A_pred, adj_label.to(opt.device))
    print("End of training!", "test_roc=", "{:.5f}".format(test_roc),
          "test_ap=", "{:.5f}".format(test_ap))
