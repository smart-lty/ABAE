import torch as t
import torch_geometric.utils as utils


def qw_score(graph):
    """
    未实现qw_score,采用度数代替
    :param graph:
    """
    score = utils.degree(graph.edge_index[0])
    return score.sort()


def pre_processing(graph, m, score, trees):
    score, indices = score
    indices.squeeze_()
    old_edges = graph.edge_index
    trees[-1] = [-1] * m

    def graft(root):
        """
        找到分值最大的2阶节点并与源节点连接
        和论文有一些不一样，会在加入二阶节点后把它视为一阶节点
        :param root: 源节点(度小于m)
        """
        nodes_1_hop, _, _, _ = utils.k_hop_subgraph(root, 1, graph.edge_index)
        if nodes_1_hop.shape[0] > m:
            return
        nodes_2_hop, _, _, _ = utils.k_hop_subgraph(root, 2, graph.edge_index)
        ma = 0
        for node in nodes_2_hop:
            if node not in nodes_1_hop:
                node = int(node.item())
                idx = t.nonzero(indices == node, as_tuple=False).item()
                ma = max(ma, idx)
        new_edge = t.tensor([[indices[ma], root], [root, indices[ma]]])
        degree[root] += 1
        graph.edge_index = t.cat((graph.edge_index, new_edge), dim=1)
        if degree[root] < m:
            graft(root)
        elif degree[root] == m:
            nodes_1_hop, _, _, _ = utils.k_hop_subgraph(root, 1, graph.edge_index)
            trees[root] = ([i.item() for i in nodes_1_hop if i != root])
            graph.edge_index = old_edges

    def prune(root):
        """
        找到分值最小的1阶节点并删除连接
        默认图为简单图
        :param root: 源节点
        """
        nodes_1_hop, _, _, mask = utils.k_hop_subgraph(root, 1, graph.edge_index)
        if nodes_1_hop.shape[0] == m + 1:
            return
        mi = graph.num_nodes + 1
        for node in nodes_1_hop:
            if node != root:
                node = int(node.item())
                idx = t.nonzero(indices == node, as_tuple=False).item()
                mi = min(idx, mi)
        mask = mask.nonzero(as_tuple=False)
        edges = graph.edge_index
        l, r = 0, 0
        for i in mask:
            i = i.item()
            if edges[0][i] == indices[mi] and edges[1][i] == root:
                l = i
            elif edges[1][i] == indices[mi] and edges[0][i] == root:
                r = i
        l, r = sorted([l, r])
        graph.edge_index = t.cat((edges[:, :l], edges[:, l + 1:r], edges[:, r + 1:]), dim=1)
        degree[root] -= 1
        if degree[root] > m:
            prune(root)
        elif degree[root] == m:
            nodes_1_hop, _, _, _ = utils.k_hop_subgraph(root, 1, graph.edge_index)
            trees[root] = ([i.item() for i in nodes_1_hop if i != root])
            graph.edge_index = old_edges

    degree = utils.degree(graph.edge_index[0])
    for node, d in enumerate(degree):
        tmp = degree[node]
        if d > m:
            prune(node)
        elif d < m:
            graft(node)
        else:
            nodes_1_hop, _, _, _ = utils.k_hop_subgraph(node, 1, graph.edge_index)
            trees[node] = ([i.item() for i in nodes_1_hop if i != node])
        degree[node] = tmp
    for tree in trees:
        while len(trees[tree]) < m:
            trees[tree].append(-1)
            # 对于孤立点对它的子树加哑节点
    graph.edge_index = old_edges
    return trees


def construct_node_tree(graph, node, trees, opt):
    """
    生成目标节点的 K_level, m_ary 树
    :param graph:
    :param node:
    :param opt:
    """
    m = opt.m
    K = opt.K
    tree = [node]
    now = 0
    for i in range(K - 1):
        for j in range(m ** i):
            root = tree[now]
            tree += trees[root]
            now += 1
    zero = t.zeros(graph.x[-1].shape)
    x = graph.x
    graph.x = t.cat([graph.x, zero[None, :]], dim=0)
    tree = graph.x[tree]
    graph.x = x
    return tree
