import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from snntorch import spikegen


cuda = True if torch.cuda.is_available() else False


def prepare_trte_data(data_folder, view_list):
    num_view = len(view_list)
    labels_tr = np.loadtxt(os.path.join(data_folder, "labels_tr.csv"), delimiter=",")
    labels_te = np.loadtxt(os.path.join(data_folder, "labels_te.csv"), delimiter=",")

    labels_tr = labels_tr.astype(int)
    labels_te = labels_te.astype(int)

    data_tr_list = []
    data_te_list = []

    for i in view_list:
        data_tr_list.append(
            np.loadtxt(os.path.join(data_folder, str(i) + "_tr.csv"), delimiter=",")
        )
        data_te_list.append(
            np.loadtxt(os.path.join(data_folder, str(i) + "_te.csv"), delimiter=",")
        )

    num_tr = data_tr_list[0].shape[0]
    num_te = data_te_list[0].shape[0]

    data_mat_list = []
    for i in range(num_view):
        data_mat_list.append(np.concatenate((data_tr_list[i], data_te_list[i]), axis=0))
    data_tensor_list = []
    for i in range(len(data_mat_list)):
        data_tensor_list.append(torch.FloatTensor(data_mat_list[i]))
        if cuda:
            data_tensor_list[i] = data_tensor_list[i].cuda()

    idx_dict = {}
    idx_dict["tr"] = list(range(num_tr))
    idx_dict["te"] = list(range(num_tr, (num_tr + num_te)))
    data_train_list = []
    data_all_list = []

    for i in range(len(data_tensor_list)):
        data_train_list.append(data_tensor_list[i][idx_dict["tr"]].clone())
        data_all_list.append(
            torch.cat(
                (
                    data_tensor_list[i][idx_dict["tr"]].clone(),
                    data_tensor_list[i][idx_dict["te"]].clone(),
                ),
                0,
            )
        )
    labels = np.concatenate((labels_tr, labels_te))

    return data_train_list, data_all_list, idx_dict, labels


def gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter):
    adj_metric = "cosine"  # cosine distance
    adj_train_list = []
    adj_test_list = []
    for i in range(len(data_tr_list)):
        adj_parameter_adaptive = cal_adj_mat_parameter(
            adj_parameter, data_tr_list[i], adj_metric
        )
        adj_train_list.append(
            gen_adj_mat_tensor(data_tr_list[i], adj_parameter_adaptive, adj_metric)
        )
        adj_test_list.append(
            gen_test_adj_mat_tensor(
                data_trte_list[i], trte_idx, adj_parameter_adaptive, adj_metric
            )
        )

    return adj_train_list, adj_test_list


def cal_sample_weight(labels, num_class, use_sample_weight=True):
    if not use_sample_weight:
        return np.ones(len(labels)) / len(labels)
    count = np.zeros(num_class)
    for i in range(num_class):
        count[i] = np.sum(labels == i)
    sample_weight = np.zeros(labels.shape)
    for i in range(num_class):
        sample_weight[np.where(labels == i)[0]] = count[i] / np.sum(count)

    return sample_weight


def one_hot_tensor(y, num_dim):
    y_onehot = torch.zeros(y.shape[0], num_dim)
    y_onehot.scatter_(1, y.view(-1, 1), 1)

    return y_onehot


def cosine_distance_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)


def to_sparse(x):
    x_typename = torch.typename(x).split(".")[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)
    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())


def cal_adj_mat_parameter(edge_per_node, data, metric="cosine"):
    assert metric == "cosine", "Only cosine distance implemented"
    dist = cosine_distance_torch(data, data)
    parameter = torch.sort(
        dist.reshape(
            -1,
        )
    ).values[edge_per_node * data.shape[0]]
    return np.ndarray.item(parameter.data.cpu().numpy())


def graph_from_dist_tensor(dist, parameter, self_dist=True):
    if self_dist:
        assert dist.shape[0] == dist.shape[1], "Input is not pairwise dist matrix"
    g = (dist <= parameter).float()
    if self_dist:
        diag_idx = np.diag_indices(g.shape[0])
        g[diag_idx[0], diag_idx[1]] = 0

    return g


def gen_adj_mat_tensor(data, parameter, metric="cosine"):
    assert metric == "cosine", "Only cosine distance implemented"
    dist = cosine_distance_torch(data, data)
    g = graph_from_dist_tensor(dist, parameter, self_dist=True)
    if metric == "cosine":
        adj = 1 - dist
    else:
        raise NotImplementedError
    adj = adj * g
    adj_T = adj.transpose(0, 1)
    I = torch.eye(adj.shape[0])
    if cuda:
        I = I.cuda()
    adj = adj + adj_T * (adj_T > adj).float() - adj * (adj_T > adj).float()
    adj = F.normalize(adj + I, p=1)
    adj = to_sparse(adj)

    return adj


def gen_test_adj_mat_tensor(data, trte_idx, parameter, metric="cosine"):
    assert metric == "cosine", "Only cosine distance implemented"
    adj = torch.zeros((data.shape[0], data.shape[0]))
    if cuda:
        adj = adj.cuda()
    num_tr = len(trte_idx["tr"])

    dist_tr2te = cosine_distance_torch(data[trte_idx["tr"]], data[trte_idx["te"]])
    g_tr2te = graph_from_dist_tensor(dist_tr2te, parameter, self_dist=False)
    if metric == "cosine":
        adj[:num_tr, num_tr:] = 1 - dist_tr2te
    else:
        raise NotImplementedError
    adj[:num_tr, num_tr:] = adj[:num_tr, num_tr:] * g_tr2te

    dist_te2tr = cosine_distance_torch(data[trte_idx["te"]], data[trte_idx["tr"]])
    g_te2tr = graph_from_dist_tensor(dist_te2tr, parameter, self_dist=False)
    if metric == "cosine":
        adj[num_tr:, :num_tr] = 1 - dist_te2tr
    else:
        raise NotImplementedError
    adj[num_tr:, :num_tr] = adj[num_tr:, :num_tr] * g_te2tr  # retain selected edges

    adj_T = adj.transpose(0, 1)
    I = torch.eye(adj.shape[0])
    if cuda:
        I = I.cuda()
    adj = adj + adj_T * (adj_T > adj).float() - adj * (adj_T > adj).float()
    adj = F.normalize(adj + I, p=1)
    adj = to_sparse(adj)

    return adj


def get_all_data_adj(
    data_folder,
    view_list,
    num_class,
):
    """
    Get all data and adjacency matrices.

    Parameters:
    - data_folder: str, path to the data folder
    - view_list: list, list of view indices
    - num_class: int, number of classes

    Returns:
    - data_tr_list: list, list of training data tensors for each view
    - data_trte_list: list, list of training and testing data tensors for each view
    - trte_idx: dict, dictionary of training and testing indices
    - labels_trte: array, labels for training and testing data
    - labels_tr_tensor: Tensor, training labels
    - onehot_labels_tr_tensor: Tensor, one-hot encoded training labels
    - adj_tr_list: list, list of training adjacency matrices for each view
    - adj_te_list: list, list of testing adjacency matrices for each view
    - dim_list: list, list of dimensions of the input data

    """

    num_view = len(view_list)  # number of views

    # if data_folder == "BRCA":
    adj_parameter = 10

    data_tr_list, data_trte_list, trte_idx, labels_trte = prepare_trte_data(
        data_folder, view_list
    )  # prepare training and testing data

    labels_tr_tensor = torch.LongTensor(labels_trte[trte_idx["tr"]])  # training labels
    onehot_labels_tr_tensor = one_hot_tensor(
        labels_tr_tensor, num_class
    )  # one-hot encoding of the training labels

    sample_weight_tr = cal_sample_weight(
        labels_trte[trte_idx["tr"]], num_class
    )  # sample weights for the training samples
    sample_weight_tr = torch.FloatTensor(sample_weight_tr)  # convert to tensor

    if cuda:
        labels_tr_tensor = labels_tr_tensor.cuda()
        onehot_labels_tr_tensor = onehot_labels_tr_tensor.cuda()
        sample_weight_tr = sample_weight_tr.cuda()

    adj_tr_list, adj_te_list = gen_trte_adj_mat(
        data_tr_list, data_trte_list, trte_idx, adj_parameter
    )  # generate adjacency matrices for training and testing data
    dim_list = [x.shape[1] for x in data_tr_list]  # dimensions of the input data

    return (
        data_tr_list,
        data_trte_list,
        trte_idx,
        labels_trte,
        labels_tr_tensor,
        onehot_labels_tr_tensor,
        adj_tr_list,
        adj_te_list,
        dim_list,
    )


def compute_spiking_node_representation(data, adj, K, num_steps, enc_type="rate"):
    """
    Compute the spiking node representations for a graph (ZHU et al.)

    Parameters:
    - data: Tensor, node features (X) for single view
    - adj: Tensor, adjacency matrix (A)
    - K: int, number of graph convolution layers
    - num_steps: int, number of steps for spike encoding
    - enc_type: str, spike encoding type (rate or latency)

    Returns:
    - H_enc: Tensor, spike encoded node representations in shape (num_steps, num_nodes, num_features)
    """

    assert enc_type in ["rate", "latency"], "Invalid spike encoding type"

    # D
    # degree matrix (D) of adj
    degree_matrix_adj = adj.sum(1).to_dense()

    # S = adj normalized
    # normalize adjacency matrix
    degree_matrix_adj_sqrt = torch.sqrt(degree_matrix_adj)
    degree_matrix_adj_inv_sqrt = 1.0 / degree_matrix_adj_sqrt

    adj_normalized = (
        adj
        * degree_matrix_adj_inv_sqrt.unsqueeze(0)
        * degree_matrix_adj_inv_sqrt.unsqueeze(1)
    )

    # H = S^K * X
    # propagate node features through K layers
    H = data
    for i in range(K):
        H = torch.mm(adj_normalized, H)

    # spike encoding
    if enc_type == "rate":
        H_enc = spikegen.rate(H, num_steps=num_steps)
    elif enc_type == "latency":
        H_enc = spikegen.latency(H, num_steps=num_steps)

    # return H_enc.permute(1, 2, 0)
    return H_enc
