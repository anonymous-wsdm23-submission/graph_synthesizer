import torch
import numpy as np
import scipy.sparse as sp

def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1), dtype='f')
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize_adjacency(adj):

    deg = torch.sum(adj, dim=1) 

    deg_inv = deg.pow(-0.5) 

    deg_inv[deg_inv == float('inf')] = 0 

    deg_inv = deg_inv * torch.eye(adj.shape[0]).type(torch.FloatTensor) 

    adj = torch.matmul(deg_inv, adj)
    adj = torch.matmul(adj, deg_inv) 

    return adj
