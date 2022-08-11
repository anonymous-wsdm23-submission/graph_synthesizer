import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from scipy.sparse import csr_matrix

class GCNNet(nn.Module):

    def __init__(self, n_layer, in_dim, hidden_dim, out_dim, num_head, pred_dim, class_cardinality, dropout_rate, attention_dropout_rate, indices_neighbor):
        super(GCNNet, self).__init__()

        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(self.dropout_rate)

        self.num_head = num_head
 
        self.activation = nn.ELU()

        self.block = GCNBlock(n_layer, in_dim, hidden_dim, class_cardinality, self.activation, num_head, dropout_rate, attention_dropout_rate, indices_neighbor)

        self.loss = nn.CrossEntropyLoss()
    
    def forward(self, x, adj, target_X, target, is_val, epoch):
        x = x.unsqueeze(0) 
        tmp_list = list()

        for i in range(self.num_head):
            tmp_list.append(x+0) 

        x = torch.cat(tmp_list, dim=0) 

        out, adj, layers_alphalist = self.block(x, adj, is_val, epoch)

        # Average multi-head
        out = torch.sum(out, dim=0) 
        y = torch.div(out, self.num_head) 
       
        loss = self.loss(y[target_X], target)
    
        return loss, y[target_X], y, layers_alphalist
 
class GCNBlock(nn.Module):

    def __init__(self, n_layer, in_dim, hidden_dim, out_dim, act, num_head, dropout_rate, attention_dropout_rate, indices_neighbor):  
        super(GCNBlock, self).__init__()

        self.num_head = num_head 

        self.activation = act

        self.layers = nn.ModuleList()

        for i in range(n_layer):
            self.layers.append(GCNLayer(in_dim if i == 0 else hidden_dim//(2*i),
                                        out_dim if i == n_layer-1 else hidden_dim//(2*(i+1)), 
                                        self.activation if i != n_layer-1 else None, 
                                        num_head,
                                        dropout_rate, attention_dropout_rate, indices_neighbor))
        
    def forward(self, x, adj, is_val, epoch):
        layers_alphalist = []
        for i, layer in enumerate(self.layers):
            out, adj, heads_alphalist = layer((x if i == 0 else out), adj, is_val, epoch, i)
            layers_alphalist.append(heads_alphalist)

        return out, adj, layers_alphalist

class GCNLayer(nn.Module):

    def __init__(self, in_dim, out_dim, act, num_head, dropout_rate, attention_dropout_rate, indices_neighbor): 
        super(GCNLayer, self).__init__()

        self.num_head = num_head

        self.attention = Attention(out_dim, num_head, indices_neighbor, dropout_rate, attention_dropout_rate)
        self.activation = act

        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(self.dropout_rate)

        self.filter_matrix_list = nn.ModuleList()
        for _ in range(self.num_head):
            self.filter_matrix_list.append(nn.Linear(in_dim, out_dim))
    
        for layer in self.filter_matrix_list:
            nn.init.xavier_uniform_(layer.weight)


    def forward(self, x, adj, is_val, epoch, layer_position):
        num_head_chunks_of_x = torch.chunk(x, self.num_head, dim=0)

        x_list = list()
        
        for chunk in num_head_chunks_of_x:
            x_list.append(chunk.squeeze(0))

        tmp_out = list()

        for (layer, chunk) in zip(self.filter_matrix_list, x_list):
            tmp_out.append(layer(chunk))

        tmp_out2 = list()
        for chunk in tmp_out:
            tmp_out2.append(chunk.unsqueeze(0))

        out = torch.cat(tmp_out2, dim=0) 

        out, heads_alphalist = self.attention(out, adj, is_val, epoch, layer_position)

        if self.activation != None:
            out = self.activation(out) 

        if self.dropout_rate > 0:
            out = self.dropout(out)

        return out, adj, heads_alphalist

class Attention(nn.Module):

    def __init__(self, out_dim, num_head, indices_neighbor, dropout_rate, attention_dropout_rate):
        super(Attention, self).__init__()

        self.indices_neighbor = indices_neighbor

        self.num_head = num_head
        self.out_dim = out_dim

        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.softmax = nn.Softmax(dim=-1)

        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(self.dropout_rate)

        self.attention_dropout_rate = attention_dropout_rate

        self.att_pattern = list()
        for _ in range(self.num_head):
            self.att_pattern.append(torch.Tensor(18405, 18405).type(torch.cuda.FloatTensor))

        for att in self.att_pattern:
            nn.init.xavier_uniform_(att)
            att.required_grad = False
           
    def forward(self, x, adj, is_val, epoch, layer_position):
        shape = list(x.size()) 

        heads = list() 

        x_chunks = torch.chunk(x, self.num_head, dim=0) 

        heads_alphalist = []
        for i in range(self.num_head):
            x_transformed = x_chunks[i].squeeze(0) 

            att_score = self.leaky_relu(self.att_pattern[i])
            zero_vec = -9e15*torch.ones_like(att_score)
            att_score = torch.where(adj > 0, att_score, zero_vec)
            att_ratio = self.softmax(att_score)

            heads_alphalist.append(csr_matrix(att_ratio.clone().detach().cpu()))

            att_ratio = F.dropout(att_ratio, self.attention_dropout_rate, training=self.training)

            x_head = torch.matmul(att_ratio, x_transformed) 

            heads.append(x_head.unsqueeze(0)) 

        out = torch.cat(heads, dim=0) 

        return out, heads_alphalist


class Classifier(nn.Module):
    def __init__(self, in_dim, out_dim, act, dropout_rate):
        super(Classifier, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(self.dropout_rate)

        self.linear = nn.Linear(self.in_dim, self.out_dim)
        nn.init.xavier_uniform_(self.linear.weight)
        
        self.activation = act

    def forward(self, x):
        out = self.linear(x)
     
        if self.activation != None:
            out = self.activation(out)

        if self.dropout_rate > 0:
            out = self.dropout(out)
    
        return out






