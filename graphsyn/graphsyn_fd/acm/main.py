import torch
import torch.nn as nn

import pickle

import argparse

import numpy as np

import pandas as pd

from sklearn.metrics import recall_score
from utils import f1_score, accuracy

from norm import normalize_adjacency, normalize_features

from model import GCNNet

import random

if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ACM',
                        help='Dataset')
    parser.add_argument('--epoch', type=int, default=50,
                        help='Training Epochs')

    parser.add_argument('--lr', type=float, default=0.005,
                        help='learning rate')

    parser.add_argument('--weight_decay', type=float, default=0.005,
                        help='weight decay factor for l2 regularization')
    parser.add_argument('--n_layer', type=int, default=2,
                        help='number of task model layer')
    parser.add_argument('--num_head', type=int, default=3,
                        help='number of attention head')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='hidden dimension')

    parser.add_argument('--dropout_rate', type=float, default=0.5,
                        help='dropout rate')

    parser.add_argument('--attention_dropout_rate', type=float, default=0.5,
                        help='attention dropout rate')
   
    parser.add_argument('--count', type=int, default=20,
                        help='count -> number of experiments')

    parser.add_argument('--split', type=float, default=1.0,
                        help='split training dataset')


    args = parser.parse_args()
    print(args)
    epochs = args.epoch

    num_head = args.num_head
    lr = args.lr
    weight_decay = args.weight_decay
    n_layer = args.n_layer

    hidden_dim = args.hidden_dim
    dropout_rate = args.dropout_rate

    attention_dropout_rate = args.attention_dropout_rate

    count = args.count

    dataset = args.dataset

    with open('data/'+dataset+'/node_features.pkl','rb') as f:
        node_features = pickle.load(f)
    with open('data/'+dataset+'/edges.pkl','rb') as f:
        edges = pickle.load(f)
    with open('data/'+dataset+'/labels.pkl','rb') as f:
        labels = pickle.load(f)
    
    num_nodes = edges[0].shape[0]

    set_of_labels = set()

    for train_label in labels[0]:
        set_of_labels.add(train_label[1]) 

    set_of_labels = sorted(set_of_labels)

    class_cardinality = len(set_of_labels)

    for i, edge in enumerate(edges):
        if i == 0:
            A = torch.from_numpy(edge.todense())
        else:
            A += torch.from_numpy(edge.todense())

    A = A.type(torch.cuda.FloatTensor)

    A = normalize_adjacency(A)

    node_features = torch.from_numpy(node_features)
    node_features = node_features.type(torch.cuda.FloatTensor)

    in_dim = node_features.shape[1]

    valid_node = torch.from_numpy(np.array(labels[1])[:,0]).type(torch.cuda.LongTensor)
    valid_target = torch.from_numpy(np.array(labels[1])[:,1]).type(torch.cuda.LongTensor)
    test_node = torch.from_numpy(np.array(labels[2])[:,0]).type(torch.cuda.LongTensor)
    test_target = torch.from_numpy(np.array(labels[2])[:,1]).type(torch.cuda.LongTensor)

    class_cardinality = 3

    split_list = [1.0]
    for split in split_list:

        node_num_label_per_class = []
        for i in range(class_cardinality):
            node_num_label_per_class.append([]*1)
        for train_label in labels[0]:
            kind_of_label = set_of_labels.index(train_label[1])
            node_num_label_per_class[kind_of_label].append([train_label[0], train_label[1]])

        node_label_0 = node_num_label_per_class[0]
        node_label_1 = node_num_label_per_class[1]
        node_label_2 = node_num_label_per_class[2]

        random.seed(0)
        random.shuffle(node_label_0)
        random.seed(0)
        random.shuffle(node_label_1)
        random.seed(0)
        random.shuffle(node_label_2)

        train_node = []
        train_target = []

        splitted = int(split * 200)
        for (n0, n1, n2) in zip(node_label_0[:splitted], node_label_1[:splitted], node_label_2[:splitted]):
            train_node.append(n0[0])
            train_node.append(n1[0])
            train_node.append(n2[0])

            train_target.append(n0[1])
            train_target.append(n1[1])
            train_target.append(n2[1])

        train_node = torch.from_numpy(np.array(train_node)).type(torch.cuda.LongTensor)
        train_target = torch.from_numpy(np.array(train_target)).type(torch.cuda.LongTensor)

        best_train_losses = []
        best_train_f1s_macro = [] 
        best_train_f1s_micro = [] 
        best_val_losses = []
        best_val_f1s_macro = []
        best_val_f1s_micro = []
        best_test_losses = []
        best_test_f1s_macro = [] 
        best_test_f1s_micro = [] 
        best_epochs = []
        best_train_accs = []
        best_val_accs = []
        best_test_accs = []

        for cnt in range(count):
            best_val_loss = 10000
            best_test_loss = 10000
            best_train_loss = 10000
            best_train_f1_macro = 0
            best_train_f1_micro = 0
            best_val_f1_macro = 0
            best_val_f1_micro = 0
            best_test_f1_macro = 0
            best_test_f1_micro = 0
            best_epoch = 0
            best_train_acc = 0
            best_valid_acc = 0
            best_test_acc = 0

            best_out = None
    
            model = GCNNet(n_layer,
                        in_dim,
                        hidden_dim,
                        0,
                        num_head,
                        0,
                        class_cardinality,
                        dropout_rate,
                        attention_dropout_rate,
                        0)

            model.cuda()

            optimizer = torch.optim.Adam([
                                        {'params':model.parameters()}
                                    ], lr=lr, weight_decay=weight_decay)

        
            for i in range(epochs):
                print('Epoch: ',i+1)
                
                model.train()
                model.zero_grad()

                loss, y_train, _, _ = model(node_features, A, train_node, train_target, 0, 0)

                loss.backward()
                optimizer.step()
    
                train_acc = accuracy(torch.argmax(y_train,dim=1), train_target)
                train_f1_macro = torch.mean(torch.tensor(recall_score(torch.argmax(y_train,dim=1).detach().cpu(), train_target.detach().cpu(), average='macro')))
                train_f1_micro = torch.mean(torch.tensor(recall_score(torch.argmax(y_train,dim=1).detach().cpu(), train_target.detach().cpu(), average='micro')))
                print('Train - Loss: {}, Macro_F1: {}, Micro_F1: {}'.format(loss.detach().cpu().numpy(), train_f1_macro, train_f1_micro))
                model.eval()
                
                with torch.no_grad():
                    val_loss, y_valid, out, alpha_list = model.forward(node_features, A, valid_node, valid_target, 0, 0)
                    test_loss, y_test, _, _ = model.forward(node_features, A, test_node, test_target, 0, 0)

                    val_acc = accuracy(torch.argmax(y_valid,dim=1), valid_target)
                    val_f1_macro = torch.mean(torch.tensor(recall_score(torch.argmax(y_valid,dim=1).detach().cpu(), valid_target.detach().cpu(), average='macro')))
                    val_f1_micro = torch.mean(torch.tensor(recall_score(torch.argmax(y_valid,dim=1).detach().cpu(), valid_target.detach().cpu(), average='micro')))
                    print('Valid - Loss: {}, Macro_F1: {}, Micro_F1: {}'.format(val_loss.detach().cpu().numpy(), val_f1_macro, val_f1_micro))
    
                    test_acc = accuracy(torch.argmax(y_test,dim=1), test_target)
                    test_f1_macro = torch.mean(torch.tensor(recall_score(torch.argmax(y_test,dim=1).detach().cpu(), test_target.detach().cpu(), average='macro')))
                    test_f1_micro = torch.mean(torch.tensor(recall_score(torch.argmax(y_test,dim=1).detach().cpu(), test_target.detach().cpu(), average='micro')))
                    print('Test - Loss: {}, Macro_F1: {}, Micro_F1: {}, Acc: {}\n'.format(test_loss.detach().cpu().numpy(), test_f1_macro, test_f1_micro, test_acc))
    
                    if val_f1_macro > best_val_f1_macro:
                        best_val_loss = val_loss.detach().cpu().numpy()
                        best_test_loss = test_loss.detach().cpu().numpy()
                        best_train_loss = loss.detach().cpu().numpy()
                        best_train_f1_macro = train_f1_macro.item()
                        best_train_f1_micro = train_f1_micro.item()
                        best_val_f1_macro   = val_f1_macro.item()
                        best_val_f1_micro   = val_f1_micro.item()
                        best_test_f1_macro  = test_f1_macro.item()
                        best_test_f1_micro  = test_f1_micro.item()
                        best_train_acc = train_acc
                        best_val_acc = val_acc
                        best_test_acc = test_acc

                        best_epoch = i
                        best_out = out

            print('---------------Best Results--------------------')
            print('Train - Loss: {}, Macro_F1: {}, Micro_F1: {}'.format(best_train_loss, best_train_f1_macro, best_train_f1_micro))
            print('Valid - Loss: {}, Macro_F1: {}, Micro_F1: {}'.format(best_val_loss, best_val_f1_macro, best_val_f1_micro))
            print('Test - Loss: {}, Macro_F1: {}, Micro_F1: {}'.format(best_test_loss, best_test_f1_macro, best_test_f1_micro))
            
            with open("cnt%s_graphsyn_fd_weight.pickle"%(str(cnt)), "wb") as f:
                pickle.dump(best_weight, f)
            with open("cnt%s_graphsyn_fd_bias.pickle"%(str(cnt)), "wb") as f:
                pickle.dump(best_bias, f)
            with open("cnt%s_graphsyn_fd_alpha.pickle"%(str(cnt)), "wb") as f:
                pickle.dump(best_alphalist, f)

            best_train_losses.append(best_train_loss)
            best_train_f1s_macro.append(best_train_f1_macro)
            best_train_f1s_micro.append(best_train_f1_micro)
            best_val_losses.append(best_val_loss)
            best_val_f1s_macro.append(best_val_f1_macro)
            best_val_f1s_micro.append(best_val_f1_micro)
            best_test_losses.append(best_test_loss)
            best_test_f1s_macro.append(best_test_f1_macro)
            best_test_f1s_micro.append(best_test_f1_micro)
            best_epochs.append(best_epoch)
            best_train_accs.append(best_train_acc)
            best_val_accs.append(best_val_acc)
            best_test_accs.append(best_test_acc)

            result = {'best_epoch':best_epoch, 'best_train_f1s_macro':best_train_f1_macro, 'best_train_f1s_micro':best_train_f1_micro, 'best_valid_f1s_macro':best_val_f1_macro, 'best_test_f1s_macro':best_test_f1_macro, 'best_test_f1s_micro':best_test_f1_micro, 'best_train_accs':best_train_acc, 'best_valid_accs':best_val_acc, 'best_test_accs':best_test_acc, 'best_train_losses':best_train_loss, 'best_valid_losses':best_val_loss, 'best_test_losses':best_test_loss}
            df = pd.DataFrame.from_dict(result, orient='index')
            df = df.transpose()
            df.to_csv('./' + dataset + '_' + str(cnt) + '_result.csv', index=False)

        result = {'best_epoch':best_epochs, 'best_train_f1s_macro':best_train_f1s_macro, 'best_train_f1s_micro':best_train_f1s_micro, 'best_valid_f1s_macro':best_val_f1s_macro, 'best_test_f1s_macro':best_test_f1s_macro, 'best_test_f1s_micro':best_test_f1s_micro, 'best_train_accs':best_train_accs, 'best_valid_accs':best_val_accs, 'best_test_accs':best_test_accs, 'best_train_losses':best_train_losses, 'best_valid_losses':best_val_losses, 'best_test_losses':best_test_losses}
        df = pd.DataFrame.from_dict(result, orient='index')
        df = df.transpose()
        df.to_csv('./' + dataset + '_result.csv', index=False)
