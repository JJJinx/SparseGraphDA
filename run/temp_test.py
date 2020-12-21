import dgl
import torch
import torch.nn as nn
import os
from data_processing.DomainData import DomainData
from data_processing.data_process import *

def evaluate(np_pred,np_label,mapping_matrix,node_label): 
    node_pair_acc = np_pred.argmax(dim=1).eq(np_label).float().mean() # np_pred [np_num, np_type_num]

    np_pred = np_pred.argmax(dim=1) # [NP,]
    node_num = node_label.shape[0]
    np_vote_node_label = mapping_matrix[np_pred].view(node_num,node_num,-1)
    print(np_vote_node_label.sum(dim=1))
    node_pred = np_vote_node_label.sum(dim=1).argmax(dim=1) # node_pred :: reshape the np_vote_node_label to [N,N,node_label_num] 
                                                            # and sum by the dim=1 and get a tensor of [N, nodel_label_num] 
                                                            # then apply argmax get [N,] node label pred
    print(node_pred)

    node_acc = node_pred.eq(node_label).float().mean()
    return node_pair_acc,node_acc


def gmm(node_label_num,np_type_num,pos_np_type_num):
    mapping_M = torch.zeros((np_type_num,node_label_num),dtype=torch.long)
    
    for src in range(node_label_num):
        for dst in range(node_label_num):

            if src > dst:
                new_src = dst
                new_dst = src

            else:
                new_src = src
                new_dst = dst

            np_type = int((node_label_num+node_label_num-(new_src-1))*((new_src-1)+1)/2+(new_dst-new_src)+1)-1
            if src==dst:
                mapping_M[np_type,src] =2
            else:
                mapping_M[np_type,src] =1
    
    mapping_M[pos_np_type_num:,:] = mapping_M[:pos_np_type_num,:]
    return mapping_M

if __name__ == "__main__":
    node_label = torch.tensor([0,1,2,0,2])  
    edge_index = torch.tensor([[0,0,0,1,1,2,2,3,4],[0,2,4,1,4,2,3,3,4]])
    g = dgl.to_bidirected(dgl.graph((edge_index[0],edge_index[1])))
    #M = gmm(3,12,6)
    M = generate_mapping_M_minus_class(3,12,6)
    #node_num,edge_index,node_label,node_label_num,adj
    node_pair,node_pair_label,max_pos_np_label,max_neg_np_label = generate_all_node_pair_minus_class(5,edge_index,node_label,3,g.adjacency_matrix())

    #np_pred = torch.randn((25,max_neg_np_label+1))
    np_pred = torch.zeros((25,max_neg_np_label+1),dtype=torch.long)
    #print(node_pair_label.view(-1))
    ############################# [ 0,  7,  2,  6,  2,  7,  3, 10,  7,  4,  2, 10,  5,  2, 11,  6,  7,  2,0,  8,  2,  4, 11,  8,  5]
    np_pred_indice = torch.tensor([ 0,  1,  3,  6,  2,  7,  3, 10,  2,  4,  7,  3,  8,  4, 11, 10,  2,  2, 8,  2,  8,  5,  6,  6, 10])
    for i in range(25):
        np_pred[i,np_pred_indice[i]] = 1
    npacc,nacc = evaluate(np_pred,node_pair_label.view(-1),M,node_label)
    print('np',npacc,'node',nacc)