import dgl
import torch
import os
from dataset.DomainData import DomainData
from data_process import *

dataset = DomainData("data/acm", name='acm')
source_data = dataset[0]
# features = source_data.x[0:5]
# node_num = 5
# edge_index = torch.tensor([[0,0,0,1,1,2,2,3],[0,2,4,1,4,2,3,3]])
# node_label = torch.tensor([0,1,2,0,2])
# g = dgl.graph((edge_index[0],edge_index[1]))
# g = dgl.add_nodes(g,5)
# print(g)
node_num = source_data.x.shape[0]
print(node_num)
self_loop = torch.arange(node_num)
self_loop = self_loop.unsqueeze(1).repeat(1,2)
edge_index = source_data.edge_index
print(edge_index)
edge_index = torch.cat([edge_index.T,self_loop])
print(edge_index)
g = dgl.graph((edge_index.T[0],edge_index.T[1]))
adj = g.adjacency_matrix()
print(adj)
