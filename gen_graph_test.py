import dgl
import torch
import torch.nn as nn
import os
from dataset.DomainData import DomainData
from data_process import *

# dataset = DomainData("data/acm", name='acm')
# source_data = dataset[0]
# # features = source_data.x[0:5]
# # node_num = 5
# # edge_index = torch.tensor([[0,0,0,1,1,2,2,3],[0,2,4,1,4,2,3,3]])
# # node_label = torch.tensor([0,1,2,0,2])
# # g = dgl.graph((edge_index[0],edge_index[1]))
# # g = dgl.add_nodes(g,5)
# # print(g)
# node_num = source_data.x.shape[0]
# print(node_num)
# self_loop = torch.arange(node_num)
# self_loop = self_loop.unsqueeze(1).repeat(1,2)
# edge_index = source_data.edge_index
# print(edge_index)
# edge_index = torch.cat([edge_index.T,self_loop])
# print(edge_index)
# g = dgl.graph((edge_index.T[0],edge_index.T[1]))
# adj = g.adjacency_matrix()
# print(adj)

# x1 = torch.ones(1, 2)*3
# x2 = torch.ones(1, 2)*2
# class m(nn.Module):
#     def __init__(self):
#         super(m,self).__init__()

#         self.mlp1 = nn.Linear(2,2) 
#         nn.init.constant(self.mlp1.weight, 1)
#         nn.init.constant(self.mlp1.bias, 1)
#         self.mlp2 = nn.Linear(2,1) 
#         nn.init.constant(self.mlp2.weight, 0.5)
#         nn.init.constant(self.mlp2.bias, 2)
#     def forward(self,x):
#         x = torch.relu(self.mlp1(x))
#         x = torch.sigmoid(self.mlp2(x))
#         return x
# model =m()
# opt = torch.optim.SGD(model.parameters(), lr=10, momentum=0.9)
# opt.zero_grad()
# # linear.weight =torch.nn.parameter.Parameter(torch.tensor([[1.0,1.0]]),requires_grad = True)
# # linear.bias = torch.nn.parameter.Parameter(torch.tensor([1.0]),requires_grad = True)
# y1 = model(x1)
# y2 = model(x2)
# loss = torch.nn.functional.binary_cross_entropy(y1,torch.ones_like(y1))+torch.nn.functional.binary_cross_entropy(y2,torch.ones_like(y2))
# print(loss)
# print(model.mlp1.weight,model.mlp1.weight.grad)
# loss.backward()
# print(model.mlp1.weight,model.mlp1.weight.grad)
# opt.step()
# print(model.mlp1.weight,model.mlp1.weight.grad)
output = torch.randn(1, 5, requires_grad = True)
print(output)
