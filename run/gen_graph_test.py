import dgl
import torch
import torch.nn as nn
import os
from DomainData import DomainData
from data_process import *


mapping_matrix = torch.tensor([
                  [0,0],
                  [1,0],
                  [1,1],
                  [0,1]],dtype = torch.long)
src = torch.tensor([0,1,2,3,0,1,2],dtype = torch.long)
dst = torch.tensor([0,1,2,3,3,2,3],dtype = torch.long)
g = dgl.graph((src,dst))
g = dgl.to_bidirected(g)


                      # 0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3
                      # 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3

np_pred = torch.tensor([1,0,0,0,0,3,2,0,0,2,1,2,2,0,2,3],dtype = torch.long)
#np_vote_node_label = mapping_matrix.gather(1,np_pred).view(5,5,-1)
np_vote_node_label = mapping_matrix[np_pred].view(4,4,-1)
print(np_vote_node_label.sum(dim=1))
node_pred = np_vote_node_label.sum(dim=1).argmax(dim=1)
print(node_pred)
# node_acc = node_pred.eq(node_label).float().mean()