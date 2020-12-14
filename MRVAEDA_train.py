# coding=utf-8
import torch
print(torch.cuda.is_available())

import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import random
import numpy as np
from torch import nn
import torch.nn.functional as F
import itertools
from argparse import ArgumentParser
from dataset.DomainData import DomainData
from data_vis import degree_level_acc
from model.MRVGAE_model import *
from data_process import *
#print all value
torch.set_printoptions(profile="full")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = ArgumentParser()
parser.add_argument("--source", type=str, default='acm')
parser.add_argument("--target", type=str, default='dblp')
parser.add_argument("--seed", type=int,default=200)
parser.add_argument("--encoder_dim", type=int, default=16)

args = parser.parse_args()
seed = args.seed
encoder_dim = args.encoder_dim

id = "source: {}, target: {}, seed: {}, encoder_dim: {}"\
    .format(args.source, args.target, seed,  encoder_dim)

print(id)

rate = 0.0
#要固定输出后两个需要；即np控制mask，而torch控制模型初始参数
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
## input data
dataset = DomainData("data/{}".format(args.source), name=args.source)
source_data = dataset[0]
dataset = DomainData("data/{}".format(args.target), name=args.target)
target_data = dataset[0]
'''
dataset 
    :: x node feature
    :: y node label min=0 max=5
    :: edge_index node pair for existing edge, start from 0
    :: train_mask mask for nodes
    :: test_mask mask for nodes
'''
label_min_index = source_data.y.min()
label_max_index = source_data.y.max()
# mapping from node type to node pair type
ntype_etype_mapping = torch.zeros([label_max_index-label_min_index+1,label_max_index-label_min_index+1],dtype=torch.long)
i = 0
for src_node_type in range(label_min_index,label_max_index+1):
    for tgt_node_type in range(src_node_type,label_max_index+1):
        ntype_etype_mapping[src_node_type,tgt_node_type] = i
        ntype_etype_mapping[tgt_node_type,src_node_type] = i
        i+=1
## data processing
# to avoid missing nodes, we should add all the self loop in the edge index and then build up the graph
self_loop = torch.arange(source_data.x.shape[0])
self_loop = self_loop.unsqueeze(1).repeat(1,2)
edge_index = source_data.edge_index
src_edge_index_sl = torch.cat([edge_index.T,self_loop]).T #[2,N]

self_loop = torch.arange(target_data.x.shape[0])
self_loop = self_loop.unsqueeze(1).repeat(1,2)
edge_index = target_data.edge_index
tgt_edge_index_sl = torch.cat([edge_index.T,self_loop]).T #[2,N]
'''
require
    source graph
        all node pair :: all node pair represent by [src_node_idx,dst_node_idx]
        node pair type :: determined by wheter existing edge, src_node_type and dst_node_type
    target graph
        ::
    mini_batch
        batch node pair :: [src_node_idx,dst_node_idx]
'''
source_node_num = source_data.x.shape[0]
target_node_num = target_data.x.shape[0]
input_dim = source_data.x.shape[1]
print(source_data.edge_index.shape)
print(target_data.edge_index.shape)
t = time.time()
src_all_node_pair,src_all_node_pair_label = generate_all_node_pair(source_node_num,src_edge_index_sl,source_data.y,ntype_etype_mapping)
tgt_all_node_pair,tgt_all_node_pair_label = generate_all_node_pair(target_node_num,tgt_edge_index_sl,target_data.y,ntype_etype_mapping)
print(src_all_node_pair_label)
print(time.time()-t)
np.savetxt('acm_all_node_pair_label.csv',src_all_node_pair_label.numpy(),delimiter=',')
np.savetxt('dblp_all_node_pair_label.csv',tgt_all_node_pair_label.numpy(),delimiter=',')
raise RuntimeError
## generate train graph
source_graph = dgl.graph(src_edge_index_sl[0],src_edge_index_sl[1])
target_graph = dgl.graph(tgt_edge_index_sl[0],tgt_edge_index_sl[1])
source_graph.ndata['feats'] = source_data.x
target_graph.ndata['feats'] = target_data.x
## model
in_dim = dataset.num_features
hidden_dim = [] #TODO
class VGAE(torch.nn.Module):
    def __init__(self, in_dim,hidden_dim, **kwargs):
        super(VGAE, self).__init__()
        self.private_encoder = private_encoder(in_dim,hidden_dim[0])
        self.shared_encoder = shared_encoder(hidden_dim[0],hidden_dim[1])
        self.VI = VI(hidden_dim[1],hidden_dim[2])
        self.shared_decoder = shared_decoder(hidden_dim[2],hidden_dim[3])
        self.private_decoder = private_decoder(hidden_dim[3],hidden_dim[4])

    def forward(self, x, edge_index):
        x = self.private_encoder(x,edge_index) 
        h = self.shared_encoder(x,edge_index)
        # h = torch.mm(h,h.T)*hij 
        z = self.VI(hij)
        h = self.shared_decoder(z)
        A = self.private_decoder(h)
        return A


loss_func = nn.CrossEntropyLoss().to(device)

models = VGAE(in_dim,hidden_dim)
optimizer = torch.optim.Adam(models.parameters(), lr=3e-3)

## training
def train():
    models.train()
    optimizer.zero_grad()
    A_pred = models(source_data.x, source_data.edge_index)
    # enc

    src_idx = node_pairs[0]
    dst_idx = node_pairs[1]
    h_src = H[src_idx]  # [num_src,dh]
    h_dst = H[dst_idx]  # [num_dst,dh]
    h_src = h_src.unsqueeze(0).repeat(h.shape[0],1,1) #[N_i,N_j,Dh]
    h_dst = h_dst.unsqueeze(1).repeat(1,h.shape[0],1) #[N_i,N_j,Dh]
    hadd = h_src+h_dst
    hcat = torch.cat((h_src,h_dst),2)
    #VI

    #dec

    #


    loss = loss_func(A_pred[source_data.train_mask], source_data.y[source_data.train_mask])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def test():
    pass

def predict():
    pass



print('------------End of training----------')