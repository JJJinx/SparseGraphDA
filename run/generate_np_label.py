# coding=utf-8
import torch
import os
import time
import psutil

import numpy as np
from torch import nn
from argparse import ArgumentParser
from data_processing.DomainData import DomainData
from data_processing.data_process import *
# TODO 查看是否每次生成的数据集是否会改变NP标签的顺序之类的


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
'''
dataset 
    :: x node feature
    :: y node label min=0 max=5
    :: edge_index node pair for existing edge, start from 0
    :: train_mask mask for nodes
    :: test_mask mask for nodes
'''
dataset = DomainData("data/{}".format(args.source), name=args.source)
source_data = dataset[0]
dataset = DomainData("data/{}".format(args.target), name=args.target)
target_data = dataset[0]

label_min_index = source_data.y.min()
label_max_index = source_data.y.max()
node_label_num = label_max_index-label_min_index+1
# mapping from node type to node pair type
# ntype_etype_mapping = torch.zeros([label_max_index-label_min_index+1,label_max_index-label_min_index+1],dtype=torch.long)
# i = 0
# for src_node_type in range(label_min_index,label_max_index+1):
#     for tgt_node_type in range(src_node_type,label_max_index+1):
#         ntype_etype_mapping[src_node_type,tgt_node_type] = i
#         ntype_etype_mapping[tgt_node_type,src_node_type] = i
#         i+=1
## data processing
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
# to avoid missing nodes, we should add all the self loop in the edge index and then build up the graph
self_loop = torch.arange(source_data.x.shape[0])
self_loop = self_loop.unsqueeze(1).repeat(1,2)
edge_index = source_data.edge_index
src_edge_index_sl = torch.cat([edge_index.T,self_loop]).T #[2,N]

self_loop = torch.arange(target_data.x.shape[0])
self_loop = self_loop.unsqueeze(1).repeat(1,2)
edge_index = target_data.edge_index
tgt_edge_index_sl = torch.cat([edge_index.T,self_loop]).T #[2,N]

source_node_num = source_data.x.shape[0]
target_node_num = target_data.x.shape[0]
print(u'当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024) )
src_all_node_pair,src_all_node_pair_label = generate_all_node_pair(source_node_num,src_edge_index_sl,source_data.y,node_label_num)
print(u'当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024) )
tgt_all_node_pair,tgt_all_node_pair_label = generate_all_node_pair(target_node_num,tgt_edge_index_sl,target_data.y,node_label_num)
print(u'当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024) )
np.savetxt('acm_all_node_pair_label.csv',src_all_node_pair_label.numpy(),delimiter=',')
np.savetxt('dblp_all_node_pair_label.csv',tgt_all_node_pair_label.numpy(),delimiter=',')
raise RuntimeError