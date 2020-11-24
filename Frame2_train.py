# coding=utf-8
import torch
print(torch.cuda.is_available())

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import random
import numpy as np
from torch import nn
import torch.nn.functional as F
import itertools
from argparse import ArgumentParser
from dual_gnn.cached_gcn_conv import CachedGCNConv
from dataset.DomainData import DomainData
from dual_gnn.ppmi_conv import PPMIConv
from data_vis import degree_level_acc
from model.MRVGAE_model import VI,private_encoder,shared_encoder,private_decoder,shared_decoder,discriminator,classifier


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = ArgumentParser()
parser.add_argument("--source", type=str, default='dblp')
parser.add_argument("--seed", type=int,default=200)
parser.add_argument("--encoder_dim", type=int, default=16)

args = parser.parse_args()
seed = args.seed
encoder_dim = args.encoder_dim

id = "source: {}, target: {}, seed: {}, UDAGCN: {}, encoder_dim: {}"\
    .format(args.source, args.target, seed, use_UDAGCN,  encoder_dim)

print(id)

rate = 0.0
#要固定输出后两个需要；即np控制mask，而torch控制模型初始参数
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
dataset = DomainData("data/{}".format(args.source), name=args.source)
source_data = dataset[0]
print(source_data)
source_data = source_data.to(device)

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
        hj = h.unsqueeze(0).repeat(h.shape[0],1,1) #[N_i,N_j,Dh],向量值仅和j相关
        hi = h.unsqueeze(1),repeat(1,h.shape[0],1) #[N_i,N_j,Dh],向量值仅和i相关
        hij = torch.cat((hi,hj),2)
        # h = torch.mm(h,h.T)*hij #向量乘后元素乘
        z = self.VI(hij)
        h = self.shared_decoder(z)
        A = self.private_decoder(h)
        return A


loss_func = nn.CrossEntropyLoss().to(device)

models = VGAE(in_dim,hidden_dim)
optimizer = torch.optim.Adam(models.parameters(), lr=3e-3)

def train():
    models.train()
    optimizer.zero_grad()
    A_pred = models(source_data.x, source_data.edge_index)
    loss = loss_func(A_pred[source_data.train_mask], source_data.y[source_data.train_mask])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def test():
    pass

def predict():
    pass


#Train
epoch = 200
for i in range(epoch):
    train()
    acc = test()

print('------------End of training----------')