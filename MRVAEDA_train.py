# coding=utf-8
import os
import time
import itertools
from argparse import ArgumentParser
import random
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from dataset.DomainData import DomainData
from data_vis import degree_level_acc
from model.MRVGAE_model import *
from data_process import *

class Node_Pair_Dataset(Dataset):
    def __init__(self,node_pairs,node_pair_labels):
        super(Node_Pair_Dataset, self).__init__()
        self.x = node_pairs # shape [N,2]
        self.y = node_pair_labels # shape [N,]
    def __getitem__(self,index):
        return self.x[index],self.y[index]
    def __len__(self):
        return self.x.shape[0]

## the da is discriminate the hidden embedding instead of the reconstruction result
class MRVAEDA(torch.nn.Module):
    def __init__(self, in_dim,hidden_dim,categorical_dim,device, **kwargs):
        super(MRVAEDA, self).__init__()
        self.private_encoder_source = private_encoder(in_dim,hidden_dim[0])
        self.private_encoder_target = private_encoder(in_dim,hidden_dim[0])
        self.shared_encoder = shared_encoder(hidden_dim[0],hidden_dim[1])
        self.VI = VI(hidden_dim[1],hidden_dim[2],categorical_dim,device,distype='Both',**kwargs)
        self.shared_decoder = shared_decoder(hidden_dim[3],hidden_dim[4])
        self.private_decoder_source = private_decoder(hidden_dim[4],hidden_dim[5],in_dim) # 2layers
        self.private_decoder_target = private_decoder(hidden_dim[4],hidden_dim[5],in_dim) # 2layers
        self.A_classifier = relation_classifier(hidden_dim[3],hidden_dim[3]//2,categorical_dim) #classify the node pair's class
        self.discriminator = discriminator(hidden_dim[3],hidden_dim[3]//2)
    def forward(self, x, edge_index,node_pair,domain):
        '''
            x :: shape [node_num,feat_dim]
            node_pair :: shape [batch_size,2]
        '''
        x = self.private_encoder_source(x,edge_index,domain) 
        h = self.shared_encoder(x,edge_index,domain) # [node_num,hidden_dim[1]]
        # h_src = h[node_pair[:,0]]  # [batch_size,dh] 
        # h_dst = h[node_pair[:,1]]  # [batch_size,dh]
        # h_src = h_src.unsqueeze(0).repeat(h.shape[0],1,1) #[N_i,N_j,Dh]
        # h_dst = h_dst.unsqueeze(1).repeat(1,h.shape[0],1) #[N_i,N_j,Dh]
        hadd = h[node_pair[:,0]]+h[node_pair[:,1]]
        M,mean,logstd,q = self.VI(hadd,temp = 0.5)
        x_recon = self.shared_decoder(M)
        x_recon = self.private_decoder_source(x_recon)
        A_pred = self.A_classifier(M)
        domain_pred = self.discriminator(M)
        return x_recon,A_pred,domain_pred,mean,logstd,q
    def inference(self): # TODO
        pass

# def focal_loss(input,targets,device,alpha=.25,gamma=2):
#     BCE_loss = F.binary_cross_entropy_with_logits(input,targets)
#     targets = targets.type(torch.long)
#     # alpha is the weight for different class
#     alpha = torch.tensor([alpha,1-alpha]).to(device)
#     print(alpha)
#     at = alpha.gather(0,target.data.view(-1))
#     print(at)
#     pt = torch.exp(-BCE_loss)
#     F_loss = at*(1-pt)**gamma * BCE_loss
#     return F_loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1).long()
        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target) # torch.gather(logpt,dim=1,index=target)
        logpt = logpt.view(-1)
        pt = torch.exp(logpt)

        if self.alpha is not None:
            if self.alpha.type()!=input.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.view(-1))
            logpt = logpt * at

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


def train(epochs,src_dataloader,tgt_dataloader,source_data,target_data,device):
    models.train()
    for epoch in range(epochs): # start from 0
        for batch_idx,data in enumerate(zip(src_dataloader,tgt_dataloader)):
            # TODO 在data process中将正负样本分开来，或者得到正负样本分别的index list
            # 每个批次有大量的负样本，也就是没有边的样本 1024中仅有30个左右正样本
            #能否改进采样方法，使得每次采样正负样本数量相同
            ## get the output from dataloader
            # node pair [src_node_index,dst_node_index]
            src_batch_np,src_batch_np_label = data[0]
            tgt_batch_np,tgt_batch_np_label = data[1]
            src_batch_np = src_batch_np.to(device)
            tgt_batch_np = tgt_batch_np.to(device)
            ## make one hot label
            src_batch_np_label = src_batch_np_label.to(device)
            tgt_batch_np_label = tgt_batch_np_label.to(device)
            ## build the reconstruction target
            src_recon_label = source_data.x[src_batch_np[:,0]]+ source_data.x[src_batch_np[:,1]]
            tgt_recon_label = target_data.x[tgt_batch_np[:,0]]+ target_data.x[tgt_batch_np[:,1]]
            ## put into the model
            src_batch_np = src_batch_np.to(device)
            tgt_batch_np = tgt_batch_np.to(device)
            # TODO  加入随epoch自适应的temp参数
            src_X_recon,src_A_pred,src_domain_pred,src_mean,src_logstd,src_q = models(source_data.x, source_data.edge_index,src_batch_np,domain='source')
            tgt_X_recon,tgt_A_pred,tgt_domain_pred,tgt_mean,tgt_logstd,tgt_q = models(target_data.x, target_data.edge_index,tgt_batch_np,domain='target')
            ## source domain cls focal loss
            focal_loss = FocalLoss(gamma=5).to(device) # there are a lot of classes so we do not give the alpha
            loss_cls = focal_loss(src_A_pred,src_batch_np_label)

            ## reconstruction loss
            loss_recon = torch.nn.functional.l1_loss(src_X_recon,src_recon_label)+ torch.nn.functional.l1_loss(tgt_X_recon,tgt_recon_label)
            #loss_recon = torch.nn.functional.mse_loss(src_X_recon,src_recon_label)+ torch.nn.functional.mse_loss(tgt_X_recon,tgt_recon_label)
            ## domain discriminator loss
            source_da_loss = F.cross_entropy(
                src_domain_pred,
                torch.zeros(src_domain_pred.size(0),dtype=torch.long).to(device)
            )
            target_da_loss = F.cross_entropy(
                tgt_domain_pred,
                torch.ones(tgt_domain_pred.size(0),dtype=torch.long).to(device)
            )
            loss_da = source_da_loss+target_da_loss
            ## KL loss
            mean = torch.cat([src_mean,tgt_mean])
            logstd = torch.cat([src_logstd,tgt_logstd]) #将正负样本沿dim0拼接后取平均
            kl_norm= torch.mean(-0.5*torch.sum(1+2*logstd-mean**2-logstd.exp()**2,dim=1),dim=0) 

            q = torch.cat([src_q,tgt_q])
            q_for_kl = F.softmax(q,dim=1)
            eps = 1e-20
            h1 = -q_for_kl*torch.log(q_for_kl + eps)#h(p)
            h2 = -q_for_kl*np.log(1./categorical_dim) #h(pq)
            kl_gumbel = torch.mean(torch.sum(h2-h1,dim=1),dim=0)
            loss_kl = kl_gumbel+kl_norm
            # backward
            loss = loss_cls+loss_recon+loss_da+loss_kl
            print(loss_cls.item,loss_recon.item,loss_da.item,loss_kl.item)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
# TODO
def test():
    pass

if __name__ == "__main__":
    # #print all value
    # torch.set_printoptions(profile="full")

    device = 'cpu'#torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = ArgumentParser()
    parser.add_argument("--source", type=str, default='acm')
    parser.add_argument("--target", type=str, default='dblp')
    parser.add_argument("--seed", type=int,default=200)
    parser.add_argument("--lr", type=float,default=0.001)

    args = parser.parse_args()
    seed = args.seed

    id = "source: {}, target: {}, seed: {}"\
        .format(args.source, args.target, seed)
    print(id)

    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    ## input data
    '''
    dataset 
        :: x node feature
        :: y node label min=0 max=5
        :: edge_index node pair for existing edge, start from 0  TODO edge_index中现在只有单向边，应当使其成为双向的边
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
    ## generate train graph
    # TODO 需要将之前的数据形式改为graph的形式，不然模型的gcn没法处理
    source_graph = dgl.graph((src_edge_index_sl[0],src_edge_index_sl[1]))
    target_graph = dgl.graph((tgt_edge_index_sl[0],tgt_edge_index_sl[1]))
    # source_graph.ndata['feats'] = source_data.x
    # target_graph.ndata['feats'] = target_data.x
    # source_graph.ndata['nlabel'] = source_data.y
    # target_graph.ndata['nlabel'] = target_data.y
    ##generate all node pair label
    source_node_num = source_data.x.shape[0]
    target_node_num = target_data.x.shape[0]
    src_all_node_pair,src_all_node_pair_label,max_np_label =generate_all_node_pair(source_node_num,src_edge_index_sl,source_data.y,
                                                                                    node_label_num,source_graph.adjacency_matrix()) # tensor,tensor
    src_all_node_pair = src_all_node_pair.view(-1,2)
    src_all_node_pair_label = src_all_node_pair_label.view(-1)
    tgt_all_node_pair,tgt_all_node_pair_label,max_np_label = generate_all_node_pair(target_node_num,tgt_edge_index_sl,target_data.y,
                                                                                    node_label_num,target_graph.adjacency_matrix())
    tgt_all_node_pair = tgt_all_node_pair.view(-1,2)
    tgt_all_node_pair_label = tgt_all_node_pair_label.view(-1)

    min_np_label = 0 # no_edge_existence
    max_np_label = max_np_label.item() 
    #np.savetxt('acm_all_node_pair_label.csv',src_all_node_pair_label.numpy(),delimiter=',')
    #np.savetxt('dblp_all_node_pair_label.csv',tgt_all_node_pair_label.numpy(),delimiter=',')
    ## dataloader
    source_np_dataset = Node_Pair_Dataset(src_all_node_pair,src_all_node_pair_label)
    target_np_dataset = Node_Pair_Dataset(tgt_all_node_pair,tgt_all_node_pair_label)
    src_dataloader = DataLoader(source_np_dataset, batch_size=512,
                            shuffle=True, num_workers=4)
    tgt_dataloader = DataLoader(target_np_dataset, batch_size=512,
                            shuffle=True, num_workers=4) 
    ## model
    categorical_dim = max_np_label-min_np_label+1
    input_dim = dataset.num_features
    hidden_dim = [1024     ,1024     ,32*categorical_dim ,32                             ,64               ,256             ]
    #              0         1            2                3                               4                 5
    #hidden_dim= [gcn1_out ,gcn2_out ,vi_mlp_out         ,decoder_mlp1_in(embedding dim) ,decoder_mlp1_out ,decoder_mlp2_out]
    models = MRVAEDA(input_dim,hidden_dim,categorical_dim,device).to(device)
    optimizer = torch.optim.Adam(models.parameters(), lr=3e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,np.arange(1,200,50), gamma=0.1, last_epoch=-1)

    ## training
    source_data = source_data.to(device)
    target_data = target_data.to(device)
    train(1,src_dataloader,tgt_dataloader,source_data,target_data,device)


    print('------------End of training----------')
