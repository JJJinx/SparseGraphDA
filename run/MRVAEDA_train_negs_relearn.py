# coding=utf-8
import os
import time
import itertools
from argparse import ArgumentParser
import random
import numpy as np

import dgl
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from data_processing.DomainData import DomainData
from data_processing.data_process import *
from MRVGAE_model import *

class Node_Pair_Dataset(Dataset):
    def __init__(self,node_pairs,node_pair_labels):
        super(Node_Pair_Dataset, self).__init__()
        self.x = node_pairs # shape [N,2]
        self.y = node_pair_labels # shape [N,]
    def __getitem__(self,index):
        return self.x[index],self.y[index]
    def __len__(self):
        return self.x.shape[0]

## TODO 先考虑在单个域上的使用
## TODO 像relearn一样考虑模型每步用不同的Loss下降，我们的模型中应当是一步 recon 一步adj
class MRVAEDA(torch.nn.Module):
    def __init__(self, in_dim,hidden_dim,categorical_dim,device):
        super(MRVAEDA, self).__init__()
        self.private_encoder_source = private_encoder(in_dim,hidden_dim[0]) # in:: in_dim / out :: hidden_dim[0]
        #self.private_encoder_target = private_encoder(in_dim,hidden_dim[0]) # in:: in_dim / out :: hidden_dim[0]
        self.shared_encoder = shared_encoder(hidden_dim[0],hidden_dim[1]) # in :: hidden_dim[0] / out :: hidden_dim[1]

        self.VI = VI_relearn(hidden_dim[1],hidden_dim[2],categorical_dim,device) # in :: hidden_dim[1] / out :: hidden_dim[2]
        self.h_dec = torch.nn.Sequential(torch.nn.Linear(input, input//2),
                                   torch.nn.ReLU(inplace=True),
                                   torch.nn.Linear(input//2, output))

        self.shared_decoder = shared_decoder(hidden_dim[2],hidden_dim[3]) # in :: hidden_dim[2] / out :: hidden_dim[3]
        #self.private_decoder_source = private_decoder(hidden_dim[3],hidden_dim[4],in_dim) # 2layers
        self.private_decoder_target = private_decoder(hidden_dim[3],hidden_dim[4],in_dim) # 2layers

        self.A_classifier = relation_classifier(hidden_dim[2],hidden_dim[2]//2,categorical_dim) #classify the node pair's class

        self.discriminator = discriminator(hidden_dim[2],hidden_dim[2]//2)

    def forward(self, x, edge_index,node_pair,domain,rate):
        '''
            x :: shape [node_num,feat_dim]
            node_pair :: shape [batch_size,2]
        '''
        if domain =='source':
            x = self.private_encoder_source(x,edge_index,domain) 
        #if domain =='target':
            #x = self.private_encoder_target(x,edge_index,domain) 
        h = self.shared_encoder(x,edge_index,domain) # [node_num,hidden_dim[1]]
        #hadd = h[node_pair[:,0]]+h[node_pair[:,1]]
        hcat = torch.cat(h[node_pair[:,0]],h[node_pair[:,1]])

        H = self.
        H0 = self.VI(hcat,temp = 0.5)

        x_recon = self.shared_decoder(N)
        if domain == 'source':
            x_recon = self.private_decoder_source(x_recon)
        # if domain == 'target':
        #     x_recon = self.private_decoder_target(x_recon)

        A_pred = self.A_classifier(N)
        domain_pred = self.discriminator(N,rate)

        return x_recon,A_pred,domain_pred,mean,logstd

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
        #print('target_max',target.max())
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

def train(models,source_node_feat,target_node_feat,source_edge_index,target_edge_index,
                src_all_node_pair_label,tgt_all_node_pair_label,device,args):
    '''
    all node pair label :: [N,N] label matric in the adj format for node pairs
    np_pos_label_idx :: positive node pairs' index in the adj format matrix
    np_neg_label_idx :: negtive node pairs' index in the adj format matrix
    pos_np_num :: number of positive node pairs
    neg_np_num :: number of negtive node pairs 
    '''
    src_np_pos_label_idx = torch.stack(torch.where(src_all_node_pair_label!=0)).T
    tgt_np_pos_label_idx = torch.stack(torch.where(tgt_all_node_pair_label!=0)).T
    src_pos_np_num = src_np_pos_label_idx.shape[0]
    tgt_pos_np_num = tgt_np_pos_label_idx.shape[0]

    ## apply negtive sampling
    src_neg_np_num = args.batch_size-src_pos_np_num
    tgt_neg_np_num = args.batch_size-tgt_pos_np_num
    src_np_neg_label_idx = torch.stack(torch.where(src_all_node_pair_label==0)).T
    tgt_np_neg_label_idx = torch.stack(torch.where(tgt_all_node_pair_label==0)).T
    indice =  torch.randperm(src_np_neg_label_idx.shape[0])[:src_neg_np_num]
    src_np_neg_label_idx = src_np_neg_label_idx[indice] # is also the node pair index
    indice =  torch.randperm(tgt_np_neg_label_idx.shape[0])[:tgt_neg_np_num]
    tgt_np_neg_label_idx = tgt_np_neg_label_idx[indice]

    src_node_pair = torch.cat((src_np_pos_label_idx,src_np_neg_label_idx))
    tgt_node_pair = torch.cat((tgt_np_pos_label_idx,tgt_np_neg_label_idx))
    src_np_label = src_all_node_pair_label[src_node_pair[:,0],src_node_pair[:,1]]
    #tgt_np_label = tgt_all_node_pair_label[tgt_node_pair[:,0],tgt_node_pair[:,1]]
    ## build the reconstruction target ##主要是这一步有大量的GPU使用
    src_recon_label = source_node_feat[src_node_pair[:,0]]+ source_node_feat[src_node_pair[:,1]]
    tgt_recon_label = target_node_feat[tgt_node_pair[:,0]]+ target_node_feat[tgt_node_pair[:,1]]
    ## put into the model
    src_node_pair = src_node_pair.to(device)
    tgt_node_pair = tgt_node_pair.to(device)
    # TODO  加入随epoch自适应的temp参数
    src_X_recon,src_A_pred,src_domain_pred,src_mean,src_logstd = models(source_node_feat, source_edge_index,src_node_pair,'source',rate)
    tgt_X_recon,tgt_A_pred,tgt_domain_pred,tgt_mean,tgt_logstd = models(target_node_feat, target_edge_index,tgt_node_pair,'target',rate)
    ## source domain cls focal loss
    focal_loss = FocalLoss(gamma=5).to(device) # there are a lot of classes so we do not give the alpha
    loss_cls = focal_loss(src_A_pred,src_np_label.to(device))

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
    loss_kl = kl_norm
    # backward
    loss = loss_cls+loss_recon+loss_da+loss_kl
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('loss',loss.item(),
            'loss_cls:',loss_cls.item(),
            'loss_recon:',loss_recon.item(),
            'loss_da:',loss_da.item(),
            'loss_kl:',loss_kl.item())

def evaluate(np_pred,np_label,mapping_matrix,node_label): 
    '''
    require
        np_pred :: prediction for node pair  [NP,cat]
        np_label :: ground truth for node pair [NP,1]
        mapping_matrix :: mapping from np type to node type vote, [cat,node_label_num],
                            example: node label num=2 cat = 4
                            nlabel    0,1         
                            cat = 0 [[0,0],
                            cat = 1  [1,0],
                            cat = 2  [1,1],
                            cat = 3  [0,1]]
        np_vote_node_label :: each node pair vote for the src node's type according to the np type [NP,node_label_num]
    '''
    # calculate the node pair corrects
    node_pair_acc = np_pred.argmax(dim=1).eq(np_label).float().mean()
    #node_pair_correct = np_pred.argmax(dim=1).eq(np_label).sum()
    # calculate the node accuracy
    np_pred = np_pred.argmax(dim=1) # [NP,]
    node_num = node_label.shape[0]
    np_vote_node_label = mapping_matrix[np_pred].view(node_num,node_num,-1)
    node_pred = np_vote_node_label.sum(dim=1).argmax(dim=1) # node_pred :: reshape the np_vote_node_label to [N,N,node_label_num] 
                                                            # and sum by the dim=1 and get a tensor of [N, nodel_label_num] 
                                                            # then apply argmax get [N,] node label pred
    node_acc = node_pred.eq(node_label).float().mean()
    return node_pair_acc,node_acc

def validate(models,dataloader,all_node_pair_label,mapping_matrix,node_feat,node_label,edge_index,domain,device,rate):
    models.eval()
    np_pred = torch.tensor([])
    for batch_np,_ in dataloader:
        _,np_pred_temp,_,_,_ = models(node_feat.to(device),edge_index.to(device),batch_np.to(device),domain,rate)
        np_pred = torch.cat((np_pred,np_pred_temp.cpu().detach()),dim=0)
    node_pair_acc,node_acc = evaluate(np_pred,all_node_pair_label.view(-1),mapping_matrix,node_label)# all node pair label is [N,N] 
    return node_acc,node_pair_acc 



if __name__ == "__main__":
    ###################################################
    ######                args                #########
    ###################################################
    device = 'cuda'#torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = ArgumentParser()
    parser.add_argument("--source", type=str, default='acm')
    parser.add_argument("--target", type=str, default='dblp')
    parser.add_argument("--seed", type=int,default=200)
    parser.add_argument("--lr", type=float,default=0.001)
    parser.add_argument("--epochs", type=int,default=500)
    parser.add_argument("--batch_size", type=int,default=65536)

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
    ## data processing
    # to avoid missing nodes, we should add all the self loop in the edge index and then build up the graph
    self_loop = torch.arange(source_data.x.shape[0])
    self_loop = self_loop.unsqueeze(1).repeat(1,2)
    src_edge_index_sl = torch.cat([source_data.edge_index.T,self_loop]).T #[2,N]

    self_loop = torch.arange(target_data.x.shape[0])
    self_loop = self_loop.unsqueeze(1).repeat(1,2)
    tgt_edge_index_sl = torch.cat([target_data.edge_index.T,self_loop]).T #[2,N]
    del self_loop
    ## generate train graph
    source_graph = dgl.to_simple(dgl.graph((src_edge_index_sl[0],src_edge_index_sl[1])))
    target_graph = dgl.to_simple(dgl.graph((tgt_edge_index_sl[0],tgt_edge_index_sl[1])))
    ## make edge index to be bidirected
    source_graph = dgl.to_bidirected(source_graph)
    target_graph = dgl.to_bidirected(target_graph)
    src_edge_index_sl = torch.vstack([source_graph.edges()[0],source_graph.edges()[1]])
    tgt_edge_index_sl = torch.vstack([target_graph.edges()[0],target_graph.edges()[1]])
    ##generate all node pair label
    source_node_num = source_data.x.shape[0]
    target_node_num = target_data.x.shape[0]
    source_node_feat = source_data.x
    target_node_feat = target_data.x
    source_node_label = source_data.y
    target_node_label = target_data.y
    del source_data,target_data

    src_all_node_pair,src_all_node_pair_label,max_np_label = generate_all_node_pair_adj(source_node_num,src_edge_index_sl,source_node_label,
                                                                        node_label_num,source_graph.adjacency_matrix())

    tgt_all_node_pair,tgt_all_node_pair_label,max_np_label = generate_all_node_pair_adj(target_node_num,tgt_edge_index_sl,target_node_label,
                                                                        node_label_num,target_graph.adjacency_matrix())
    min_np_label = 0 # no_edge_existence
    max_np_label = max_np_label.item() 
    categorical_dim = max_np_label-min_np_label+1
    mapping_matrix = generate_mapping_M(node_label_num,categorical_dim)
    #np.savetxt('acm_all_node_pair_label.csv',src_all_node_pair_label.numpy(),delimiter=',')
    #np.savetxt('dblp_all_node_pair_label.csv',tgt_all_node_pair_label.numpy(),delimiter=',')
    ## dataloader
    src_all_node_pair = src_all_node_pair.view(-1,2)
    tgt_all_node_pair = tgt_all_node_pair.view(-1,2)
    source_np_dataset = Node_Pair_Dataset(src_all_node_pair,src_all_node_pair_label.view(-1))
    target_np_dataset = Node_Pair_Dataset(tgt_all_node_pair,tgt_all_node_pair_label.view(-1))
    src_dataloader = DataLoader(source_np_dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=4)
    tgt_dataloader = DataLoader(target_np_dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=4) 
    ## model
    input_dim = dataset.num_features
    hidden_dim = [1024     ,1024     ,    256    ,          512,            1024]   
    #              0         1            2                3                 4
    #hidden_dim= [gcn1_out ,gcn2_out ,vi_mlp_out,decoder_mlp1_out ,decoder_mlp1_out ]  vi_out <= gcn2_out//2
    models = MRVAEDA(input_dim,hidden_dim,categorical_dim,device).to(device)
    optimizer = torch.optim.Adam(models.parameters(), lr=3e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,np.arange(1,args.batch_size,args.batch_size//4), gamma=0.1, last_epoch=-1)
    ###################################################
    ######                training            #########
    ###################################################
    for epoch in range(args.epochs): # start from 0
        models.train()
        rate = min((epoch + 1) / args.epochs, 0.05)
        t = time.time()
        # during training we use the negtive sampling and do not use the dataloader 
        train(models,source_node_feat.to(device),target_node_feat.to(device),
                    src_edge_index_sl.to(device),tgt_edge_index_sl.to(device),
                    src_all_node_pair_label,tgt_all_node_pair_label,device,args)
        # train(models,src_dataloader,tgt_dataloader,
        #         source_node_feat.to(device),src_edge_index_sl.to(device),
        #         target_node_feat.to(device),tgt_edge_index_sl.to(device),device)
        print('train_time used:',time.time()-t)

        if (epoch+1) %10 == 0:
            ## validate
            t = time.time()
            src_node_acc,src_node_pair_acc=validate(models,src_dataloader,src_all_node_pair_label,mapping_matrix,
                                                    source_node_feat,source_node_label,src_edge_index_sl,
                                                    'source',device,rate)
            tgt_node_acc,tgt_node_pair_acc=validate(models,tgt_dataloader,tgt_all_node_pair_label,mapping_matrix,
                                                    target_node_feat,target_node_label,tgt_edge_index_sl,
                                                    'target',device,rate)

            print('source node acc:{},source node pair acc:{},target node acc:{},target node pair acc:{}'\
                    .format(src_node_acc,src_node_pair_acc,tgt_node_acc,tgt_node_pair_acc))
            print('val_time used:',time.time()-t)

    print('------------End of training----------')
