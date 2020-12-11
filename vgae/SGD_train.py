# -*- coding: utf-8 -*-
import os
import time
import utils
import pickle
import tqdm

import torch
import torch.nn.functional as F
from torch.optim import Adam
import dgl
import dgl.function as fn

import numpy as np
import sklearn.linear_model
import sklearn.metrics

import args
from processing import *
from model import SGD_MRVGAE,ScorePredictor


def train(model,predictor,device,args,train_g):
    best_accuracy = 0
    best_model_path = 'model.pt'

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    negative_sampler = dgl.dataloading.negative_sampler.Uniform(5)
    dataloader = dgl.dataloading.EdgeDataLoader(
        train_g, torch.arange(train_g.number_of_edges()), sampler,
        negative_sampler=negative_sampler,
        batch_size=args.batchsize,
        shuffle=True,
        drop_last=False,
        num_workers=4
    )
    node_features = train_g.ndata['feat']
    for epoch in range(args.num_epoch):
        model.train()
        with tqdm.tqdm(dataloader) as tq:
            for step, (input_nodes, pos_graph, neg_graph, bipartites) in enumerate(tq):
                bipartites = [b.to(torch.device(device)) for b in bipartites]
                pos_graph = pos_graph.to(torch.device(device))
                neg_graph = neg_graph.to(torch.device(device))
                inputs = node_features[input_nodes].to(device)
                [posA,negA,posX,negX,pos_mean,neg_mean,pos_logstd,neg_logstd,posq,negq] = model(bipartites, inputs,pos_graph,neg_graph,temp=0.5)
                # TODO 因为进入解码器模型的是正负节点对的属性，所以输出的解码也是正负节点对，而不是节点特征，所以要重建也是重建节点对，且分类具体的节点对类别
                label = torch.cat([torch.ones_like(posA), torch.zeros_like(negA)])
                A = torch.cat([posA,negA])
                loss_A = F.binary_cross_entropy_with_logits(A, label)  
                ## KL loss
                mean = torch.cat([pos_mean,neg_mean])
                logstd = torch.cat([pos_logstd,neg_logstd])
                kl_norm= torch.mean(-0.5*torch.sum(1+2*logstd-mean**2-logstd.exp(),dim=1),dim=0)

                q = torch.cat([posq,negq])
                q_for_kl = F.softmax(q,dim=1)
                eps = 1e-20
                h1 = -q_for_kl*torch.log(q_for_kl + eps)#h(p)
                h2 = -q_for_kl*np.log(1./args.categorical_dim) #h(pq)
                kl_gumbel = torch.mean(torch.sum(h2-h1,dim=(1,2)),dim=0)
                loss_VAE = kl_gumbel+kl_norm
                # TODO 重建损失

                loss = loss_A+loss_VAE
                opt.zero_grad()
                loss.backward()
                opt.step()
                
                tq.set_postfix({'loss': '%.03f' % loss.item()}, refresh=False)
        
        model.eval()
        posA,posX,negA,negX = model.inference(train_g,val_pos_graph,val_neg_graph,node_features,temp=0.5)
        #TODO 用VGAE的方式做检测
        # valid_acc, test_acc = evaluate(emb.numpy(), node_labels.numpy())
        # print('Epoch {} Validation Accuracy {} Test Accuracy {}'.format(epoch, valid_acc, test_acc))
        # if best_accuracy < valid_acc:
        #     best_accuracy = valid_acc
        #     torch.save(model.state_dict(), best_model_path)

if __name__ == "__main__":
    device = args.device
    # get data split
    dataset = dgl.data.CoraGraphDataset()
    #graph = dataset.graph
    graph = dataset[0]
    train_g,val_pos_graph,val_neg_graph,test_pos_graph,test_neg_graph = data_process(graph)
    model = SGD_MRVGAE(args.input_dim,args.n_hidden,args.input_dim,args.device,distype='Both',categorical_dim=args.categorical_dim).to(device) 
    predictor = ScorePredictor().to(device)
    opt = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()))
    train(model,predictor,device,args,train_g)
    print('--------end of scripts---------')


