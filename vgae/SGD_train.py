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
from sklearn.metrics import roc_auc_score, average_precision_score

import args
from processing import *
from model import SGD_MRVGAE,ScorePredictor

def get_scores(posA, negA, pos_graph,neg_graph):
    """
        posA:: the prediction for the positive node pair
        negA:: the prediction for the negtive node pair
        posgraph:: the graph including ground truth positive edges
        neggraph:: the graph including ground truth negtive edges 
    """
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    preds = posA.detach().numpy()
    pos = np.zeros_like(preds,dtype=np.int16)
    pos[:,1] = 1
    preds_neg = negA.detach().numpy()
    neg = np.zeros_like(preds_neg,dtype=np.int16)
    neg[:,0] = 1

    preds_all = np.vstack([preds,preds_neg])
    labels_all = np.vstack([pos,neg])

    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score

def train(model,predictor,device,args,train_g):
    best_accuracy = 0
    best_model_path = 'model.pt'

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    negative_sampler = dgl.dataloading.negative_sampler.Uniform(5)
    dataloader = dgl.dataloading.EdgeDataLoader(
        train_g, torch.arange(train_g.number_of_edges()), sampler,
        negative_sampler=negative_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=4
    )
    node_features = train_g.ndata['feat']
    for epoch in range(args.num_epoch):
        t = time.time()
        model.train()
        with tqdm.tqdm(dataloader) as tq:
            for step, (input_nodes, pos_graph, neg_graph, bipartites) in enumerate(tq):
                bipartites = [b.to(torch.device(device)) for b in bipartites]
                pos_graph = pos_graph.to(torch.device(device))
                neg_graph = neg_graph.to(torch.device(device))
                inputs = node_features[input_nodes].to(device)
                [posA,negA,posX,negX,pos_mean,neg_mean,pos_logstd,neg_logstd,posq,negq] = model(bipartites, inputs,pos_graph,neg_graph,temp=0.5)
                #one hot形式编码，无边为第0维，有边为第1维，设置标签，标签值为gt值
                A = torch.cat([posA,negA])
                label = torch.cat([torch.ones(posA.shape[0],dtype=torch.long),torch.zeros(negA.shape[0],dtype=torch.long)])
                loss_A = F.cross_entropy(A, label)  
                ## KL loss
                mean = torch.cat([pos_mean,neg_mean])
                logstd = torch.cat([pos_logstd,neg_logstd]) #将正负样本沿dim0拼接后取平均
                kl_norm= torch.mean(-0.5*torch.sum(1+2*logstd-mean**2-logstd.exp()**2,dim=1),dim=0) 

                q = torch.cat([posq,negq])
                q_for_kl = F.softmax(q,dim=1)
                eps = 1e-20
                h1 = -q_for_kl*torch.log(q_for_kl + eps)#h(p)
                h2 = -q_for_kl*np.log(1./args.categorical_dim) #h(pq)
                kl_gumbel = torch.mean(torch.sum(h2-h1,dim=1),dim=0)
                loss_VAE = kl_gumbel+kl_norm

                neg_graph.ndata['feat'] = pos_graph.ndata['feat'] #仅有pos graph有特征
                pos_graph.apply_edges(dgl.function.u_add_v('feat', 'feat', 'np')) 
                neg_graph.apply_edges(dgl.function.u_add_v('feat', 'feat', 'np'))
                recon = torch.cat([posX,negX])
                recon_label = torch.cat([pos_graph.edata['np'],neg_graph.edata['np']])
                loss_recon = F.mse_loss(recon,recon_label)
                loss = loss_A+loss_VAE+loss_recon
                opt.zero_grad()
                loss.backward()
                opt.step()
                
                tq.set_postfix({'loss': '%.03f' % loss.item()}, refresh=False)
        
        model.eval()
        posA,posX,negA,negX = model.inference(train_g,val_pos_graph,val_neg_graph,node_features,temp=0.5)
        print(loss_A,loss_VAE,loss_recon)
        val_roc, val_ap = get_scores(posA,negA,val_pos_graph,val_neg_graph)        
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss.item()), 
            "val_roc=", "{:.5f}".format(val_roc),"val_ap=", "{:.5f}".format(val_ap),
            "time=", "{:.5f}".format(time.time() - t))

    #test_roc, test_ap = get_scores(test_edges, test_edges_false, A_pred)
    #print("End of training!", "test_roc=", "{:.5f}".format(test_roc),
    #    "test_ap=", "{:.5f}".format(test_ap))


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


