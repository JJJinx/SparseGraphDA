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

def inference(model, graph, input_features, batch_size):
    '''
        graph:: validation graph

    '''
    nodes = torch.arange(graph.number_of_nodes())
    
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler([2])  # one layer at a time, taking all neighbors
    dataloader = dgl.dataloading.NodeDataLoader(
        graph, nodes, sampler,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0)
    
    with torch.no_grad():
        for l, layer in enumerate(model.layers):
            # Allocate a buffer of output representations for every node
            # Note that the buffer is on CPU memory.
            output_features = torch.zeros(graph.number_of_nodes(), model.n_hidden)

            for input_nodes, output_nodes, bipartites in tqdm.tqdm(dataloader):
                bipartite = bipartites[0].to(torch.device('cuda'))

                x = input_features[input_nodes].cuda()

                # the following code is identical to the loop body in model.forward()
                x = layer(bipartite, x)
                if l != model.n_layers - 1:
                    x = F.relu(x)

                output_features[output_nodes] = x.cpu()
            input_features = output_features
    return output_features

def evaluate(emb, label, train_nids, valid_nids, test_nids):
    classifier = sklearn.linear_model.LogisticRegression(solver='lbfgs', multi_class='multinomial', verbose=1, max_iter=1000)
    classifier.fit(emb[train_nids], label[train_nids])
    valid_pred = classifier.predict(emb[valid_nids])
    test_pred = classifier.predict(emb[test_nids])
    valid_acc = sklearn.metrics.accuracy_score(label[valid_nids], valid_pred)
    test_acc = sklearn.metrics.accuracy_score(label[test_nids], test_pred)
    return valid_acc, test_acc

def train(model,predictor,device,args,train_g,val_g):
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
        
        # TODO 后面考虑如何inference    
        model.eval()
        emb = inference(model, val_g, node_features, args.batch_size)
        valid_acc, test_acc = evaluate(emb.numpy(), node_labels.numpy())
        print('Epoch {} Validation Accuracy {} Test Accuracy {}'.format(epoch, valid_acc, test_acc))
        if best_accuracy < valid_acc:
            best_accuracy = valid_acc
            torch.save(model.state_dict(), best_model_path)

if __name__ == "__main__":
    device = args.device

    # get data split
    # TODO 所以就得到一个删去val和test边的adj_train，用这个adj_train构建一个graph对象(dgl.from_scipy)后将这个graph对象放入进行训练，val的时候就是将推测的值
    #dataset = dgl.data.FB15k237Dataset()
    dataset = dgl.data.CoraGraphDataset()
    #graph = dataset.graph
    graph = dataset[0]
    data_process(graph)
    model = SGD_MRVGAE(args.input_dim,args.n_hidden,args.input_dim,args.device,distype='Both',categorical_dim=args.categorical_dim).to(device) 
    predictor = ScorePredictor().to(device)
    opt = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()))
    train(model,predictor,device,args,train_g,val_g)
    print('--------end of scripts---------')


