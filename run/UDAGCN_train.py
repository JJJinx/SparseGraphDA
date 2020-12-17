# coding=utf-8
import torch
print(torch.cuda.is_available())

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from argparse import ArgumentParser
from DomainData import DomainData
from model.UDAGCN_model import *
from torch import nn
from torch import nn.functional as F
import random
import numpy as np
import itertools

def data_processing(device,args):
    seed = args.seed
    encoder_dim = args.encoder_dim

    id = "source: {}, target: {}, seed: {}, encoder_dim: {}"\
    .format(args.source, args.target, seed,  encoder_dim)
    print(id)

    rate = 0.0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dataset = DomainData("data/{}".format(args.source), name=args.source)
    source_data = dataset[0]
    print(source_data)

    dataset = DomainData("data/{}".format(args.target), name=args.target)
    target_data = dataset[0]
    print(target_data)

    source_data = source_data.to(device)
    target_data = target_data.to(device)
    return source_data,target_data,dataset.num_classes,dataset.num_feat

def train(model,optimizer,source_data,target_data,epoch):
    model.train()
    optimizer.zero_grad()
    loss_func = nn.CrossEntropyLoss().to(device)

    global rate
    rate = min((epoch + 1) / epochs,0.05)

    source_result = model(source_data,'source',rate)
    target_result = model(target_data,'target',rate)

    source_class_predict,source_domain_predict = source_result[1],source_result[2]
    target_class_predict,target_domain_predict = target_result[1],target_result[2]

    # use source classifier loss:
    source_cls_loss = loss_func(source_class_predict,source_data.y)
    for name, param in model.named_parameters():
        if "weight" in name:
            source_cls_loss = source_cls_loss + param.mean() * 3e-3
    
    # use domain classifier loss:
    source_domain_loss = loss_func(
            source_domain_predict,
            torch.zeros(source_domain_predict.size(0)).type(torch.LongTensor).to(device)
        )
    target_domain_loss = loss_func(
            target_domain_predict,
            torch.ones(target_domain_predict.size(0)).type(torch.LongTensor).to(device)
        )
    grl_loss = source_domain_loss+target_domain_loss
    
    # use target classifier loss:
    target_probs = F.softmax(target_class_predict,dim=-1)
    target_probs = torch.clamp(target_probs,min=1e-9,max=1.0)
    target_entropy_loss = torch.mean(torch.sum(-target_probs*torch.log(target_probs),dim = -1))
    
    loss =  source_cls_loss + target_entropy_loss*(epoch / epochs * 0.01) + grl_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
def test(model, data,cache_name, mask=None):
    model.eval()
    result = model(data,cache_name,rate)
    preds = result[1]
    labels = data.y if mask is None else data.y[mask]
    accuracy = evaluate(preds, labels)
    return accuracy

def evaluate(preds, labels):
    corrects = preds.eq(labels)
    accuracy = corrects.float().mean()
    return accuracy

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = ArgumentParser()
    parser.add_argument("--source", type=str, default='acm')
    parser.add_argument("--target", type=str, default='ratiodel_dblp0.4_1')
    parser.add_argument("--name", type=str, default='UDAGCN')
    parser.add_argument("--seed", type=int,default=200)
    parser.add_argument("--encoder_dim", type=int, default=16)
    args = parser.parse_args()

    source_data,target_data,num_classes,num_feat=data_processing(device,args)

    best_source_acc = 0.0
    best_target_acc = 0.0
    best_epoch = 0.0
    model = UDAGCN(device,args.encoder_dim,num_classes,num_feat)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    epochs = 200
    for epoch in range(1,epochs):
        train(model,optimizer,source_data,target_data,epoch)
        source_correct = test(model,source_data, "source", source_data.test_mask)
        target_correct = test(model,target_data, "target")
        print("Epoch: {}, source_acc: {}, target_acc: {}".format(epoch, source_correct, target_correct))
        if target_correct > best_target_acc:
            best_target_acc = target_correct
            best_source_acc = source_correct
            best_epoch = epoch
print("=============================================================")
line = "{} - Epoch: {}, best_source_acc: {}, best_target_acc: {}"\
    .format(id, best_epoch, best_source_acc, best_target_acc)

print(line)