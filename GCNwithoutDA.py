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

# from torch.utils.tensorboard import SummaryWriter
# data_path = os.path.join(os.getcwd(),'datastastic')
# logfilename = 'ACMd0.2(1)2DBLPrd0.2(3)'+'_log'
# writer = SummaryWriter(os.path.join(data_path,logfilename))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = ArgumentParser()
parser.add_argument("--source", type=str, default='dblp')
parser.add_argument("--target", type=str, default='ratiodel_acm0.2_1')
parser.add_argument("--name", type=str, default='UDAGCN')
parser.add_argument("--seed", type=int,default=200)
parser.add_argument("--UDAGCN", type=bool,default=False)
parser.add_argument("--encoder_dim", type=int, default=16)


args = parser.parse_args()
seed = args.seed
use_UDAGCN = args.UDAGCN
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
dataset = DomainData("data/{}".format(args.target), name=args.target)
target_data = dataset[0]
print(target_data)

source_data = source_data.to(device)
target_data = target_data.to(device)

class GNN(torch.nn.Module):
    def __init__(self, base_model=None, type="gcn", **kwargs):
        super(GNN, self).__init__()

        if base_model is None:
            weights = [None, None]
            biases = [None, None]
        else:
            weights = [conv_layer.weight for conv_layer in base_model.conv_layers]
            biases = [conv_layer.bias for conv_layer in base_model.conv_layers]


        self.dropout_layers = [nn.Dropout(0.1) for _ in weights]
        self.type = type

        model_cls = PPMIConv if type == "ppmi" else CachedGCNConv

        self.conv_layers = nn.ModuleList([
            model_cls(dataset.num_features, 128,
                     weight=weights[0],
                     bias=biases[0],
                      **kwargs),
            model_cls(128, encoder_dim,
                     weight=weights[1],
                     bias=biases[1],
                      **kwargs)
        ])

    def forward(self, x, edge_index, cache_name):
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x, edge_index, cache_name)
            if i < len(self.conv_layers) - 1:
                x = F.relu(x)
                x = self.dropout_layers[i](x)
        return x



loss_func = nn.CrossEntropyLoss().to(device)

encoder = GNN(type="gcn").to(device)

cls_model = nn.Sequential(
    nn.Linear(encoder_dim, dataset.num_classes),
).to(device)

models = [encoder, cls_model]
params = itertools.chain(*[model.parameters() for model in models])
optimizer = torch.optim.Adam(params, lr=3e-3)


def gcn_encode(data, cache_name, mask=None):
    encoded_output = encoder(data.x, data.edge_index, cache_name)
    if mask is not None:
        encoded_output = encoded_output[mask]
    return encoded_output

def encode(data, cache_name, mask=None):
    gcn_output = gcn_encode(data, cache_name, mask)
    return gcn_output

def predict(data, cache_name, mask=None):
    encoded_output = encode(data, cache_name, mask)
    logits = cls_model(encoded_output)
    return logits


def evaluate(preds, labels):
    corrects = preds.eq(labels)
    accuracy = corrects.float().mean()
    return accuracy


def test(data, cache_name, mask=None):
    for model in models:
        model.eval()
    logits = predict(data, cache_name, mask)
    preds = logits.argmax(dim=1)
    labels = data.y if mask is None else data.y[mask]
    accuracy = evaluate(preds, labels)
    return accuracy



epochs = 200
def train(epoch):
    for model in models:
        model.train()
    optimizer.zero_grad()

    global rate
    rate = min((epoch + 1) / epochs, 0.05)

    encoded_source = encode(source_data, "source")
    encoded_target = encode(target_data, "target")
    source_logits = cls_model(encoded_source)

    # use source classifier loss:
    # 仅计算训练集的loss
    cls_loss = loss_func(source_logits[source_data.train_mask], source_data.y[source_data.train_mask])

    for model in models:
        for name, param in model.named_parameters():
            if "weight" in name:
                cls_loss = cls_loss + param.mean() * 3e-3


    # use target classifier loss:
    target_logits = cls_model(encoded_target)
    target_probs = F.softmax(target_logits, dim=-1)
    target_probs = torch.clamp(target_probs, min=1e-9, max=1.0)
    loss_entropy = torch.mean(torch.sum(-target_probs * torch.log(target_probs), dim=-1))
    loss = cls_loss + loss_entropy* (epoch / epochs * 0.01)
    #loss = cls_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

best_source_acc = 0.0
best_target_acc = 0.0
best_epoch = 0.0
for epoch in range(1, epochs):
    train(epoch)
    source_correct = test(source_data, "source", source_data.test_mask)
    target_correct = test(target_data, "target")
    # writer.add_scalars('ACC', { 'src ACC': source_correct,'tgt ACC': target_correct}, epoch)
    # writer.flush()
    print("Epoch: {}, source_acc: {}, target_acc: {}".format(epoch, source_correct, target_correct))
    if target_correct > best_target_acc:
        best_target_acc = target_correct
        best_source_acc = source_correct
        best_epoch = epoch
#创建文件夹
data_vis_path = './datastastic/degree_related_acc/{}2{}gcnnoda/'.format(args.source,args.target)
isnrpExist=os.path.exists(data_vis_path)
# print(isnrpExist,data_vis_path)
# raise RuntimeError
if not isnrpExist:
    os.makedirs(data_vis_path)
#对于源域的节点级别的度相关性分析
for model in models:
    model.eval()
encoded_output = encode(source_data, 'source', source_data.test_mask)
logits = cls_model(encoded_output)
preds = logits.argmax(dim=1)
degree_level_acc("data/{}/raw".format(args.source),args.source,preds,source_data.y,\
    savefig_path=os.path.join(data_vis_path+'{}.png'.format(args.source)),mask=source_data.test_mask)
#对于目标域的节点级别的度相关性分析
for model in models:
    model.eval()
encoded_output = encode(target_data, 'target')
logits = cls_model(encoded_output)
preds = logits.argmax(dim=1)
degree_level_acc("data/{}/raw".format(args.target),args.target,preds,target_data.y,\
    savefig_path=os.path.join(data_vis_path+'{}.png'.format(args.target)))
print("=============================================================")
line = "{} - Epoch: {}, best_source_acc: {}, best_target_acc: {}"\
    .format(id, best_epoch, best_source_acc, best_target_acc)
print(line)
# writer.close()