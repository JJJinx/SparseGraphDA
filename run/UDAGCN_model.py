# coding=utf-8
import torch
from layers import CachedGCNConv,PPMIConv
from torch import nn
import torch.nn.functional as F


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
            model_cls(num_features, 128,
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

#把grl层换成这个，将p传入
class ReverseLayerF(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, p):
        ctx.p = p
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.p
        return output, None


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * rate
        return grad_output, None


class GRL(nn.Module):
    def forward(self, input):
        return GradReverse.apply(input)


class Attention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.dense_weight = nn.Linear(in_channels, 1)
        self.dropout = nn.Dropout(0.1)


    def forward(self, inputs):
        stacked = torch.stack(inputs, dim=1)
        weights = F.softmax(self.dense_weight(stacked), dim=1)
        outputs = torch.sum(stacked * weights, dim=1)
        return outputs
class UDAGCN(nn.Module):
    def __init__(self,device,encoder_dim,num_classes,num_feat):
        super(nn.Module,self).__init__()
        self.encoder = GNN(type='gcn',num_features = num_feat)
        self.ppmi_encoder = GNN(base_model=self.encoder,type="ppmi", path_len=10)
        self.cls_model = nn.Sequential(nn.Linear(encoder_dim,num_classes))
        self.domain_model = nn.Sequential(
            #GRL(),
            nn.Linear(encoder_dim,40),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(40,2),
        )
        self.att = Attention(encoder_dim)

    def forward(self,data,cache_name,rate,mask = None):
        gcn_output = self.gcn_encode(data,cache_name,mask)
        ppmi_output = self.ppmi_encode(data,cache_name,mask)
        emb = self.att([gcn_output,ppmi_output])

        class_predict = self.cls_model(emb) 
        domain_predict = self.domain_model(ReverseLayerF.apply(emb, rate))
        result = [emb,class_predict,domain_predict]
        return  result

    def gcn_encode(self,data,cache_name,mask=None):
        encoded_output = self.encoder(data.x,data.edge_index,cache_name)
        if mask is not None:
            encoded_output = encoded_output[mask]
        return encoded_output
    
    def ppmi_encode(self,data,cache_name,mask=None):
        encoded_output = self.ppmi_encoder(data.x,data.edge_index,cache_name)
        if mask is not None:
            encoded_output = encoded_output[mask]
        return encoded_output
