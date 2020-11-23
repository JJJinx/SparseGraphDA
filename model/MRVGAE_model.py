import torch
from torch import nn
from dual_gnn.cached_gcn_conv import CachedGCNConv


class meta_relation_VGAE(nn.Module):

    def __init__(self,num_features,out_dim,base_model=None,**kwargs):
        super(meta_relation_VGAE,self).__init__()

        if base_model is None:
            weights = [None, None]
            biases = [None, None]
        else:
            weights = [conv_layer.weight for conv_layer in base_model.conv_layers]
            biases = [conv_layer.bias for conv_layer in base_model.conv_layers]

        self.dropout_layers = [nn.Dropout(0.1) for _ in weights]
        self.encoder = nn.ModuleList([
            CachedGCNConv(num_features, 128,
                     weight=weights[0],
                     bias=biases[0],
                      **kwargs),
            CachedGCNConv(128, out_dim,
                     weight=weights[1],
                     bias=biases[1],
                      **kwargs)
        ])
        self.mlp_mean   = nn.Linear(2*out_dim,out_dim) #输入是N*N*2outdim的向量，仅对最后一维操作
        self.mlp_logstd = nn.Linear(2*out_dim,out_dim)

        self.decoder1=nn.Linear(out_dim,out_dim)
        self.att = None
        self.decoder2_adj = None
        self.decoder2_x = None

    def encode(self,X,adj):

        for i, conv_layer in enumerate(self.encoder):
            x = conv_layer(x, adj)
            if i < len(self.conv_layers) - 1:
                x = F.relu(x)
                x = self.dropout_layers[i](x)
        h = self.encoder()


    def forward(self,X,adj)
        Z = self.encode(X,adj)
        Z = self.decoder1(Z)

        Z_adj = self.decoder2_adj(Z)
        A_pred = dot_product_decode(Z) #TODO

        Z_x = self.att(Z)
        X = self.decoder2_x(Z_x)
        return A_pred,X

            