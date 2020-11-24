import torch
from torch import nn
from dual_gnn.cached_gcn_conv import CachedGCNConv

class VI(nn.Module):
    #模型地变分推断部分
    # 根据VAE,fc层不需要dropout操作
    def __init__(self,in_dim,out_dim,distype='Norm',**kwargs):
        super(VI,self).__init__()
        self.type = distype
        self.out_dim = out_dim
        if distype == 'Norm':
            #变分分布为高斯
            self.mlp_mean   = nn.Linear(in_dim,out_dim) 
            self.mlp_logstd = nn.Linear(in_dim,out_dim)
        if distype == 'Bern':
            #变分分布为伯努利分布
            self.mlp_p = nn.Linear(in_dim,out_dim)

    def forward(self,H)
        if self.type == 'Norm':
            mean = self.mlp_mean(H)
            logstd = self.mlp_logstd(H)
            gausian_noise = torch.randn(H.size(0),self.out_dim)
            Z = gausian_noise*torch.exp(self.logstd) + self.mean #加exp保证方差为正，所以限制学到的是logstd
        if self.type == 'Bern':
            # TODO 如何重参数
            p = self.mlp_p(H)
        return Z

class private_encoder(nn.Module):
    def __init__(self,num_features,out_dim,**kwargs):
        super(private_encoder,self).__init__()
        self.gcn = CachedGCNConv(num_features, out_dim,**kwargs)
        self.dropout_layers = nn.Dropout(0.1)
    def forward(self,x,adj):
        x = self.gcn(x, adj)
        x = self.dropout_layers(x)
        return x

class shared_encoder(nn.Module):
    def __init__(self,in_dim,out_dim,**kwargs):
        super(private_encoder,self).__init__()
        self.gcn = CachedGCNConv(in_dim, out_dim,**kwargs)
    def forward(self,x,adj):
        x = self.gcn(x, adj)
        return x

class shared_decoder(nn.Module):
    def __init__(self,in_dim,out_dim):
        super(shared_decoder,self).__init__()
        self.mlp = nn.Linear(in_dim,out_dim)
        self.dropout_layers = nn.Dropout(0.1)
    def forward(self,x):
        x = self.mlp(x)
        x = self.dropout_layers(x)
        return x 
            
class private_decoder(nn.Module):
    def __init__(self,in_dim,out_dim):
        super(private_decoder,self).__init__()
        self.mlp = nn.Linear(in_dim,out_dim)
    def forward(self,x):
        x = self.mlp(x)
        return x


class R_classifier():
    def __init__():
        super(private_decoder,self).__init__()
        pass
    def forward():
        pass


class discriminator():
    def __init__():
        pass
    def forward():
        pass



# class allinone(nn.Module):
#     #整个模型
#     def __init__(self,num_features,out_dim,base_model=None,**kwargs):
#         super(VI,self).__init__()

#         if base_model is None:
#             weights = [None, None]
#             biases = [None, None]
#         else:
#             weights = [conv_layer.weight for conv_layer in base_model.conv_layers]
#             biases = [conv_layer.bias for conv_layer in base_model.conv_layers]

#         self.dropout_layers = [nn.Dropout(0.1) for _ in weights]
#         self.encoder = nn.ModuleList([
#             CachedGCNConv(num_features, 128,
#                      weight=weights[0],
#                      bias=biases[0],
#                       **kwargs),
#             CachedGCNConv(128, out_dim,
#                      weight=weights[1],
#                      bias=biases[1],
#                       **kwargs)
#         ])
#         self.mlp_mean   = nn.Linear(2*out_dim,out_dim) #建立伯努利分布
#         self.mlp_logstd = nn.Linear(2*out_dim,out_dim)

#         self.decoder1=nn.Linear(out_dim,out_dim)
#         self.att = None
#         self.decoder2_adj = None
#         self.decoder2_x = None

#     def encode(self,X,adj):

#         for i, conv_layer in enumerate(self.encoder):
#             h = conv_layer(X, adj)
#             if i < len(self.conv_layers) - 1:
#                 h = F.relu(h)
#                 h = self.dropout_layers[i](h)#h [N,Dh]
#         hj = h.unsqueeze(0).repeat(h.shape[0],1,1) #[N_i,N_j,Dh],向量值仅和j相关
#         hi = h.unsqueeze(1),repeat(1,h.shape[0],1) #[N_i,N_j,Dh],向量值仅和i相关
#         hij = torch.cat((hi,hj),2)
#         # h = torch.mm(h,h.T)*hij #向量乘后元素乘


#     def forward(self,X,adj)
#         Z = self.encode(X,adj)
#         Z = self.decoder1(Z)

#         Z_adj = self.decoder2_adj(Z)
#         A_pred = dot_product_decode(Z) #TODO

#         Z_x = self.att(Z)
#         X = self.decoder2_x(Z_x)
#         return A_pred,X