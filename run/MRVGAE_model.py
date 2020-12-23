
import torch
from torch import nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn
from layers import CachedGCNConv
'''
DGL GCN ENCODER
class private_encoder(nn.Module):
    def __init__(self,in_feats,out_dim,**kwargs):
        super(private_encoder,self).__init__()
        self.gcn = dglnn.GraphConv(in_feats,out_dim)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
    def forward(self,graph,x):
        x = self.gcn(graph,x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class shared_encoder(nn.Module):
    def __init__(self,in_feats,out_dim,**kwargs):
        super(shared_encoder,self).__init__()
        self.gcn = dglnn.GraphConv(in_feats,out_dim)
        self.relu = nn.ReLU()
    def forward(self,graph,x):
        x = self.gcn(graph,x)
        x = self.relu(x)
        return x
'''
class private_encoder(nn.Module):
    def __init__(self,in_feats,out_dim,**kwargs):
        super(private_encoder,self).__init__()
        self.gcn = CachedGCNConv(in_feats,out_dim)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x,edge_index,cache_name):
        x = self.gcn(x,edge_index,cache_name)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class shared_encoder(nn.Module):
    def __init__(self,in_feats,out_dim,**kwargs):
        super(shared_encoder,self).__init__()
        self.gcn = CachedGCNConv(in_feats,out_dim)
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x,edge_index,cache_name):
        x = self.gcn(x,edge_index,cache_name)
        x = self.relu(x)
        return x

class VI(nn.Module):
    # 根据VAE,fc层不需要dropout操作
    def __init__(self,in_dim,out_dim,categorical_dim,device,distype='Norm',**kwargs):
        super(VI,self).__init__()
        self.device = device
        self.type = distype
        self.out_dim = out_dim
        self.cat = categorical_dim

        if distype == 'Norm':
            self.mean   = nn.Linear(in_dim,out_dim) 
            self.logstd = nn.Linear(in_dim,out_dim)
        if distype == 'Both':
            self.mean   = nn.Linear(in_dim,out_dim) 
            self.logstd = nn.Linear(in_dim,out_dim)
            self.q = nn.Linear(in_dim,categorical_dim)

    def forward(self,H,temp):
        '''
            H :: shape [node_pair_num,hidden_dim[1]]
        '''
        if self.type == 'Norm':
            mean = self.mean(H)  # [node_pair_num,dN]
            logstd = self.logstd(H)
            gausian_noise = torch.randn_like(mean)
            N = gausian_noise*torch.exp(logstd) + mean   # [node_pair_num,dN]
            return N,mean,logstd
        if self.type == 'Both':
            mean = self.mean(H)  # [node_pair_num,dn*cat_dim]
            logstd = self.logstd(H)
            gausian_noise = torch.randn_like(mean)
            N = gausian_noise*torch.exp(logstd) + mean   # [node_pair_num,dn*cat_dim]
            N = N.view(H.shape[0],self.cat,-1) #[node_pair_num,cat_dim,dN]

            q = self.q(H) # [node_pair_num,cat_dim]
            eps = 1e-7
            uniform = torch.rand_like(q) # [node_pair_num,cat_dim] sample from uniform distribution
            gumbel = -torch.log(-torch.log(uniform+eps)+eps)  
            Z = nn.functional.softmax((q+gumbel)/temp,dim=-1) # [node_pair_num,catdim]
            Z = Z.view(-1,1,self.cat)  # [node_pair_num,1,cat_dim]
            M = torch.matmul(Z,N).squeeze() #[node_pair_num,dn]  
            return M,mean,logstd,q

#现在这个还没有像relearn那样多一个encoder1出来
class VI_relearn(nn.Module):
    def __init__(self,in_dim,out_dim,categorical_dim,device):
        super(VI_relearn,self).__init__()
        self.device = device
        self.out_dim = out_dim
        self.cat = categorical_dim

        self.mean   = nn.Parameter(torch.FloatTensor(categorical_dim, out_dim).uniform_(-0.5 / out_dim, 0.5 / out_dim)) # init the mean of GMM
        self.logstd = nn.Parameter(torch.FloatTensor(categorical_dim, out_dim).uniform_(-0.5 / out_dim, 0.5 / out_dim)) # init the logstd of GMM
        self.stdnorm = torch.distributions.Normal(torch.zeros(categorical_dim, out_dim), torch.ones(categorical_dim, out_dim)) # standard norm distribution

        self.mlp = nn.Sequential(nn.Linear(in_dim, in_dim//2),
                                 nn.ReLU(inplace=True), # inplace 节约空间
                                 nn.Linear(in_dim//2, categorical_dim))
        self.dropout = nn.Dropout(0.5)

    def forward(self,G,temp):
        # gumbel
        z = F.gumbel_softmax(self.mlp(G),tau=temp,hard=False)
        # GMM
        std_norm = self.stdnorm.sample().to(self.device)
        GMM = self.mean+self.logstd.exp() * std_norm
        GMM = self.dropout(GMM)
        H0 = z @ GMM  # @ is matrix mul
        return H0,z


class shared_decoder(nn.Module):
    def __init__(self,in_feats,out_dim,**kwargs):
        super(shared_decoder,self).__init__()
        self.mlp = nn.Linear(in_feats,out_dim) 
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        x = self.mlp(x)
        x = self.relu(x)
        return x

class private_decoder(nn.Module):
    def __init__(self,in_feats,hid_dim,out_dim,**kwargs):
        super(private_decoder,self).__init__()
        self.mlp1 = nn.Linear(in_feats,hid_dim) 
        self.mlp2 = nn.Linear(hid_dim,out_dim)
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        x = self.mlp1(x)
        x = self.relu(x)
        x = self.mlp2(x)
        x = self.relu(x)
        return x

class relation_classifier(nn.Module):
    def __init__(self,in_feats,hidden_dim,out_dim,**kwargs):
        super(relation_classifier,self).__init__()
        self.mlp1 = nn.Linear(in_feats,hidden_dim) 
        self.mlp2 = nn.Linear(hidden_dim,out_dim)
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        x = self.mlp1(x)
        x = self.relu(x)
        x = self.mlp2(x)
        #x = nn.functional.softmax(x,dim=-1) the softmax is implemented in the focal loss function
        return x 

class discriminator(nn.Module):
    def __init__(self,in_feats,hidden_dim,**kwargs):
        super(discriminator,self).__init__()
        self.grl = GRL()
        self.mlp1 = nn.Linear(in_feats,hidden_dim) 
        self.mlp2 = nn.Linear(hidden_dim,2)
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x,rate):
        x = self.grl(x,rate)
        x = self.mlp1(x)
        x = self.relu(x)
        x = self.mlp2(x)
        #x = torch.softmax(x) the softmax is implemented in the crossentropy loss function
        return x

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x,rate):
        ctx.rate = rate
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.rate
        return grad_output, None

class GRL(nn.Module):
    def forward(self, input,rate):
        return GradReverse.apply(input,rate)
