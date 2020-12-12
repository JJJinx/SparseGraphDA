import torch
import dgl
import dgl.nn.pytorch as dglnn
import dgl.function as fn
from torch import nn

class ScorePredictor(nn.Module):
    def forward(self, subgraph, x):
        with subgraph.local_scope():
            subgraph.ndata['x'] = x
            subgraph.apply_edges(dgl.function.u_dot_v('x', 'x', 'score')) 
            return subgraph.edata['score']

class SGD_MRVGAE(nn.Module):
    def __init__(self,in_feats,n_hidden,out_feats,device,distype = 'Norm',categorical_dim=None,**kwargs):
        super(SGD_MRVGAE, self).__init__()
        self.n_hidden = n_hidden
        self.cat = categorical_dim
        self.distype = distype
        self.device = device

        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        ## encoder
        self.enc_gcn = nn.ModuleList()
        self.enc_gcn.append(dglnn.GraphConv(in_feats,n_hidden[0]))
        self.enc_gcn.append(dglnn.GraphConv(n_hidden[0],n_hidden[1]))
        #self.enc_gcn1 = dglnn.GraphConv(in_feats,n_hidden[0])
        #self.enc_gcn2 = dglnn.GraphConv(n_hidden[0],n_hidden[1])
        if distype =='Rel':
            ## variation inference
            pass
            ## decoder
            pass
        if distype == 'Both':
            # variation inference
            # instead of n_hidden[1], the input dimention is n_hidden[2]
            self.vi_mlp_mean   = nn.Linear(n_hidden[1],n_hidden[2]) 
            self.vi_mlp_logstd = nn.Linear(n_hidden[1],n_hidden[2]) 
            self.vi_q          = nn.Linear(n_hidden[1],categorical_dim)
            ## decoder
            self.dec_mlp1 = nn.Linear(n_hidden[3],n_hidden[4]) 
            self.dec_mlpX = nn.Linear(n_hidden[4],out_feats)
            ## edge classifier
            self.cls_mlpA = nn.Linear(n_hidden[3],categorical_dim)#输出one-hot 编码
            
    
    def inference(self,graph,pos_graph,neg_graph,x,temp):
        '''
            model:: the trained model
            graph:: train graph
            input_features:: all node features
            pos_nodepair:: graph that regards the pos nodepair as edges  TODO
            neg_nodepair:: graph that regards the neg nodepair as edges  TODO
        '''
        ## encode
        h = self.enc_gcn[0](graph,x)
        h = self.enc_gcn[1](graph,h)
        pos_graph.ndata['h'] = h
        pos_graph.apply_edges(dgl.function.u_add_v('h', 'h', 'npair_emb')) 
        pos_npemb = pos_graph.edata['npair_emb']
        neg_graph.ndata['h'] = h
        neg_graph.apply_edges(dgl.function.u_add_v('h', 'h', 'npair_emb')) 
        neg_npemb = neg_graph.edata['npair_emb']
        ## variation inference
        if self.distype=='Rel':
            pass
        if self.distype=='Both':
            ## for pos graph
            pos_mean = self.vi_mlp_mean(pos_npemb)  # [pos,dn*cat_dim]
            pos_logstd = self.vi_mlp_logstd(pos_npemb)
            gausian_noise = torch.randn(pos_mean.size(0),pos_mean.size(1)).to(self.device)
            posN = gausian_noise*torch.exp(pos_logstd) + pos_mean   # [pos,dn*cat_dim]
            posN = posN.view(pos_npemb.shape[0],self.cat,-1) #[pos,cat_dim,dN]

            posq = self.vi_q(pos_npemb) #shape [pos,cat_dim]
            eps = 1e-7
            uniform = torch.rand_like(posq) # shape[pos,cat_dim] sample from uniform distribution
            gumbel = -torch.log(-torch.log(uniform+eps)+eps)  
            posZ = nn.functional.softmax((posq+gumbel)/temp,dim=-1) # shape [pos,catdim]
            posZ = posZ.view(-1,1,self.cat)  # [pos,1,cat_dim]
            posM = torch.matmul(posZ,posN).squeeze() #[pos,dn]  
            ## for neg graph
            neg_mean = self.vi_mlp_mean(neg_npemb)
            neg_logstd = self.vi_mlp_logstd(neg_npemb)
            gausian_noise = torch.randn(neg_mean.size(0),neg_mean.size(1)).to(self.device)
            negN = gausian_noise*torch.exp(neg_logstd) + neg_mean   # [neg,dn*cat_dim]
            negN = negN.view(neg_npemb.shape[0],self.cat,-1) #[neg,cat_dim,dN]

            negq = self.vi_q(neg_npemb) #shape [neg,cat_dim]
            eps = 1e-7
            uniform = torch.rand_like(negq) # shape[neg,cat_dim] sample from uniform distribution
            gumbel = -torch.log(-torch.log(uniform+eps)+eps)  
            negZ = nn.functional.softmax((negq+gumbel)/temp,dim=-1) # shape [neg,catdim]
            negZ = negZ.view(-1,1,self.cat)  # [neg,1,cat_dim]

            negM = torch.matmul(negZ,negN).squeeze() #[neg,dn]  
        # decode and classify
        ## for pos graph
        posX = self.dec_mlp1(posM)
        posX = self.relu(posX)
        posX = self.dec_mlpX(posX)
        posX = self.relu(posX)

        posA = self.cls_mlpA(posM)
        posA = nn.functional.softmax(posA,dim=-1)
        ## for neg graph
        negX = self.dec_mlp1(negM)
        negX = self.relu(negX)
        negX = self.dec_mlpX(negX)
        negX = self.relu(negX)

        negA = self.cls_mlpA(negM)
        negA = nn.functional.softmax(negA,dim=-1)

        return posA,posX,negA,negX

    def forward(self, blocks, x,pos_graph,neg_graph,temp):
        h = x
        ## encode
        for l,(layer, block) in enumerate(zip(self.enc_gcn,blocks)):
            h = layer(block,h)
            if l != 1:
                h = self.relu(h)
                h = self.dropout(h) # h shape [num_dst_nodes for each batch,dh]
        ## calculating the dst noda pair embedding
        pos_graph.ndata['h'] = h
        pos_graph.apply_edges(dgl.function.u_add_v('h', 'h', 'npair_emb')) 
        pos_npemb = pos_graph.edata['npair_emb']
        neg_graph.ndata['h'] = h
        neg_graph.apply_edges(dgl.function.u_add_v('h', 'h', 'npair_emb')) 
        neg_npemb = neg_graph.edata['npair_emb']
        ## variation inference
        if self.distype=='Rel':
            pass
        if self.distype=='Both':
            ## for pos graph
            pos_mean = self.vi_mlp_mean(pos_npemb)  # [pos,dn*cat_dim]
            pos_logstd = self.vi_mlp_logstd(pos_npemb)
            gausian_noise = torch.randn(pos_mean.size(0),pos_mean.size(1)).to(self.device)
            posN = gausian_noise*torch.exp(pos_logstd) + pos_mean   # [pos,dn*cat_dim]
            posN = posN.view(pos_npemb.shape[0],self.cat,-1) #[pos,cat_dim,dN]

            posq = self.vi_q(pos_npemb) #shape [pos,cat_dim]
            eps = 1e-7
            uniform = torch.rand_like(posq) # shape[pos,cat_dim] sample from uniform distribution
            gumbel = -torch.log(-torch.log(uniform+eps)+eps)  
            posZ = nn.functional.softmax((posq+gumbel)/temp,dim=-1) # shape [pos,catdim]
            posZ = posZ.view(-1,1,self.cat)  # [pos,1,cat_dim]
            posM = torch.matmul(posZ,posN).squeeze() #[pos,dn]  
            ## for neg graph
            neg_mean = self.vi_mlp_mean(neg_npemb)
            neg_logstd = self.vi_mlp_logstd(neg_npemb)
            gausian_noise = torch.randn(neg_mean.size(0),neg_mean.size(1)).to(self.device)
            negN = gausian_noise*torch.exp(neg_logstd) + neg_mean   # [neg,dn*cat_dim]
            negN = negN.view(neg_npemb.shape[0],self.cat,-1) #[neg,cat_dim,dN]

            negq = self.vi_q(neg_npemb) #shape [neg,cat_dim]
            eps = 1e-7
            uniform = torch.rand_like(negq) # shape[neg,cat_dim] sample from uniform distribution
            gumbel = -torch.log(-torch.log(uniform+eps)+eps)  
            negZ = nn.functional.softmax((negq+gumbel)/temp,dim=-1) # shape [neg,catdim]
            negZ = negZ.view(-1,1,self.cat)  # [neg,1,cat_dim]

            negM = torch.matmul(negZ,negN).squeeze() #[neg,dn]  
        # decode
        ## for pos graph
        posX = self.dec_mlp1(posM)
        posX = self.relu(posX)
        posX = self.dec_mlpX(posX)
        posX = self.relu(posX)

        posA = self.cls_mlpA(posM)
        posA = nn.functional.softmax(posA,dim=-1)
        ## for neg graph
        negX = self.dec_mlp1(negM)
        negX = self.relu(negX)
        negX = self.dec_mlpX(negX)
        negX = self.relu(negX)

        negA = self.cls_mlpA(negM)
        negA = nn.functional.softmax(negA,dim=-1)

        return [posA,negA,posX,negX,pos_mean,neg_mean,pos_logstd,neg_logstd,posq,negq]
        
'''
class MRVGAE(torch.nn.Module):
    def __init__(self, in_dim,hidden_dim,adj,device,distype ='Norm',categorical_dim=None,**kwargs):
        super(MRVGAE, self).__init__()
        self.adj = adj
        self.device = device
        self.distype = distype

        self.gcn1 = GraphConvSparse(in_dim,hidden_dim[0],adj) 
        self.dropout_layers = nn.Dropout(0.1)
        self.gcn2 = GraphConvSparse(hidden_dim[0],hidden_dim[1],adj)

        self.VI = VI(hidden_dim[1],hidden_dim[2],self.device,distype=distype,categorical_dim=categorical_dim)
        if distype == 'both':
            self.mlp = nn.ConvTranspose1d
            self.mlp_A = nn.Linear(hidden_dim[3],1) 
            self.mlp_X = nn.Linear(hidden_dim[3],hidden_dim[4])
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()

        else:
            self.mlp1 = nn.Linear(2*hidden_dim[2],hidden_dim[3])
            self.relu = nn.ReLU()
            self.mlp2 = nn.Linear(hidden_dim[3],hidden_dim[4])
            self.sm = nn.Sigmoid()

    def forward(self, x,temperature=None):
        h = self.gcn1(x)
        h = self.dropout_layers(h)
        h = self.gcn2(h) #8

        hj = h.unsqueeze(0).repeat(h.shape[0],1,1) #[N_i,N_j,Dh],vector value only related to j
        hi = h.unsqueeze(1).repeat(1,h.shape[0],1) #[N_i,N_j,Dh],vector value only related to i
        #hij = torch.cat((hi,hj),2) 
        hij = hi+hj #[N,N,Dh]
        # h = torch.mm(h,h.T)*hij
        if self.distype == 'Norm':
            z,self.mean,self.logstd = self.VI(hij)

            a = self.mlp1(z)
            a = self.relu(a)
            #a = self.dropout_layers(a)
            a = self.mlp2(a)
            A = self.sm(a)
            return A

        if self.distype == 'Gumbel':
            hij = torch.flatten(hij,end_dim=1) #[N*N,Dh]
            [q,z] = self.VI(hij,temperature)
            a = self.mlp1(z) #[N*N,hid_dim]
            a = self.relu(a)
            #a = self.dropout_layers(a)
            a = self.mlp2(a)
            A = self.sm(a).view(h.shape[0],h.shape[0],-1)
            return [A,q]

        if self.distype == 'both':
            #the parameter of gaussian is not in the NN

            hij = torch.flatten(hij,end_dim=1) # [N*N, Dh]
            [M,Z,q,N,mean,logstd] = self.VI(hij,temperature)
            
            a = self.mlp(M)
            a = self.relu(a)
            A = self.mlp_A(a)
            A = self.sigmoid(A)
            X = self.mlp_X(a)
            X = self.relu(X)

            return [A,X,q,mean,logstd]

class VI(nn.Module):
    def __init__(self,in_dim,out_dim,device,distype='Norm',categorical_dim=None):
        super(VI,self).__init__()
        self.device = device
        self.type = distype
        self.cat = categorical_dim        
        self.out_dim = out_dim

        if distype == 'Norm':
            self.mlp_mean   = nn.Linear(in_dim,out_dim) 
            self.mlp_logstd = nn.Linear(in_dim,out_dim)
        if distype == 'Gumbel':
            self.q = nn.Linear(in_dim,out_dim*self.cat)# shape [N*N,outdim*categorical_dim]
        if distype == 'both':
            # set mean and logstd as torch variable instead of learning from the nn,they are global parameters
            self.mean   = torch.tensor((self.cat,self.out_dim),requires_grad=True) #[cat,dz]
            self.logstd   = torch.tensor((self.cat,self.out_dim),requires_grad=True) #[cat,dz]
            torch.nn.init.xavier_normal_(self.mean, gain=1.0)
            torch.nn.init.xavier_normal_(self.logstd, gain=1.0)
            self.q          = nn.Linear(in_dim,categorical_dim)

    def forward(self,H,temperature=None):
        if self.type == 'Norm':
            mean = self.mlp_mean(H)
            logstd = self.mlp_logstd(H)
            gausian_noise = torch.randn(H.size(0),self.out_dim).to(self.device)
            Z = gausian_noise*torch.exp(logstd).to(self.device) + mean #make sure that the std is positive,hence learn the logstd
            return Z,mean,logstd
        if self.type == 'Gumbel':
            q = self.q(H) # shape [N*N,dz*2]
            q = q.view(-1,self.out_dim,self.cat) # shape[N*N,dz,2]
            eps = 1e-7
            # sample from gumbel
            u = torch.rand_like(q) # shape[N*N,dz,2] sample from uniform distribution
            g = -torch.log(-torch.log(u+eps)+eps)  

            # Gumbel-Softmax sample
            z = nn.functional.softmax((q+g)/temperature,dim=-1)
            z = z.view(-1,self.out_dim*self.cat)
            return [q,z]
        if self.type == 'both':
            gausian_noise = torch.randn(self.cat,self.out_dim).to(self.device)
            N = gausian_noise*torch.exp(self.logstd).to(self.device) + self.mean   # [cat_dim,outdim]

            q = self.q(H) #shape [N*N,cat_dim]
            eps = 1e-7
            uniform = torch.rand_like(q) # shape[N*N,cat_dim] sample from uniform distribution
            gumbel = -torch.log(-torch.log(uniform+eps)+eps)  
            Z = nn.functional.softmax((q+gumbel)/temperature,dim=-1) # shape [N*N,catdim]            
            M = torch.matmul(Z,N).squeeze() #[N*N,outdim]  
            return [M,Z,q,N,mean,logstd]
'''