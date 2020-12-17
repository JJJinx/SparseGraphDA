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
        h = self.relu(h)
        h = self.enc_gcn[1](graph,h)
        h = self.relu(h)
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
            h = self.relu(h)
            if l != 1:
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

class SGD_MRVGAE2(nn.Module):
    def __init__(self,in_feats,n_hidden,out_feats,device,distype = 'Norm',categorical_dim=None,**kwargs):
        super(SGD_MRVGAE2, self).__init__()
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
            ## decoder
            self.dec_mlp1 = nn.Linear(n_hidden[2],n_hidden[4]) 
            self.dec_mlpX = nn.Linear(n_hidden[4],out_feats)
            ## edge classifier
            self.cls_mlp1 = nn.Linear(n_hidden[2],int(n_hidden[2]/2))
            self.cls_mlpA = nn.Linear(int(n_hidden[2]/2),categorical_dim)#输出one-hot 编码
            
    
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
        h = self.relu(h)
        h = self.enc_gcn[1](graph,h)
        h = self.relu(h)
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
            pos_mean = self.vi_mlp_mean(pos_npemb)  # [pos,dn]
            pos_logstd = self.vi_mlp_logstd(pos_npemb)
            gausian_noise = torch.randn(pos_mean.size(0),pos_mean.size(1)).to(self.device)
            posN = gausian_noise*torch.exp(pos_logstd) + pos_mean   # [pos,dn]
            #posN = posN.view(pos_npemb.shape[0],self.cat,-1) #[pos,cat_dim,dN]
            ## for neg graph
            neg_mean = self.vi_mlp_mean(neg_npemb)
            neg_logstd = self.vi_mlp_logstd(neg_npemb)
            gausian_noise = torch.randn(neg_mean.size(0),neg_mean.size(1)).to(self.device)
            negN = gausian_noise*torch.exp(neg_logstd) + neg_mean   # [neg,dn]
            #negN = negN.view(neg_npemb.shape[0],self.cat,-1) #[neg,cat_dim,dN]

        # decode and classify
        ## for pos graph
        posX = self.dec_mlp1(posN)
        posX = self.relu(posX)
        posX = self.dec_mlpX(posX)
        posX = self.relu(posX)

        posA = self.cls_mlp1(posN)
        posA = self.relu(posA)
        posA = self.cls_mlpA(posA)
        posA = nn.functional.softmax(posA,dim=-1)
        ## for neg graph
        negX = self.dec_mlp1(negN)
        negX = self.relu(negX)
        negX = self.dec_mlpX(negX)
        negX = self.relu(negX)

        negA = self.cls_mlp1(negN)
        negA = self.relu(negA)
        negA = self.cls_mlpA(negA)
        negA = nn.functional.softmax(negA,dim=-1)

        return posA,posX,negA,negX

    def forward(self, blocks, x,pos_graph,neg_graph,temp):
        h = x
        ## encode
        for l,(layer, block) in enumerate(zip(self.enc_gcn,blocks)):
            h = layer(block,h)
            h = self.relu(h)
            if l != 1:
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
            pos_mean = self.vi_mlp_mean(pos_npemb)  # [pos,dn]
            pos_logstd = self.vi_mlp_logstd(pos_npemb)
            gausian_noise = torch.randn(pos_mean.size(0),pos_mean.size(1)).to(self.device)
            posN = gausian_noise*torch.exp(pos_logstd) + pos_mean   # [pos,dn]
            #posN = posN.view(pos_npemb.shape[0],self.cat,-1) #[pos,cat_dim,dN]

            ## for neg graph
            neg_mean = self.vi_mlp_mean(neg_npemb)
            neg_logstd = self.vi_mlp_logstd(neg_npemb)
            gausian_noise = torch.randn(neg_mean.size(0),neg_mean.size(1)).to(self.device)
            negN = gausian_noise*torch.exp(neg_logstd) + neg_mean   # [neg,dn]
            #negN = negN.view(neg_npemb.shape[0],self.cat,-1) #[neg,cat_dim,dN]

        # decode
        ## for pos graph
        posX = self.dec_mlp1(posN)
        posX = self.relu(posX)
        posX = self.dec_mlpX(posX)
        posX = self.relu(posX)

        posA = self.cls_mlp1(posN)
        posA = self.relu(posA)
        posA = self.cls_mlpA(posA)
        posA = nn.functional.softmax(posA,dim=-1)
        ## for neg graph
        negX = self.dec_mlp1(negN)
        negX = self.relu(negX)
        negX = self.dec_mlpX(negX)
        negX = self.relu(negX)

        negA = self.cls_mlp1(negN)
        negA = self.relu(negA)
        negA = self.cls_mlpA(negA)
        negA = nn.functional.softmax(negA,dim=-1)

        return [posA,negA,posX,negX,pos_mean,neg_mean,pos_logstd,neg_logstd]
