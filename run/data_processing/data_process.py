import dgl
import torch
import numpy as np

def edge_index_to_adj(node_num,edge_index):
    '''
        edge_index :: [n,2]
    '''
    adj = torch.zeros([node_num,node_num],dtype=torch.long)
    for i in edge_index:
        adj[i[0],i[1]] = 1
    return adj

def Is_self_loop_exist(edge_index,node_num):
    num_self_loop = torch.eq(edge_index[0]-edge_index[1],0).sum()
    symble = (num_self_loop==node_num)
    return symble

def generate_all_node_pair_adj(node_num,edge_index,node_label,node_label_num,adj):
    '''
    input
        edge_index :: tensor shape [2,N_e] 
        node_label_num :: tensor shape [1]
        adj :: sparse format for adjacent matrix
    return
        all_node_pair_label :: tensor shape [N,N]. For all node pairs,no matter whether the edge exists
    '''
    edge_index = edge_index.T
    row = torch.arange(node_num)
    col = torch.arange(node_num)
    row = row.unsqueeze(1).repeat(1,node_num).unsqueeze(2)
    col = col.unsqueeze(0).repeat(node_num,1).unsqueeze(2)
    all_node_pair = torch.cat((row,col),dim=2)
    ntype_label_for_each_pair = node_label[all_node_pair]
    #make [n_i,n_j]和[n_j,n_i] has the same np label
    src = torch.min(ntype_label_for_each_pair,dim=-1).values # shape [N,N]
    dst = torch.max(ntype_label_for_each_pair,dim=-1).values # shape [N,N]
    #label the existence edges; make not existence edges' label =0
    all_node_pair_label = ((node_label_num+node_label_num-(src-1))*((src-1)+1)/2+(dst-src)+1).type(torch.int32)
    max_np_label = all_node_pair_label.max()
    all_node_pair_label = all_node_pair_label*adj.to_dense().type(torch.int32)
    return all_node_pair,all_node_pair_label,max_np_label

def generate_all_node_pair(node_num,edge_index,node_label,node_label_num,adj):
    '''
    input
        edge_index :: tensor shape [2,N_e] 
        node_label_num :: tensor shape [1]
        adj :: sparse format for adjacent matrix
    return
        all_node_pair :: tensor shape [N,N,2]
        all_node_pair_label :: tensor shape [N,N]. For all node pairs,no matter whether the edge exists
    '''
    #row = np.expand_dims(np.expand_dims(np.arange(node_num),axis=1).repeat(node_num,axis=1),axis=2)
    #col = np.expand_dims(np.expand_dims(np.arange(node_num),axis=0).repeat(node_num,axis=0),axis=2)
    #all_node_pair = np.concatenate((row,col),axis=2)
    edge_index = edge_index.T
    row = torch.arange(node_num)
    col = torch.arange(node_num)
    row = row.unsqueeze(1).repeat(1,node_num).unsqueeze(2)
    col = col.unsqueeze(0).repeat(node_num,1).unsqueeze(2)
    all_node_pair = torch.cat((row,col),dim=2)

    #ntype_label_for_each_pair = node_label.numpy()[all_node_pair]
    ntype_label_for_each_pair = node_label[all_node_pair]
    #make [n_i,n_j]和[n_j,n_i] has the same np label
    src = torch.min(ntype_label_for_each_pair,dim=-1).values # shape [N,N]
    dst = torch.max(ntype_label_for_each_pair,dim=-1).values # shape [N,N]
    # src = ntype_label_for_each_pair.min(axis=-1)
    # dst = ntype_label_for_each_pair.max(axis=-1)
    #label the existence edges; make not existence edges' label =0
    all_node_pair_label = ((node_label_num+node_label_num-(src-1))*((src-1)+1)/2+(dst-src)+1).type(torch.int32)
    max_np_label = all_node_pair_label.max()
    all_node_pair_label = all_node_pair_label*adj.to_dense().type(torch.int32)
    #all_node_pair_label = ((int(node_label_num)+int(node_label_num)-(src-1))*((src-1)+1)/2+(dst-src)+1)*adj.to_dense().numpy()
    return all_node_pair,all_node_pair_label,max_np_label

def generate_all_node_pair_minus_class(node_num,edge_index,node_label,node_label_num,adj):
    edge_index = edge_index.T
    row = torch.arange(node_num)
    col = torch.arange(node_num)
    row = row.unsqueeze(1).repeat(1,node_num).unsqueeze(2)
    col = col.unsqueeze(0).repeat(node_num,1).unsqueeze(2)
    all_node_pair = torch.cat((row,col),dim=2)

    #ntype_label_for_each_pair = node_label.numpy()[all_node_pair]
    ntype_label_for_each_pair = node_label[all_node_pair]
    #make [n_i,n_j]和[n_j,n_i] has the same np label
    src = torch.min(ntype_label_for_each_pair,dim=-1).values # shape [N,N]
    dst = torch.max(ntype_label_for_each_pair,dim=-1).values # shape [N,N]
    # src = ntype_label_for_each_pair.min(axis=-1)
    # dst = ntype_label_for_each_pair.max(axis=-1)
    #label the existence edges; make not existence edges' label =0
    all_node_pair_label = ((node_label_num+node_label_num-(src-1))*((src-1)+1)/2+(dst-src)+1).type(torch.int32)-1
    max_pos_np_label = all_node_pair_label.max()
    adj = adj.to_dense().type(torch.int32)
    adj_one_indice = torch.where(adj!=0)
    adj_zero_indice = torch.where(adj==0)
    adj[adj_one_indice[0],adj_one_indice[1]] = 0 # start label for pos node pair
    adj[adj_zero_indice[0],adj_zero_indice[1]] = max_pos_np_label+1 # start label for neg node pair
    #
    all_node_pair_label = all_node_pair_label+adj
    max_neg_np_label = all_node_pair_label.max()
    #all_node_pair_label = ((int(node_label_num)+int(node_label_num)-(src-1))*((src-1)+1)/2+(dst-src)+1)*adj.to_dense().numpy()
    return all_node_pair,all_node_pair_label,max_pos_np_label,max_neg_np_label

def generate_mapping_M(node_label_num,np_type_num):
    mapping_M = torch.zeros((np_type_num,node_label_num),dtype=torch.long)
    for src in range(node_label_num):
        for dst in range(node_label_num):
            if src > dst:
                new_src = dst
                new_dst = src
            else:
                new_src = src
                new_dst = dst
            np_type = int((node_label_num+node_label_num-(new_src-1))*((new_src-1)+1)/2+(new_dst-new_src)+1)
            mapping_M[np_type,src] =1 
    return mapping_M

## TODO generate the mapping matrix for considering the minus classes
def generate_mapping_M_minus_class(node_label_num,np_type_num,pos_np_type_num):
    mapping_M = torch.zeros((np_type_num,node_label_num),dtype=torch.long)
    
    for src in range(node_label_num):
        for dst in range(node_label_num):

            if src > dst:
                new_src = dst
                new_dst = src

            else:
                new_src = src
                new_dst = dst

            np_type = int((node_label_num+node_label_num-(new_src-1))*((new_src-1)+1)/2+(new_dst-new_src)+1)-1
            mapping_M[np_type,src] =1
    
    mapping_M[pos_np_type_num:,:] = mapping_M[:pos_np_type_num,:]
    return mapping_M



if __name__ == "__main__":
    node_label = torch.tensor([0,1,2,0,2])
    edge_index = torch.tensor([[0,0,0,1,1,2,2,3,4],[0,2,4,1,4,2,3,3,4]])
    g = dgl.to_bidirected(dgl.graph((edge_index[0],edge_index[1])))
    # ntype_etype_mapping = torch.zeros([3,3],dtype=torch.long)
    # i = 1
    # for src_node_type in range(0,3):
    #     for tgt_node_type in range(src_node_type,3):
    #         ntype_etype_mapping[src_node_type,tgt_node_type] = i
    #         ntype_etype_mapping[tgt_node_type,src_node_type] = i
    #         i+=1
    max_np_label,node_pair_label,max_pos_np_label,max_neg_np_label = generate_all_node_pair_minus_class(5,edge_index,node_label,3,g.adjacency_matrix())
    M = generate_mapping_M_minus_class(4,20,10)
    #print(node_pair_label,max_np_label)