import torch
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

def generate_all_node_pair(node_num,edge_index,node_label,ntype_etype_mapping):
    '''
    return
        all_node_pair :: shape [N ,2]
        all_node_pair_label :: shape [N]
    '''
    #
    edge_index = edge_index.T
    row = torch.arange(node_num)
    col = torch.arange(node_num)
    row = row.unsqueeze(1).repeat(1,node_num).unsqueeze(2)
    col = col.unsqueeze(0).repeat(node_num,1).unsqueeze(2)
    all_node_pair = torch.cat((row,col),dim=2).view(-1,2)
    all_node_pair_label = torch.zeros(all_node_pair.shape[0],dtype=torch.long)
    # traverse the edge index is faster
    '''
    for node_pair in edge_index:
        src = node_pair[0]  #source node index
        dst = node_pair[1]  #destination node index
        ntype_pair = [node_label[src],node_label[dst]] # [souce node type,destination node type]
        node_pair_label = ntype_etype_mapping[ntype_pair] # this node pair's label, looking up from the dict
        # return the corresponding index of this node pair in all_node_pair
        i = torch.nonzero(torch.all(torch.eq(all_node_pair,node_pair),dim=-1))
        all_node_pair_label[i] = node_pair_label # value this node_pair's type
    '''
    node_label_num = torch.tensor([node_label.max()-node_label.min()+1]*all_node_pair.shape[0],dtype=torch.long)
    #首先获得节点对中每对节点,得到其节点标签的pair TODO
    ntype_label_for_each_pair = node_label[all_node_pair]
    #利用节点标签的pair获取src和dst，目的是使得[n_i,n_j]和[n_j,n_i]拥有相同的标签 TODO
    src = torch.min(ntype_label_for_each_pair,dim=-1).values
    dst = torch.max(ntype_label_for_each_pair,dim=-1).values
    #利用关系式求出node pair的标签all_node_pair_label  TODO
    all_node_pair_label = ((node_label_num+node_label_num-(src-1))*((src-1)+1)/2+(dst-src)+1).long()
    print(all_node_pair_label.view(5,5))
    raise RuntimeError
    return all_node_pair,all_node_pair_label

if __name__ == "__main__":
    node_label = torch.tensor([0,1,2,0,2])
    edge_index = torch.tensor([[0,0,0,1,1,2,2,3,4],[0,2,4,1,4,2,3,3,4]])
    ntype_etype_mapping = torch.zeros([3,3],dtype=torch.long)
    i = 1
    for src_node_type in range(0,3):
        for tgt_node_type in range(src_node_type,3):
            ntype_etype_mapping[src_node_type,tgt_node_type] = i
            ntype_etype_mapping[tgt_node_type,src_node_type] = i
            i+=1
    node_pair,node_pair_label = generate_all_node_pair(5,edge_index,node_label,ntype_etype_mapping)
    print(node_pair)
    print(node_pair_label.view(5,5))
    ##检查一对元素是否在edge index中
    # check = torch.tensor([5,3])
    # print(edge_index.T)
    # print(torch.eq(edge_index.T,check))
    # print(torch.all(torch.eq(edge_index.T,check),dim=-1))
    # print(torch.all(torch.eq(edge_index.T,check),dim=-1).any())
    
