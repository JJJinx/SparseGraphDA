import numpy as np
import torch
import dgl
import scipy.sparse as sp
from scipy.sparse import coo_matrix


def sparse_to_tuple(sparse_mx):
    '''
       return:: 
            coords:: edges cordinates
            values:: edges value
            shape::  adj's shape
    '''
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def mask_test_edges(adj):
    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0
    # 保留上三角的元素
    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0] #[row,col] 仅有单向边
    edges_all = sparse_to_tuple(adj)[0] #[row,col] 包含了双向的边
    num_test = int(np.floor(edges.shape[0] / 10.)) #10%
    num_val = int(np.floor(edges.shape[0] / 20.))  #0.05%

    all_edge_idx = list(range(edges.shape[0])) # each edge's index 仅有单向边
    np.random.shuffle(all_edge_idx) #shuffle the index
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5): # check whether a is member of b
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)
    # negtive sampling
    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j: #首先不能是自环
            continue
        if ismember([idx_i, idx_j], edges_all):#第二不可以是positive的边，即这个对节点不应当连接
            continue
        if ismember([idx_j, idx_i], edges_all):#第二不可以是positive的边，即这个对节点不应当连接
            continue
        if test_edges_false: # if [idx_i, idx_j] not in edges_all and test_edges_false is not empty 
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if ismember([idx_j, idx_i], edges_all):
            continue
        # if ismember([idx_i, idx_j], val_edges):
        #     continue
        # if ismember([idx_j, idx_i], val_edges):
        #     continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges) #make sure each set has no intersection
    weight = np.ones(train_edges.shape[0]) #仅有单向边的train set中每条边的权重

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    raise RuntimeError

def neg_sampling(train_pos_nodepair,val_pos_nodepair,test_pos_nodepair,graph):
    num_node = graph.nodes().shape[0]
    val_neg_nodepair = []
    def ismember(a, b, tol=5): # check whether a is member of b 
        # a :: a list
        # b :: a array
        ##for test
        # a = np.array([1,2])
        # b = np.array([[2,3],[1,2],[0,0]]).T
        # print((np.array(a) - b.T))
        # print(np.all(np.round(np.array(a) - b.T)==0,axis=1))
        # print(np.any(np.all(np.round(np.array(a) - b.T)==0,axis=1)))
        # raise RuntimeError
        rows_close = np.all(np.round(np.array(a) - b, tol) == 0, axis=1)
        return np.any(rows_close)
    ## for val neg
    while len(val_neg_nodepair)<val_pos_nodepair.shape[0]:
        idx_i = np.random.randint(0, num_node)
        idx_j = np.random.randint(0, num_node)
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_pos_nodepair):
            continue
        if ismember([idx_j, idx_i], train_pos_nodepair):
            continue
        if ismember([idx_i, idx_j], val_pos_nodepair):
            continue
        if ismember([idx_j, idx_i], val_pos_nodepair):
            continue
        if val_neg_nodepair:
            if ismember([idx_j, idx_i], np.array(val_neg_nodepair)):
                continue
            if ismember([idx_i, idx_j], np.array(val_neg_nodepair)):
                continue
        val_neg_nodepair.append([idx_i, idx_j])
    val_neg_nodepair = torch.tensor(val_neg_nodepair)
    ## for test neg 
    test_neg_nodepair = []
    while len(test_neg_nodepair)<test_pos_nodepair.shape[0]:
        idx_i = np.random.randint(0, num_node)
        idx_j = np.random.randint(0, num_node)
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_pos_nodepair):
            continue
        if ismember([idx_j, idx_i], train_pos_nodepair):
            continue
        if ismember([idx_i, idx_j], val_pos_nodepair):
            continue
        if ismember([idx_j, idx_i], val_pos_nodepair):
            continue
        if ismember([idx_i, idx_j], test_pos_nodepair):
            continue
        if ismember([idx_j, idx_i], test_pos_nodepair):
            continue        
        if test_neg_nodepair:
            if ismember([idx_j, idx_i], np.array(test_neg_nodepair)):
                continue
            if ismember([idx_i, idx_j], np.array(test_neg_nodepair)):
                continue
        test_neg_nodepair.append([idx_i, idx_j])
    test_neg_nodepair = torch.tensor(test_neg_nodepair)
    return val_neg_nodepair,test_neg_nodepair

def data_process(graph):
    # train pos edges::graph形式 用于训练与inference阶段的推断
    # val pos edges::节点对形式 ，用于模型指标的计算
    # val neg edges:: 负采样得到，节点对形式，用于模型指标的计算
    # test pos edges::节点对形式，用于模型指标的计算
    # test neg edges:: 负采样得到，节点对形式，用于模型指标的计算
    row = graph.edges()[0]
    col = graph.edges()[1]
    data = np.ones_like(row)
    adj_coo = coo_matrix((data,(row,col)),shape=(2708,2708))
    ## test
    edge_all = sparse_to_tuple(adj_coo)[0] #即节点对

    edge_id_shuffle = graph.edge_ids(graph.edges()[0],graph.edges()[1]).numpy()
    np.random.shuffle(edge_id_shuffle)
    num_test = int(edge_id_shuffle.shape[0]*0.05)
    num_val  = int(edge_id_shuffle.shape[0]*0.10)
    val_idx = np.sort(edge_id_shuffle[:num_val])
    test_idx = np.sort(edge_id_shuffle[num_val:(num_val+num_test)])
    train_idx = np.sort(edge_id_shuffle[(num_val+num_test):])
    train_mask = torch.zeros_like(graph.edges()[0],dtype=torch.bool)
    train_mask[train_idx] = True
    test_mask = torch.zeros_like(graph.edges()[0],dtype=torch.bool)
    test_mask[test_idx] = True
    val_mask = torch.zeros_like(graph.edges()[0],dtype=torch.bool)
    val_mask[val_idx] = True
    #TODO save the mask
    graph.edata['train_mask'] = train_mask
    graph.edata['val_mask'] = val_mask
    graph.edata['test_mask'] = test_mask
    ##########
    train_set = torch.arange(graph.number_of_edges())[train_mask]
    val_set = torch.arange(graph.number_of_edges())[val_mask] # val的边在原始图中的序号
    test_set = torch.arange(graph.number_of_edges())[test_mask]
    #build train_g
    train_edges = train_set
    train_g = graph.edge_subgraph(train_edges,preserve_nodes=True)
    train_edges_src = train_g.edges()[0]
    train_edges_dst = train_g.edges()[1]
    train_pos_nodepair = torch.vstack([train_edges_src,train_edges_dst]).numpy().T # array,not include self-loop
    #add self-loop edge
    train_g = dgl.add_self_loop(train_g)
    #build val_pos_graph
    val_edges = val_set
    val_pos_graph = graph.edge_subgraph(val_edges,preserve_nodes=True)
    val_edges_src = val_pos_graph.edges()[0]
    val_edges_dst = val_pos_graph.edges()[1]
    val_pos_nodepair = torch.vstack([val_edges_src,val_edges_dst]).numpy().T # array 应该是单向边
    # build test_pos_graph
    test_edges = test_set
    test_pos_graph = graph.edge_subgraph(test_edges,preserve_nodes=True)
    test_edges_src = test_pos_graph.edges()[0]
    test_edges_dst = test_pos_graph.edges()[1]
    test_pos_nodepair = torch.vstack([test_edges_src,test_edges_dst]).numpy().T # array
    ### 负采样的方法就是给定首尾节点的idx，然后确定在节点对(graph.edges())中不存在即可以认为是合适的负样本
    val_neg_nodepair,test_neg_nodepair = neg_sampling(train_pos_nodepair,val_pos_nodepair,test_pos_nodepair,graph)
    ## 得到负样本对后为val和test都生成neg subgraph
    edges = torch.arange(graph.number_of_edges())
    val_neg_graph = graph.__copy__()
    val_neg_graph.remove_edges(edges)
    val_neg_graph = dgl.add_edges(val_neg_graph,val_neg_nodepair.T[0],val_neg_nodepair.T[1])
    test_neg_graph = graph.__copy__()
    test_neg_graph.remove_edges(edges)
    test_neg_graph = dgl.add_edges(test_neg_graph,test_neg_nodepair.T[0],test_neg_nodepair.T[1])

    return train_g,val_pos_graph,val_neg_graph,test_pos_graph,test_neg_graph
