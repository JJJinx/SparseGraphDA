import os
import shutil
import numpy as np

def dele_fun(data_path,raw_path,name):
    rate = 1
    filename = name+'_edgelist.txt'
    edge_mat =[]
    with open(os.path.join(raw_path,filename),"r") as f:
        for line in f.readlines():
            temp = line.strip('\n').split(',')
            edge_mat.append(list(map(int,temp)))
    edge_mat = np.array(edge_mat)
    array_len = edge_mat.shape[0]
    sample_index =  np.sort(np.random.choice(array_len,int(rate*array_len),False))
    print('number of edges',len(sample_index))
    sampled_edge_mat = edge_mat[sample_index]

    ##判断是否存在删除边后的文件夹，不存在就创建
    new_name = 'del_'+name+str(rate)+'_3'
    new_path = os.path.join(data_path,new_name)
    new_raw_path = os.path.join(new_path,'raw')
    isnpExist=os.path.exists(new_path)
    isnrpExist=os.path.exists(new_raw_path)
    if not isnpExist:
        os.makedirs(new_path)
    if not isnrpExist:
        os.makedirs(new_raw_path)
    ##复制标签文件和节点特征
    X_source = os.path.join(raw_path,name+'_docs.txt')
    Y_source = os.path.join(raw_path,name+'_labels.txt')
    X_target = os.path.join(new_raw_path,new_name+'_docs.txt')
    Y_target = os.path.join(new_raw_path,new_name+'_labels.txt')
    shutil.copyfile(X_source,X_target)
    shutil.copyfile(Y_source,Y_target)

    np.savetxt(os.path.join(new_raw_path,new_name+'_edgelist.txt'),sampled_edge_mat,fmt='%d',delimiter=',')

def classify_and_del_edge(input_list):
    '''
    根据边的起始点和终点的节点类型，为边进行分类；在分类后，根据不同类别的边的占比对边进行有选择的删除
    类别为0~5，总共6个类，所以边和边总共分为36个类

    '''
    rate = 0.8
    data_path,raw_path,name = input_list

    labels_file = name+'_labels.txt'
    edge_file = name+'_edgelist.txt'

    label_vec = np.loadtxt(os.path.join(raw_path,labels_file),dtype=np.int16)
    edge_mat =[]
    with open(os.path.join(raw_path,edge_file),"r") as f:
        for line in f.readlines():
            temp = line.strip('\n').split(',')
            edge_mat.append(list(map(int,temp)))
    edge_mat = np.array(edge_mat)

    edgetype_vec = [] #edge_mat中每条边对应的类型
    for single_edge in edge_mat: 
        source_node = single_edge[0]
        target_node = single_edge[1]
        source_type = label_vec[source_node]
        target_type = label_vec[target_node]
        edgetype_vec.append(source_type*10+target_type)
    # Count the proportion of each type of edge
    total_edge = len(edgetype_vec) #边的总数量
    edgetypes = {}
    edgetypes_index = {}
    sample_index_dict = {}
    for i in range(6):
        for j in range(6):
            edgetypes[i*10+j] = 0 #这个类型的边的数量
            edgetypes_index[i*10+j] = []  #具体的这个类对应的边的索引
            sample_index_dict[i*10+j] = []  #采样以后的边的索引
    for index,single_edgetype in enumerate(edgetype_vec): #遍历边的索引和每条边对应的类型，然后对edgetypes进行操作
        edgetypes[single_edgetype] += 1
        edgetypes_index[single_edgetype].append(index)
    ## Random sampling with a ratio of 0.2 for each type of edge
    ## get the final index array as the output
    sampled_edge_index = np.array([],dtype=np.int16) #采样的边的序号，应当指的是在原来的edge
    for etype in edgetypes.keys():
        type_num = len(edgetypes_index[etype])
        sample_index_dict[etype] =  np.sort(np.random.choice(edgetypes_index[etype],int(rate*type_num),False)) #从edgetypes_index[etype]中采出指定数量的边保留
        sampled_edge_index = np.sort(np.concatenate((sampled_edge_index,sample_index_dict[etype]),axis=0)).astype(np.int16)
    #sampled_edge_index = np.sort(sampled_edge_index).astype(np.int16)
    sampled_edge_mat = edge_mat[sampled_edge_index]
    ##判断是否存在删除边后的文件夹，不存在就创建 
    new_name = 'ratiodel_'+name+str(rate)+'_3'
    new_path = os.path.join(data_path,new_name)
    new_raw_path = os.path.join(new_path,'raw')
    isnpExist=os.path.exists(new_path)
    isnrpExist=os.path.exists(new_raw_path)
    if not isnpExist:
        os.makedirs(new_path)
    if not isnrpExist:
        os.makedirs(new_raw_path)
    ##复制标签文件和节点特征
    X_source = os.path.join(raw_path,name+'_docs.txt')
    Y_source = os.path.join(raw_path,name+'_labels.txt')
    X_target = os.path.join(new_raw_path,new_name+'_docs.txt')
    Y_target = os.path.join(new_raw_path,new_name+'_labels.txt')
    shutil.copyfile(X_source,X_target)
    shutil.copyfile(Y_source,Y_target)
    ##保存新的邻接矩阵文件
    np.savetxt(os.path.join(new_raw_path,new_name+'_edgelist.txt'),sampled_edge_mat,fmt='%d',delimiter=',')


if __name__ == "__main__":
    root = os.getcwd()
    data_path = os.path.join(root,'data')
    choice = 'dblp'
    if choice == 'acm':
        acm_path = os.path.join(data_path,'acm')
        acm_raw_path = os.path.join(acm_path,'raw')
        #dele_fun(data_path,acm_raw_path,'acm')
        input_list = [data_path,acm_raw_path,'acm']
        classify_and_del_edge(input_list)
    if choice == 'dblp':
        dblp_path = os.path.join(data_path,'dblp')
        dblp_raw_path = os.path.join(dblp_path,'raw')
        #dele_fun(data_path,dblp_raw_path,'dblp')
        input_list = [data_path,dblp_raw_path,'dblp']
        classify_and_del_edge(input_list)