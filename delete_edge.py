import os
import shutil
import numpy as np

def dele_fun(data_path,raw_path,name):

    filename = name+'_edgelist.txt'
    ##判断是否存在删除边后的文件夹，不存在就创建
    new_path = os.path.join(data_path,'deledge_'+name)
    new_raw_path = os.path.join(new_path,'raw')
    isnpExist=os.path.exists(new_path)
    isnrpExist=os.path.exists(new_raw_path)

    if not isnpExist:
        os.makedirs(new_path)
    if not isnrpExist:
        os.makedirs(new_raw_path)
    ##
    edge_mat =[]
    with open(os.path.join(raw_path,filename),"r") as f:
        for line in f.readlines():
            temp = line.strip('\n').split(',')
            edge_mat.append(list(map(int,temp)))
    edge_mat = np.array(edge_mat)
    array_len = edge_mat.shape[0]
    sample_index =  np.sort(np.random.choice(array_len,int(0.2*array_len),False))
    sampled_edge_mat = edge_mat[sample_index]
    np.savetxt(os.path.join(new_raw_path,'deledge_'+name+'_edgelist.txt'),sampled_edge_mat,fmt='%d',delimiter=',')

def classify_and_del_edge(input_list):
    '''
    根据边的起始点和终点的节点类型，为边进行分类；在分类后，根据不同类别的边的占比对边进行有选择的删除
    类别为0~5，总共6个类，所以边和边总共分为36个类

    '''
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

    edgetype_vec = []
    for single_edge in edge_mat:
        #source_node和target_node对应label_vec中的index
        source_node = single_edge[0]
        target_node = single_edge[1]
        source_type = label_vec[source_node]
        target_type = label_vec[target_node]
        edgetype_vec.append(source_type*10+target_type)
    # Count the proportion of each type of edge
    total_edge = len(edgetype_vec)
    edgetypes = {}
    edgetypes_index = {}
    sample_index_dict = {}
    for i in range(6):
        for j in range(6):
            edgetypes[i*10+j] = 0
            edgetypes_index[i*10+j] = []
            sample_index_dict[i*10+j] = []
    for index,single_edgetype in enumerate(edgetype_vec):
        edgetypes[single_edgetype] += 1
        edgetypes_index[single_edgetype].append(index)
    ## Random sampling with a ratio of 0.2 for each type of edge
    ## get the final index array as the output
    sampled_edge_index = np.array([],dtype=np.int16)
    for etype in edgetypes.keys():
        type_num = len(edgetypes_index[etype])
        sample_index_dict[etype] =  np.sort(np.random.choice(type_num,int(0.2*type_num),False))
        sampled_edge_index = np.concatenate((sampled_edge_index,sample_index_dict[etype]),axis=0)
    sampled_edge_index = np.sort(sampled_edge_index)
    sampled_edge_mat = edge_mat[sampled_edge_index]
    ##判断是否存在删除边后的文件夹，不存在就创建 
    new_path = os.path.join(data_path,'ratiodel_'+name)
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
    X_target = os.path.join(new_raw_path,'ratiodel_'+name+'_docs.txt')
    Y_target = os.path.join(new_raw_path,'ratiodel_'+name+'_labels.txt')
    shutil.copyfile(X_source,X_target)
    shutil.copyfile(Y_source,Y_target)
    ##保存新的邻接矩阵文件
    np.savetxt(os.path.join(new_raw_path,'ratiodel_'+name+'_edgelist.txt'),sampled_edge_mat,fmt='%d',delimiter=',')


if __name__ == "__main__":
    root = os.getcwd()
    data_path = os.path.join(root,'data')
    #acmfname = 'acm_edgelist.txt'           # 22270条边
    #dblpfname = 'dblp_edgelist.txt'         # 14682条边 选取0.2以后就是2936左右的边数
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