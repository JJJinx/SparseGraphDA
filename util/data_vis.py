import os
import numpy as np
import matplotlib.pyplot as plt


def draw_hist(raw_path,filename):

    edge_index =[]
    with open(os.path.join(raw_path,filename),"r") as f:
        for line in f.readlines():
            temp = line.strip('\n').split(',')
            edge_index.append(list(map(int,temp)))
    edge_index = np.array(edge_index)
    Vindex_min = edge_index.min()#节点序号的最小值
    Vindex_max = edge_index.max()
    degree_matrix = np.zeros((Vindex_max+1,Vindex_max+1))
    for i in edge_index:
        degree_matrix[i[0],i[1]]+=1 
    #按行求和 出度
    degree_sumr = np.sum(degree_matrix,axis=1)
    #按列求和 入度
    degree_sumc = np.sum(degree_matrix,axis=0)

    plt.figure(facecolor='#FFFFFF', figsize=(16,12))   #将图的外围设为白色
    plt.bar(np.arange(degree_sumc.shape[0]),degree_sumc)
    plt.show()

    binss = np.arange(200)
    hist,bins = np.histogram(degree_sumc,bins = binss) 
    print(hist)
    print(bins)
    plt.figure(facecolor='#FFFFFF', figsize=(16,12))  
    plt.hist(degree_sumc, binss) 
    plt.title("histogram") 
    plt.show()


def degree_level_acc(raw_path,name,prediction,label,savefig_path = None,mask=None):
    #TODO确定prediction和label是否是np
    #输入的后三者都是cuda tensor

    labels_file = name+'_labels.txt'
    edge_file = name+'_edgelist.txt'

    label_vec = np.loadtxt(os.path.join(raw_path,labels_file),dtype=np.int16)
    edge_mat =[]
    each_node_degree_vec = np.zeros_like(label_vec,dtype=np.int16) #每个节点地度的数量
    with open(os.path.join(raw_path,edge_file),"r") as f:
        for line in f.readlines():
            temp = line.strip('\n').split(',')
            edge_mat.append(list(map(int,temp)))
    edge_mat = np.array(edge_mat)
    #求每个节点的入度和出度之和
    for edge in edge_mat:
        source_node_index = edge[0]
        target_node_index = edge[1]
        each_node_degree_vec[source_node_index] += 1
        each_node_degree_vec[target_node_index] += 1
    #获得prediction 中正确的部分的序号值
    if mask is not None:
        corrects = prediction.eq(label[mask])
        degree_vec = each_node_degree_vec[mask.cpu().numpy()]
    else:
        corrects = prediction.eq(label)
        degree_vec = each_node_degree_vec
    min_degree = degree_vec.min()
    max_degree = degree_vec.max()
    #计算不同度对应的准确率
    accs = {}
    num_of_degree = [] #节点数不为0的度
    node_num_per_degree = []#每个度有多少节点    
    for d in range(min_degree,max_degree+1):  
        degree_index = np.where(degree_vec==d)[0]
        selected_corrects = corrects[degree_index]
        if degree_index.shape[0] != 0:
            num_of_degree.append(d) #横轴
            node_num_per_degree.append(degree_index.shape[0]) #线图
            accuracy = selected_corrects.float().mean().item() #柱状图
            accs[d] = accuracy
    # 绘图
    fig,ax1 = plt.subplots(figsize=(16,12)) #初始化一个画板和一个画布
    #柱状图表示准确率
    x = np.arange(len(num_of_degree)) #横轴
    y = np.array(list(accs.values())) #准确率
    xticks1 = num_of_degree  #横坐标的每个标签
    plt.bar(x,y)
    #打印柱状图数据
    # for a,b in zip(x,y):
    #     plt.text(a,b,'%0.2f' %b,ha='center',va='bottom',fontsize=10)
    #折线图表示节点关于度的分布
    ax2 = ax1.twinx()
    z = np.array(node_num_per_degree) #每个度的节点数
    ax2.plot(x,z,c='y') 
    #打印线图数据
    for a,b in zip(x,z):
        plt.text(a,b,'%d' %b,ha='center',va='bottom',fontsize=10)
    plt.savefig(savefig_path)

if __name__ == "__main__":
    root = os.getcwd()
    data_path = os.path.join(root,'data')
    #origin
    acm_path = os.path.join(data_path,'acm')
    acm_raw_path = os.path.join(acm_path,'raw')
    dblp_path = os.path.join(data_path,'dblp')
    dblp_raw_path = os.path.join(dblp_path,'raw')

    acmfname = 'acm_edgelist.txt'
    dblpfname = 'dblp_edgelist.txt'
    # random del
    del_acm_path = os.path.join(data_path,'deledge_acm')
    del_acm_raw_path = os.path.join(del_acm_path,'raw')
    del_dblp_path = os.path.join(data_path,'deledge_dblp')
    del_dblp_raw_path = os.path.join(del_dblp_path,'raw')

    deledge_acmfname = 'deledge_acm_edgelist.txt'
    deledge_dblpfname = 'deledge_dblp_edgelist.txt'
    # ratio del
    add_item = '0.8_1'
    ratiodel_acm_path = os.path.join(data_path,'ratiodel_acm'+add_item)
    ratiodel_acm_raw_path = os.path.join(ratiodel_acm_path,'raw')
    ratiodel_dblp_path = os.path.join(data_path,'ratiodel_dblp'+add_item)
    ratiodel_dblp_raw_path = os.path.join(ratiodel_dblp_path,'raw')

    ratiodel_acmfname = 'ratiodel_acm'+add_item+'_edgelist.txt'
    ratiodel_dblpfname = 'ratiodel_dblp'+add_item+'_edgelist.txt'

    draw_hist(ratiodel_dblp_raw_path,ratiodel_dblpfname)

