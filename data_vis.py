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
    Vindex_min = edge_index.min()
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
    ratiodel_acm_path = os.path.join(data_path,'ratiodel_acm')
    ratiodel_acm_raw_path = os.path.join(ratiodel_acm_path,'raw')
    ratiodel_dblp_path = os.path.join(data_path,'ratiodel_dblp')
    ratiodel_dblp_raw_path = os.path.join(ratiodel_dblp_path,'raw')

    ratiodel_acmfname = 'ratiodel_acm_edgelist.txt'
    ratiodel_dblpfname = 'ratiodel_dblp_edgelist.txt'

    draw_hist(del_dblp_raw_path,deledge_dblpfname)

