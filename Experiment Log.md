# Experiment Log

UDAGCN：训练集采样比例为0.8，测试集为0.2

原数据
ACM:     e=22270 n=7410  平均每个节点三条边
DBLP： e=14682 n=5578    平均每个节点两条边

随机删除边
ACM_D0.2(即 del_acm0.x_x): 4454
DBLP_D0.2(即 del_dblp0.x_x)：2936  
      0.6                    8809

等比例删除边

ACM_RD0.2(即 ratiodel_acm0.x_x)：4436
DBLP_RD0.2(即 ratiodel_dblp0.x_x)：2931
DBLP_RD0.4(即 ratiodel_dblp0.x_x)：5867
DBLP_RD0.6(即 ratiodel_dblp0.x_x)：8803 
DBLP_RD0.8(即 ratiodel_dblp0.x_x)：11739

> 为了避免过多的实验，仅考虑DBLP和ACM作为源域的情况



* 考虑到之前有研究显示低度的节点趋向于拥有更低的准确度，为了考察，在高度分布图中学得的权重矩阵，能够是否在低度节点上也能有效地运用；考察源域不同度节点上地准确率，以及目标域不同节点上地准确率分布；换句话说就是，删边带来地表现下降是否可以从度地角度去考察？



* 11.22 将改变后的VGAE进行实验，然后就是实验无PPMI和无迁移gcn还有无迁移vgae的效果

  ==重做所有baseline，使用最后一个epoch的结果，而不是best的结果==

  无PPMI有迁移 
  
  
  
  无PPMI无迁移 
  
  
  
  有PPMI无迁移

* 针对节点对的VGAE自编码器效果

  效果上来说

  MRVGAE test_roc= 0.70311 test_ap= 0.70560  hid = [64,32 32,8]

  ​				求和的方式 test_roc= 0.71255 test_ap= 0.72889 hid = [64,32 32,8]

  VGAE   test_roc= 0.92175 test_ap= 0.93952  hid = [64,32]  后者的话隐层的大小对于其效果不会有很大影响，但是前者增大隐层会提升性能

  改变拼接为加和然后增大隐层（原始的VGAE无法预测有向图的边，因为调换乘法前后的顺序不影响结果，但是concat具有这样的潜能）

* 理解代码里是如何进行连接性测试的



* 验证双路重建的效果



* 验证加入adv 损失后的分类效果





## 问题设定

源域  $X_S,A_S$

目标域 $X_T,A_T$ 其中邻接矩阵具有大量缺失值

meta-relation的定义   <s,e,t>三元组，考虑到文章所处理的问题中，边不包含feature且仅有一类边，故该三元组可以表示为<s,t>

## 框图的解释

#### F1

要不停的生成A，再用A去更新目标域的Z

对于$\mu*\mu^{T},\sigma*\sigma^{T}$的正态约束应当是N维的，N为节点个数

#### F2

对于所有节点对，由于其重建结果是对于A和X的观察，所以隐变量中的节点对的分布可以是高斯分布；但是这样做的结果很可能是重建的图连接性很差，即成对节点很多但是缺少联通的路径；分类器的标签为节点对的标签

#### F3

每个节点对的伯努利分布的p都不相同，然后每个节点对的分布都是高维空间中以p为参数的高维伯努利分布；分类器的标签为节点对的类别

## Note

* 关于cycle consistency的具体做法是什么？

> 一个是重建损失上的cycle,也就是$X_{TST}-X_T$的重建损失，使用TST不用STS是应为某些S中的样本可能再少量的T域中没有对应的样本作为监督
>
> 一个是task上的cycle，也就是用$X_{ST}$得到的中间隐变量$Z_{ST}$的分类结果（用原本S域的标签）

VGAE中使用CE损失来为每个节点对之间边的存在性进行判定

* 如何考虑引入PPMI，在VGAE中引入？

* 是否有必要去考虑节点在网络中的position对于链接预测的影响，特别是在目标域连接很稀疏的时候？还有就是解码时的置换不变性？

* 判断连接时应当加入sparsity的考虑，也就是由于邻接矩阵非常稀疏导致正负样本失衡的问题，Loss中给正样本较大权重之类的；准确的说对源域是使用正负样本的思路，而对目标域则要引入低秩约束

* 如何处理源域和目标域节点数量不同的情况

>  节点对的形式可以处理，但这样就不能再使用图卷积了

* 实际上我们的观测值是A和X，由于A是0，1的值选一个（不能认为中间的NxNx2D是观测值），所以只能使用Gumbel分布，而不是高斯分布作为一个先验；

  换个思路就是观察值为A*concat(Xi,Xj)，这样就比较好使用高斯去做变分,但这样网络参数极大，须要用batch了（或者使用上采样卷积）

  支路1：Gumbel分布如何进行重参数

  ​	

  支路2：用AX的重建，如何减少运算复杂度=卷积上采样操作，batch操作

* 如果按照现有的方法实际上做的就是先做link prediction在做node classification，那为什么不用其他的baseline做link prediction后再做node prediction试一试呢，就实验里应该有这个，就是baseline不做link prediction那么模型本身也就不应该做；

  换一个思路，目标域连接性不稀疏，使用VGAE增加来自源域的解码器生成的边，是否会增加效果？

* 离散的变分应该用GUMBEL分布



* Loss应该要换掉，换成FOCAL loss 现有的这种无法对负类做出约束；但是实际上要做的是细分边的种类



* 关于DGL的子图采样

  https://archwalker.github.io/blog/2019/07/07/GNN-Framework-DGL-NodeFlow.html

  运行中会采出一个pos_graph和neg_graph，以及一个dst_node集合；其中pos_graph是dst_node的positive边集（即实际存在），neg_graph则反之；而在计算中为了求出dst_node的hidden embedding，就需要其邻居的参与，而对于两层的GCN则需要二跳邻居来更新一跳邻居的hidden embedding，这在代码中就体现为两个block，第一个block用于更新一跳邻居，第二个block用于更新dst_node,这样子网络最后的输出就是dst_node的embedding。

  此外对于函数subgraph.apply_edges(dgl.function.u_dot_v('x', 'x', 'score'))，虽然看起来是求了所有节点对的分数score，但是实际上只会给subgraph中存在的边添加属性
  
* pytorch的使用过程中的一些记录

  * data loader，在非if main的情况下使用多num_workers的情况倾向于同时打开多个script,会导致内存不足，一定要使用if main来运行dataloader
  * grad是在loss.backward之后才进行计算的，所以在同一个batch中调用两次forward，能够正确地计算相应的梯度（而不是只计算后一次的）；此外，pytorch利用计算图保存loss相对于w的梯度计算式，见https://zhuanlan.zhihu.com/p/84890656

* 为什么会DA效果下降，从acm2dblp和acm2del_dblp0.2_1的degree_related_acc的图片来看，主要原因是1.度为0的节点大大增加2.度为1的节点准确率大大下降（0.6->0.4）

  

  


## TODO

- [ ] 检查UDAGCN中出来的ppmi encoder 和encoder最后的权重是否相同

- [ ] 查一下一般的训练设置(是取最后还是取最好的)

- [ ] python的当前路径和系统搜索路径要搞清楚·····

- [ ] 就现在这种框架下，即使用所有的负节点对作为0我觉得应该是不如将源域的负节点对也做分类再迁移到目标域上的；从当前的结果来说也是这样的，应当加入负节点对的类别一起训练才比较合适

- [ ] 找一下自适应的temp的写法

  
