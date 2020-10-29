# Experiment Log

|         | ACM   | DBLP   | ACM_D  | DBLP_D | ACM_RD | DBLP_RD |
| ------- | ----- | ------ | ------ | ------ | ------ | ------- |
| ACM     |       | 0.7644 |        | 0.7533 |        |         |
| ACM_D   |       | 0.6779 |        | 0.674  |        |         |
| ACM_RD  |       | 0.65   |        |        |        |         |
| DBLP    | 0.79  |        | 0.6841 |        |        |         |
| DBLP_D  | 0.493 |        | 0.5070 |        |        |         |
| DBLP_RD | 0.18  |        |        |        |        |         |

UDAGCN：训练集采样比例为0.8，测试集为0.2

原数据
ACM: 22270
DBLP： 14682

随机删除边
ACM_D0.2(即 del_acm0.x_x): 4454
DBLP_D0.2(即 del_dblp0.x_x)：2936

等比例删除边

ACM_RD0.2(即 ratiodel_acm0.x_x)：4436
DBLP_RD0.2(即 ratiodel_dblp0.x_x)：2931


## TODO

- [ ] 修改数据读取的代码，固定label和X的输入路径
- [ ] dblp0.2的表现太差了，尝试dblp0.4，查看结果
- [ ] 检查UDAGCN中出来的ppmi encoder 和encoder最后的权重是否相同
- [ ] 查一下一般的训练设置(是取最后还是取最好的)
- [ ] 针对结果节点级（不同度节点的准确率等）分析

