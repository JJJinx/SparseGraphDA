# Experiment Log

|         | ACM   | DBLP   | ACM_D  | DBLP_D | ACM_RD | DBLP_RD |
| ------- | ----- | ------ | ------ | ------ | ------ | ------- |
| ACM     |       | 0.7644 |        | 0.7533 |        |         |
| ACM_D   |       | 0.6779 |        | 0.674  |        |         |
| ACM_RD  |       | 0.65   |        |        |        |         |
| DBLP    | 0.79  |        | 0.6841 |        |        |         |
| DBLP_D  | 0.493 |        | 0.5070 |        |        |         |
| DBLP_RD | 0.18  |        |        |        |        |         |

ACM、DBLP：原数据

ACM_D、DBLP_D：随机删除边

ACM_RD、DBLP_RD：等比例删除边

## TODO

- [ ] 修改数据读取的代码，固定label和X的输入路径
- [ ] 检查UDAGCN中出来的ppmi encoder 和encoder最后的权重是否相同
- [ ] 查一下一般的训练设置
- [ ] 针对结果节点级（不同度节点的准确率等）分析

