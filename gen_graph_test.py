import dgl
import torch
import torch.nn as nn
import os
from dataset.DomainData import DomainData
from data_process import *

import dgl
import torch as th
g = dgl.graph((th.tensor([0,1,2,0, 1, 2]), th.tensor([0,1,2,1, 2, 0])))
print(torch.vstack([g.edges()[0],g.edges()[1]]))
bg1 = dgl.to_bidirected(g)
print(torch.vstack([bg1.edges()[0],bg1.edges()[1]]))
