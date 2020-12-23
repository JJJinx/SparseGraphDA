import dgl
import torch
import torch.nn as nn
import os
from data_processing.DomainData import DomainData
from data_processing.data_process import *

a = torch.randn((3,2))
b = torch.randn((2,5))
c = a@b
d = torch.mm(a,b)
print(a)
print(a.pow(2).sum(1))