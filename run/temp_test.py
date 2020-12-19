import dgl
import torch
import torch.nn as nn
import os
from data_processing.DomainData import DomainData
from data_processing.data_process import *

x = torch.randn(3, 2)
y = torch.ones(3, 2)
print(x)
print(torch.where(x>0))