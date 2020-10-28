import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, average_precision_score
import scipy.sparse as sp
import numpy as np
import os
import time
import random
import itertools

from vgae.model import VGAE 


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = ArgumentParser()
parser.add_argument("--source", type=str, default='deledge_dblp')
parser.add_argument("--target", type=str, default='deledge_acm')
parser.add_argument("--name", type=str, default='UDAGCN')
parser.add_argument("--seed", type=int,default=200)
parser.add_argument("--encoder_dim", type=int, default=16)


args = parser.parse_args()
