import torch
import numpy as np
from tqdm import tqdm
from collections import Counter
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.nn.inits import glorot, zeros



class CachedGCNConv(MessagePassing):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels, out_channels,
                 weight=None,
                 bias=None,
                 improved=False,
                 use_bias=True, **kwargs):
        super().__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cache_dict = {}

        # self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        #
        # if bias:
        #     self.bias = Parameter(torch.Tensor(out_channels))
        # else:
        #     self.register_parameter('bias', None)


        if weight is None:
            self.weight = Parameter(torch.Tensor(in_channels, out_channels).to(torch.float32))
            glorot(self.weight)
        else:
            self.weight = weight
            print("use shared weight")

        if bias is None:
            if use_bias:
                self.bias = Parameter(torch.Tensor(out_channels).to(torch.float32))
            else:
                self.register_parameter('bias', None)
            zeros(self.bias)
        else:
            self.bias = bias
            print("use shared bias")

        # self.reset_parameters()

    # def reset_parameters(self):
    #     glorot(self.weight)
    #     zeros(self.bias)
        # self.cached_result = None
        # self.cached_num_edges = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False,
             dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, cache_name="default_cache", edge_weight=None):
        """"""

        x = torch.matmul(x, self.weight)

        if not cache_name in self.cache_dict:
            edge_index, norm = self.norm(edge_index, x.size(0), edge_weight,
                                         self.improved, x.dtype)
            self.cache_dict[cache_name] = edge_index, norm
        else:
            edge_index, norm = self.cache_dict[cache_name]


        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)



class PPMIConv(CachedGCNConv):

    def __init__(self, in_channels, out_channels,
                 weight=None, bias=None, improved=False, use_bias=True,
                 path_len=5,
                 **kwargs):
        super().__init__(in_channels, out_channels, weight, bias, improved, use_bias, **kwargs)
        self.path_len = path_len



    def norm(self, edge_index, num_nodes, edge_weight=None, improved=False,
             dtype=None):

        adj_dict = {}

        def add_edge(a, b):
            if a in adj_dict:
                neighbors = adj_dict[a]
            else:
                neighbors = set()
                adj_dict[a] = neighbors
            if b not in neighbors:
                neighbors.add(b)

        cpu_device = torch.device("cpu")
        gpu_device = torch.device("cuda")
        for a, b in edge_index.t().detach().to(cpu_device).numpy():
            a = int(a)
            b = int(b)
            add_edge(a, b)
            add_edge(b, a)

        adj_dict = {a: list(neighbors) for a, neighbors in adj_dict.items()}

        def sample_neighbor(a):
            neighbors = adj_dict[a]
            random_index = np.random.randint(0, len(neighbors))
            return neighbors[random_index]


        # word_counter = Counter()
        walk_counters = {}

        def norm(counter):
            s = sum(counter.values())
            new_counter = Counter()
            for a, count in counter.items():
                new_counter[a] = counter[a] / s
            return new_counter

        for _ in tqdm(range(40)):
            for a in adj_dict:
                current_a = a
                current_path_len = np.random.randint(1, self.path_len + 1)
                for _ in range(current_path_len):
                    b = sample_neighbor(current_a)
                    if a in walk_counters:
                        walk_counter = walk_counters[a]
                    else:
                        walk_counter = Counter()
                        walk_counters[a] = walk_counter

                    walk_counter[b] += 1

                    current_a = b

        normed_walk_counters = {a: norm(walk_counter) for a, walk_counter in walk_counters.items()}

        prob_sums = Counter()

        for a, normed_walk_counter in normed_walk_counters.items():
            for b, prob in normed_walk_counter.items():
                prob_sums[b] += prob

        ppmis = {}

        for a, normed_walk_counter in normed_walk_counters.items():
            for b, prob in normed_walk_counter.items():
                ppmi = np.log(prob / prob_sums[b] * len(prob_sums) / self.path_len)
                ppmis[(a, b)] = ppmi

        new_edge_index = []
        edge_weight = []
        for (a, b), ppmi in ppmis.items():
            new_edge_index.append([a, b])
            edge_weight.append(ppmi)

        edge_index = torch.tensor(new_edge_index).t().to(gpu_device)
        edge_weight = torch.tensor(edge_weight).to(gpu_device)


        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, (deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]).type(torch.float32)
