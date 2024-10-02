from typing import Callable

import torch
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import OptTensor, PairOptTensor, PairTensor
from torch_geometric.utils import softmax

try:
    from torch_cluster import knn
except ImportError:
    knn = None


class DynamicEdgeConv(MessagePassing):
    def __init__(
        self, nn: Callable, k: int, in_channels: int, aggr: str = "max", num_workers: int = 1, **kwargs
    ):
        super().__init__(aggr=aggr, flow="source_to_target", **kwargs)

        if knn is None:
            raise ImportError("`DynamicEdgeConv` requires `torch-cluster`.")

        self.nn = nn
        self.k = k
        self.in_channels = in_channels
        self.num_workers = num_workers
        self.edge_index = None

        # Attention mechanism parameters
        self.att_weight = torch.nn.Parameter(torch.Tensor(2 * in_channels, 1))
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=0.2)

        self.reset_parameters()

    def reset_parameters(self):chatgpt.com
        self.nn.reset_parameters()
        torch.nn.init.xavier_uniform_(self.att_weight)

    def get_edge_index(self):
        return self.edge_index

    def forward(
        self,
        x: Tensor | PairTensor,
        batch: OptTensor | PairTensor | None = None,
    ) -> Tensor:
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        if x[0].dim() != 2:
            raise ValueError("Static graphs not supported in DynamicEdgeConv")

        b: PairOptTensor = (None, None)
        if isinstance(batch, Tensor):
            b = (batch, batch)
        elif isinstance(batch, tuple):
            assert batch is not None
            b = (batch[0], batch[1])

        self.edge_index = knn(x[0], x[1], self.k, b[0], b[1]).flip([0])

        # propagate_type: (x: PairTensor)
        return self.propagate(self.edge_index, x=x, size=None), self.edge_index

    def message(self, x_i: Tensor, x_j: Tensor, edge_index_i: Tensor) -> Tensor:
        z = torch.cat([x_i, x_j - x_i], dim=-1)
        alpha = (z @ self.att_weight).squeeze(-1)
        alpha = self.leaky_relu(alpha)
        alpha = softmax(alpha, edge_index_i)
        h = self.nn(z)
        return h * alpha.view(-1, 1)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(nn={self.nn}, k={self.k})"
