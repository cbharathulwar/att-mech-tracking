from typing import Callable, Optional

import torch
from torch import Tensor
from torch.nn import Module
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import OptTensor, PairOptTensor, PairTensor
from torch_geometric.utils import softmax

try:
    from torch_cluster import knn
except ImportError:
    knn = None


class DynamicEdgeConv(MessagePassing):
    def __init__(
        self,
        nn: Callable,
        k: int,
        in_channels: int,
        edge_filter_model: Optional[Module] = None,
        aggr: str = "max",
        num_workers: int = 1,
        **kwargs,
    ):
        super().__init__(aggr=aggr, flow="source_to_target", **kwargs)

        if knn is None:
            raise ImportError("`DynamicEdgeConv` requires `torch-cluster`.")

        self.nn = nn
        self.k = k
        self.in_channels = in_channels
        self.num_workers = num_workers
        self.edge_filter_model = edge_filter_model  # Edge filtering model
        self.edge_index = None

        # Attention mechanism parameters
        self.att_weight = torch.nn.Parameter(torch.Tensor(2 * in_channels, 1))
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=0.2)

        self.reset_parameters()

    def reset_parameters(self):
        self.nn.reset_parameters()
        torch.nn.init.xavier_uniform_(self.att_weight)
        if self.edge_filter_model is not None:
            self.edge_filter_model.reset_parameters()

    def get_edge_index(self):
        return self.edge_index

    def forward(
        self,
        x: Tensor | PairTensor,
        batch: OptTensor | PairTensor | None = None,
        edge_attr: Optional[Tensor] = None,
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

        # Compute k-NN edge indices
        self.edge_index = knn(x[0], x[1], self.k, b[0], b[1]).flip([0])

        # Edge Filtering (Graph Sparsification)
        if self.edge_filter_model is not None:
            # Prepare data for edge filtering model
            edge_features = self.compute_edge_features(x[0], self.edge_index, edge_attr)

            # Predict edge weights
            edge_weights = self.edge_filter_model(edge_features)

            # Apply threshold to filter edges
            threshold = 0.5  # You can adjust this threshold as needed
            mask = edge_weights > threshold
            self.edge_index = self.edge_index[:, mask]
            # Optionally, update edge_attr if used
            if edge_attr is not None:
                edge_attr = edge_attr[mask]

        # propagate_type: (x: PairTensor)
        return self.propagate(self.edge_index, x=x, size=None), self.edge_index

    def compute_edge_features(
        self, x: Tensor, edge_index: Tensor, edge_attr: Optional[Tensor]
    ):
        i, j = edge_index
        x_i = x[i]
        x_j = x[j]
        features = [x_i, x_j - x_i]
        if edge_attr is not None:
            features.append(edge_attr)
        edge_features = torch.cat(features, dim=-1)
        return edge_features

    def message(self, x_i: Tensor, x_j: Tensor, edge_index_i: Tensor) -> Tensor:
        z = torch.cat([x_i, x_j - x_i], dim=-1)
        alpha = (z @ self.att_weight).squeeze(-1)
        alpha = self.leaky_relu(alpha)
        alpha = softmax(alpha, edge_index_i)
        h = self.nn(z)
        return h * alpha.view(-1, 1)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(nn={self.nn}, k={self.k})"
