"""Models for embeddings used for graph construction with combined edge distances and attention scores."""

# Ignore unused arguments because of save_hyperparameters
# ruff: noqa: ARG002

from typing import Callable

import torch
import torch.nn
from pytorch_lightning.core.mixins.hparams_mixin import HyperparametersMixin
from torch import Tensor as T
from torch import nn
from torch.jit import script as jit
from torch_cluster import knn_graph
from torch_geometric.data import Data

from gnn_tracking.models.mlp import MLP, HeterogeneousResFCNN, ResFCNN
from gnn_tracking.models.resin import ResIN
from gnn_tracking.utils.asserts import assert_feat_dim
from gnn_tracking.utils.lightning import get_model, obj_from_or_to_hparams
from gnn_tracking.utils.log import logger
from gnn_tracking.utils.torch_utils import freeze_if


class GraphConstructionFCNN(ResFCNN, HyperparametersMixin):
    def __init__(
        self,
        *,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        depth: int,
        alpha: float = 0.6,
    ):
        """Fully connected neural network for graph construction.
        Contains additional normalization parameter for the latent space.
        """
        super().__init__(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            depth=depth,
            alpha=alpha,
            bias=False,
        )
        self._latent_normalization = torch.nn.Parameter(
            torch.tensor([1.0]), requires_grad=True
        )
        self.save_hyperparameters()

    def forward(self, data: Data) -> dict[str, T]:
        out = super().forward(data.x) * self._latent_normalization
        return {"H": out}


class GraphConstructionHeteroResFCNN(HeterogeneousResFCNN, HyperparametersMixin):
    def __init__(
        self,
        *,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        depth: int,
        alpha: float = 0.6,
    ):
        """Fully connected neural network for graph construction.
        Fully heterogeneous (i.e., two separate MLPs for node and edge features).
        Contains additional normalization parameter for the latent space.
        """
        super().__init__(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            depth=depth,
            alpha=alpha,
            bias=False,
        )
        self._latent_normalization = torch.nn.Parameter(
            torch.tensor([1.0]), requires_grad=True
        )
        self.save_hyperparameters()

    def forward(self, data: Data) -> dict[str, T]:
        out = super().forward(data.x, layer=data.layer) * self._latent_normalization
        return {"H": out}


class GraphConstructionHeteroEncResFCNN(nn.Module, HyperparametersMixin):
    def __init__(
        self,
        *,
        in_dim: int,
        hidden_dim_enc: int,
        hidden_dim: int,
        out_dim: int,
        depth_enc: int,
        depth: int,
        alpha: float = 0.6,
    ):
        """Fully connected neural network for graph construction.
        Heterogeneous encoding.
        Contains additional normalization parameter for the latent space.
        """
        super().__init__()
        self.encoder = HeterogeneousResFCNN(
            in_dim=in_dim,
            hidden_dim=hidden_dim_enc,
            out_dim=hidden_dim,
            depth=depth_enc,
            alpha=alpha,
            bias=False,
        )
        self.fcnn = ResFCNN(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            depth=depth,
            alpha=alpha,
            bias=False,
        )
        self._latent_normalization = torch.nn.Parameter(
            torch.tensor([1.0]), requires_grad=True
        )
        self.save_hyperparameters()

    def forward(self, data: Data) -> dict[str, T]:
        assert_feat_dim(data.x, self.hparams.in_dim)
        enc = torch.nn.functional.relu(self.encoder(data.x, layer=data.layer))
        assert_feat_dim(enc, self.hparams.hidden_dim)
        out = self.fcnn(enc)
        assert_feat_dim(out, self.hparams.out_dim)
        out *= self._latent_normalization
        return {"H": out}


class GraphConstructionResIN(nn.Module, HyperparametersMixin):
    # noinspection PyUnusedLocal
    def __init__(
        self,
        *,
        node_indim: int,
        edge_indim: int,
        h_outdim: int = 8,
        hidden_dim: int = 40,
        alpha: float = 0.5,
        n_layers: int = 1,
        alpha_fcnn: float = 0.5,
    ):
        """Graph construction refinement with a stack of interaction network with
        residual connections between them. Refinement means that we assume that we're
        starting of the latent space and edges from `GraphConstructionFCNN`.

        Args:
            node_indim: Input dimension of the node features
            edge_indim: Input dimension of the edge features
            h_outdim: Output dimension of the node features
            hidden_dim: All other dimensions
            alpha: Strength of residual connections for the INs
            n_layers: Number of INs
            alpha_fcnn: Strength of residual connections connecting back to the initial
                latent space from the FCNN (assuming that the first `h_outdim` features
                are its latent space output)
        """
        super().__init__()
        self.save_hyperparameters()
        self._node_encoder = jit(
            MLP(
                input_size=node_indim,
                output_size=hidden_dim,
                hidden_dim=hidden_dim,
                L=2,
                bias=False,
            )
        )
        self._edge_encoder = jit(
            MLP(
                input_size=edge_indim,
                output_size=hidden_dim,
                hidden_dim=hidden_dim,
                L=2,
                bias=False,
            )
        )
        self._resin = ResIN(
            node_dim=hidden_dim,
            edge_dim=hidden_dim,
            object_hidden_dim=hidden_dim,
            relational_hidden_dim=hidden_dim,
            n_layers=n_layers,
            alpha=alpha,
        )
        self._decoder: Callable = jit(  # type: ignore
            MLP(
                input_size=hidden_dim,
                output_size=h_outdim,
                hidden_dim=hidden_dim,
                L=2,
                bias=False,
            )
        )
        self._latent_normalization = torch.nn.Parameter(
            torch.tensor([1.0]), requires_grad=True
        )

    def forward(self, data: Data) -> dict[str, T]:
        x_fcnn = data.x[:, : self.hparams.h_outdim]
        assert_feat_dim(data.x, self.hparams.node_indim)
        assert_feat_dim(data.edge_attr, self.hparams.edge_indim)
        x = self._node_encoder(data.x)
        assert_feat_dim(x, self.hparams.hidden_dim)
        edge_attr = self._edge_encoder(data.edge_attr)
        assert_feat_dim(edge_attr, self.hparams.hidden_dim)
        x, _, _ = self._resin(x, data.edge_index, edge_attr)
        assert_feat_dim(x, self.hparams.hidden_dim)
        delta = self._decoder(x)
        assert_feat_dim(delta, self.hparams.h_outdim)
        h = self.hparams.alpha_fcnn * x_fcnn + (1 - self.hparams.alpha_fcnn) * delta
        h *= self._latent_normalization
        return {"H": h}


def knn_with_max_radius(x: T, k: int, max_radius: float | None = None) -> T:
    """A version of kNN that excludes edges with a distance larger than a given radius.

    Args:
        x:
        k: Number of neighbors
        max_radius:
    Returns:
        edge index
    """
    edge_index = knn_graph(x, k=k)
    if max_radius is not None:
        dists = (x[edge_index[0]] - x[edge_index[1]]).norm(dim=-1)
        edge_index = edge_index[:, dists < max_radius]
    return edge_index


class MLGraphConstruction(nn.Module, HyperparametersMixin):
    # noinspection PyUnusedLocal
    def __init__(
        self,
        ml: torch.nn.Module | None = None,
        *,
        ec: torch.nn.Module | None = None,
        max_radius: float = 1,
        max_num_neighbors: int = 256,
        use_embedding_features=False,
        ratio_of_false=None,
        build_edge_features=True,
        ec_threshold=None,
        ml_freeze: bool = True,
        ec_freeze: bool = True,
        embedding_slice: tuple[int | None, int | None] = (None, None),
        edge_threshold: float | None = None,  # Distance threshold for pruning edges
        x_dim: int = None,
        attention_hidden_dim: int = 64,
        attention_threshold: float | None = None,  # Threshold for attention scores
    ):
        """Builds graph from embedding space using both edge distances and attention scores.

        Args:
            ml: Metric learning embedding module. If not specified, it is assumed that
                the node features from the data object are already the embedding
                coordinates. To use a subset of the embedding coordinates, use
                ``embedding_slice``.
            ec: Directly apply edge filter
            max_radius: Maximum radius for kNN
            max_num_neighbors: Number of neighbors for kNN
            use_embedding_features: Add embedding space features to node features
                (only if ``ml`` is not None)
            ratio_of_false: Subsample false edges (using truth information)
            build_edge_features: Build edge features (as difference and sum of node
                features).
            ec_threshold: EC threshold for edge filter/classification
            ml_freeze: Freeze the metric learning model parameters
            ec_freeze: Freeze the edge classifier model parameters
            embedding_slice: Used if ``ml`` is None. If not None, all node features
                are used. If a tuple, the first element is the start index and the
                second element is the end index.
            edge_threshold: Distance threshold for pruning edges based on embedding space
            x_dim: Dimension of the node features after optional concatenation
            attention_hidden_dim: Hidden dimension size for the attention MLP
            attention_threshold: Threshold for attention scores to prune edges
        """
        super().__init__()
        self.save_hyperparameters(ignore=["ml", "ec"])
        self._ml = freeze_if(obj_from_or_to_hparams(self, "ml", ml), ml_freeze)
        self._ef = freeze_if(obj_from_or_to_hparams(self, "ec", ec), ec_freeze)
        self._validate_config()

        if self.hparams.x_dim is None:
            raise ValueError("x_dim must be specified for attention mechanism.")

        # Define the attention mechanism
        self.attention_mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * self.hparams.x_dim, attention_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(attention_hidden_dim, 1),
        )

    def _validate_config(self) -> None:
        """Ensure that config makes sense."""
        if self._ef is not None and self.hparams.ec_threshold is None:
            msg = "ec_threshold must be set if ec/ef is not None"
            raise ValueError(msg)
        if self._ml is None and self.hparams.use_embedding_features:
            msg = "use_embedding_features requires ml to be not None"
            raise ValueError(msg)
        if self._ml is not None and self.hparams.embedding_slice != (None, None):
            msg = "embedding_slice requires ml to be None"
            raise ValueError(msg)
        if self.hparams.build_edge_features and self.hparams.ratio_of_false:
            logger.warning(
                "Subsampling false edges. This might not make sense"
                " for message passing."
            )

    @classmethod
    def from_chkpt(
        cls,
        ml_chkpt_path: str = "",
        ec_chkpt_path: str = "",
        *,
        ml_class_name: str = "gnn_tracking.training.ml.MLModule",
        ec_class_name: str = "gnn_tracking.training.ec.ECModule",
        ml_model_only: bool = True,
        ec_model_only: bool = True,
        **kwargs,
    ) -> "MLGraphConstruction":
        """Build `MLGraphConstruction` from checkpointed models.

        Args:
            ml_chkpt_path: Path to metric learning checkpoint
            ec_chkpt_path: Path to edge filter checkpoint. If empty, no EC will be
                used.
            ml_class_name: Class name of metric learning lightning module
                (default should almost always be fine)
            ec_class_name: Class name of edge filter lightning module
                (default should almost always be fine)
            ml_model_only: Only the torch model is loaded (excluding preprocessing
                steps from the lightning module)
            ec_model_only: See ``ml_model_only``
            **kwargs: Additional arguments passed to `MLGraphConstruction`
        """
        ml = get_model(ml_class_name, ml_chkpt_path, whole_module=not ml_model_only)
        ec = get_model(ec_class_name, ec_chkpt_path, whole_module=not ec_model_only)
        return cls(
            ml=ml,
            ec=ec,
            **kwargs,
        )

    @property
    def out_dim(self) -> tuple[int, int]:
        """Returns node, edge, output dims"""
        if self._ml is None:
            msg = "Cannot infer output dimension without ML model"
            raise RuntimeError(msg)
        node_dim: int = self._ml.in_dim  # type: ignore
        if self.hparams.use_embedding_features:
            node_dim += self._ml.out_dim  # type: ignore
        edge_dim = 2 * node_dim if self.hparams.build_edge_features else 0
        return node_dim, edge_dim

    def forward(self, data: Data) -> Data:
        if not hasattr(data, "true_edge_index"):
            # For the point cloud data, we unfortunately saved the true edges
            # simply as edge_index. We need to make sure that we keep the
            # true edge information around and don't overwrite it.
            data.true_edge_index = data.edge_index
        if self._ml is not None:
            mo = self._ml(data)
            embedding_features = mo["H"]
        else:
            s = self.hparams.embedding_slice
            embedding_features = data.x[:, s[0] : s[1]]
        edge_index = knn_with_max_radius(
            embedding_features,
            max_radius=self.hparams.max_radius,
            k=self.hparams.max_num_neighbors,
        )
        y: T = (data.particle_id[edge_index[0]] == data.particle_id[edge_index[1]]) & (
            data.particle_id[edge_index[0]] > 0
        )

        # Compute distances between nodes connected by edges
        distances = (
            embedding_features[edge_index[0]] - embedding_features[edge_index[1]]
        ).norm(dim=1)

        if not self._ml or not self.hparams.use_embedding_features:
            x = data.x
        else:
            x = torch.cat((mo["H"], data.x), dim=1)

        if self.hparams.build_edge_features:
            edge_features = torch.cat(
                (
                    x[edge_index[0]] - x[edge_index[1]],
                    x[edge_index[0]] + x[edge_index[1]],
                ),
                dim=1,
            )
        else:
            edge_features = None

        # Compute attention scores
        attention_scores = self.attention_mlp(edge_features).squeeze(-1)

        # Threshold-based pruning using distances and attention scores
        mask = torch.ones(
            edge_index.size(1), dtype=torch.bool, device=edge_index.device
        )
        if self.hparams.edge_threshold is not None:
            mask &= distances < self.hparams.edge_threshold
        if self.hparams.attention_threshold is not None:
            mask &= attention_scores > self.hparams.attention_threshold

        # Apply mask
        edge_index = edge_index[:, mask]
        y = y[mask]
        distances = distances[mask]
        attention_scores = attention_scores[mask]
        if edge_features is not None:
            edge_features = edge_features[mask]

        if self.hparams.ratio_of_false and self.training:
            num_true = y.sum()
            num_false_to_keep = int(num_true * self.hparams.ratio_of_false)
            false_mask = ~y
            false_indices = false_mask.nonzero(as_tuple=False).squeeze()
            true_indices = y.nonzero(as_tuple=False).squeeze()
            selected_false_indices = false_indices[:num_false_to_keep]
            selected_indices = torch.cat((selected_false_indices, true_indices))
            edge_index = edge_index[:, selected_indices]
            y = y[selected_indices]
            if edge_features is not None:
                edge_features = edge_features[selected_indices]
            distances = distances[selected_indices]
            attention_scores = attention_scores[selected_indices]

        if self._ef is not None:
            w = self._ef(edge_features)["W"]
            mask = w > self.hparams.ec_threshold
            edge_index = edge_index[:, mask]
            y = y[mask]
            if edge_features is not None:
                edge_features = edge_features[mask]
            distances = distances[mask]
            attention_scores = attention_scores[mask]

        # Optionally, store distances and attention_scores in the Data object
        data.distances = distances
        data.attention_scores = attention_scores

        return Data(
            x=x,
            edge_index=edge_index,
            true_edges=data.true_edge_index,
            y=y.long(),
            pt=data.pt,
            particle_id=data.particle_id,
            sector=data.sector,
            reconstructable=data.reconstructable,
            edge_attr=edge_features,
            eta=data.eta,
            layer=data.layer,
        )


class MLGraphConstructionFromChkpt(nn.Module, HyperparametersMixin):
    def __new__(cls, *args, **kwargs) -> MLGraphConstruction:
        """Alias for `MLGraphConstruction.from_chkpt` for use in yaml files"""
        return MLGraphConstruction.from_chkpt(*args, **kwargs)


class MLPCTransformer(nn.Module, HyperparametersMixin):
    # noinspection PyUnusedLocal
    def __init__(
        self,
        model: nn.Module,
        *,
        original_features: bool = False,
        freeze: bool = True,
    ):
        """Transforms a point cloud (PC) using a metric learning (ML) model.
        This is just a thin wrapper around the ML module with specification of what
        to do with the resulting latent space.
        In contrast to `MLGraphConstructionFromChkpt`, this class does not build a
        graph from the latent space but returns a transformed point cloud.
        Use `MLPCTransformer.from_ml_chkpt` to build from a checkpointed ML model.

        .. warning::
            In the current implementation, the original ``Data`` object is modified
            by `forward`.

        Args:
            model: Metric learning model. Should return latent space with key "H"
            original_features: Include original node features as node features (after
                the transformed ones)
        """
        super().__init__()
        self._ml: nn.Module = freeze_if(
            obj_from_or_to_hparams(self, "ml", model), freeze
        )
        self.save_hyperparameters(ignore=["model"])

    @classmethod
    def from_ml_chkpt(
        cls,
        chkpt_path: str,
        *,
        class_name: str = "gnn_tracking.training.ml.MLModule",
        **kwargs,
    ):
        """Build `MLPCTransformer` from checkpointed ML model.

        Args:
            chkpt_path: Path to checkpoint
            class_name: Lightning module class name that was used for training.
                Probably default covers most cases.
            **kwargs: Additional kwargs passed to `MLPCTransformer` constructor
        """
        ml_model = get_model(class_name, chkpt_path)
        return cls(
            ml_model,
            **kwargs,
        )

    def forward(self, data: Data) -> Data:
        out = self._ml(data)
        if self.hparams.original_features:
            data.x = torch.cat((out["H"], data.x), dim=1)
        else:
            data.x = out["H"]
        return data


class MLPCTransformerFromMLChkpt(MLPCTransformer):
    def __new__(cls, *args, **kwargs) -> MLPCTransformer:
        """Alias for `MLPCTransformer.from_ml_chkpt` for use in yaml configs"""
        return MLPCTransformer.from_ml_chkpt(*args, **kwargs)  # type: ignore
