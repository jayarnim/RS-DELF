import torch
import torch.nn as nn
from .components.embedding.builder import embedding_builder
from .components.matching.builder import matching_fn_builder
from .components.fusion import FusionLayer
from .components.prediction import ProjectionLayer
from .components.att.model import AttentionMechanism


class Module(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        hidden_dim: list,
        dropout: float,
        histories: dict[str, torch.Tensor],
    ):
        """
        DELF: A dual-embedding based deep latent factor model for recommendation (Cheng et al., 2018)
        -----
        Implements the base structure of Dual Embedding based Deep Latent Factor Model (DELF),
        MF & id embedding based latent factor model,
        applying attention mechanism to aggregate histories.

        Args:
            num_users (int):
                total number of users in the dataset, U.
            num_items (int):
                total number of items in the dataset, I.
            embedding_dim (int):
                dimensionality of user and item latent representation vectors, K.
            hidden_dim (int):
                layer dimensions for the MLP-based matching function.
                (e.g., [64, 32, 16, 8])
            dropout (float):
                dropout rate applied to MLP layers for regularization.
            histories (dict[str, torch.Tensor]):
                interaction histories.
                    - `user`: item history for each user.
                    (shape: [U, max_history_length])
                    - `item`: user history for each item. 
                    (shape: [I, max_history_length])
        """
        super().__init__()

        # attr dictionary for load
        self.init_args = locals().copy()
        del self.init_args["self"]
        del self.init_args["__class__"]

        # global attr
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.histories = histories

        # generate layers
        self._set_up_components()

    def forward(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        user_emb, item_emb, mask = self.embedding(user_idx, item_idx)

        kwargs = dict(
            Q=user_emb["query"],
            K=user_emb["history"],
            V=user_emb["history"],
            mask=mask["user"],
        )
        user_pooled = self.pooling["user"](**kwargs)

        kwargs = dict(
            Q=item_emb["query"],
            K=item_emb["history"],
            V=item_emb["history"],
            mask=mask["item"],
        )
        item_pooled = self.pooling["item"](**kwargs)

        args = (
            self.matching["anchor"](user_emb["anchor"], item_emb["anchor"]),
            self.matching["pooled"](user_pooled, item_pooled),
            self.matching["user"](user_emb["anchor"], item_pooled),
            self.matching["item"](user_pooled, item_emb["anchor"]),
        )
        X_pred = self.fusion(*args)

        return X_pred

    def estimate(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        """
        Training Method

        Args:
            user_idx (torch.Tensor): target user idx (shape: [B,])
            item_idx (torch.Tensor): target item idx (shape: [B,])
        
        Returns:
            logit (torch.Tensor): (u,i) pair interaction logit (shape: [B,])
        """
        X_pred = self.forward(user_idx, item_idx)
        logit = self.prediction(X_pred).squeeze(-1)
        return logit

    @torch.no_grad()
    def predict(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        """
        Evaluation Method

        Args:
            user_idx (torch.Tensor): target user idx (shape: [B,])
            item_idx (torch.Tensor): target item idx (shape: [B,])

        Returns:
            prob (torch.Tensor): (u,i) pair interaction probability (shape: [B,])
        """
        X_pred = self.forward(user_idx, item_idx)
        logit = self.prediction(X_pred).squeeze(-1)
        return logit

    def _set_up_components(self):
        self._create_components()

    def _create_components(self):
        kwargs = dict(
            num_users=self.num_users,
            num_items=self.num_items,
            embedding_dim=self.embedding_dim,
            histories=self.histories,
        )
        self.embedding = embedding_builder(**kwargs)

        kwargs = dict(
            dim=self.embedding_dim,
        )
        components = dict(
            user=AttentionMechanism(**kwargs),
            item=AttentionMechanism(**kwargs),
        )
        self.pooling = nn.ModuleDict(components)
        
        kwargs = dict(
            input_dim=self.embedding_dim*2,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        )
        components = dict(
            anchor=matching_fn_builder(**kwargs),
            pooled=matching_fn_builder(**kwargs),
            user=matching_fn_builder(**kwargs),
            item=matching_fn_builder(**kwargs),
        )
        self.matching = nn.ModuleDict(components)

        self.fusion = FusionLayer()

        kwargs = dict(
            dim=self.hidden_dim[-1]*4,
        )
        self.prediction = ProjectionLayer(**kwargs)