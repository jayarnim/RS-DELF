import torch
import torch.nn as nn
from components.histories import Histories
from components.base import BaseModel
from .layers.embedding import build as build_embedding_layer
from .layers.matching import build as build_matching_layer
from .layers.fusion import ConcatenationLayer
from .layers.prediction import ProjectionLayer
from .layers.att import AttentionMechanism


class DualEmbeddingDeepLatentFactorModel(BaseModel):
    def __init__(
        self,
        histories: dict[str, Histories],
        num_users: int,
        num_items: int,
        embedding_dim: int,
        hidden_dim: list,
        dropout: float,
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
            histories (dict[str, Histories]):
                interaction histories.
                    - `user`: item history for each user.
                    (shape: [U, max_history_length])
                    - `item`: user history for each item. 
                    (shape: [I, max_history_length])
        """
        super().__init__(locals())

        self.histories = nn.ModuleDict(histories)

        # IDX EMBEDDING ==========
        kwargs = dict(
            name="idx",
            num_users=num_users,
            num_items=num_items,
            embedding_dim=embedding_dim,
        )
        components = dict(
            target=build_embedding_layer(**kwargs),
            history=build_embedding_layer(**kwargs),
        )
        self.embedding = nn.ModuleDict(components)

        # GLOBAL QUERY VECTORS ==========
        kwargs = dict(
            num_embeddings=1, 
            embedding_dim=embedding_dim,
        )
        components = dict(
            user=nn.Embedding(**kwargs),
            item=nn.Embedding(**kwargs),
        )
        self.query = nn.ModuleDict(components)

        # KEY TRANSFORM FUNCTION ==========
        components = dict(
            user=nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.Tanh(),
            ),
            item=nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.Tanh(),
            ),
        )
        self.transform = nn.ModuleDict(components)

        # HISTORY POOLING ==========
        kwargs = dict(
            score="dot",
            dim=embedding_dim,
            beta=1.0,
            dropout=dropout,
        )
        components = dict(
            user=AttentionMechanism(**kwargs),
            item=AttentionMechanism(**kwargs),
        )
        self.pooling = nn.ModuleDict(components)

        # MATCHING FUNCTION LEARNING ==========
        kwargs = dict(
            name="ncf",
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        components = dict(
            target=build_matching_layer(**kwargs),
            pooled=build_matching_layer(**kwargs),
            user=build_matching_layer(**kwargs),
            item=build_matching_layer(**kwargs),
        )
        self.matching = nn.ModuleDict(components)

        # FUSION ==========
        self.fusion = ConcatenationLayer()

        # PREDICTION ==========
        self.prediction = ProjectionLayer(
            dim=hidden_dim[-1]*4,
        )

    def forward(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ) -> torch.Tensor:
        # SEARCH USER & ITEM HISTORY IDX ==========
        user_hist_idx, user_hist_mask = self.histories["user"](user_idx, item_idx)
        item_hist_idx, item_hist_mask = self.histories["item"](item_idx, user_idx)

        # EMBEDDING SLICE ==========
        user_emb, item_emb = self.embedding["target"](user_idx, item_idx)
        item_hist_emb, user_hist_emb = self.embedding["history"](item_hist_idx, user_hist_idx)

        # POOLING USER HISTORY ==========
        user_pooled = self.pooling["user"](
            q=self.query["user"].weight,
            k=self.transform["user"](user_hist_emb),
            v=user_hist_emb,
            mask=user_hist_mask,
        )

        # POOLING ITEM HISTORY ==========
        item_pooled = self.pooling["item"](
            q=self.query["item"].weight,
            k=self.transform["item"](item_hist_emb),
            v=item_hist_emb,
            mask=item_hist_mask,
        )

        # MATCHING FUNCTION LEARNING ==========
        args = (
            self.matching["target"](user_emb, item_emb),
            self.matching["pooled"](user_pooled, item_pooled),
            self.matching["user"](user_emb, item_pooled),
            self.matching["item"](user_pooled, item_emb),
        )

        # MATCHING AGGREGATION ==========
        X_pred = self.fusion(*args)

        # PRED VEC ==========
        return X_pred

    def predict(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate Method

        Args:
            user_idx (torch.Tensor): target user idx (shape: [B,])
            item_idx (torch.Tensor): target item idx (shape: [B,])
        
        Returns:
            logit (torch.Tensor): (u,i) pair interaction logit (shape: [B,])
        """
        # INTERACTION MODELING ==========
        X_pred = self.forward(user_idx, item_idx)
        # PREDICTION ==========
        logit = self.prediction(X_pred)
        return logit