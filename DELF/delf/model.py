import torch
import torch.nn as nn
from . import rl, ml


class Module(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_factors: int,
        hidden: list,
        dropout: float,
        user_hist: torch.Tensor,
        item_hist: torch.Tensor,
    ):
        """
        DELF: A dual-embedding based deep latent factor model for recommendation (Cheng et al., 2018)
        -----
        Implements the base structure of Dual Embedding based Deep Latent Factor Model (DELF),
        MF & id embedding based latent factor model,
        applying attention mechanism to aggregate histories.

        Args:
            n_users (int):
                total number of users in the dataset, U.
            n_items (int):
                total number of items in the dataset, I.
            n_factors (int):
                dimensionality of user and item latent representation vectors, K.
            hidden (int):
                layer dimensions for the MLP-based matching function.
                (e.g., [64, 32, 16, 8])
            dropout (float):
                dropout rate applied to MLP layers for regularization.
            user_hist (torch.Tensor): 
                historical item interactions for each user, represented as item indices.
                (shape: [U, history_length])
            item_hist (torch.Tensor): 
                historical user interactions for each item, represented as user indices.
                (shape: [I, history_length])
        """
        super().__init__()

        # attr dictionary for load
        self.init_args = locals().copy()
        del self.init_args["self"]
        del self.init_args["__class__"]

        # global attr
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.hidden = hidden
        self.dropout = dropout
        self.register_buffer(
            name="user_hist", 
            tensor=user_hist,
        )
        self.register_buffer(
            name="item_hist", 
            tensor=item_hist,
        )

        # generate layers
        self._set_up_components()

    def forward(
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
        return self.score(user_idx, item_idx)

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
        logit = self.score(user_idx, item_idx)
        prob = torch.sigmoid(logit)
        return prob

    def score(self, user_idx, item_idx):
        # representation
        kwargs = dict(
            user_idx=user_idx,
            item_idx=item_idx,
        )
        id_cat, hist_cat, user_cat, item_cat = self.rl(**kwargs)

        # matching
        kwargs = dict(
            id_cat=id_cat,
            hist_cat=hist_cat,
            user_cat=user_cat,
            item_cat=item_cat,
        )
        pred_vector = self.ml(**kwargs)

        # predict
        logit = self.pred_layer(pred_vector).squeeze(-1)
        
        return logit

    def _set_up_components(self):
        self._create_layers()

    def _create_layers(self):
        kwargs = dict(
            n_users=self.n_users,
            n_items=self.n_items,
            n_factors=self.n_factors,
            user_hist=self.user_hist,
            item_hist=self.item_hist,
        )
        self.rl = rl.Module(**kwargs)

        kwargs = dict(
            n_factors=self.n_factors,
            hidden=self.hidden,
            dropout=self.dropout,
        )
        self.ml = ml.Module(**kwargs)

        kwargs = dict(
            in_features=self.hidden[-1] * 4,
            out_features=1,
        )
        self.pred_layer = nn.Linear(**kwargs)