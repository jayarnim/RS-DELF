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
        super(Module, self).__init__()
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
        self._init_layers()

    def forward(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        """
        user_idx: (B,)
        item_idx: (B,)
        """
        return self.score(user_idx, item_idx)

    def predict(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        """
        user_idx: (B,)
        item_idx: (B,)
        """
        with torch.no_grad():
            logit = self.score(user_idx, item_idx)
            pred = torch.sigmoid(logit)
        return pred

    def score(self, user_idx, item_idx):
        kwargs = dict(
            user_idx=user_idx,
            item_idx=item_idx,
        )
        id_cat, hist_cat, user_cat, item_cat = self.rl(**kwargs)

        kwargs = dict(
            id_cat=id_cat,
            hist_cat=hist_cat,
            user_cat=user_cat,
            item_cat=item_cat,
        )
        pred_vector = self.ml(**kwargs)
        
        logit = self.logit_layer(pred_vector).squeeze(-1)
        
        return logit

    def _init_layers(self):
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
        self.logit_layer = nn.Linear(**kwargs)