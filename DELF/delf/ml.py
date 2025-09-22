import torch
import torch.nn as nn


class MatchingFunction(nn.Module):
    def __init__(
        self,
        n_factors: int,
        hidden: list,
        dropout: float,
    ):
        super(MatchingFunction, self).__init__()
        # attr dictionary for load
        self.init_args = locals().copy()
        del self.init_args["self"]
        del self.init_args["__class__"]

        # global attr
        self.n_factors = n_factors
        self.hidden = hidden
        self.dropout = dropout

        # debugging args error
        self._assert_arg_error()

        # generate layers
        self._init_layers()

    def forward(
        self, 
        id_cat: torch.Tensor,
        hist_cat: torch.Tensor,
        user_cat: torch.Tensor,
        item_cat: torch.Tensor,
    ):
        pred_vec_id = self.mlp_id(id_cat)
        pred_vec_hist = self.mlp_hist(hist_cat)
        pred_vec_user = self.mlp_user(user_cat)
        pred_vec_item = self.mlp_item(item_cat)
        pred_vector = torch.cat([pred_vec_id, pred_vec_hist, pred_vec_user, pred_vec_item], dim=-1)
        return pred_vector

    def _init_layers(self):
        self.mlp_id = nn.Sequential(
            *list(self._generate_layers(self.hidden))
        )
        self.mlp_hist = nn.Sequential(
            *list(self._generate_layers(self.hidden))
        )
        self.mlp_user = nn.Sequential(
            *list(self._generate_layers(self.hidden))
        )
        self.mlp_item = nn.Sequential(
            *list(self._generate_layers(self.hidden))
        )

    def _generate_layers(self, hidden):
        idx = 1
        while idx < len(hidden):
            yield nn.Linear(hidden[idx-1], hidden[idx])
            yield nn.LayerNorm(hidden[idx])
            yield nn.ReLU()
            yield nn.Dropout(self.dropout)
            idx += 1

    def _assert_arg_error(self):
        CONDITION = (self.hidden[0] == self.n_factors * 2)
        ERROR_MESSAGE = f"First MLP layer must match input size: {self.n_factors * 2}"
        assert CONDITION, ERROR_MESSAGE