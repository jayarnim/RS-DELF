import torch
import torch.nn as nn


class Module(nn.Module):
    def __init__(
        self,
        n_factors: int,
        hidden: list,
        dropout: float,
    ):
        super().__init__()

        # global attr
        self.n_factors = n_factors
        self.hidden = hidden
        self.dropout = dropout

        # debugging args error
        self._assert_arg_error()

        # generate layers
        self._set_up_components()

    def forward(
        self, 
        id_cat: torch.Tensor,
        hist_cat: torch.Tensor,
        user_cat: torch.Tensor,
        item_cat: torch.Tensor,
    ):
        pred_vec_id = self.matching_fn_id(id_cat)
        pred_vec_hist = self.matching_fn_hist(hist_cat)
        pred_vec_user = self.matching_fn_user(user_cat)
        pred_vec_item = self.matching_fn_item(item_cat)
        pred_vector = torch.cat([pred_vec_id, pred_vec_hist, pred_vec_user, pred_vec_item], dim=-1)
        return pred_vector

    def _set_up_components(self):
        self._create_layers()

    def _create_layers(self):
        components = list(self._yield_layers(self.hidden))
        self.matching_fn_id = nn.Sequential(*components)

        components = list(self._yield_layers(self.hidden))
        self.matching_fn_hist = nn.Sequential(*components)

        components = list(self._yield_layers(self.hidden))
        self.matching_fn_user = nn.Sequential(*components)

        components = list(self._yield_layers(self.hidden))
        self.matching_fn_item = nn.Sequential(*components)

    def _yield_layers(self, hidden):
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