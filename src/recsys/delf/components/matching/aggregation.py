import torch
import torch.nn as nn


class Concatenation(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(
        self, 
        user_emb: torch.Tensor, 
        item_emb: torch.Tensor,
    ):
        kwargs = dict(
            tensors=(user_emb, item_emb), 
            dim=-1,
        )
        return torch.cat(**kwargs)