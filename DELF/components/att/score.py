import torch
import torch.nn as nn


class Dot(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(
        self, 
        Q: torch.Tensor, 
        K: torch.Tensor,
    ):
        """
        Q: (B,K,D)
        K: (B,K,D)
        """
        # (B,K,D) -> (B,K)
        return (Q * K).sum(dim=-1)