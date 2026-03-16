import torch
import torch.nn as nn
from .score import Dot
from .simplex import SoftmaxProjection
from .transform import TransformLayer


class AttentionMechanism(nn.Module):
    def __init__(
        self,
        dim: int,
    ):
        super().__init__()

        self.dim = dim

        self._set_up_components()

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor,
    ):
        """
        Q: (B,D)
        K: (B,H,D)
        V: (B,H,D)
        mask: (B,H)
        """
        # Q: (B,D) -> (B,1,D) -> (B,H,D)
        Q_exp = Q.unsqueeze(1).expand_as(K)
        # K: (B,H,D) -> (B,H,D)
        K_transform = self.transform(K)
        # attention scores: (B,H)
        scores = self.score_fn(Q_exp, K_transform)
        # masking: (B,H) -> (B,H)
        scores_masked = scores.masked_fill(~mask, float('-inf'))
        # simplex projection: (B,H) -> (B,H)
        weights = self.simplex_fn(scores_masked)
        # context vector: (B,H,1) x (B,H,D) -> (B,H,D) -> (B,D)
        context = torch.sum(weights.unsqueeze(-1) * V, dim=1)
        return context

    def _set_up_components(self):
        self.transform = TransformLayer(self.dim)
        self.score_fn = Dot()
        self.simplex_fn = SoftmaxProjection()