import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionMechanism(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        Q: torch.Tensor,  # (B, D)
        K: torch.Tensor,  # (B, H, D)
        V: torch.Tensor,  # (B, H, D)
        mask: torch.Tensor,  # (B, H)
    ):
        # Q: (B,D) -> (B,1,D) -> (B,H,D)
        Q_exp = Q.unsqueeze(1).expand_as(K)

        # Attention scores: (B,H)
        scores = (Q_exp * K).sum(dim=-1)

        # Masking: (B,H) or (H,) -> (B,H)
        masked_scores = scores.masked_fill(~mask.expand_as(scores), float('-inf'))

        # Simplex projection: (B,H) -> (B,H)
        weights = F.softmax(masked_scores, dim=-1)

        # stabilize -inf
        valid = torch.isfinite(weights)
        weights = weights.masked_fill(~valid, 0.0)

        # Context vector: (B,H) x (B,H,D) -> (B,D)
        context = torch.sum(weights.unsqueeze(-1) * V, dim=1)

        return context