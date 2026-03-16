import torch
import torch.nn as nn


class SoftmaxProjection(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, scores):
        # numerator
        numerator = torch.exp(scores)
        # denominator
        numerator_sum = numerator.sum(dim=-1, keepdim=True)
        denominator = numerator_sum + 1e-8
        # attention weight
        weights = numerator / denominator
        # stabilize -inf, +inf
        valid = torch.isfinite(weights)
        weights = weights.masked_fill(~valid, 0.0)
        return weights