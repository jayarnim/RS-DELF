import torch
import torch.nn as nn


class TransformLayer(nn.Module):
    def __init__(
        self,
        dim: int,
    ):
        super().__init__()

        self.dim = dim

        self._set_up_components()

    def forward(
        self, 
        X: torch.Tensor, 
    ):
        return self.transform(X)

    def _set_up_components(self):
        self.transform = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.Tanh(),
        )