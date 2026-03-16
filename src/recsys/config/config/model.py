from dataclasses import dataclass


@dataclass
class DELFCfg:
    num_users: int
    num_items: int
    embedding_dim: int
    hidden_dim: list
    dropout: float