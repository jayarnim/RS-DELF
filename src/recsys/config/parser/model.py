from ..config.model import (
    DELFCfg,
)


def auto(cfg):
    model = cfg["model"]["name"]
    if model=="delf":
        return delf(cfg)
    else:
        raise ValueError("invalid model name in .yaml config")


def delf(cfg):
    return DELFCfg(
        num_users=cfg["data"]["entity"]["num_users"],
        num_items=cfg["data"]["entity"]["num_items"],
        embedding_dim=cfg["model"]["embedding_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        dropout=cfg["model"]["dropout"],
    )