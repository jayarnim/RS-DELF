import torch
import torch.nn as nn
from ..attn.model import AttentionMechanism


class RepresentationFunction(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_factors: int,
        user_hist: torch.Tensor,
        item_hist: torch.Tensor,
    ):
        super(RepresentationFunction, self).__init__()
        # attr dictionary for load
        self.init_args = locals().copy()
        del self.init_args["self"]
        del self.init_args["__class__"]

        # global attr
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.register_buffer(
            name="user_hist", 
            tensor=user_hist,
        )
        self.register_buffer(
            name="item_hist", 
            tensor=item_hist,
        )

        # generate layers
        self._init_layers()

    def forward(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        """
        user_idx: (B,)
        item_idx: (B,)
        """
        user_id_embed = self.user_embed_target(user_idx)
        item_id_embed = self.item_embed_target(item_idx)
        user_hist_embed = self.user_hist_embed(user_idx, item_idx)
        item_hist_embed = self.item_hist_embed(user_idx, item_idx)

        id_cat = torch.cat([user_id_embed, item_id_embed], dim=-1)
        hist_cat = torch.cat([user_hist_embed, item_hist_embed], dim=-1)
        user_cat = torch.cat([user_id_embed, user_hist_embed], dim=-1)
        item_cat = torch.cat([item_id_embed, item_hist_embed], dim=-1)

        return id_cat, hist_cat, user_cat, item_cat

    def user_hist_embed(self, user_idx, item_idx):
        kwargs = dict(
            target_idx=user_idx, 
            target_hist=self.user_hist, 
            counterpart_padding_value=self.n_items,
        )
        refer_idx = self._hist_slicer(**kwargs)

        kwargs = dict(
            counterpart_idx=item_idx, 
            target_hist_slice=refer_idx,
            counterpart_padding_value=self.n_items,
        )
        mask = self._mask_generator(**kwargs)

        query = self.user_embed_global.unsqueeze(0)
        refer = self.item_embed_hist(refer_idx)

        kwargs = dict(
            Q=query,
            K=self.proj_u(refer),
            V=refer,
            mask=mask,
        )
        context = self.attn_u(**kwargs)
        return context

    def item_hist_embed(self, user_idx, item_idx):
        kwargs = dict(
            target_idx=item_idx, 
            target_hist=self.item_hist, 
            counterpart_padding_value=self.n_users,
        )
        refer_idx = self._hist_slicer(**kwargs)

        kwargs = dict(
            counterpart_idx=user_idx, 
            target_hist_slice=refer_idx,
            counterpart_padding_value=self.n_users,
        )
        mask = self._mask_generator(**kwargs)

        query = self.item_embed_global.unsqueeze(0)
        refer = self.user_embed_hist(refer_idx)
        
        kwargs = dict(
            Q=query,
            K=self.proj_i(refer),
            V=refer,
            mask=mask,
        )
        context = self.attn_i(**kwargs)

        return context

    def _mask_generator(self, counterpart_idx, target_hist_slice, counterpart_padding_value):
        # mask to current target item from history
        mask_counterpart = target_hist_slice == counterpart_idx.unsqueeze(1)
        # mask to padding
        mask_padded = target_hist_slice == counterpart_padding_value
        # final mask
        mask = mask_counterpart | mask_padded
        return mask

    def _hist_slicer(self, target_idx, target_hist, counterpart_padding_value):
        # target hist slice
        target_hist_slice = target_hist[target_idx]
        # calculate max hist in batch
        lengths = (target_hist_slice != counterpart_padding_value).sum(dim=1)
        max_len = lengths.max().item()
        # drop padding values
        target_hist_slice_trunc = target_hist_slice[:, :max_len]
        return target_hist_slice_trunc

    def _init_layers(self):
        kwargs = dict(
            num_embeddings=self.n_users+1, 
            embedding_dim=self.n_factors,
            padding_idx=self.n_users,
        )
        self.user_embed_target = nn.Embedding(**kwargs)
        self.user_embed_hist = nn.Embedding(**kwargs)
        self.user_embed_global = nn.Parameter(torch.randn(self.n_factors))

        nn.init.normal_(self.user_embed_target.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.user_embed_hist.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.user_embed_global, mean=0.0, std=0.01)

        kwargs = dict(
            num_embeddings=self.n_items+1, 
            embedding_dim=self.n_factors,
            padding_idx=self.n_items,
        )
        self.item_embed_target = nn.Embedding(**kwargs)
        self.item_embed_hist = nn.Embedding(**kwargs)
        self.item_embed_global = nn.Parameter(torch.randn(self.n_factors))

        nn.init.normal_(self.item_embed_target.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_embed_hist.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_embed_global, mean=0.0, std=0.01)

        self.proj_u = nn.Sequential(
            nn.Linear(self.n_factors, self.n_factors),
            nn.Tanh(),
        )
        self.proj_i = nn.Sequential(
            nn.Linear(self.n_factors, self.n_factors),
            nn.Tanh(),
        )

        self.attn_u = AttentionMechanism()
        self.attn_i = AttentionMechanism()