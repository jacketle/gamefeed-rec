"""Matched exposure classifiers: no sequence, masked mean, target attention."""
import torch
from torch import nn
from torch.nn import functional as F


class SequenceRanker(nn.Module):
    def __init__(self, num_users, num_items, num_tags, num_tabs, mode, history_id_dropout=0.0,
                 num_authors=0, match_dim=0, paired_initialization=False):
        super().__init__()
        if mode not in ('B0', 'B1', 'B2'):
            raise ValueError(mode)
        self.mode = mode
        if not 0 <= history_id_dropout <= 1:
            raise ValueError('history_id_dropout must be between zero and one')
        self.history_id_dropout = float(history_id_dropout)
        if match_dim not in (0, 3):
            raise ValueError('match_dim must be zero or three')
        self.match_dim = match_dim
        self.paired_initialization = bool(paired_initialization)
        extra = (16 if num_authors else 0) + match_dim
        self.item = nn.Embedding(num_items, 16, padding_idx=0)
        self.tag = nn.Embedding(num_tags, 16, padding_idx=0)
        self.user = nn.Embedding(num_users, 16, padding_idx=0)
        self.tab = nn.Embedding(num_tabs, 8, padding_idx=0)
        self.content = nn.Linear(32, 32)
        self.output = nn.Sequential(nn.Linear(92 if paired_initialization else 92+extra, 64), nn.ReLU(),
                                    nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        # Initialize shared layers first: paired B1/B2 seeds start identically.
        self.attention = nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1)) if mode == 'B2' else None
        self.register_buffer('trained_items', torch.zeros(num_items, dtype=torch.bool))
        if paired_initialization:
            # Preserve the baseline RNG stream and all shared initial parameters.
            with torch.random.fork_rng(devices=[]):
                if extra:
                    expanded = nn.Linear(92+extra, 64)
                    with torch.no_grad():
                        expanded.weight.zero_()
                        expanded.weight[:, :92].copy_(self.output[0].weight)
                        expanded.bias.copy_(self.output[0].bias)
                    self.output[0] = expanded
                self.author = nn.Embedding(num_authors, 16, padding_idx=0) if num_authors else None
        else:
            self.author = nn.Embedding(num_authors, 16, padding_idx=0) if num_authors else None

    def encode(self, ids, tags):
        # Explicit zero fallback: unknown ID vectors never remain random.
        iv = self.item(ids) * ids.gt(1).unsqueeze(-1)
        tv = self.tag(tags) * tags.gt(1).unsqueeze(-1)
        tv = tv.sum(-2) / tags.gt(0).sum(-1, keepdim=True).clamp_min(1)
        return self.content(torch.cat([iv, tv], -1))

    def item_l2(self, batch):
        """Mean squared coordinates of distinct, active item IDs in this batch.

        Shared candidate/history table; excludes padding and UNK. This is a
        batch-local L2 penalty, not DIN's frequency-corrected MBA regularizer.
        """
        if self.history_id_dropout:
            raise ValueError('Item L2 trial must not also mask history IDs')
        ids = batch['item']
        if self.mode != 'B0':
            ids = torch.cat([ids, batch['history'][batch['mask']]])
        ids = ids[ids.gt(1)].unique()
        if not ids.numel():
            return self.item.weight.new_zeros(())
        return self.item(ids).square().mean()

    def forward(self, batch):
        ids = batch['item']
        history = batch['history']
        mask = batch['mask']
        if self.training:
            if self.mode != 'B0' and self.history_id_dropout:
                drop = (torch.rand(history.shape, device=history.device) < self.history_id_dropout) & mask
                # Zero only the ID component via UNK; retain tags and valid positions.
                history = torch.where(drop, torch.ones_like(history), history)
            with torch.no_grad():
                self.trained_items[ids] = True
                if self.mode != 'B0':
                    self.trained_items[history[mask & history.gt(1)]] = True
        else:
            ids = torch.where(self.trained_items[ids], ids, torch.ones_like(ids))
            history = torch.where(self.trained_items[history], history, torch.ones_like(history))
        candidate = self.encode(ids, batch['tags'])
        if self.mode == 'B0':
            pooled = torch.zeros_like(candidate)
        else:
            hv = self.encode(history, batch['history_tags'])
            if self.mode == 'B1':
                pooled = (hv * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
            else:
                cv = candidate[:, None, :].expand_as(hv)
                scores = self.attention(torch.cat([hv,cv,hv-cv,hv*cv],-1)).squeeze(-1)
                weights = torch.softmax(scores.masked_fill(~mask, -1e9), dim=1) * mask
                pooled = (weights.unsqueeze(-1) * hv).sum(1)
        x = torch.cat([self.user(batch['user']), candidate, pooled,
                       self.tab(batch['tab']), batch['numeric']], -1)
        if self.author is not None:
            author = self.author(batch['author']) * batch['author'].gt(0).unsqueeze(-1)
            x = torch.cat([x, author], -1)
        if self.match_dim:
            x = torch.cat([x, batch['matches']], -1)
        if self.paired_initialization:
            # The original GEMM shape keeps baseline logits exactly reproducible.
            layer = self.output[0]
            hidden = F.linear(x[:, :92].contiguous(), layer.weight[:, :92].contiguous(), layer.bias)
            if x.shape[-1] > 92:
                hidden = hidden + F.linear(x[:, 92:].contiguous(), layer.weight[:, 92:].contiguous())
            for layer in self.output[1:]:
                hidden = layer(hidden)
            return hidden.squeeze(-1)
        return self.output(x).squeeze(-1)
