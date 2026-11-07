"""Frozen-encoder bilingual dependency parser: trainable source-graph and cross-lingual layers.

All trainable parameters are outside XLM-R; the final parsing head is target-only.
No Hindi parsing head is instantiated. Input embeddings are computed by cache module.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class ParallelParser(nn.Module):
    def __init__(self, embedding_dim: int, n_labels: int, hidden: int = 192,
                 layers: int = 2, dropout: float = 0.15, alignment_bias: float = 0.6,
                 alignment_dropout: float = 0.15):
        super().__init__()
        if hidden < 8 or layers < 1 or n_labels < 2:
            raise ValueError("hidden>=8, layers>=1, n_labels>=2 required")
        self.config = {"embedding_dim": embedding_dim, "n_labels": n_labels,
                       "hidden": hidden, "layers": layers, "dropout": dropout,
                       "alignment_bias": alignment_bias,
                       "alignment_dropout": alignment_dropout}
        self.src_proj = nn.Linear(embedding_dim, hidden)
        self.tgt_proj = nn.Linear(embedding_dim, hidden)
        self.graph_self = nn.Linear(hidden, hidden)
        self.graph_parent = nn.Linear(hidden, hidden, bias=False)
        self.graph_child = nn.Linear(hidden, hidden, bias=False)
        self.graph_norm = nn.LayerNorm(hidden)
        self.x_queries = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers))
        self.x_keys = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers))
        self.x_values = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers))
        self.gates = nn.ModuleList(nn.Linear(hidden * 2 + 1, hidden) for _ in range(layers))
        self.x_norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.arc_dep = nn.Linear(hidden, hidden)
        self.arc_head = nn.Linear(hidden, hidden)
        self.arc_U = nn.Parameter(torch.empty(hidden, hidden))
        nn.init.xavier_uniform_(self.arc_U)
        self.label_dep = nn.Linear(hidden, hidden)
        self.label_head = nn.Linear(hidden, hidden)
        self.label_U = nn.Parameter(torch.empty(n_labels, hidden, hidden))
        nn.init.xavier_uniform_(self.label_U)
        self.arc_bias = nn.Linear(hidden, 1)
        self.root = nn.Parameter(torch.randn(hidden) * 0.02)
        self.dropout = nn.Dropout(dropout)

    def _source_graph(self, x: torch.Tensor, heads: torch.Tensor,
                      mask: torch.Tensor) -> torch.Tensor:
        # heads: 0=root; 1..M = gold Hindi word ID. No learnable source parser.
        b, m, d = x.shape
        idx = (heads - 1).clamp(min=0)
        parent = x.gather(1, idx.unsqueeze(-1).expand(-1, -1, d))
        parent = parent * (heads != 0).unsqueeze(-1)
        child_sum = torch.zeros_like(x)
        child_sum.scatter_add_(1, idx.unsqueeze(-1).expand(-1, -1, d),
                               x * ((heads > 0) & mask).unsqueeze(-1))
        degree = torch.zeros((b, m, 1), device=x.device, dtype=x.dtype)
        degree.scatter_add_(1, idx.unsqueeze(-1), ((heads > 0) & mask).unsqueeze(-1).to(x.dtype))
        child_mean = child_sum / degree.clamp(min=1)
        out = self.graph_norm(x + F.gelu(self.graph_self(x)
                              + self.graph_parent(parent) + self.graph_child(child_mean)))
        return out * mask.unsqueeze(-1)

    def forward(self, source: torch.Tensor, target: torch.Tensor,
                source_heads: torch.Tensor, source_mask: torch.Tensor,
                target_mask: torch.Tensor, align: torch.Tensor,
                reliability: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return arc logits[B,N,N+1], relation logits[B,N,N+1,R], cross-attn[B,N,M]."""
        sx = self._source_graph(self.src_proj(source), source_heads, source_mask)
        tx = self.tgt_proj(target)
        attn = None
        if self.training and self.config["alignment_dropout"]:
            # Drop entire alignment rows, not random individual *gold* arcs; this
            # directly models missing/wrong alignment as absence of evidence.
            present = (torch.rand_like(reliability) >= self.config["alignment_dropout"])
            row_rel = reliability * present
            guide = align * present.unsqueeze(-1)
        else:
            row_rel, guide = reliability, align
        for q, k, v, gate, norm in zip(self.x_queries, self.x_keys,
                                        self.x_values, self.gates, self.x_norms):
            scores = torch.bmm(q(tx), k(sx).transpose(1, 2)) / math.sqrt(sx.size(-1))
            scores = scores + self.config["alignment_bias"] * torch.log(guide + 0.02)
            scores = scores.masked_fill(~source_mask[:, None, :], -1e4)
            attn = F.softmax(scores, dim=-1)
            context = torch.bmm(attn, v(sx))
            gating = torch.sigmoid(gate(torch.cat([tx, context, row_rel.unsqueeze(-1)], -1)))
            tx = norm(tx + self.dropout(gating * context * row_rel.unsqueeze(-1)))
            tx = tx * target_mask.unsqueeze(-1)
        deps = self.arc_dep(self.dropout(tx))
        heads = self.arc_head(torch.cat([self.root[None, None, :].expand(tx.size(0), 1, -1), tx], 1))
        arc_logits = torch.einsum("bnd,dh,bmh->bnm", deps, self.arc_U, heads)
        arc_logits = arc_logits / math.sqrt(deps.size(-1)) + self.arc_bias(tx)
        head_mask = torch.cat([torch.ones((tx.size(0), 1), dtype=torch.bool, device=tx.device),
                               target_mask], dim=1)
        arc_logits = arc_logits.masked_fill(~head_mask[:, None, :], -1e4)
        i = torch.arange(tx.size(1), device=tx.device)
        arc_logits[:, i, i + 1] = -1e4
        # All possible heads are scored for weak joint arc/relation supervision.
        relation_logits = torch.einsum("bnd,rdh,bmh->bnmr",
                                      self.label_dep(tx), self.label_U,
                                      self.label_head(torch.cat([
                                          self.root[None, None, :].expand(tx.size(0), 1, -1), tx], 1)))
        relation_logits = relation_logits / math.sqrt(deps.size(-1))
        return arc_logits, relation_logits, attn


def weak_loss(arcs: torch.Tensor, labels: torch.Tensor,
              q_joint: torch.Tensor, weights: torch.Tensor,
              *, label_weight: float = 0.5) -> tuple[torch.Tensor, dict]:
    """Weighted cross entropy against *soft noisy* projection targets (not gold LAS)."""
    mass = weights.sum()
    if float(mass.detach()) <= 0:
        # The trainer must reject this batch/corpus, not fabricate gold labels.
        return arcs.sum() * 0, {"arc": 0.0, "relation": 0.0, "mass": 0.0}
    q_arc = q_joint.sum(dim=-1)
    loss_arc = -(weights * (q_arc * F.log_softmax(arcs, dim=-1)).sum(-1)).sum() / mass
    loss_rel = -(weights * (q_joint * F.log_softmax(labels, dim=-1)).sum((-1, -2))).sum() / mass
    value = loss_arc + label_weight * loss_rel
    return value, {"arc": float(loss_arc.detach()), "relation": float(loss_rel.detach()),
                   "mass": float(mass.detach())}
