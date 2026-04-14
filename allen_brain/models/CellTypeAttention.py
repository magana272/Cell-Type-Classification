import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedEmbedding(nn.Module):
    def __init__(self, n_genes: int, n_pathways: int, embed_dim: int, mask: torch.Tensor):
        super().__init__()
        self.n_genes     = n_genes
        self.n_pathways  = n_pathways
        self.embed_dim   = embed_dim
        self.register_buffer('mask', mask)
        self.weight = nn.Parameter(torch.randn(embed_dim, n_genes, n_pathways) * 0.01)
        self.bias   = nn.Parameter(torch.zeros(embed_dim, n_pathways))

    def forward(self, x):
        W = self.weight * self.mask.unsqueeze(0)
        out = torch.einsum('bg,egp->bep', x, W) + self.bias
        return out


class TOSICA(nn.Module):

    def __init__(self, n_genes, n_pathways, n_classes, mask,
                 embed_dim=48, n_heads=4, n_layers=2, dropout=0.1,
                 unknown_threshold=0.95):
        super().__init__()
        self.n_pathways        = n_pathways
        self.embed_dim         = embed_dim
        self.unknown_threshold = unknown_threshold

        self.embedding = MaskedEmbedding(n_genes, n_pathways, embed_dim, mask)

        self.cls_token = nn.Parameter(torch.randn(1, embed_dim, 1))

        self.pos_embed = nn.Parameter(torch.randn(1, embed_dim, n_pathways + 1) * 0.01)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self._attn_weights = None

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, n_classes),
        )

    def forward(self, x, return_attention=False):

        batch = x.size(0)

        tokens = self.embedding(x)

        cls = self.cls_token.expand(batch, -1, -1)
        tokens = torch.cat([cls, tokens], dim=2)

        tokens = tokens + self.pos_embed

        tokens = tokens.permute(0, 2, 1)

        out = self.transformer(tokens)

        cls_out = out[:, 0, :]

        logits = self.classifier(cls_out)

        if return_attention:

            pathway_out = out[:, 1:, :]
            cls_rep     = cls_out.unsqueeze(1)
            attn_scores = torch.bmm(cls_rep, pathway_out.transpose(1, 2))
            attn_scores = F.softmax(attn_scores.squeeze(1) / math.sqrt(self.embed_dim), dim=-1)
            return logits, attn_scores

        return logits

    def predict_with_unknown(self, x, threshold=None):
        if threshold is None:
            threshold = self.unknown_threshold
        logits = self.forward(x)
        probs  = F.softmax(logits, dim=-1)
        max_p, preds = probs.max(dim=-1)
        preds[max_p < threshold] = -1  
        return preds, max_p
