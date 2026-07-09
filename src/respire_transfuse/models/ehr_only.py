import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)

        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class AttentionPool(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()

        hidden = max(d_model // 2, 16)

        self.score = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, tokens, key_padding_mask=None):
        logits = self.score(tokens).squeeze(-1).float()

        if key_padding_mask is not None:
            logits = logits.masked_fill(
                key_padding_mask.bool(),
                torch.finfo(logits.dtype).min,
            )

        attn = torch.softmax(logits, dim=1).to(tokens.dtype)
        pooled = torch.sum(tokens * attn.unsqueeze(-1), dim=1)

        return pooled, attn


class EHRTransformerRiskModel(nn.Module):
    def __init__(
        self,
        n_features,
        d_model=128,
        n_heads=4,
        n_layers=3,
        dim_feedforward=384,
        dropout=0.25,
        use_mask_channel=True,
        use_cls_token=True,
        ehr_token_dim=48,
    ):
        super().__init__()

        self.n_features = int(n_features)
        self.d_model = int(d_model)
        self.use_mask_channel = bool(use_mask_channel)
        self.use_cls_token = bool(use_cls_token)

        self.feature_dim = int(d_model) * 2
        self.ehr_token_dim = int(ehr_token_dim)
        self.token_dim = self.ehr_token_dim
        self.summary_dim = self.ehr_token_dim

        input_dim = self.n_features * 2 if self.use_mask_channel else self.n_features

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        else:
            self.cls_token = None

        self.pos = PositionalEncoding(
            d_model=d_model,
            max_len=512,
            dropout=dropout,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )

        self.attn_pool = AttentionPool(d_model, dropout=dropout)

        self.fusion_token_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, self.ehr_token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.fusion_summary_proj = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, self.ehr_token_dim),
        )

        self.head = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def encode_tokens(self, x, m):
        if x.dim() != 3:
            raise ValueError(f"Expected x [B,T,F], got {tuple(x.shape)}")

        if m.dim() != 3:
            raise ValueError(f"Expected m [B,T,F], got {tuple(m.shape)}")

        if x.shape != m.shape:
            raise ValueError(f"x and m shape mismatch. x={tuple(x.shape)}, m={tuple(m.shape)}")

        if x.shape[-1] != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {x.shape[-1]}")

        if self.use_mask_channel:
            z = torch.cat([x, m], dim=-1)
        else:
            z = x

        z = self.input_proj(z)

        time_observed = m.sum(dim=-1) > 0
        src_key_padding_mask = ~time_observed

        all_missing = src_key_padding_mask.all(dim=1)
        if all_missing.any():
            src_key_padding_mask = src_key_padding_mask.clone()
            src_key_padding_mask[all_missing, 0] = False

        if self.use_cls_token:
            batch_size = z.shape[0]
            cls = self.cls_token.expand(batch_size, -1, -1).to(dtype=z.dtype, device=z.device)
            z = torch.cat([cls, z], dim=1)

            cls_mask = torch.zeros(
                batch_size,
                1,
                dtype=torch.bool,
                device=z.device,
            )
            src_key_padding_mask = torch.cat([cls_mask, src_key_padding_mask], dim=1)

        z = self.pos(z)
        z = self.encoder(z, src_key_padding_mask=src_key_padding_mask)

        if self.use_cls_token:
            cls_vec = z[:, 0]
            token_z = z[:, 1:]
            token_mask = src_key_padding_mask[:, 1:]
        else:
            pooled_tmp, _ = self.attn_pool(z, key_padding_mask=src_key_padding_mask)
            cls_vec = pooled_tmp
            token_z = z
            token_mask = src_key_padding_mask

        return cls_vec, token_z, token_mask

    def forward_all(self, x, m):
        cls_vec, token_z, token_mask = self.encode_tokens(x, m)
        pooled_vec, attn = self.attn_pool(token_z, key_padding_mask=token_mask)

        ehr_features = torch.cat([cls_vec, pooled_vec], dim=-1)
        ehr_logit = self.head(ehr_features).squeeze(-1)

        ehr_tokens = self.fusion_token_proj(token_z)
        ehr_summary = self.fusion_summary_proj(ehr_features)

        return {
            "logit": ehr_logit,
            "attn": attn,
            "ehr_features": ehr_features,
            "ehr_tokens_raw": token_z,
            "ehr_tokens": ehr_tokens,
            "ehr_summary": ehr_summary,
            "token_mask": token_mask,
            "cls": cls_vec,
            "pooled": pooled_vec,
        }

    def forward_features(self, x, m):
        out = self.forward_all(x, m)
        return out["ehr_features"], out["attn"]

    def tokens_features_logit(self, x, m):
        out = self.forward_all(x, m)
        return (
            out["ehr_tokens"],
            out["ehr_features"],
            out["logit"],
            out["token_mask"],
            out["attn"],
        )

    def forward(self, x, m, return_all=False):
        out = self.forward_all(x, m)

        if return_all:
            return out

        return {
            "logit": out["logit"],
            "attn": out["attn"],
        }
