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

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class LocalTemporalBlock(nn.Module):
    def __init__(self, d_model, dropout=0.1, scale_init=-1.1):
        super().__init__()

        self.norm = nn.LayerNorm(d_model)

        self.conv3 = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=3,
            padding=1,
            groups=d_model,
        )

        self.conv5 = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=5,
            padding=2,
            groups=d_model,
        )

        self.merge = nn.Conv1d(
            d_model * 2,
            d_model,
            kernel_size=1,
        )

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        self.scale_logit = nn.Parameter(
            torch.tensor(float(scale_init))
        )

    def forward(self, x, key_padding_mask=None):
        residual = x
        z = self.norm(x).transpose(1, 2)

        local3 = self.conv3(z)
        local5 = self.conv5(z)

        z = torch.cat([local3, local5], dim=1)
        z = self.merge(z).transpose(1, 2)
        z = self.dropout(self.activation(z))

        if key_padding_mask is not None:
            z = z.masked_fill(
                key_padding_mask.unsqueeze(-1),
                0.0,
            )

        scale = torch.sigmoid(self.scale_logit)
        return residual + scale * z, scale


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

        nn.init.normal_(
            self.score[-1].weight,
            mean=0.0,
            std=0.01,
        )
        nn.init.zeros_(self.score[-1].bias)

    def forward(self, tokens, key_padding_mask=None):
        logits = self.score(tokens).squeeze(-1).float()

        if key_padding_mask is not None:
            mask = key_padding_mask.bool()
            logits = logits.masked_fill(
                mask,
                torch.finfo(logits.dtype).min,
            )

            all_masked = mask.all(dim=1)

            if all_masked.any():
                logits = logits.clone()
                logits[all_masked] = 0.0

        attn = torch.softmax(logits, dim=1)

        if key_padding_mask is not None:
            attn = attn.masked_fill(mask, 0.0)
            attn = attn / attn.sum(
                dim=1,
                keepdim=True,
            ).clamp_min(1e-12)

        attn = attn.to(tokens.dtype)
        pooled = torch.sum(
            tokens * attn.unsqueeze(-1),
            dim=1,
        )

        return pooled, attn


class SummaryMixer(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.score = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    def forward(self, summaries):
        logits = self.score(summaries).squeeze(-1)
        weights = torch.softmax(logits, dim=1)
        pooled = torch.sum(
            summaries * weights.unsqueeze(-1),
            dim=1,
        )
        return pooled, weights


class EHRTransformerRiskModel(nn.Module):
    def __init__(
        self,
        n_features,
        d_model=128,
        n_heads=4,
        n_layers=2,
        dim_feedforward=320,
        dropout=0.35,
        use_mask_channel=True,
        use_cls_token=True,
        ehr_token_dim=48,
        local_scale_init=-1.1,
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

        input_dim = (
            self.n_features * 2
            if self.use_mask_channel
            else self.n_features
        )

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.local_block = LocalTemporalBlock(
            d_model=d_model,
            dropout=dropout,
            scale_init=local_scale_init,
        )

        self.pos = PositionalEncoding(
            d_model=d_model,
            max_len=512,
            dropout=dropout,
        )

        if self.use_cls_token:
            self.cls_token = nn.Parameter(
                torch.zeros(1, 1, d_model)
            )
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        else:
            self.cls_token = None

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

        self.attn_pool = AttentionPool(
            d_model=d_model,
            dropout=dropout,
        )

        self.summary_mixer = SummaryMixer(d_model)

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

        head_hidden = max(64, d_model * 3 // 4)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def set_fusion_adapters_trainable(self, trainable):
        trainable = bool(trainable)

        for module in [
            self.fusion_token_proj,
            self.fusion_summary_proj,
        ]:
            for parameter in module.parameters():
                parameter.requires_grad_(trainable)

    @staticmethod
    def masked_mean(tokens, key_padding_mask=None):
        if key_padding_mask is None:
            return tokens.mean(dim=1)

        valid = (~key_padding_mask.bool()).to(tokens.dtype)
        denominator = valid.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1.0)

        return (
            tokens * valid.unsqueeze(-1)
        ).sum(dim=1) / denominator

    @staticmethod
    def last_observed(tokens, key_padding_mask=None):
        if key_padding_mask is None:
            return tokens[:, -1]

        valid = ~key_padding_mask.bool()
        batch_size, time_steps = valid.shape

        positions = torch.arange(
            time_steps,
            device=tokens.device,
        ).unsqueeze(0).expand(batch_size, -1)

        positions = positions.masked_fill(~valid, -1)
        last_index = positions.max(dim=1).values
        all_missing = last_index < 0
        last_index = last_index.clamp_min(0)

        gathered = tokens.gather(
            1,
            last_index.view(-1, 1, 1).expand(
                -1,
                1,
                tokens.shape[-1],
            ),
        ).squeeze(1)

        if all_missing.any():
            gathered = gathered.clone()
            gathered[all_missing] = 0.0

        return gathered

    def encode_tokens(self, x, m):
        if x.dim() != 3:
            raise ValueError(
                f"Expected x [B,T,F], got {tuple(x.shape)}"
            )

        if m.dim() != 3:
            raise ValueError(
                f"Expected m [B,T,F], got {tuple(m.shape)}"
            )

        if x.shape != m.shape:
            raise ValueError(
                "x and m shape mismatch. "
                f"x={tuple(x.shape)}, m={tuple(m.shape)}"
            )

        if x.shape[-1] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features, "
                f"got {x.shape[-1]}"
            )

        if self.use_mask_channel:
            z = torch.cat([x, m], dim=-1)
        else:
            z = x

        z = self.input_proj(z)

        time_observed = m.sum(dim=-1) > 0
        token_mask = ~time_observed

        z, local_scale = self.local_block(
            z,
            key_padding_mask=token_mask,
        )

        z = self.pos(z)

        if self.use_cls_token:
            batch_size = z.shape[0]
            cls = self.cls_token.expand(
                batch_size,
                -1,
                -1,
            ).to(dtype=z.dtype, device=z.device)

            z = torch.cat([cls, z], dim=1)

            cls_mask = torch.zeros(
                batch_size,
                1,
                dtype=torch.bool,
                device=z.device,
            )

            encoder_mask = torch.cat(
                [cls_mask, token_mask],
                dim=1,
            )
        else:
            encoder_mask = token_mask
            all_missing = encoder_mask.all(dim=1)

            if all_missing.any():
                encoder_mask = encoder_mask.clone()
                encoder_mask[all_missing, 0] = False

        z = self.encoder(
            z,
            src_key_padding_mask=encoder_mask,
        )

        if self.use_cls_token:
            cls_vec = z[:, 0]
            token_z = z[:, 1:]
        else:
            cls_vec = self.masked_mean(z, encoder_mask)
            token_z = z
            token_mask = encoder_mask

        return cls_vec, token_z, token_mask, local_scale

    def forward_all(self, x, m):
        cls_vec, token_z, token_mask, local_scale = self.encode_tokens(x, m)

        attention_vec, attn = self.attn_pool(
            token_z,
            key_padding_mask=token_mask,
        )

        mean_vec = self.masked_mean(
            token_z,
            token_mask,
        )

        last_vec = self.last_observed(
            token_z,
            token_mask,
        )

        summaries = torch.stack(
            [attention_vec, mean_vec, last_vec],
            dim=1,
        )

        pooled_vec, summary_weights = self.summary_mixer(
            summaries
        )

        ehr_features = torch.cat(
            [cls_vec, pooled_vec],
            dim=-1,
        )

        ehr_logit = self.head(ehr_features).squeeze(-1)

        ehr_tokens = self.fusion_token_proj(token_z)
        ehr_summary = self.fusion_summary_proj(ehr_features)

        return {
            "logit": ehr_logit,
            "attn": attn,
            "summary_weights": summary_weights,
            "local_scale": local_scale,
            "ehr_features": ehr_features,
            "ehr_tokens_raw": token_z,
            "ehr_tokens": ehr_tokens,
            "ehr_summary": ehr_summary,
            "token_mask": token_mask,
            "cls": cls_vec,
            "attention_pooled": attention_vec,
            "mean_pooled": mean_vec,
            "last_pooled": last_vec,
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
            "summary_weights": out["summary_weights"],
            "local_scale": out["local_scale"],
        }
