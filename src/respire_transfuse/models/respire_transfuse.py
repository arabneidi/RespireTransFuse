import torch
import torch.nn as nn

from respire_transfuse.models.ehr_only import EHRTransformerRiskModel
from respire_transfuse.models.image_only import ConservativeImageModel


def masked_mean(tokens, token_mask=None):
    if token_mask is None:
        return tokens.mean(dim=1)

    valid = (~token_mask.bool()).to(tokens.dtype).unsqueeze(-1)
    denom = valid.sum(dim=1).clamp_min(1.0)
    return (tokens * valid).sum(dim=1) / denom


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        n_heads=4,
        dim_feedforward=96,
        dropout=0.45,
        residual_scale=0.40,
    ):
        super().__init__()

        self.residual_scale = float(residual_scale)

        self.q_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=int(dim),
            num_heads=int(n_heads),
            dropout=float(dropout),
            batch_first=True,
        )

        self.drop_attn = nn.Dropout(float(dropout))

        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(dim_feedforward, dim),
            nn.Dropout(float(dropout)),
        )

    def forward(self, query_tokens, context_tokens, context_padding_mask=None, need_weights=False):
        q = self.q_norm(query_tokens)
        kv = self.kv_norm(context_tokens)

        attn_out, attn_weights = self.cross_attn(
            query=q,
            key=kv,
            value=kv,
            key_padding_mask=context_padding_mask,
            need_weights=need_weights,
            average_attn_weights=False,
        )

        x = query_tokens + self.residual_scale * self.drop_attn(attn_out)
        x = x + self.residual_scale * self.ffn(self.ffn_norm(x))

        return x, attn_weights


class BidirectionalCrossAttentionLayer(nn.Module):
    def __init__(
        self,
        dim,
        n_heads=4,
        dim_feedforward=96,
        dropout=0.45,
        residual_scale=0.40,
    ):
        super().__init__()

        self.image_queries_ehr = CrossAttentionBlock(
            dim=dim,
            n_heads=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            residual_scale=residual_scale,
        )

        self.ehr_queries_image = CrossAttentionBlock(
            dim=dim,
            n_heads=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            residual_scale=residual_scale,
        )

    def forward(self, image_tokens, ehr_tokens, ehr_padding_mask=None, need_weights=False):
        image_new, attn_image_to_ehr = self.image_queries_ehr(
            query_tokens=image_tokens,
            context_tokens=ehr_tokens,
            context_padding_mask=ehr_padding_mask,
            need_weights=need_weights,
        )

        ehr_new, attn_ehr_to_image = self.ehr_queries_image(
            query_tokens=ehr_tokens,
            context_tokens=image_tokens,
            context_padding_mask=None,
            need_weights=need_weights,
        )

        return image_new, ehr_new, attn_image_to_ehr, attn_ehr_to_image


class RespireTransFuse(nn.Module):
    def __init__(
        self,
        image_branch,
        ehr_branch,
        fusion_dim=48,
        n_heads=4,
        dim_feedforward=96,
        dropout=0.45,
        residual_scale=0.40,
        delta_bound=3.0,
        delta_scale_start=0.12,
        delta_scale_end=0.35,
        detach_ehr_fusion_features=False,
    ):
        super().__init__()

        self.image_branch = image_branch
        self.ehr_branch = ehr_branch

        self.fusion_dim = int(fusion_dim)
        self.delta_bound = float(delta_bound)
        self.delta_scale_start = float(delta_scale_start)
        self.delta_scale_end = float(delta_scale_end)
        self.current_delta_scale = float(delta_scale_start)
        self.detach_ehr_fusion_features = bool(detach_ehr_fusion_features)

        self.cross_layers = nn.ModuleList([
            BidirectionalCrossAttentionLayer(
                dim=self.fusion_dim,
                n_heads=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                residual_scale=residual_scale,
            )
        ])

        self.fusion_head = nn.Sequential(
            nn.LayerNorm(self.fusion_dim * 4),
            nn.Dropout(float(dropout)),
            nn.Linear(self.fusion_dim * 4, self.fusion_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.LayerNorm(self.fusion_dim),
            nn.Linear(self.fusion_dim, self.fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.fusion_dim // 2, 1),
        )

    def set_delta_scale(self, value):
        self.current_delta_scale = float(value)

    def initialize_fusion_prior(self, prevalence):
        prevalence = float(prevalence)
        prevalence = min(max(prevalence, 1e-5), 1.0 - 1e-5)

        last = self.fusion_head[-1]

        if isinstance(last, nn.Linear):
            nn.init.normal_(last.weight, mean=0.0, std=1e-4)
            nn.init.constant_(last.bias, 0.0)

        return {
            "prevalence": float(prevalence),
            "fusion_delta_bias": 0.0,
            "reason": "ehr_anchored_residual_delta_starts_near_zero",
        }

    def forward(self, image, ehr_x, ehr_m, return_all=False, return_attention=False):
        image_out = self.image_branch(image, return_all=True)
        ehr_out = self.ehr_branch(ehr_x, ehr_m, return_all=True)

        image_tokens = image_out["image_tokens"]
        image_summary = image_out["image_summary"]
        image_logit = image_out["logit"]

        ehr_tokens = ehr_out["ehr_tokens"]
        ehr_summary = ehr_out["ehr_summary"]
        ehr_logit = ehr_out["logit"]
        ehr_mask = ehr_out["token_mask"]

        if self.detach_ehr_fusion_features:
            ehr_tokens_for_fusion = ehr_tokens.detach()
            ehr_summary_for_fusion = ehr_summary.detach()
        else:
            ehr_tokens_for_fusion = ehr_tokens
            ehr_summary_for_fusion = ehr_summary

        attn_image_to_ehr = None
        attn_ehr_to_image = None

        for layer in self.cross_layers:
            image_tokens, ehr_tokens_for_fusion, attn_image_to_ehr, attn_ehr_to_image = layer(
                image_tokens=image_tokens,
                ehr_tokens=ehr_tokens_for_fusion,
                ehr_padding_mask=ehr_mask,
                need_weights=bool(return_attention),
            )

        image_cross = image_tokens.mean(dim=1)
        ehr_cross = masked_mean(ehr_tokens_for_fusion, ehr_mask)

        fusion_vector = torch.cat(
            [
                image_summary,
                ehr_summary_for_fusion,
                image_cross,
                ehr_cross,
            ],
            dim=-1,
        )

        fusion_delta_raw_logit = self.fusion_head(fusion_vector).squeeze(-1)
        fusion_delta_logit = self.delta_bound * torch.tanh(
            fusion_delta_raw_logit / self.delta_bound
        )

        fusion_anchor_logit = ehr_logit.detach()
        fusion_logit = fusion_anchor_logit + self.current_delta_scale * fusion_delta_logit

        out = {
            "fusion_logit": fusion_logit,
            "image_logit": image_logit,
            "ehr_logit": ehr_logit,
            "fusion_delta_logit": fusion_delta_logit,
            "fusion_delta_raw_logit": fusion_delta_raw_logit,
        }

        if return_all or return_attention:
            out.update(
                {
                    "image_tokens": image_tokens,
                    "ehr_tokens": ehr_tokens_for_fusion,
                    "image_summary": image_summary,
                    "ehr_summary": ehr_summary,
                    "image_cross": image_cross,
                    "ehr_cross": ehr_cross,
                    "attn_image_to_ehr": attn_image_to_ehr,
                    "attn_ehr_to_image": attn_ehr_to_image,
                }
            )

        return out


def build_respire_transfuse_from_config(cfg, n_ehr_features):
    image_cfg = cfg["image_branch"]
    ehr_cfg = cfg["ehr_branch"]
    model_cfg = cfg["model"]

    image_branch = ConservativeImageModel(
        backbone_name=image_cfg["backbone"],
        pretrained=bool(image_cfg["pretrained"]),
        dropout=float(image_cfg["dropout"]),
        hidden_mult=float(image_cfg["hidden_mult"]),
        image_token_dim=int(image_cfg.get("image_token_dim", 48)),
        token_grid_size=int(image_cfg.get("token_grid_size", 2)),
    )

    ehr_branch = EHRTransformerRiskModel(
        n_features=int(n_ehr_features),
        d_model=int(ehr_cfg["d_model"]),
        n_heads=int(ehr_cfg["n_heads"]),
        n_layers=int(ehr_cfg["n_layers"]),
        dim_feedforward=int(ehr_cfg["dim_feedforward"]),
        dropout=float(ehr_cfg["dropout"]),
        use_mask_channel=bool(ehr_cfg["use_mask_channel"]),
        use_cls_token=bool(ehr_cfg["use_cls_token"]),
        ehr_token_dim=int(ehr_cfg.get("ehr_token_dim", 48)),
    )

    return RespireTransFuse(
        image_branch=image_branch,
        ehr_branch=ehr_branch,
        fusion_dim=int(model_cfg["fusion_dim"]),
        n_heads=int(model_cfg["n_heads"]),
        dim_feedforward=int(model_cfg.get("dim_feedforward", 96)),
        dropout=float(model_cfg["dropout"]),
        residual_scale=float(model_cfg.get("residual_scale", 0.40)),
        delta_bound=float(model_cfg.get("delta_bound", 3.0)),
        delta_scale_start=float(model_cfg.get("delta_scale_start", 0.12)),
        delta_scale_end=float(model_cfg.get("delta_scale_end", 0.35)),
        detach_ehr_fusion_features=bool(model_cfg.get("detach_ehr_fusion_features", False)),
    )


def set_requires_grad(module, value):
    for p in module.parameters():
        p.requires_grad = bool(value)


def configure_respire_transfuse_trainability(model, freeze_cfg):
    set_requires_grad(model, True)

    if bool(freeze_cfg.get("freeze_image_backbone", True)):
        set_requires_grad(model.image_branch.backbone, False)

    if bool(freeze_cfg.get("freeze_image_classifier", False)):
        set_requires_grad(model.image_branch.classifier, False)

    if bool(freeze_cfg.get("freeze_ehr_branch", False)):
        set_requires_grad(model.ehr_branch, False)

    if bool(freeze_cfg.get("train_projection_layers", True)):
        set_requires_grad(model.image_branch.token_proj, True)
        set_requires_grad(model.image_branch.summary_proj, True)
        set_requires_grad(model.ehr_branch.fusion_token_proj, True)
        set_requires_grad(model.ehr_branch.fusion_summary_proj, True)

    if bool(freeze_cfg.get("train_fusion_layers", True)):
        set_requires_grad(model.cross_layers, True)
        set_requires_grad(model.fusion_head, True)