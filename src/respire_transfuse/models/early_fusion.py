
import torch
import torch.nn as nn

from respire_transfuse.models.image_only import ConservativeImageModel
from respire_transfuse.models.ehr_only import EHRTransformerRiskModel
from respire_transfuse.models.respire_transfuse import configure_respire_transfuse_trainability


class EarlyFusionNoGate(nn.Module):
    def __init__(
        self,
        image_branch,
        ehr_branch,
        fusion_dim=48,
        dropout=0.45,
        delta_bound=3.0,
        delta_scale_start=0.12,
        delta_scale_target=0.35,
    ):
        super().__init__()

        self.image_branch = image_branch
        self.ehr_branch = ehr_branch

        self.fusion_dim = int(fusion_dim)
        self.delta_bound = float(delta_bound)
        self.delta_scale_start = float(delta_scale_start)
        self.delta_scale_target = float(delta_scale_target)
        self.current_delta_scale = float(delta_scale_start)

        self.modality_gate = nn.Identity()
        self.cross_layers = nn.ModuleList()

        self.fusion_head = nn.Sequential(
            nn.LayerNorm(self.fusion_dim * 4),
            nn.Dropout(float(dropout)),
            nn.Linear(self.fusion_dim * 4, self.fusion_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.LayerNorm(self.fusion_dim),
            nn.Linear(self.fusion_dim, max(16, self.fusion_dim // 2)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(max(16, self.fusion_dim // 2), 1),
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

    @staticmethod
    def masked_mean(tokens, token_mask=None):
        if token_mask is None:
            return tokens.mean(dim=1)

        valid = token_mask.float()
        denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (tokens * valid.unsqueeze(-1)).sum(dim=1) / denom

    def forward(self, image, ehr_x, ehr_m, return_all=False, return_attention=False):
        image_out = self.image_branch.forward_all(image)
        ehr_out = self.ehr_branch.forward_all(ehr_x, ehr_m)

        image_tokens = image_out["image_tokens"]
        ehr_tokens = ehr_out["ehr_tokens"]

        image_summary = image_out["image_summary"]
        ehr_summary = ehr_out["ehr_summary"]

        image_logit = image_out["logit"].view(-1)
        ehr_logit = ehr_out["logit"].view(-1)

        token_mask = ehr_out.get("token_mask", None)

        image_cross = image_tokens.mean(dim=1)
        ehr_cross = self.masked_mean(ehr_tokens, token_mask)

        fusion_vector = torch.cat(
            [
                image_summary,
                ehr_summary,
                image_cross,
                ehr_cross,
            ],
            dim=1,
        )

        fusion_delta_raw_logit = self.fusion_head(fusion_vector).view(-1)
        fusion_delta_logit = self.delta_bound * torch.tanh(
            fusion_delta_raw_logit / self.delta_bound
        )

        fusion_anchor_logit = ehr_logit.detach()
        fusion_logit = fusion_anchor_logit + self.current_delta_scale * fusion_delta_logit

        ones = torch.ones_like(fusion_logit)

        out = {
            "fusion_logit": fusion_logit,
            "fusion_delta_logit": fusion_delta_logit,
            "fusion_delta_raw_logit": fusion_delta_raw_logit,
            "fusion_anchor_logit": fusion_anchor_logit,
            "image_logit": image_logit,
            "ehr_logit": ehr_logit,
            "image_weight": ones,
            "ehr_weight": ones,
            "gate_weights": torch.stack([ones, ones], dim=1),
            "gate_image": torch.ones_like(image_summary),
            "gate_ehr": torch.ones_like(ehr_summary),
        }

        if return_all or return_attention:
            out.update(
                {
                    "image_tokens": image_tokens,
                    "ehr_tokens": ehr_tokens,
                    "image_summary": image_summary,
                    "ehr_summary": ehr_summary,
                    "image_cross": image_cross,
                    "ehr_cross": ehr_cross,
                    "fusion_vector": fusion_vector,
                    "token_mask": token_mask,
                    "attn_image_to_ehr": None,
                    "attn_ehr_to_image": None,
                }
            )

        return out


def _get_model_cfg(cfg):
    return cfg.get("model", cfg)


def build_early_fusion_from_config(cfg, n_ehr_features):
    model_cfg = _get_model_cfg(cfg)

    image_branch = ConservativeImageModel(
        backbone_name=model_cfg.get("backbone", model_cfg.get("backbone_name", "tf_efficientnet_b0_ns")),
        pretrained=bool(model_cfg.get("pretrained", True)),
        dropout=float(model_cfg.get("image_dropout", model_cfg.get("dropout_image", 0.75))),
        hidden_mult=float(model_cfg.get("image_hidden_mult", 0.02)),
    )

    ehr_branch = EHRTransformerRiskModel(
        n_features=int(n_ehr_features),
        d_model=int(model_cfg.get("ehr_d_model", model_cfg.get("d_model", 128))),
        n_heads=int(model_cfg.get("ehr_n_heads", model_cfg.get("n_heads", 4))),
        n_layers=int(model_cfg.get("ehr_n_layers", model_cfg.get("n_layers", 3))),
        dim_feedforward=int(model_cfg.get("ehr_dim_feedforward", model_cfg.get("dim_feedforward", 384))),
        dropout=float(model_cfg.get("ehr_dropout", model_cfg.get("dropout", 0.25))),
        use_mask_channel=bool(model_cfg.get("use_mask_channel", True)),
        use_cls_token=bool(model_cfg.get("use_cls_token", True)),
    )

    return EarlyFusionNoGate(
        image_branch=image_branch,
        ehr_branch=ehr_branch,
        fusion_dim=int(model_cfg.get("fusion_dim", 48)),
        dropout=float(model_cfg.get("fusion_dropout", model_cfg.get("dropout_fusion", 0.45))),
        delta_bound=float(model_cfg.get("delta_bound", model_cfg.get("fusion_delta_tanh_bound", 3.0))),
        delta_scale_start=float(model_cfg.get("delta_scale_start", model_cfg.get("fusion_delta_scale_start", 0.12))),
        delta_scale_target=float(model_cfg.get("delta_scale_target", model_cfg.get("fusion_delta_scale", 0.35))),
    )


def configure_early_fusion_trainability(model, freeze_cfg):
    configure_respire_transfuse_trainability(model, freeze_cfg)
    return model
