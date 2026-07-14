import torch
import torch.nn as nn
import torch.nn.functional as F

from respire_transfuse.models.image_only import (
    ImageOnlyModel,
)

from respire_transfuse.models.ehr_only import (
    EHRTransformerRiskModel,
)


class EarlyFusionModel(nn.Module):
    def __init__(
        self,
        image_branch,
        ehr_branch,
        fusion_dim=8,
        fusion_dropout=0.0,
    ):
        super().__init__()

        self.image_branch = image_branch
        self.ehr_branch = ehr_branch

        self.fusion_dim = int(fusion_dim)
        self.fusion_dropout = float(
            fusion_dropout
        )

        if self.fusion_dim < 1:
            raise ValueError(
                "fusion_dim must be at least 1."
            )

        self.image_fusion_proj = nn.Sequential(
            nn.LayerNorm(
                self.image_branch.num_features
            ),
            nn.Linear(
                self.image_branch.num_features,
                self.fusion_dim,
                bias=True,
            ),
            nn.GELU(),
        )

        self.ehr_fusion_proj = nn.Sequential(
            nn.LayerNorm(
                self.ehr_branch.feature_dim
            ),
            nn.Linear(
                self.ehr_branch.feature_dim,
                self.fusion_dim,
                bias=True,
            ),
            nn.GELU(),
        )

        self.fusion_head = nn.Sequential(
            nn.LayerNorm(
                self.fusion_dim * 2
            ),
            nn.Dropout(
                self.fusion_dropout
            ),
            nn.Linear(
                self.fusion_dim * 2,
                1,
                bias=True,
            ),
        )

        nn.init.xavier_uniform_(
            self.fusion_head[-1].weight
        )

        nn.init.zeros_(
            self.fusion_head[-1].bias
        )

        self.modality_gate = nn.Identity()
        self.cross_layers = nn.ModuleList()

    def forward(
        self,
        image,
        ehr_x,
        ehr_m,
        return_all=False,
        return_attention=False,
    ):
        image_out = self.image_branch(
            image,
            return_all=True,
        )

        ehr_out = self.ehr_branch.forward_all(
            ehr_x,
            ehr_m,
        )

        image_features = image_out[
            "image_features"
        ]

        ehr_features = ehr_out[
            "ehr_features"
        ]

        image_summary = self.image_fusion_proj(
            image_features
        )

        ehr_summary = self.ehr_fusion_proj(
            ehr_features
        )

        image_summary = F.normalize(
            image_summary,
            p=2,
            dim=1,
            eps=1e-6,
        )

        ehr_summary = F.normalize(
            ehr_summary,
            p=2,
            dim=1,
            eps=1e-6,
        )

        fusion_vector = torch.cat(
            [
                image_summary,
                ehr_summary,
            ],
            dim=1,
        )

        fusion_logit = self.fusion_head(
            fusion_vector
        ).view(-1)

        image_logit = image_out[
            "logit"
        ].view(-1)

        ehr_logit = ehr_out[
            "logit"
        ].view(-1)

        zeros = torch.zeros_like(
            fusion_logit
        )

        half = torch.full_like(
            fusion_logit,
            0.5,
        )

        out = {
            "fusion_logit": fusion_logit,
            "fusion_delta_logit": zeros,
            "fusion_delta_raw_logit": zeros,
            "fusion_anchor_logit": zeros,
            "image_logit": image_logit,
            "ehr_logit": ehr_logit,
            "image_weight": half,
            "ehr_weight": half,
            "gate_weights": torch.stack(
                [
                    half,
                    half,
                ],
                dim=1,
            ),
            "gate_image": torch.full_like(
                image_summary,
                0.5,
            ),
            "gate_ehr": torch.full_like(
                ehr_summary,
                0.5,
            ),
        }

        if return_all or return_attention:
            out.update(
                {
                    "image_features": image_features,
                    "ehr_features": ehr_features,
                    "image_summary": image_summary,
                    "ehr_summary": ehr_summary,
                    "fusion_vector": fusion_vector,
                    "image_tokens": (
                        image_summary.unsqueeze(1)
                    ),
                    "ehr_tokens": (
                        ehr_summary.unsqueeze(1)
                    ),
                    "image_cross": image_summary,
                    "ehr_cross": ehr_summary,
                    "token_mask": ehr_out.get(
                        "token_mask",
                        None,
                    ),
                    "image_attention_map": (
                        image_out.get(
                            "attention_map",
                            None,
                        )
                    ),
                    "ehr_attention": ehr_out.get(
                        "attn",
                        None,
                    ),
                    "attn_image_to_ehr": None,
                    "attn_ehr_to_image": None,
                }
            )

        return out


def build_early_fusion_from_config(
    cfg,
    n_ehr_features,
):
    image_cfg = cfg["image_branch"]
    ehr_cfg = cfg["ehr_branch"]
    model_cfg = cfg["model"]

    image_branch = ImageOnlyModel(
        backbone_name=image_cfg[
            "backbone"
        ],
        pretrained=bool(
            image_cfg["pretrained"]
        ),
        hidden_dim=int(
            image_cfg["hidden_dim"]
        ),
        dropout=float(
            image_cfg["dropout"]
        ),
    )

    ehr_branch = (
        EHRTransformerRiskModel(
            n_features=int(
                n_ehr_features
            ),
            d_model=int(
                ehr_cfg["d_model"]
            ),
            n_heads=int(
                ehr_cfg["n_heads"]
            ),
            n_layers=int(
                ehr_cfg["n_layers"]
            ),
            dim_feedforward=int(
                ehr_cfg["dim_feedforward"]
            ),
            dropout=float(
                ehr_cfg["dropout"]
            ),
            use_mask_channel=bool(
                ehr_cfg["use_mask_channel"]
            ),
            use_cls_token=bool(
                ehr_cfg["use_cls_token"]
            ),
            ehr_token_dim=int(
                ehr_cfg["ehr_token_dim"]
            ),
            local_scale_init=float(
                ehr_cfg["local_scale_init"]
            ),
        )
    )

    ehr_branch.set_fusion_adapters_trainable(
        False
    )

    return EarlyFusionModel(
        image_branch=image_branch,
        ehr_branch=ehr_branch,
        fusion_dim=int(
            model_cfg["fusion_dim"]
        ),
        fusion_dropout=float(
            model_cfg["fusion_dropout"]
        ),
    )


def configure_early_fusion_trainability(
    model,
    freeze_cfg,
):
    expected = {
        "freeze_image_backbone": True,
        "freeze_image_classifier": False,
        "freeze_ehr_branch": False,
        "train_projection_layers": True,
        "train_fusion_layers": True,
    }

    for key, expected_value in expected.items():
        actual_value = bool(
            freeze_cfg.get(
                key,
                expected_value,
            )
        )

        if actual_value != expected_value:
            raise RuntimeError(
                f"{key} must remain "
                f"{expected_value}."
            )

    for parameter in (
        model.image_branch.backbone.parameters()
    ):
        parameter.requires_grad_(False)

    for parameter in (
        model.image_branch.attention_score.parameters()
    ):
        parameter.requires_grad_(True)

    model.image_branch.attention_mix_logit.requires_grad_(
        True
    )

    for parameter in (
        model.image_branch.classifier.parameters()
    ):
        parameter.requires_grad_(True)

    for parameter in (
        model.ehr_branch.parameters()
    ):
        parameter.requires_grad_(True)

    model.ehr_branch.set_fusion_adapters_trainable(
        False
    )

    for module in [
        model.image_fusion_proj,
        model.ehr_fusion_proj,
        model.fusion_head,
    ]:
        for parameter in module.parameters():
            parameter.requires_grad_(True)

    return model
