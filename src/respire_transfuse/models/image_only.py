import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConservativeImageModel(nn.Module):
    def __init__(
        self,
        backbone_name,
        pretrained=True,
        dropout=0.75,
        hidden_mult=0.02,
        image_token_dim=48,
        token_grid_size=2,
    ):
        super().__init__()

        import timm

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=bool(pretrained),
            num_classes=0,
            global_pool="avg",
        )

        num_features = int(getattr(self.backbone, "num_features"))
        hidden = max(16, int(num_features * float(hidden_mult)))

        self.classifier = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Dropout(float(dropout)),
            nn.Linear(num_features, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, 1),
        )

        self.token_grid_size = int(token_grid_size)
        self.image_token_dim = int(image_token_dim)

        self.token_pool = nn.AdaptiveAvgPool2d(
            (self.token_grid_size, self.token_grid_size)
        )

        self.token_proj = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Linear(num_features, self.image_token_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )

        self.summary_proj = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Linear(num_features, self.image_token_dim),
        )

        self.num_features = num_features
        self.hidden = hidden
        self.feature_dim = num_features
        self.summary_dim = self.image_token_dim
        self.token_dim = self.image_token_dim
        self.num_tokens = self.token_grid_size * self.token_grid_size

    def initialize_prior(self, prevalence):
        prevalence = float(prevalence)
        prevalence = min(max(prevalence, 1e-5), 1.0 - 1e-5)

        bias = math.log(prevalence / (1.0 - prevalence))

        last = self.classifier[-1]

        if isinstance(last, nn.Linear):
            nn.init.normal_(last.weight, mean=0.0, std=1e-4)
            nn.init.constant_(last.bias, bias)

        return {
            "prevalence": float(prevalence),
            "bias": float(bias),
        }

    def extract_feature_map(self, x):
        feat = self.backbone.forward_features(x)

        if isinstance(feat, (list, tuple)):
            feat = feat[-1]

        if feat.dim() != 4:
            raise RuntimeError(f"Expected EfficientNet feature map [B,C,H,W], got {tuple(feat.shape)}")

        return feat

    def extract_features(self, x):
        feature_map = self.extract_feature_map(x)
        pooled = F.adaptive_avg_pool2d(feature_map, 1).flatten(1)
        return pooled

    def make_tokens(self, feature_map):
        pooled_grid = self.token_pool(feature_map)
        tokens = pooled_grid.flatten(2).transpose(1, 2).contiguous()
        tokens = self.token_proj(tokens)
        return tokens

    def forward_all(self, x):
        feature_map = self.extract_feature_map(x)
        pooled_feat = F.adaptive_avg_pool2d(feature_map, 1).flatten(1)

        logit = self.classifier(pooled_feat).squeeze(1)

        image_tokens = self.make_tokens(feature_map)
        image_summary = self.summary_proj(pooled_feat)

        return {
            "logit": logit,
            "image_features": pooled_feat,
            "image_tokens": image_tokens,
            "image_summary": image_summary,
            "feature_map": feature_map,
        }

    def forward_features(self, x):
        out = self.forward_all(x)
        return out["image_features"], out["logit"]

    def forward(self, x, return_all=False):
        out = self.forward_all(x)

        if return_all:
            return out

        return {
            "logit": out["logit"],
        }


def set_backbone_trainable(model, trainable):
    for p in model.backbone.parameters():
        p.requires_grad = bool(trainable)


def create_ema_model(model):
    ema = copy.deepcopy(model)
    ema.eval()

    for p in ema.parameters():
        p.requires_grad_(False)

    return ema


@torch.no_grad()
def update_ema_model(ema_model, model, decay):
    ema_state = ema_model.state_dict()
    model_state = model.state_dict()

    for key in ema_state.keys():
        ema_value = ema_state[key]
        model_value = model_state[key]

        if torch.is_floating_point(ema_value):
            ema_value.mul_(float(decay)).add_(model_value.detach(), alpha=1.0 - float(decay))
        else:
            ema_value.copy_(model_value)
