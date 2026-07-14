import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageOnlyModel(nn.Module):
    def __init__(
        self,
        backbone_name,
        pretrained=True,
        hidden_dim=128,
        dropout=0.50,
    ):
        super().__init__()

        import timm

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=bool(pretrained),
            num_classes=0,
            global_pool="avg",
        )

        self.num_features = int(
            getattr(
                self.backbone,
                "num_features",
            )
        )

        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)

        self.attention_score = nn.Sequential(
            nn.Conv2d(
                self.num_features,
                4,
                kernel_size=1,
                bias=True,
            ),
            nn.GELU(),
            nn.Conv2d(
                4,
                1,
                kernel_size=1,
                bias=True,
            ),
        )

        nn.init.kaiming_normal_(
            self.attention_score[0].weight,
            mode="fan_in",
            nonlinearity="linear",
        )
        nn.init.zeros_(
            self.attention_score[0].bias
        )
        nn.init.zeros_(
            self.attention_score[2].weight
        )
        nn.init.zeros_(
            self.attention_score[2].bias
        )

        self.attention_mix_logit = nn.Parameter(
            torch.tensor(
                -2.0,
                dtype=torch.float32,
            )
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(
                self.num_features
            ),
            nn.Dropout(
                self.dropout
            ),
            nn.Linear(
                self.num_features,
                self.hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(
                self.dropout
            ),
            nn.Linear(
                self.hidden_dim,
                1,
            ),
        )

        self.freeze_backbone()

    def freeze_backbone(self):
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)

    def trainable_parameters(self):
        return (
            list(
                self.attention_score.parameters()
            )
            + [self.attention_mix_logit]
            + list(
                self.classifier.parameters()
            )
        )

    def train(self, mode=True):
        super().train(mode)

        if mode:
            self.backbone.eval()
            self.attention_score.train()
            self.classifier.train()

        return self

    def extract_feature_map(self, x):
        feature_map = (
            self.backbone.forward_features(x)
        )

        if isinstance(
            feature_map,
            (list, tuple),
        ):
            feature_map = feature_map[-1]

        if feature_map.dim() != 4:
            raise RuntimeError(
                "Expected EfficientNet feature map "
                f"[B,C,H,W], got "
                f"{tuple(feature_map.shape)}"
            )

        return feature_map

    @staticmethod
    def global_average_pool(
        feature_map,
    ):
        return F.adaptive_avg_pool2d(
            feature_map,
            1,
        ).flatten(1)

    def attention_pool(
        self,
        feature_map,
    ):
        score_map = self.attention_score(
            feature_map
        )

        batch_size, _, height, width = (
            score_map.shape
        )

        attention = torch.softmax(
            score_map.flatten(2),
            dim=-1,
        ).view(
            batch_size,
            1,
            height,
            width,
        )

        focused_features = (
            feature_map * attention
        ).flatten(2).sum(dim=-1)

        return (
            focused_features,
            attention,
        )

    def forward(
        self,
        x,
        return_all=False,
    ):
        feature_map = (
            self.extract_feature_map(x)
        )

        global_features = (
            self.global_average_pool(
                feature_map
            )
        )

        focused_features, attention = (
            self.attention_pool(
                feature_map
            )
        )

        attention_mix = torch.sigmoid(
            self.attention_mix_logit
        )

        pooled_features = (
            global_features
            + attention_mix
            * (
                focused_features
                - global_features
            )
        )

        logit = self.classifier(
            pooled_features
        ).squeeze(1)

        output = {
            "logit": logit,
        }

        if return_all:
            output.update(
                {
                    "image_features": pooled_features,
                    "global_features": global_features,
                    "focused_features": focused_features,
                    "attention_map": attention,
                    "attention_mix": attention_mix,
                    "feature_map": feature_map,
                }
            )

        return output


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

    def train(self, mode=True):
        super().train(mode)

        if mode and not any(
            parameter.requires_grad
            for parameter in self.backbone.parameters()
        ):
            self.backbone.eval()

        return self

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
            raise RuntimeError(
                "Expected EfficientNet feature map "
                f"[B,C,H,W], got {tuple(feat.shape)}"
            )

        return feat

    def extract_features(self, x):
        feature_map = self.extract_feature_map(x)
        pooled = F.adaptive_avg_pool2d(
            feature_map,
            1,
        ).flatten(1)
        return pooled

    def make_tokens(self, feature_map):
        pooled_grid = self.token_pool(feature_map)
        tokens = pooled_grid.flatten(2).transpose(1, 2).contiguous()
        tokens = self.token_proj(tokens)
        return tokens

    def forward_all(self, x):
        feature_map = self.extract_feature_map(x)
        pooled_feat = F.adaptive_avg_pool2d(
            feature_map,
            1,
        ).flatten(1)

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
        if return_all:
            return self.forward_all(x)

        pooled_feat = self.extract_features(x)
        logit = self.classifier(pooled_feat).squeeze(1)

        return {
            "logit": logit,
        }


def set_backbone_trainable(model, trainable):
    for parameter in model.backbone.parameters():
        parameter.requires_grad = bool(trainable)


def create_ema_model(model):
    ema_model = copy.deepcopy(model)
    ema_model.eval()

    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)

    return ema_model


@torch.no_grad()
def update_ema_model(ema_model, model, decay):
    ema_state = ema_model.state_dict()
    model_state = model.state_dict()

    for key in ema_state.keys():
        ema_value = ema_state[key]
        model_value = model_state[key]

        if torch.is_floating_point(ema_value):
            ema_value.mul_(float(decay)).add_(
                model_value.detach(),
                alpha=1.0 - float(decay),
            )
        else:
            ema_value.copy_(model_value)
