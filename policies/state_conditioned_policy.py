"""Explicit state-conditioned hybrid policy: image_encoder + state_mlp + fusion + action_head.

Unlike ACT, this architecture forces the state vector to participate in action prediction:
- image_encoder(wrist_img) → image_feat
- state_mlp(observation.state) → state_feat
- concat(image_feat, state_feat) → action_mlp → action

There is no cross-attention, no learned queries, no way for the model to ignore state.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightCNN(nn.Module):
    """Small CNN for wrist camera images. Input: (B, 3, H, W). Output: (B, feat_dim)."""

    def __init__(self, in_channels=3, feat_dim=256, input_h=480, input_w=640):
        super().__init__()
        self.feat_dim = feat_dim
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /4

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),  # /8
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),  # /16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),  # /32
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, feat_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.fc(x)
        return x


class StateMLP(nn.Module):
    """Encode observation.state into a feature vector."""

    def __init__(self, input_dim=7, hidden_dim=128, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ActionHead(nn.Module):
    """Fusion + action prediction MLP."""

    def __init__(self, image_feat_dim=256, state_feat_dim=128, hidden_dim=256, output_dim=7):
        super().__init__()
        in_dim = image_feat_dim + state_feat_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, image_feat, state_feat):
        x = torch.cat([image_feat, state_feat], dim=1)
        return self.net(x)


class StateConditionedPolicy(nn.Module):
    """Explicit state-conditioned hybrid image+state policy.

    Forces action to depend on state by design — no way to ignore state input.
    """

    def __init__(self, state_dim=7, action_dim=7, img_feat_dim=256, state_feat_dim=128,
                 state_hidden=128, action_hidden=256, use_global_img=False):
        super().__init__()
        self.use_global_img = use_global_img
        self.image_encoder = LightweightCNN(in_channels=3, feat_dim=img_feat_dim)
        if use_global_img:
            self.global_encoder = LightweightCNN(in_channels=3, feat_dim=img_feat_dim)
        self.state_mlp = StateMLP(input_dim=state_dim, hidden_dim=state_hidden,
                                  output_dim=state_feat_dim)
        fusion_image_dim = img_feat_dim * (2 if use_global_img else 1)
        self.action_head = ActionHead(image_feat_dim=fusion_image_dim,
                                      state_feat_dim=state_feat_dim,
                                      hidden_dim=action_hidden,
                                      output_dim=action_dim)

    def forward(self, wrist_img, state, global_img=None):
        img_feat = self.image_encoder(wrist_img)
        if self.use_global_img and global_img is not None:
            g_feat = self.global_encoder(global_img)
            img_feat = torch.cat([img_feat, g_feat], dim=1)
        state_feat = self.state_mlp(state)
        return self.action_head(img_feat, state_feat)

    @property
    def config(self):
        """Minimal config stub for compatibility."""
        class Cfg:
            input_features = {"observation.state": type("S", (), {"shape": (7,)})()}
            chunk_size = 1
            n_action_steps = 1
        return Cfg()
