"""Hybrid v4: Lookahead Delta Policy.

Predicts delta from current state to future action (K steps ahead), not absolute action.
Key differences from v3:
- Output: 6D arm delta (no gripper)
- base_delta = state_head(state)  — state-only trajectory
- image_residual = residual_scale * tanh(delta_head(img, state))  — bounded visual correction
- residual_scale is small and fixed, preventing image from dominating
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightCNN(nn.Module):
    """Small CNN for wrist camera images. Input: (B, 3, H, W). Output: (B, feat_dim)."""

    def __init__(self, in_channels=3, feat_dim=256):
        super().__init__()
        self.feat_dim = feat_dim
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
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
    """Encode observation.state into a compact feature vector."""

    def __init__(self, input_dim=7, hidden_dim=128, output_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class HybridDeltaPolicy(nn.Module):
    """Hybrid v4: predict lookahead delta with bounded image residual.

    base_delta = state_head(LN(state_feat))
    image_residual = residual_scale * tanh(delta_head(LN(img) || LN(state)))
    pred_delta = base_delta + image_residual  (6D arm only)
    """

    def __init__(self, state_dim=7, delta_dim=6, img_feat_dim=256, state_feat_dim=64,
                 state_hidden=128, action_hidden=256, use_global_img=False):
        super().__init__()
        self.use_global_img = use_global_img
        self.img_feat_dim = img_feat_dim
        self.state_feat_dim = state_feat_dim
        self.delta_dim = delta_dim

        self.image_encoder = LightweightCNN(in_channels=3, feat_dim=img_feat_dim)
        if use_global_img:
            self.global_encoder = LightweightCNN(in_channels=3, feat_dim=img_feat_dim)

        self.state_mlp = StateMLP(input_dim=state_dim, hidden_dim=state_hidden,
                                  output_dim=state_feat_dim)

        self.ln_img = nn.LayerNorm(img_feat_dim)
        self.ln_state = nn.LayerNorm(state_feat_dim)

        # State head: state → base delta (6D arm only)
        self.state_head = nn.Sequential(
            nn.Linear(state_feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, delta_dim),
        )

        # Image delta head: image+state → bounded residual correction
        fusion_in = img_feat_dim * (2 if use_global_img else 1) + state_feat_dim
        self.image_delta_head = nn.Sequential(
            nn.Linear(fusion_in, action_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(action_hidden, action_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(action_hidden // 2, delta_dim),
        )

        # Per-joint residual scale — small, fixed, prevents image from dominating
        # J1=0.05, J2=0.03, J3=0.05, J4=0.02, J5=0.02, J6=0.02
        self.register_buffer("residual_scale",
                             torch.tensor([0.05, 0.03, 0.05, 0.02, 0.02, 0.02]))

    def forward(self, wrist_img, state, global_img=None):
        img_feat = self.image_encoder(wrist_img)
        if self.use_global_img and global_img is not None:
            g_feat = self.global_encoder(global_img)
            img_feat = torch.cat([img_feat, g_feat], dim=1)

        state_feat = self.state_mlp(state)

        img_norm = self.ln_img(img_feat)
        state_norm = self.ln_state(state_feat)

        # Base delta: state-only trajectory (always available, provides main progress)
        base_delta = self.state_head(state_norm)

        # Image residual: bounded visual correction
        fused = torch.cat([img_norm, state_norm], dim=1)
        image_residual_raw = self.image_delta_head(fused)
        image_residual = self.residual_scale * torch.tanh(image_residual_raw)

        return base_delta + image_residual

    def forward_with_internals(self, wrist_img, state, global_img=None):
        """Forward pass returning base_delta and image_residual separately for diagnostics."""
        img_feat = self.image_encoder(wrist_img)
        if self.use_global_img and global_img is not None:
            g_feat = self.global_encoder(global_img)
            img_feat = torch.cat([img_feat, g_feat], dim=1)

        state_feat = self.state_mlp(state)

        img_norm = self.ln_img(img_feat)
        state_norm = self.ln_state(state_feat)

        base_delta = self.state_head(state_norm)

        fused = torch.cat([img_norm, state_norm], dim=1)
        image_residual_raw = self.image_delta_head(fused)
        image_residual = self.residual_scale * torch.tanh(image_residual_raw)

        return base_delta + image_residual, base_delta, image_residual

    @property
    def config(self):
        class Cfg:
            input_features = {"observation.state": type("S", (), {"shape": (7,)})()}
            chunk_size = 1
            n_action_steps = 1
        return Cfg()
