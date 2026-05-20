"""Hybrid v3: state-conditioned policy with explicit image contribution.

Key improvements over v2:
- LayerNorm on img_feat and state_feat before fusion (fixes 12x magnitude imbalance)
- Learnable gates: img_gate (init 2.0), state_gate (init 0.5) — biases toward image early
- Image residual head: base_action (state-only) + image_delta (image+state)
- Reduced state_feat_dim 128→64 to give image more relative weight

Architecture:
  image_encoder(wrist_img) → img_feat (256D) → LN → img_norm
  state_mlp(state) → state_feat (64D) → LN → state_norm
  state_head(state_norm) → base_action (7D)
  img_gate * img_norm + state_gate * state_norm → concat → image_delta_head → delta (7D)
  action = base_action + delta
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


class StateConditionedPolicyV3(nn.Module):
    """Hybrid v3: image residual policy with gated fusion.

    base_action = state_head(LN(state_feat))          — state-only trajectory
    image_delta = delta_head(gate_img*LN(img) || gate_state*LN(state))  — position correction
    action = base_action + image_delta
    """

    def __init__(self, state_dim=7, action_dim=7, img_feat_dim=256, state_feat_dim=64,
                 state_hidden=128, action_hidden=256, use_global_img=False):
        super().__init__()
        self.use_global_img = use_global_img
        self.img_feat_dim = img_feat_dim
        self.state_feat_dim = state_feat_dim

        self.image_encoder = LightweightCNN(in_channels=3, feat_dim=img_feat_dim)
        if use_global_img:
            self.global_encoder = LightweightCNN(in_channels=3, feat_dim=img_feat_dim)

        self.state_mlp = StateMLP(input_dim=state_dim, hidden_dim=state_hidden,
                                  output_dim=state_feat_dim)

        # LayerNorm for feature balance
        self.ln_img = nn.LayerNorm(img_feat_dim)
        self.ln_state = nn.LayerNorm(state_feat_dim)

        # Learnable gates: bias toward image early, model can adjust
        self.img_gate = nn.Parameter(torch.tensor(2.0))
        self.state_gate = nn.Parameter(torch.tensor(0.5))

        # State head: state-only → base action (trajectory progress)
        self.state_head = nn.Sequential(
            nn.Linear(state_feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, action_dim),
        )

        # Image delta head: gated image+state → position-specific correction
        fusion_in = img_feat_dim * (2 if use_global_img else 1) + state_feat_dim
        self.image_delta_head = nn.Sequential(
            nn.Linear(fusion_in, action_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(action_hidden, action_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(action_hidden // 2, action_dim),
        )

    def forward(self, wrist_img, state, global_img=None):
        img_feat = self.image_encoder(wrist_img)
        if self.use_global_img and global_img is not None:
            g_feat = self.global_encoder(global_img)
            img_feat = torch.cat([img_feat, g_feat], dim=1)

        state_feat = self.state_mlp(state)

        # LayerNorm for balanced magnitudes
        img_norm = self.ln_img(img_feat)
        state_norm = self.ln_state(state_feat)

        # Base action: state-only trajectory (always available)
        base_action = self.state_head(state_norm)

        # Gated fusion: image + state → position-specific delta
        gated_img = self.img_gate * img_norm
        gated_state = self.state_gate * state_norm
        fused = torch.cat([gated_img, gated_state], dim=1)
        image_delta = self.image_delta_head(fused)

        return base_action + image_delta

    @property
    def config(self):
        """Minimal config stub for compatibility."""
        class Cfg:
            input_features = {"observation.state": type("S", (), {"shape": (7,)})()}
            chunk_size = 1
            n_action_steps = 1
        return Cfg()
