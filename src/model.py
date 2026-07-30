"""
model.py — DrunkSoberNet

CNN(ResNet18) + GRU model for binary drunk/sober video classification.
Input:  [B, T, 3, 224, 224]  (sequence of face-cropped frames)
Output: [B]                  (raw logits — pair with BCEWithLogitsLoss)
"""

import torch
import torch.nn as nn
import torchvision.models as models


class DrunkSoberNet(nn.Module):
    def __init__(self, hidden_size=128, gru_layers=1, dropout=0.3):
        super().__init__()

        # --- CNN backbone: ResNet18 pretrained on ImageNet ---
        backbone = models.resnet18(weights="IMAGENET1K_V1")
        backbone.fc = nn.Identity()  # drop the 1000-class head, keep 512-d features
        self.backbone = backbone

        # --- Temporal modeling: GRU over per-frame feature vectors ---
        self.gru = nn.GRU(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=gru_layers,
            batch_first=True,
        )

        # --- Output block: Dense -> Dropout -> Dense(1) ---
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),  # single logit; do NOT apply sigmoid here
        )

    def forward(self, x):
        # x: [B, T, 3, 224, 224]
        B, T, C, H, W = x.shape

        # Fold batch and time together so the CNN processes all frames at once
        x = x.view(B * T, C, H, W)
        feats = self.backbone(x)          # [B*T, 512]
        feats = feats.view(B, T, -1)      # [B, T, 512]

        # GRU consumes the sequence; h_n is the final hidden state
        _, h_n = self.gru(feats)          # h_n: [gru_layers, B, hidden_size]
        out = self.head(h_n[-1])          # [B, 1]

        return out.squeeze(-1)            # [B] — raw logits

    # ---- Two-phase fine-tuning helpers ----

    def freeze_backbone(self):
        """Phase 1: freeze all ResNet18 weights, train GRU + head only."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_layer4(self):
        """Phase 2: unfreeze only layer4 (most task-specific features)."""
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True


if __name__ == "__main__":
    # Quick sanity check with dummy data
    model = DrunkSoberNet()
    dummy_input = torch.randn(2, 10, 3, 224, 224)  # batch=2, T=10 frames
    logits = model(dummy_input)
    print("Output shape:", logits.shape)  # expected: torch.Size([2])

    probs = torch.sigmoid(logits)
    print("Probabilities:", probs)