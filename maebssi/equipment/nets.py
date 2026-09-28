import torch
import torch.nn as nn


class FusionNet(nn.Module):
    """열화상 썸네일(1x30x40) CNN + 센서/시계열 특징 MLP 결합 분류기 (4클래스)."""

    def __init__(self, n_tab: int, n_cls: int = 4):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.GELU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(n_tab, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.GELU(),
        )
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(64 + 128, 64), nn.GELU(), nn.Linear(64, n_cls))

    def forward(self, tab: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.cnn(img), self.mlp(tab)], dim=1))
