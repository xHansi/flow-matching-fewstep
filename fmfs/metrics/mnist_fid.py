from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from fmfs.data import make_loaders


class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.head = nn.Linear(128, 10)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x))


def train_classifier(device: torch.device, epochs: int = 3, cache: str = "results/mnist_clf.pt"):
    clf = MNISTClassifier().to(device)
    path = Path(cache)
    if path.exists():
        clf.load_state_dict(torch.load(path, map_location=device))
        return clf.eval()

    train, _ = make_loaders("mnist", batch_size=128, num_workers=2)
    opt = torch.optim.Adam(clf.parameters(), 1e-3)
    for _ in range(epochs):
        clf.train()
        for x, y in train:
            loss = F.cross_entropy(clf(x.to(device)), y.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(clf.state_dict(), path)
    return clf.eval()


class MNISTFID:
    def __init__(self, device: torch.device, epochs: int = 3):
        self.device = device
        self.clf = train_classifier(device, epochs=epochs)

    @torch.no_grad()
    def features(self, images: torch.Tensor) -> np.ndarray:
        return self.clf.features(images.to(self.device)).cpu().numpy()
