from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import MNIST

DATASET_META = {
    "mnist": dict(channels=1, num_classes=10, image_size=32),
    # Roman coins: grayscale, centered, treated as unconditional (single class).
    "coins": dict(channels=1, num_classes=1, image_size=32),
}
DATASETS = set(DATASET_META)

_IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _mnist_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # -> [-1, 1]
        ]
    )


def _coins_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Grayscale(1),
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )


class FlatImageDataset(Dataset):
    """All images under `root` (recursively), grayscale, single class (label 0)."""

    def __init__(self, root: str, transform: transforms.Compose):
        self.paths = sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in _IMG_EXT)
        if not self.paths:
            raise FileNotFoundError(f"no images found under {root!r}")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        return self.transform(Image.open(self.paths[i]).convert("L")), 0


def make_loaders(
    dataset: str = "mnist",
    data_root: str = "data",
    image_size: int = 32,
    batch_size: int = 128,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader]:
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}, expected one of {sorted(DATASETS)}")

    if dataset == "mnist":
        tf = _mnist_transform(image_size)
        train: Dataset = MNIST(data_root, train=True, download=True, transform=tf)
        test: Dataset = MNIST(data_root, train=False, download=True, transform=tf)
    else:  # coins: 90/10 deterministic split of a flat image folder
        full = FlatImageDataset(data_root, _coins_transform(image_size))
        n_test = max(1, len(full) // 10)
        gen = torch.Generator().manual_seed(0)
        perm = torch.randperm(len(full), generator=gen).tolist()
        test = torch.utils.data.Subset(full, perm[:n_test])
        train = torch.utils.data.Subset(full, perm[n_test:])

    pin = torch.cuda.is_available()
    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin, drop_last=True)
    train_loader = DataLoader(train, shuffle=True, **common)
    test_loader = DataLoader(test, shuffle=False, **common)
    return train_loader, test_loader
