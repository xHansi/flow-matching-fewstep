import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

DATASET_META = {"mnist": dict(channels=1, num_classes=10, image_size=32)}
DATASETS = set(DATASET_META)

_BUILDERS = {"mnist": MNIST}


def _transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # -> [-1, 1]
        ]
    )


def make_loaders(
    dataset: str = "mnist",
    data_root: str = "data",
    image_size: int = 32,
    batch_size: int = 128,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader]:
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}, expected one of {sorted(DATASETS)}")

    tf = _transform(image_size)
    builder = _BUILDERS[dataset]
    train = builder(data_root, train=True, download=True, transform=tf)
    test = builder(data_root, train=False, download=True, transform=tf)

    pin = torch.cuda.is_available()
    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin, drop_last=True)
    train_loader = DataLoader(train, shuffle=True, **common)
    test_loader = DataLoader(test, shuffle=False, **common)
    return train_loader, test_loader
