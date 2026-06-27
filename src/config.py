from dataclasses import dataclass


@dataclass
class Config:
    seed: int = 0

    dataset: str = "mnist"
    data_root: str = "data"
    image_size: int = 32
    channels: int = 1
    num_classes: int = 10

    batch_size: int = 128
    num_workers: int = 2
