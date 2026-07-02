import numpy as np
import torch
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
from torch import nn


def frechet(feats_real: np.ndarray, feats_gen: np.ndarray) -> float:
    mu1, s1 = feats_real.mean(0), np.cov(feats_real, rowvar=False)
    mu2, s2 = feats_gen.mean(0), np.cov(feats_gen, rowvar=False)
    return float(calculate_frechet_distance(mu1, s1, mu2, s2))


def features_over(extractor, images: torch.Tensor, batch: int = 256) -> np.ndarray:
    feats = [extractor(images[i : i + batch]) for i in range(0, len(images), batch)]
    return np.concatenate(feats)


@torch.no_grad()
def generate(
    method, model, meta: dict, n: int, steps: int, cfg_scale: float, batch: int = 256, **extra
) -> torch.Tensor:
    device = next(model.parameters()).device
    out, remaining = [], n
    while remaining > 0:
        b = min(batch, remaining)
        y = torch.randint(0, meta["num_classes"], (b,), device=device)
        x = method.sample(
            model,
            y,
            steps=steps,
            image_size=meta["image_size"],
            channels=meta["channels"],
            cfg_scale=cfg_scale,
            **extra,
        )
        out.append(x.cpu())
        remaining -= b
    return torch.cat(out)[:n]


class InceptionFID:
    def __init__(self, device: torch.device):
        idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.model: nn.Module = InceptionV3([idx]).to(device).eval()
        self.device = device

    @torch.no_grad()
    def features(self, images: torch.Tensor) -> np.ndarray:
        x = (images.clamp(-1, 1) + 1) / 2
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        f = self.model(x.to(self.device))[0].squeeze(-1).squeeze(-1)
        return f.cpu().numpy()
