import torch

from fmfs.flow import make_method
from fmfs.models import UNet


def load_checkpoint(path: str, device: torch.device, use_ema: bool = True):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = UNet(**ckpt["arch"]).to(device)
    model.load_state_dict(ckpt["ema"] if use_ema else ckpt["model"])
    model.eval()
    return model, make_method(ckpt["method"]), ckpt


def sample_kwargs(method_name: str, eta: float) -> dict:
    return {"eta": eta} if method_name == "ddpm" else {}
