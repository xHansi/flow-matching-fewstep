import numpy as np
import torch
from PIL import Image

from fmfs.data import make_loaders
from fmfs.flow import make_method
from fmfs.models import UNet
from fmfs.utils import EMA, set_seed


def tiny_model() -> UNet:
    return UNet(base=16, num_classes=3, image_size=32).eval()


def test_unet_forward_shape():
    m = tiny_model()
    x = torch.randn(2, 1, 32, 32)
    out = m(x, torch.rand(2) * 1000, torch.randint(0, 4, (2,)))
    assert out.shape == x.shape


def test_flow_loss_and_sampling():
    m = tiny_model()
    rf = make_method("flow")
    x1, y = torch.randn(4, 1, 32, 32), torch.randint(0, 3, (4,))
    loss = rf.loss(m, x1, y)
    assert loss.ndim == 0 and loss.requires_grad
    loss.backward()
    for steps in (1, 2, 8):
        assert rf.sample(m, y, steps=steps).shape == (4, 1, 32, 32)
    guided = rf.sample(m, y, steps=4, cfg_scale=2.5)
    assert torch.isfinite(guided).all()


def test_ddpm_loss_and_samplers():
    m = tiny_model()
    ddpm = make_method("ddpm")
    x0, y = torch.randn(4, 1, 32, 32), torch.randint(0, 3, (4,))
    ddpm.loss(m, x0, y).backward()
    for eta in (0.0, 1.0):
        assert ddpm.sample(m, y, steps=8, eta=eta).shape == (4, 1, 32, 32)
    assert torch.isfinite(ddpm.sample(m, y, steps=4, cfg_scale=2.5)).all()


def test_ema_tracks_model():
    m = tiny_model()
    ema = EMA(m, decay=0.9)
    key = next(iter(ema.shadow))
    before = ema.shadow[key].clone()
    with torch.no_grad():
        for p in m.parameters():
            p.add_(1.0)
    ema.update(m)
    assert not torch.equal(ema.shadow[key], before)


def test_sampling_is_deterministic():
    m = tiny_model()
    rf = make_method("flow")
    y = torch.randint(0, 3, (4,))
    set_seed(0)
    a = rf.sample(m, y, steps=4)
    set_seed(0)
    b = rf.sample(m, y, steps=4)
    assert torch.allclose(a, b)


def test_sample_trajectory_shape():
    m = tiny_model()
    y = torch.randint(0, 3, (2,))
    for name in ("flow", "ddpm"):
        x, traj = make_method(name).sample(m, y, steps=5, return_trajectory=True)
        assert x.shape == (2, 1, 32, 32)
        assert traj.shape == (6, 2, 1, 32, 32)  # steps + 1 states


def test_coins_flat_loader(tmp_path):
    for i in range(8):
        arr = (np.random.rand(40, 40) * 255).astype("uint8")
        Image.fromarray(arr, mode="L").save(tmp_path / f"coin_{i}.png")
    train, test = make_loaders("coins", data_root=str(tmp_path), batch_size=2, num_workers=0)
    x, y = next(iter(train))
    assert x.shape == (2, 1, 32, 32)
    assert x.min() >= -1.0 and x.max() <= 1.0
    assert torch.equal(y, torch.zeros(2, dtype=y.dtype))
    assert len(test.dataset) >= 1
