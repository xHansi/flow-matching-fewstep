from .rectified_flow import RectifiedFlow

__all__ = ["RectifiedFlow", "make_method"]


def make_method(name: str):
    if name == "flow":
        return RectifiedFlow()
    if name == "ddpm":
        from .ddpm import DDPM

        return DDPM()
    raise ValueError(f"unknown method {name!r}")
