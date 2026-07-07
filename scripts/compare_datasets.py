import argparse
import json
from pathlib import Path

from fmfs.utils import save_comparison


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Side-by-side FID-vs-steps comparison across datasets.")
    p.add_argument(
        "--panels",
        nargs="+",
        default=["MNIST=figures/mnist/fid.json", "Fashion-MNIST=figures/fashion/fid.json"],
        help="entries of the form label=path/to/fid.json",
    )
    p.add_argument("--metric", default="inception", help="metric key inside the json")
    p.add_argument("--steps", default="1,2,4,8,16,50,100")
    p.add_argument("--out", default="figures/comparison/fid_vs_steps_inception.png")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    steps = [int(s) for s in args.steps.split(",")]
    panels = []
    for entry in args.panels:
        label, path = entry.split("=", 1)
        panels.append((label, json.load(open(path))[args.metric]))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    title = f"FID vs. sampling steps ({args.metric}) — domain shift"
    save_comparison(panels, steps, str(out), title)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
