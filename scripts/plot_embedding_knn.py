"""
Bar chart from gen_embeds.py accuracy.json (k-NN on embeddings).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_scripts = Path(__file__).resolve().parent
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_utils import ensure_figures_dir, repo_root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--accuracy-json",
        type=str,
        default="./data/embeddings/CIFAR10-dino_vitb16/accuracy.json",
    )
    ap.add_argument("--out-dir", type=str, default="./data/embeddings/CIFAR10-dino_vitb16/figures")
    ap.add_argument("--out-name", type=str, default="knn_accuracy.png")
    args = ap.parse_args()

    os.chdir(repo_root())
    jpath = Path(args.accuracy_json)
    if not jpath.is_file():
        raise FileNotFoundError(f"Missing {jpath}. Run gen_embeds.py first.")

    with open(jpath, encoding="utf-8") as f:
        data = json.load(f)

    # knn_classifier returns accuracy in 0–100 (percent)
    top1 = float(data.get("top1", 0))
    top5 = float(data.get("top5", 0))

    ensure_figures_dir(args.out_dir)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.bar(["top-1", "top-5"], [top1, top5], color=["C0", "C1"])
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("k-NN on embeddings")
    for i, v in enumerate([top1, top5]):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=10)
    out = Path(args.out_dir) / args.out_name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
