"""
Plot TensorBoard scalar 'Train loss epoch' from cluster training (train_cluster_heads.py).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_scripts = Path(__file__).resolve().parent
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from plot_utils import ensure_figures_dir, repo_root


def load_tensorboard_loss(path: Path) -> pd.DataFrame:
    tag = "Train loss epoch"
    event_acc = EventAccumulator(str(path))
    event_acc.Reload()
    scalars = event_acc.Tags().get("scalars") or []
    if tag in scalars:
        return pd.DataFrame([{"Epoch": ev.step, "loss": ev.value} for ev in event_acc.Scalars(tag)]).set_index(
            "Epoch"
        )

    dfs = []
    for p in path.rglob("Train loss*/event*"):
        event_acc = EventAccumulator(str(p.parent))
        event_acc.Reload()
        if tag in (event_acc.Tags().get("scalars") or []):
            dfs.append(
                pd.DataFrame([{"Epoch": ev.step, "loss": ev.value} for ev in event_acc.Scalars(tag)]).set_index(
                    "Epoch"
                )
            )
    if not dfs:
        raise ValueError(
            f"No TensorBoard scalars with tag {tag!r} under {path}. "
            "Remove stale events or re-run train_cluster_heads.py."
        )
    df = pd.concat(dfs)
    return df.groupby("Epoch").min()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", type=str, default="./data-model/CIFAR10", help="TensorBoard log dir")
    ap.add_argument("--out-dir", type=str, default="./data-model/CIFAR10/figures")
    ap.add_argument("--out-name", type=str, default="tensorboard_train_loss.png")
    args = ap.parse_args()

    os.chdir(repo_root())
    logdir = Path(args.logdir)
    ensure_figures_dir(args.out_dir)

    try:
        losses_df = load_tensorboard_loss(logdir)
    except ValueError as e:
        print(f"[skip] plot_tensorboard_loss: {e}")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(losses_df.index, losses_df["loss"], "-", color="C0")
    ax.set_xlabel("epoch")
    ax.set_ylabel("Train loss epoch")
    ax.set_title("Cluster heads (TensorBoard)")
    ax.grid(True, alpha=0.3)

    out = Path(args.out_dir) / args.out_name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
