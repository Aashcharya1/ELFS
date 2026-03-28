"""
Plot NMI, ARI, cluster accuracy vs epoch from eval_cluster.py checkpoint_metrics.csv.
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

from plot_utils import ensure_figures_dir, repo_root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--metrics-csv",
        type=str,
        default="./data-model/CIFAR10/checkpoint_metrics.csv",
        help="Path from eval_cluster.py",
    )
    ap.add_argument("--out-dir", type=str, default="./data-model/CIFAR10/figures")
    ap.add_argument("--out-name", type=str, default="cluster_metrics.png")
    args = ap.parse_args()

    os.chdir(repo_root())
    csv_path = Path(args.metrics_csv)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing {csv_path}. Run eval_cluster.py first.")

    df = pd.read_csv(csv_path, index_col=0)
    ensure_figures_dir(args.out_dir)

    cols_plot = [c for c in ("cluster_acc", "nmi", "ari", "train_loss") if c in df.columns]
    if not cols_plot:
        raise ValueError(f"No expected columns in {csv_path.name}; got {list(df.columns)}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for c in cols_plot:
        ax.plot(df.index, df[c], "-o", label=c, markersize=3)
    ax.set_xlabel("epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title("Cluster metrics (eval_cluster)")

    out = Path(args.out_dir) / args.out_name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
