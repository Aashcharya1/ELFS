"""
Load data-score-<task>.pickle from generate_importance_score.py and save AUM / GMM / hybrid figures.
Run: python scripts/plot_data_scores.py --base-dir ./data-model/CIFAR10 --task-name proxy-run
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

_scripts = Path(__file__).resolve().parent
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from plot_utils import ensure_figures_dir, repo_root

import plots_aux as _plots_aux

# Avoid non-interactive backend warning from plt.show() inside plots_aux
_plots_aux.plt.show = lambda *a, **k: None

from plots_aux import (
    plot_data_score_distribution,
    plot_misclassification_rates,
)


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", type=str, default="./data-model/CIFAR10")
    ap.add_argument("--task-name", type=str, default="proxy-run")
    ap.add_argument(
        "--pickle-path",
        type=str,
        default=None,
        help="Override path (default: <base-dir>/<task>/data-score-<task>.pickle)",
    )
    ap.add_argument(
        "--pseudo-label-path",
        type=str,
        default=None,
        help="e.g. ./data-model/CIFAR10/pseudo_label.pt for misclassification bar chart",
    )
    ap.add_argument("--out-dir", type=str, default=None)
    args = ap.parse_args()

    os.chdir(repo_root())

    if args.pickle_path:
        pkl = Path(args.pickle_path)
    else:
        pkl = Path(args.base_dir) / args.task_name / f"data-score-{args.task_name}.pickle"

    if not pkl.is_file():
        raise FileNotFoundError(f"Pickle not found: {pkl}")

    out_dir = args.out_dir or str(Path(args.base_dir) / args.task_name / "figures")
    ensure_figures_dir(out_dir)
    out_dir = Path(out_dir)

    with open(pkl, "rb") as f:
        d = pickle.load(f)

    aum = _to_numpy(d["accumulated_margin"]).ravel()

    plot_data_score_distribution(
        aum,
        f"AUM ({args.task_name})",
        filename=str(out_dir / f"aum_distribution_{args.task_name}.png"),
    )
    plt.close("all")

    if "gmm_thresholds" in d:
        t_low, t_high = d["gmm_thresholds"]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(aum, bins=80, color="#3a5bd4", alpha=0.75, edgecolor="white", linewidth=0.3)
        ax.axvline(t_low, color="C1", linestyle="--", linewidth=2, label=f"Hard|Useful ({t_low:.4g})")
        ax.axvline(t_high, color="C2", linestyle="--", linewidth=2, label=f"Useful|Easy ({t_high:.4g})")
        ax.set_xlabel("accumulated margin (AUM)")
        ax.set_ylabel("count")
        ax.set_title(f"GMM thresholds ({args.task_name})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(out_dir / f"aum_gmm_thresholds_{args.task_name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_dir / f'aum_gmm_thresholds_{args.task_name}.png'}")

    if "hybrid_score" in d:
        hyb = _to_numpy(d["hybrid_score"]).ravel()
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(aum, hyb, s=4, alpha=0.25, c="C0")
        ax.set_xlabel("AUM")
        ax.set_ylabel("hybrid score")
        ax.set_title(f"Hybrid vs AUM ({args.task_name})")
        ax.grid(True, alpha=0.3)
        fig.savefig(out_dir / f"hybrid_vs_aum_{args.task_name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_dir / f'hybrid_vs_aum_{args.task_name}.png'}")

    pseudo_path = args.pseudo_label_path
    if pseudo_path and Path(pseudo_path).is_file():
        try:
            pl = torch.load(pseudo_path, map_location="cpu", weights_only=False)
        except TypeError:
            pl = torch.load(pseudo_path, map_location="cpu")
        pseudo = _to_numpy(pl).astype(np.int64).ravel()
        gt = _to_numpy(d["targets"]).astype(np.int64).ravel()
        n = min(len(pseudo), len(gt))
        try:
            plot_misclassification_rates(
                gt[:n],
                pseudo[:n],
                f"Pseudo vs GT ({args.task_name})",
                filename=str(out_dir / f"misclassification_by_class_{args.task_name}.png"),
            )
            plt.close("all")
            print(f"Saved {out_dir / f'misclassification_by_class_{args.task_name}.png'}")
        except (IndexError, ValueError) as e:
            print(
                f"[skip] misclassification plot: pseudo/GT label ranges may not match "
                f"(len pseudo={len(pseudo)}, len gt={len(gt)}): {e}"
            )

    print("plot_data_scores done.")


if __name__ == "__main__":
    main()
