"""
Parse log-train-<task>.log and save training / test accuracy curves.
Run from repo root: python scripts/plot_training_log.py --base-dir ./data-model/CIFAR10 --task-name final-adaptive
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

_scripts = Path(__file__).resolve().parent
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_utils import ensure_figures_dir, repo_root


def parse_log(text: str):
    train_epochs, train_loss, train_acc = [], [], []
    test_loss, test_acc = [], []

    re_epoch_loss = re.compile(r">>\s*Epoch\s*\[(\d+)\]:\s*Loss:\s*([\d.]+)")
    re_epoch_tr = re.compile(r">>\s*Epoch\s*\[(\d+)\]:\s*Training Accuracy:\s*([\d.]+)")
    re_test_loss = re.compile(r"^Loss:\s*([\d.]+)\s*$", re.MULTILINE)
    re_test_acc = re.compile(r"^Test Accuracy:\s*([\d.]+)\s*$", re.MULTILINE)

    for m in re_epoch_loss.finditer(text):
        train_epochs.append(int(m.group(1)))
        train_loss.append(float(m.group(2)))
    for m in re_epoch_tr.finditer(text):
        train_acc.append(float(m.group(2)))

    for m in re_test_loss.finditer(text):
        test_loss.append(float(m.group(1)))
    for m in re_test_acc.finditer(text):
        test_acc.append(float(m.group(1)))

    return {
        "train_epochs": train_epochs,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
    }


def plot_parsed(data: dict, out_prefix: Path, title: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    if data["train_epochs"] and data["train_loss"]:
        axes[0].plot(data["train_epochs"], data["train_loss"], "-", label="train loss", color="C0")
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("train loss (sum)")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
    else:
        axes[0].text(0.5, 0.5, "no epoch loss lines", ha="center")

    if data["test_acc"]:
        axes[1].plot(range(1, len(data["test_acc"]) + 1), data["test_acc"], "-o", label="test acc", color="C1")
        axes[1].set_xlabel("test index (order in log)")
        axes[1].set_ylabel("test accuracy (%)")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
    elif data["train_acc"]:
        axes[1].plot(range(1, len(data["train_acc"]) + 1), data["train_acc"], "-o", label="train acc", color="C2")
        axes[1].set_xlabel("epoch index")
        axes[1].set_ylabel("train accuracy (%)")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, "no test accuracy lines", ha="center")

    fig.suptitle(title)
    plt.tight_layout()
    out = Path(out_prefix).with_suffix(".png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    if data["test_acc"]:
        print(f"  last Test Accuracy: {data['test_acc'][-1]:.2f}%")


def main():
    ap = argparse.ArgumentParser(description="Plot curves from train.py log file.")
    ap.add_argument("--log-file", type=str, default=None, help="Full path to log-train-*.log")
    ap.add_argument("--base-dir", type=str, default="./data-model/CIFAR10")
    ap.add_argument("--task-name", type=str, default="final-adaptive")
    ap.add_argument("--out-dir", type=str, default=None, help="Figures directory (default: <base-dir>/<task>/figures)")
    args = ap.parse_args()

    os.chdir(repo_root())

    if args.log_file:
        log_path = Path(args.log_file)
    else:
        log_path = Path(args.base_dir) / args.task_name / f"log-train-{args.task_name}.log"

    if not log_path.is_file():
        raise FileNotFoundError(f"Log not found: {log_path}")

    out_dir = args.out_dir or str(Path(args.base_dir) / args.task_name / "figures")
    ensure_figures_dir(out_dir)

    text = log_path.read_text(encoding="utf-8", errors="replace")
    data = parse_log(text)
    prefix = Path(out_dir) / f"training_curves_{args.task_name}"
    plot_parsed(data, prefix, title=f"Training log: {args.task_name}")


if __name__ == "__main__":
    main()
