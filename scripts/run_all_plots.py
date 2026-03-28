"""
Run all visualization scripts with pipeline-friendly defaults.
Usage (from repo root):
  python scripts/run_all_plots.py
  python scripts/run_all_plots.py --base-dir ./data-model/CIFAR10 --score-task proxy-run
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def run_py(script: str, extra: list[str], dry_run: bool) -> int:
    cmd = [sys.executable, str(REPO / "scripts" / script)] + extra
    print("\n>>>", " ".join(cmd))
    if dry_run:
        return 0
    r = subprocess.run(cmd, cwd=str(REPO))
    return r.returncode


def main():
    ap = argparse.ArgumentParser(description="Generate all report figures (PNG) under figures/ dirs.")
    ap.add_argument("--base-dir", type=str, default="./data-model/CIFAR10")
    ap.add_argument("--score-task", type=str, default="proxy-run", help="Task with data-score-*.pickle")
    ap.add_argument(
        "--log-tasks",
        type=str,
        default="final-adaptive,proxy-run",
        help="Comma-separated task names for log-train-*.log",
    )
    ap.add_argument("--metrics-csv", type=str, default=None, help="Default: <base-dir>/checkpoint_metrics.csv")
    ap.add_argument("--tb-logdir", type=str, default=None, help="Default: <base-dir>/")
    ap.add_argument("--pseudo-label", type=str, default=None, help="Default: <base-dir>/pseudo_label.pt")
    ap.add_argument(
        "--accuracy-json",
        type=str,
        default="./data/embeddings/CIFAR10-dino_vitb16/accuracy.json",
    )
    ap.add_argument("--out-cluster-root", type=str, default=None, help="Figures dir for cluster/TB plots")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--fail-fast", action="store_true", help="Stop on first non-zero exit")
    args = ap.parse_args()

    base = Path(args.base_dir)
    metrics_csv = args.metrics_csv or str(base / "checkpoint_metrics.csv")
    tb_dir = args.tb_logdir or str(base)
    pseudo = args.pseudo_label or str(base / "pseudo_label.pt")
    cluster_fig = args.out_cluster_root or str(base / "figures")

    failed = []

    # 1) Cluster metrics CSV
    if Path(metrics_csv).is_file():
        code = run_py(
            "plot_cluster_metrics.py",
            ["--metrics-csv", metrics_csv, "--out-dir", cluster_fig, "--out-name", "cluster_metrics.png"],
            args.dry_run,
        )
        if code != 0:
            failed.append("plot_cluster_metrics")
            if args.fail_fast:
                sys.exit(code)
    else:
        print(f"[skip] plot_cluster_metrics: not found {metrics_csv}")

    # 2) TensorBoard loss
    if Path(tb_dir).is_dir():
        code = run_py(
            "plot_tensorboard_loss.py",
            ["--logdir", tb_dir, "--out-dir", cluster_fig, "--out-name", "tensorboard_train_loss.png"],
            args.dry_run,
        )
        if code != 0:
            failed.append("plot_tensorboard_loss")
            if args.fail_fast:
                sys.exit(code)
    else:
        print(f"[skip] plot_tensorboard_loss: not a directory {tb_dir}")

    # 3) Training logs
    for task in [t.strip() for t in args.log_tasks.split(",") if t.strip()]:
        log_f = base / task / f"log-train-{task}.log"
        if log_f.is_file():
            code = run_py(
                "plot_training_log.py",
                ["--base-dir", str(base), "--task-name", task],
                args.dry_run,
            )
            if code != 0:
                failed.append(f"plot_training_log({task})")
                if args.fail_fast:
                    sys.exit(code)
        else:
            print(f"[skip] plot_training_log: not found {log_f}")

    # 4) Data scores (AUM / GMM / hybrid)
    pkl = base / args.score_task / f"data-score-{args.score_task}.pickle"
    if pkl.is_file():
        extra = [
            "--base-dir",
            str(base),
            "--task-name",
            args.score_task,
            "--pseudo-label-path",
            pseudo if Path(pseudo).is_file() else "",
        ]
        # remove empty pseudo if missing
        if not Path(pseudo).is_file():
            extra = ["--base-dir", str(base), "--task-name", args.score_task]
            print(f"[warn] plot_data_scores: no pseudo labels at {pseudo} (misclassification plot skipped)")
        code = run_py("plot_data_scores.py", extra, args.dry_run)
        if code != 0:
            failed.append("plot_data_scores")
            if args.fail_fast:
                sys.exit(code)
    else:
        print(f"[skip] plot_data_scores: not found {pkl}")

    # 5) Embedding k-NN JSON
    aj = Path(args.accuracy_json)
    if aj.is_file():
        emb_fig = aj.parent / "figures"
        code = run_py(
            "plot_embedding_knn.py",
            ["--accuracy-json", str(aj), "--out-dir", str(emb_fig), "--out-name", "knn_accuracy.png"],
            args.dry_run,
        )
        if code != 0:
            failed.append("plot_embedding_knn")
            if args.fail_fast:
                sys.exit(code)
    else:
        print(f"[skip] plot_embedding_knn: not found {aj}")

    if failed:
        print("\nCompleted with errors:", ", ".join(failed))
        sys.exit(1)
    print("\nAll available plots generated.")
    sys.exit(0)


if __name__ == "__main__":
    main()
