#!/usr/bin/env python3
"""
Final training phase: continues from the best checkpoint (ckpt-best.pt) for the same task.

This script does not change train.py. It runs train.py with --load-from-best and an iteration
budget you choose for this phase. train.py will load weights from:

  <base-dir>/<task-name>/ckpt-best.pt

Notes (train.py behavior):
  - Optimizer and LR schedule start fresh for this run; only model weights are restored.
  - --iterations here is the total iteration budget for this invocation (e.g. 3000 more steps),
    not added to a previous counter.

Example:
  python train_final_phase.py --extra-iterations 3000
  python train_final_phase.py --extra-iterations 6000 --gpuid 1
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser(
        description='Resume from ckpt-best.pt and run an additional training phase (train.py --load-from-best).'
    )
    ap.add_argument(
        '--extra-iterations',
        type=int,
        default=3000,
        help='Iteration budget for this phase (passed to train.py --iterations). Default: 3000.',
    )
    ap.add_argument('--gpuid', type=str, default='0', help='CUDA device id (default: 0).')
    ap.add_argument(
        '--task-name',
        type=str,
        default='final-adaptive',
        help='Must match the task that produced ckpt-best.pt (default: final-adaptive).',
    )
    ap.add_argument(
        '--base-dir',
        type=str,
        default='./data-model/CIFAR10/',
        help='Base directory containing the task folder (default: ./data-model/CIFAR10/).',
    )
    args = ap.parse_args()

    train_py = ROOT / 'train.py'
    if not train_py.is_file():
        print(f'ERROR: train.py not found next to this script: {train_py}', file=sys.stderr)
        return 1

    cmd = [
        sys.executable,
        str(train_py),
        '--dataset',
        'cifar10',
        '--gpuid',
        args.gpuid,
        '--iterations',
        str(args.extra_iterations),
        '--batch-size',
        '512',
        '--task-name',
        args.task_name,
        '--base-dir',
        args.base_dir,
        '--coreset',
        '--coreset-mode',
        'adaptive',
        '--data-score-path',
        './data-model/CIFAR10/proxy-run/data-score-proxy-run.pickle',
        '--coreset-key',
        'hybrid_score',
        '--ignore-td',
        '--load-from-best',
    ]

    print('Command:', ' '.join(cmd))
    print(f'Working directory: {ROOT}')
    return subprocess.call(cmd, cwd=str(ROOT))


if __name__ == '__main__':
    raise SystemExit(main())
