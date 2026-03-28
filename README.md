# Effective Label-Free Selection (ELFS)

Code implementation for ICLR 2025 paper: [ELFS: Label-Free Coreset Selection with Proxy Training Dynamics](https://openreview.net/forum?id=yklJpvB7Dq).

ELFS is a label-free coreset selection method that uses deep clustering and **proxy training dynamics** (e.g. accumulated margin, forgetting) to score examples **without** ground-truth labels at selection time, then trains on a chosen subset with real labels for evaluation.

If you find this work useful in your research, please consider citing:

```
@inproceedings{zheng2025elfs,
  title={ELFS: Label-Free Coreset Selection with Proxy Training Dynamics},
  author={Zheng, Haizhong and Tsai, Elisa and Lu, Yifu and Sun, Jiachen and Bartoldson, Brian R and Kailkhura, Bhavya and Prakash, Atul},
  booktitle={Proceedings of the International Conference on Learning Representations (ICLR)},
  year={2025}
}
```

---

## Extensions in this repository

This fork extends the baseline ELFS + [CCS](https://github.com/haizhongzheng/Coverage-centric-coreset-selection)-style pipeline with the following **optional** features:

| Feature | Where | What it does |
|--------|--------|----------------|
| **GMM adaptive thresholds** | `generate_importance_score.py` | Fits a **3-component Gaussian Mixture Model** on **AUM** scores and stores decision boundaries between *hard/noisy*, *ambiguous/useful*, and *easy/redundant* regions. |
| **Adaptive coreset** | `core/data/Coreset.py`, `train.py` | `--coreset-mode adaptive` keeps samples whose score lies in the **useful** band using `gmm_thresholds` from the score pickle (no fixed `--coreset-ratio` / `--mis-ratio`). |
| **Hybrid score** | `generate_importance_score.py` | `--hybrid` combines **normalized AUM** with **embedding-space density** (distance to k-means centroids in DINO space): `hybrid = α·AUM_norm + (1−α)·density_norm`. Use `--coreset-key hybrid_score` when selecting. |
| **Early-stop training dynamics** | `train.py` | `--early-stop-td-ratio` (e.g. `0.2`) stops **logging** training dynamics after that fraction of epochs; optimization continues. Reduces disk and I/O for long proxy runs. |

**Note:** GMM is fit on **AUM**; the same numeric thresholds are applied when you threshold **`hybrid_score`** under adaptive mode. If most hybrid values fall inside the band, adaptive selection may keep **100%** of the data—inspect logs and adjust strategy (e.g. `budget` mode, or `--coreset-key accumulated_margin`) if you need a smaller coreset.

See `context.md` for a concise codebase map.

---

## Environment

```bash
conda env create -f environment.yml
```

**Python packages used by the clustering / eval path:** `tensorboard` is required for `train_cluster_heads.py` and TensorBoard log reading in `eval_cluster.py`:

```bash
pip install tensorboard
```

Use a **CUDA**-enabled PyTorch build for GPU training. On **Windows**, `train.py` uses `num_workers=0` for DataLoaders by default to avoid multiprocessing spawn issues.

---

## End-to-end pipeline (CIFAR-10 + DINO + novelties)

From the repo root on **Windows**:

```bat
run_pipeline.bat
```

The batch file sets tunable variables at the top (`FINAL_ITER`, `TRAIN_BATCH`, `SCORE_BATCH`, `PROXY_BATCH`). **Do not** use raw `&` inside `echo` lines in CMD (use `^&` or quotes).

**Equivalent manual steps** (Linux/macOS or step-by-step):

1. **Embeddings + k-NN**

```bash
python gen_embeds.py --dataset CIFAR10 --arch dino_vitb16
```

2. **Cluster heads (TEMI on precomputed embeddings)**

```bash
python train_cluster_heads.py --dataset CIFAR10 --arch dino_vitb16 --precomputed --disable_ddp \
  --num_workers 0 --knn_path ./data/embeddings/CIFAR10-dino_vitb16/knn.pt --epochs 200 \
  --output_dir ./data-model/CIFAR10 --loss TEMI --loss-args beta=0.6
```

On Linux with DDP you may omit `--disable_ddp` and increase `--num_workers`.

3. **Pseudo-labels**

```bash
python eval_cluster.py --dataset CIFAR10 --arch dino_vitb16 --ckpt_folder ./data-model/CIFAR10/
```

4. **Proxy training + dynamics** (optional early TD stop)

```bash
python train.py --dataset cifar10 --gpuid 0 --epochs 200 --lr 0.1 --network resnet18 --batch-size 256 \
  --task-name proxy-run --base-dir ./data-model/CIFAR10/ \
  --load-pseudo --pseudo-train-label-path ./data-model/CIFAR10/pseudo_label.pt \
  --early-stop-td-ratio 0.2
```

5. **Scores (GMM + optional hybrid) + final adaptive training**

```bash
python generate_importance_score.py --dataset cifar10 --gpuid 0 --base-dir ./data-model/CIFAR10/ \
  --task-name proxy-run --load-pseudo --pseudo-train-label-path ./data-model/CIFAR10/pseudo_label.pt \
  --hybrid --hybrid-alpha 0.5 \
  --hybrid-embedding-path ./data/embeddings/CIFAR10-dino_vitb16/embeddings.pt

python train.py --dataset cifar10 --gpuid 0 --iterations 40000 --batch-size 256 \
  --task-name final-adaptive --base-dir ./data-model/CIFAR10/ \
  --coreset --coreset-mode adaptive \
  --data-score-path ./data-model/CIFAR10/proxy-run/data-score-proxy-run.pickle \
  --coreset-key hybrid_score --ignore-td
```

Artifacts are written under `./data-model/CIFAR10/` (and embeddings under `./data/embeddings/`). **Do not commit** large checkpoints or data—see `.gitignore`.

---

## Original usage examples (budget / AUM)

These match the classic CCS-style **budget** coreset (fixed `--coreset-ratio` and `--mis-ratio`). **Importance scoring** must be run first so `data-score-*.pickle` exists.

**Training dynamics (proxy) with pseudo-labels**

```bash
python train.py --dataset cifar10 --gpuid 0 --epochs 200 --lr 0.1 \
  --network resnet18 --batch-size 256 --task-name all-data \
  --base-dir ./data-model/cifar10 \
  --load-pseudo --pseudo-train-label-path <path-to-cifar10_label.pt> \
  --pseudo-test-label-path <path-to-cifar10_label-test.pt>
```

**Importance scores**

```bash
python generate_importance_score.py --dataset cifar10 --gpuid 0 \
  --base-dir ./data-model/cifar10 --task-name all-data \
  --load-pseudo --pseudo-train-label-path <path-to-cifar10_label.pt>
```

Scores are saved under `./data-model/cifar10/<task>/data-score-<task>.pickle` by default.

**ELFS with AUM (budget)**

```bash
python train.py --dataset cifar10 --gpuid 0 --iterations 40000 --task-name budget-0.1 \
  --base-dir ./data-model/cifar10/ --coreset --coreset-mode budget \
  --data-score-path <path-to-data-score.pickle> \
  --coreset-key accumulated_margin \
  --coreset-ratio 0.1 --mis-ratio 0.4 --ignore-td
```

**ELFS with forgetting (budget)**

```bash
python train.py --dataset cifar10 --gpuid 0 --iterations 40000 --task-name budget-0.1-forgetting \
  --base-dir ./data-model/cifar10/ --coreset --coreset-mode budget \
  --data-score-path <path-to-data-score.pickle> --coreset-key forgetting --data-score-descending 1 \
  --coreset-ratio 0.1 --mis-ratio 0.4 --ignore-td
```

More sampling methods (random, EL2N, etc.) are described in the [CCS repository](https://github.com/haizhongzheng/Coverage-centric-coreset-selection).

---

## ImageNet training

```bash
# Train classifier and collect training dynamics
python train_imagenet.py --epochs 60 --lr 0.1 --scheduler cosine --task-name pseudo_dino \
  --base-dir ./data-model/imagenet --data-dir <path-to-imagenet-data> --network resnet34 \
  --batch-size 256 --gpuid 0,1 --load-pseudo \
  --pseudo-train-label-path <path-to-imagenet-pseudo-labels.pt> \
  --pseudo-test-label-path <path-to-imagenet-pseudo-labels-test.pt>

# Per-example scores
python generate_importance_score_imagenet.py --data-dir <path-to-imagenet-data> \
  --base-dir ./data-model/imagenet --task-name pseudo_dino --load_pseudo \
  --pseudo-train-label-path <path-to-imagenet-pseudo-labels.pt> \
  --data-score-path ./imagenet_dino_score.pt

# Train with ELFS coreset (example: 90% pruning)
python train_imagenet.py --iterations 300000 --iterations-per-testing 5000 --lr 0.1 \
  --scheduler cosine --task-name budget-0.1 --data-dir <path-to-imagenet-data> \
  --base-dir ./data-model/imagenet --coreset --coreset-mode budget \
  --data-score-path ./imagenet_dino_score.pt --coreset-key accumulated_margin \
  --network resnet34 --batch-size 256 --coreset-ratio 0.1 --mis-ratio 0.3 \
  --data-score-descending 1 --gpuid 0,1 --ignore-td
```

---

## Contributing / git

Large artifacts (checkpoints, `*.pt`, pickles, downloaded data) are listed in **`.gitignore`**. Track **source code** and small configs; regenerate data and models locally.

---

## Acknowledgements

Thanks to the authors of [Exploring the Limits of Deep Image Clustering using Pretrained Models](https://github.com/HHU-MMBS/TEMI-official-BMVC2023) — the pseudo-label generation pipeline builds on that line of work.

Thanks to the authors of [Coverage-centric Coreset Selection for High Pruning Rates](https://github.com/haizhongzheng/Coverage-centric-coreset-selection). Much of this codebase is adapted from their code.

## Other baselines

* Random sampling  
* [Active Learning (BADGE)](https://decile-team-distil.readthedocs.io/en/latest/ActStrategy/distil.active_learning_strategies.html#module-distil.active_learning_strategies.badge)  
* Prototypicality ([SWaV](https://github.com/facebookresearch/swav) + k-means)  
* [D2 Pruning](https://github.com/adymaharana/d2pruning/tree/master)
