@echo off
setlocal enabledelayedexpansion

REM ========== Speed vs quality (tune here) ==========
REM Final train: lower FINAL_ITER = faster, less training; raise TRAIN_BATCH if VRAM allows (3050: try 512, reduce to 256 if OOM).
set "FINAL_ITER=20000"
set "TRAIN_BATCH=512"
set "SCORE_BATCH=512"
REM Proxy step (4/5): larger batch speeds epochs if memory allows
set "PROXY_BATCH=512"

echo ===================================================
echo ELFS End-to-End Pipeline (with Novelties)
echo ===================================================

echo [1/5] Generating embeddings (gen_embeds.py)...
python gen_embeds.py --dataset CIFAR10 --arch dino_vitb16
if !ERRORLEVEL! NEQ 0 (
    echo Error in gen_embeds.py! Exiting.
    exit /b !ERRORLEVEL!
)

echo [2/5] Training cluster heads (train_cluster_heads.py)...
python train_cluster_heads.py --dataset CIFAR10 --arch dino_vitb16 --precomputed --disable_ddp --num_workers 0 --knn_path ./data/embeddings/CIFAR10-dino_vitb16/knn.pt --epochs 200 --output_dir ./data-model/CIFAR10 --loss TEMI --loss-args beta=0.6
if !ERRORLEVEL! NEQ 0 (
    echo Error in train_cluster_heads.py! Exiting.
    exit /b !ERRORLEVEL!
)

echo [3/5] Extracting pseudo-labels (eval_cluster.py)...
python eval_cluster.py --dataset CIFAR10 --arch dino_vitb16 --ckpt_folder ./data-model/CIFAR10/
if !ERRORLEVEL! NEQ 0 (
    echo Error in eval_cluster.py! Exiting.
    exit /b !ERRORLEVEL!
)

echo [4/5] Proxy training with early-stop TD (train.py)...
python train.py --dataset cifar10 --gpuid 0 --epochs 200 --lr 0.1 --network resnet18 --batch-size %PROXY_BATCH% --task-name proxy-run --base-dir ./data-model/CIFAR10/ --load-pseudo --pseudo-train-label-path ./data-model/CIFAR10/pseudo_label.pt --early-stop-td-ratio 0.2
if !ERRORLEVEL! NEQ 0 (
    echo Error in proxy training! Exiting.
    exit /b !ERRORLEVEL!
)

echo [5/5] Scoring ^& Final Adaptive Training...
python generate_importance_score.py --dataset cifar10 --gpuid 0 --base-dir ./data-model/CIFAR10/ --task-name proxy-run --load-pseudo --pseudo-train-label-path ./data-model/CIFAR10/pseudo_label.pt --hybrid --hybrid-alpha 0.5 --hybrid-embedding-path ./data/embeddings/CIFAR10-dino_vitb16/embeddings.pt --batch-size %SCORE_BATCH%
if !ERRORLEVEL! NEQ 0 (
    echo Error in generating scores! Exiting.
    exit /b !ERRORLEVEL!
)

python train.py --dataset cifar10 --gpuid 0 --iterations %FINAL_ITER% --batch-size %TRAIN_BATCH% --task-name final-adaptive --base-dir ./data-model/CIFAR10/ --coreset --coreset-mode adaptive --data-score-path ./data-model/CIFAR10/proxy-run/data-score-proxy-run.pickle --coreset-key hybrid_score --ignore-td
if !ERRORLEVEL! NEQ 0 (
    echo Error in final training! Exiting.
    exit /b !ERRORLEVEL!
)

echo ===================================================
echo Pipeline completed successfully!
echo ===================================================
