# Context & Codebase Overview: ELFS (Effective Label-Free Selection)

## 1. Introduction
- **Paper**: "[ELFS: Label-Free Coreset Selection with Proxy Training Dynamics](https://openreview.net/forum?id=yklJpvB7Dq)" (ICLR 2025).
- **Objective**: Select a coreset (a highly representative subset of a dataset) *without* requiring ground-truth labels. This is achieved by generating pseudo-labels via self-supervised embeddings and deep clustering, simulating proxy training dynamics on these pseudo-labels, and filtering out redundant or uninformative data based on those dynamics (like margin or forgetting scores).

## 2. Directory Structure
- **`augs/`**: Contains data augmentation pipelines. `augs.py` defines various `IMAGE_AUGMENTATIONS` and `EMBED_AUGMENTATIONS` (e.g., multi-crop augmentations for clustering).
- **`core/`**: Contains the core learning and data management components.
  - **`core/data/`**: Dataset wrappers and the coreset selection logic (`Coreset.py`, `MiscDataset.py`). `CoresetSelection` implements strategies like `budget`, `stratified`, `swav`, `badge`.
  - **`core/model_generator/`**: Neural network architecture definitions like ResNet, WideResNet, PreActResNet (`resnet.py`, `wideresnet.py`, etc.).
  - **`core/training/`**: Training loop helpers. `Trainer.py` logic handles the forward/backward passes and logs proxy training dynamics. `TrainingDynamicsLogger` interacts with `Trainer` to record predictions over epochs.
  - **`core/utils/`**: Generic utility functions like logging, metric calculation, formatting, and clustering helpers.
- **`loaders/`**: Dataloaders for specific datasets (`cifar20.py`, `imagenet.py`, `embedNN.py`). `embedNN.py` provides `EmbedNN` and `PrecomputedEmbeddingDataset`, which are crucial for loading embeddings and fetching nearest neighbors during deep clustering.
- **`losses/`**: Implementations of deep clustering objectives (like TEMI) inside `mi.py` and `multihead_losses.py`.
- **`model_builders/`**: Helper utilities to construct projection heads and models for clustering and feature extraction (`model_builders.py`, `multi_head.py`).
- **`figs/`**: Contains figures generally used for the documentation/README.

## 3. Key Scripts and Their Roles

### 3.1 Pseudo-Label Generation Pipeline
These scripts handle generating pseudo-labels using a pretrained self-supervised model (e.g., DINO) followed by deep clustering, drawing heavy inspiration from the TEMI clustering logic.
- **`gen_embeds.py` / `parall_gen_embeds.py`**: Uses a pretrained vision model (e.g., DINO ViT) to extract features/embeddings for the entire dataset and computes K-nearest neighbors (k-NN) for each sample.
- **`train_cluster_heads.py`**: Trains multiple cluster heads in a teacher-student framework using Exponential Moving Average (EMA) on top of the precomputed embeddings and their k-NN relationships. This maps the continuous representation space into discrete proxy classes.
- **`eval_cluster.py` / `eval_cluster_utils.py`**: Evaluates the trained cluster heads (measures NMI, ARI, Accuracy against actual labels using bipartite matching/Hungarian algorithm for logging) and saves out the discrete class predictions as the final pseudo-labels (`pseudo_label.pt`, `pseudo_label-test.pt`).

### 3.2 Proxy Training Dynamics & Importance Scoring
These scripts use the generated pseudo-labels to estimate how "hard," "easy," or "ambiguous" each data point is by observing a model training on those pseudo-labels over time.
- **`train.py` / `train_imagenet.py`**: The primary training loops. For ELFS, they run in a "proxy mode" (`--load-pseudo`) where a standard architecture (like ResNet-18) is trained on the pseudo-labels. Throughout training, it collects training dynamics (output probabilities per epoch, forgetting occurrences) via the `TrainingDynamicsLogger`. **[Novelty 3]** It now supports `--early-stop-td-ratio` to halt dynamics logging early (e.g., after 20% of epochs) while the model continues training, saving computation. Can also be used later for normal training using ground truth labels on the selected coreset.
- **`generate_importance_score.py` / `generate_importance_score_imagenet.py`**: Reads the saved proxy training dynamics (`td-[task-name].pickle`) and calculates importance scores per sample. Metrics calculated include Accumulated Margin (AUM), Forgetting Events, EL2N, or basic Entropy. **[Novelty 1]** It now fits a 3-component Gaussian Mixture Model (GMM) to the AUM scores to dynamically find the threshold boundaries between noisy, useful, and redundant data. **[Novelty 2]** It also supports `--hybrid`, grouping standard AUM dynamics with spatial embedding density to form a robust, composite `hybrid_score`. Output is consistently saved as a `data-score-[task-name].pickle` dictionary.

### 3.3 Coreset Selection and Final Evaluation
Once sample-wise scores are computed, the codebase allows applying selection limits to prune the dataset.
- **`core/data/Coreset.py`**: Processes the `data-score` attributes and selects a subset of indices. Available logic implementations include:
  - `budget`: The core ELFS selection method, which chops off samples from the ends of the difficulty distribution based on predefined ratios (mislabel filtering and redundancy/easy dropping).
  - `adaptive`: **[Novelty 1]** Automatically filters out the high-confidence noisy and redundant segments using the GMM thresholds pre-calculated, requiring no fixed pruning percentage.
  - `stratified`: Samples uniformly across score quantiles.
  - `swav`: Prototypicality based selection.
  - `badge`: Simulates BADGE active-learning constraints based on scoring.
- **`train.py` (Evaluation mode)**: Rerunning `train.py` with the parameter `--coreset` and `--ignore-td`, coupled with the raw dataset labels. The dataloader uses `Coreset.py` to trim down to the selected indices.

## 4. End-to-End Workflow Summarized
1. **Representation & Clustering**: `gen_embeds.py` (Embedding Generation) -> `train_cluster_heads.py` (Deep Clustering) -> `eval_cluster.py` (Pseudo Label Extraction).
2. **Proxy Dynamics**: `train.py` (Train structural model on Pseudo-labels, record dynamics — optionally stopping early with `--early-stop-td-ratio`).
3. **Scoring & Selection**: `generate_importance_score.py` (Calculate AUM / Forgetting scores + Hybrid density, and fit GMM) -> `Coreset.py` (Select indices based on budget ratios or automatically with `--coreset-mode adaptive`).
4. **Final Model Training**: `train.py` (Train from scratch strictly on the chosen coreset labeled with ground-truth labels for evaluation).
