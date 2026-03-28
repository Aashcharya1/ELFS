import torch
import numpy as np
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import os, sys
_NUM_DL_WORKERS = 0 if sys.platform == 'win32' else 16
import argparse
import pickle
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture

from core.model_generator import wideresnet, preact_resnet, resnet
from core.training import Trainer, TrainingDynamicsLogger
from core.data import IndexDataset, CIFARDataset, SVHNDataset, CINIC10Dataset, STL10Dataset
from core.utils import print_training_info, find_centroid_kmeans, calculate_distances


def fit_gmm_thresholds(scores, n_components=3, random_state=42):
    """Fit a GMM to the 1-D score array and return adaptive thresholds.

    The three components are interpreted (after sorting by mean) as:
        lowest-mean  -> Hard / Noisy
        middle-mean  -> Ambiguous / Useful  (keep these)
        highest-mean -> Easy / Redundant

    Returns
    -------
    threshold_low  : float  – boundary between Hard and Useful
    threshold_high : float  – boundary between Useful and Easy
    gmm            : fitted GaussianMixture object
    """
    X = scores.reshape(-1, 1)
    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    gmm.fit(X)

    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.flatten())
    weights = gmm.weights_.flatten()
    order = np.argsort(means)
    means, stds, weights = means[order], stds[order], weights[order]

    # Gaussian intersection between adjacent components
    def _gaussian_intersection(mu1, s1, mu2, s2):
        """Find the intersection point between two 1-D Gaussians (mu1 < mu2)."""
        if np.isclose(s1, s2):
            return (mu1 + mu2) / 2.0
        a = 1.0 / (2 * s1**2) - 1.0 / (2 * s2**2)
        b = mu2 / (s2**2) - mu1 / (s1**2)
        c = mu1**2 / (2 * s1**2) - mu2**2 / (2 * s2**2) - np.log(s2 / s1)
        disc = b**2 - 4 * a * c
        if disc < 0:
            return (mu1 + mu2) / 2.0
        roots = [(-b + np.sqrt(disc)) / (2 * a), (-b - np.sqrt(disc)) / (2 * a)]
        # pick the root that lies between the two means
        for r in roots:
            if mu1 <= r <= mu2:
                return r
        return (mu1 + mu2) / 2.0

    threshold_low = _gaussian_intersection(means[0], stds[0], means[1], stds[1])
    threshold_high = _gaussian_intersection(means[1], stds[1], means[2], stds[2])

    print(f'\n=== Adaptive GMM Thresholds ===')
    print(f'  Component means (sorted): {means}')
    print(f'  Component stds  (sorted): {stds}')
    print(f'  Component weights (sorted): {weights}')
    print(f'  Threshold (Hard|Useful):  {threshold_low:.6f}')
    print(f'  Threshold (Useful|Easy):  {threshold_high:.6f}')

    n_hard = int((scores < threshold_low).sum())
    n_useful = int(((scores >= threshold_low) & (scores <= threshold_high)).sum())
    n_easy = int((scores > threshold_high).sum())
    total = len(scores)
    print(f'  Hard/Noisy:       {n_hard} ({n_hard/total*100:.1f}%)')
    print(f'  Ambiguous/Useful: {n_useful} ({n_useful/total*100:.1f}%)')
    print(f'  Easy/Redundant:   {n_easy} ({n_easy/total*100:.1f}%)')
    print(f'  Effective keep ratio: {n_useful/total*100:.1f}%')
    print(f'================================\n')

    return threshold_low, threshold_high, gmm


def compute_hybrid_score(aum_scores, embedding_path, num_classes, alpha=0.5):
    """Combine AUM with embedding-space density into a hybrid importance score."""
    embeddings = torch.load(embedding_path, map_location='cpu')
    print(f'Loaded embeddings from {embedding_path}, shape={embeddings.shape}')

    centroids, cluster_labels = find_centroid_kmeans(embeddings.numpy(), num_classes)
    distances_list = calculate_distances(embeddings.numpy(), cluster_labels, centroids)
    # distances_list is list of [index, distance, label]
    density_scores = np.zeros(len(distances_list))
    for item in distances_list:
        density_scores[int(item[0])] = item[1]

    # Normalize both to [0, 1]
    def _minmax(arr):
        mn, mx = arr.min(), arr.max()
        if np.isclose(mn, mx):
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    aum_norm = _minmax(aum_scores)
    density_norm = _minmax(density_scores)

    hybrid = alpha * aum_norm + (1.0 - alpha) * density_norm
    print(f'Hybrid score computed (alpha={alpha}): mean={hybrid.mean():.4f}, std={hybrid.std():.4f}')
    return torch.tensor(hybrid, dtype=torch.float32), torch.tensor(density_scores, dtype=torch.float32)



parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

######################### Data Setting #########################
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training.')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'tiny', 'svhn', 'cinic10', 'stl10'])

######################### Path Setting #########################
parser.add_argument('--data-dir', type=str, default='../data/',
                    help='The dir path of the data.')
parser.add_argument('--base-dir', type=str,
                    help='The base dir of this project.')
parser.add_argument('--task-name', type=str, default='tmp',
                    help='The name of the training task.')

######################### GPU Setting #########################
parser.add_argument('--gpuid', type=str, default='0',
                    help='The ID of GPU.')

######################### Importance Score Generation Scheme #########################
parser.add_argument('--from-td', type=int, default=1,
                    help='Set 0 to calculate score for prototypicality.')
parser.add_argument('--importance-scheme', type=str, default='td', choices=['td', 'prototypicality'])  # 
parser.add_argument('--embedding-path', type=str, help='Path for the embedding') # for swav, simclr, etc.

################### Load Pseudo Labels from DL models ###################
parser.add_argument('--load-pseudo', action='store_true', default=False)
parser.add_argument('--pseudo-train-label-path', type=str, help='Path for the pseudo train labels')

######################### Novelty Flags #########################
parser.add_argument('--hybrid', action='store_true', default=False,
                    help='Compute hybrid importance score (AUM + embedding density).')
parser.add_argument('--hybrid-alpha', type=float, default=0.5,
                    help='Weight for AUM in hybrid score: S = alpha*AUM + (1-alpha)*density.')
parser.add_argument('--hybrid-embedding-path', type=str, default=None,
                    help='Path to precomputed embeddings .pt file for hybrid density scoring.')

args = parser.parse_args()

######################### Set path variable #########################
task_dir = os.path.join(args.base_dir, args.task_name)
ckpt_path = os.path.join(task_dir, f'ckpt-last.pt')
td_path = os.path.join(task_dir, f'td-{args.task_name}.pickle')
data_score_path = os.path.join(task_dir, f'data-score-{args.task_name}.pickle')

######################### Print setting #########################
print_training_info(args, all=True)

#########################
dataset = args.dataset
print(f"Dataset is {dataset}")
if dataset in ['cifar10', 'svhn', 'cinic10', 'stl10']:
    num_classes=10
elif dataset == 'cifar100':
    num_classes=100
    

######################### Ftn definition #########################
"""Calculate loss and entropy"""
def post_training_metrics(model, dataloader, data_importance, device):
    model.eval()
    data_importance['entropy'] = torch.zeros(len(dataloader.dataset))
    data_importance['loss'] = torch.zeros(len(dataloader.dataset))

    for batch_idx, (idx, (inputs, targets)) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        logits = model(inputs)
        prob = nn.Softmax(dim=1)(logits)

        entropy = -1 * prob * torch.log(prob + 1e-10)
        entropy = torch.sum(entropy, dim=1).detach().cpu()

        loss = nn.CrossEntropyLoss(reduction='none')(logits, targets).detach().cpu()

        data_importance['entropy'][idx] = entropy
        data_importance['loss'][idx] = loss

"""Calculate td metrics"""
def training_dynamics_metrics(td_log, dataset, data_importance):
    targets = []
    data_size = len(dataset)

    for i in range(data_size):
        _, (_, y) = dataset[i]
        targets.append(y)
    targets = torch.tensor(targets)
    data_importance['targets'] = targets.type(torch.int32)

    data_importance['correctness'] = torch.zeros(data_size).type(torch.int32)
    data_importance['forgetting'] = torch.zeros(data_size).type(torch.int32)
    data_importance['last_correctness'] = torch.zeros(data_size).type(torch.int32)
    data_importance['accumulated_margin'] = torch.zeros(data_size).type(torch.float32)

    def record_training_dynamics(td_log):
        #output = torch.exp(td_log['output'].type(torch.float))
        output = torch.tensor(td_log['output'], dtype=torch.float32)
        output = F.softmax(output, dim=-1)

        predicted = output.argmax(dim=1)
        index = td_log['idx'].type(torch.long)

        label = targets[index]

        correctness = (predicted == label).type(torch.int)
        data_importance['forgetting'][index] += torch.logical_and(data_importance['last_correctness'][index] == 1, correctness == 0)
        data_importance['last_correctness'][index] = correctness
        data_importance['correctness'][index] += data_importance['last_correctness'][index]

        batch_idx = range(output.shape[0])
        target_prob = output[batch_idx, label]
        output[batch_idx, label] = 0
        other_highest_prob = torch.max(output, dim=1)[0]
        margin = target_prob - other_highest_prob
        data_importance['accumulated_margin'][index] += margin

    for i, item in enumerate(td_log):
        if i % 10000 == 0:
            print(i)
        record_training_dynamics(item)

"""Calculate td metrics"""
def EL2N(td_log, dataset, data_importance, max_epoch=10):
    targets = []
    data_size = len(dataset)

    for i in range(data_size):
        _, (_, y) = dataset[i]
        targets.append(y)
    targets = torch.tensor(targets)
    data_importance['targets'] = targets.type(torch.int32)
    data_importance['el2n'] = torch.zeros(data_size).type(torch.float32)
    l2_loss = torch.nn.MSELoss(reduction='none')

    def record_training_dynamics(td_log):
        output = torch.tensor(td_log['output'] , dtype=torch.float32)  
        output = F.softmax(output, dim=1)
        predicted = output.argmax(dim=1)
        index = td_log['idx'].type(torch.long)

        label = targets[index]

        label_onehot = torch.nn.functional.one_hot(label, num_classes=num_classes)
        el2n_score = torch.sqrt(l2_loss(label_onehot,output).sum(dim=1))

        data_importance['el2n'][index] += el2n_score

    for i, item in enumerate(td_log):
        if i % 10000 == 0:
            print(i)
        if item['epoch'] == max_epoch:
            return
        record_training_dynamics(item)
        
#########################

GPUID = args.gpuid
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUID)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform_identical = transforms.Compose([
            transforms.ToTensor(),
        ])

data_dir =  os.path.join(args.data_dir, dataset)
print(f'dataset: {dataset}, data_dir: {data_dir}')
valset = None
if dataset == 'cifar10':
    trainset = CIFARDataset.get_cifar10_train(data_dir, transform = transform_identical)
elif dataset == 'cifar100':
    trainset = CIFARDataset.get_cifar100_train(data_dir, transform = transform_identical)
elif dataset == 'svhn':
    trainset = SVHNDataset.get_svhn_train(data_dir, transform = transform_identical)
elif dataset == 'stl10':
    trainset = STL10Dataset.get_stl10_train(data_dir, transform = transform_identical)
elif args.dataset == 'cinic10':
    trainset = CINIC10Dataset.get_cinic10_train(data_dir, transform = transform_identical)
    valset = CINIC10Dataset.get_cinic10_train(data_dir, transform = transform_identical, is_val=True)

if args.from_td == 1:
    if args.load_pseudo:
        if "cifar" in args.dataset:
            #--pseudo_train_label_path example: ../datasets/cifar-100-python/label.pt 
            print(f"Loading Pseudo dataset labels from {args.pseudo_train_label_path}")
            trainset = CIFARDataset.load_custom_labels(trainset, args.pseudo_train_label_path)
        if "svhn" in args.dataset:
            print(f"Loading Pseudo dataset labels from {args.pseudo_train_label_path}")
            trainset = SVHNDataset.load_custom_labels(trainset, args.pseudo_train_label_path)
        if "stl" in args.dataset:
            print(f"Loading Pseudo dataset labels from {args.pseudo_train_label_path}")
            trainset = STL10Dataset.load_custom_labels(trainset, args.pseudo_train_label_path)
        if "cinic" in args.dataset:
            print(f"Loading Pseudo dataset labels from {args.pseudo_train_label_path}")
            trainset = CINIC10Dataset.load_custom_labels(trainset, args.pseudo_train_label_path)
            print(f"Loading Pseudo dataset labels from {args.pseudo_train_label_path}")
            valset = CINIC10Dataset.load_custom_labels(valset, args.pseudo_train_label_path)

    if valset:
        # merge trainset and valset
        trainset = torch.utils.data.ConcatDataset([trainset, valset])

    trainset = IndexDataset(trainset)
    print(f"Trainset size: {len(trainset)}")

    if args.load_pseudo:
        pl = torch.load(args.pseudo_train_label_path, weights_only=False)
        num_classes = int(torch.as_tensor(pl).long().max().item()) + 1
        print(f"Number of classes: {num_classes} (from pseudo labels)")

    data_importance = {}

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=False, num_workers=_NUM_DL_WORKERS)

    model = resnet('resnet18', num_classes=num_classes, device=device)
    model = model.to(device)


    # print(f'Ckpt path: {ckpt_path}.')
    # checkpoint = torch.load(ckpt_path)['model_state_dict']
    # checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}

    # model.load_state_dict(checkpoint)
    # model.eval()

    with open(td_path, 'rb') as f:
        pickled_data = pickle.load(f)

    training_dynamics = pickled_data['training_dynamics']

    # post_training_metrics(model, trainloader, data_importance, device)
    training_dynamics_metrics(training_dynamics, trainset, data_importance)
    EL2N(training_dynamics, trainset, data_importance, max_epoch=10)

    ################### Novelty 1: GMM Adaptive Thresholds ###################
    aum_scores = data_importance['accumulated_margin'].numpy()
    threshold_low, threshold_high, gmm_model = fit_gmm_thresholds(aum_scores)
    data_importance['gmm_thresholds'] = (threshold_low, threshold_high)
    data_importance['gmm_means'] = gmm_model.means_.flatten()[np.argsort(gmm_model.means_.flatten())]
    data_importance['gmm_labels'] = torch.tensor(gmm_model.predict(aum_scores.reshape(-1, 1)))

    ################### Novelty 2: Hybrid Scoring ###################
    if args.hybrid:
        assert args.hybrid_embedding_path is not None, \
            '--hybrid-embedding-path is required when --hybrid is set.'
        hybrid_score, density_score = compute_hybrid_score(
            aum_scores, args.hybrid_embedding_path, num_classes, alpha=args.hybrid_alpha)
        data_importance['hybrid_score'] = hybrid_score
        data_importance['density_score'] = density_score

    print(f'Saving data score at {data_score_path}')
    with open(data_score_path, 'wb') as handle:
        pickle.dump(data_importance, handle)

elif args.importance_scheme == 'prototypicality':
    print("Calculating prototypicality score")
    embeddings = torch.load(args.embedding_path, map_location='cpu')
    print(f"Loading embeddings from {args.embedding_path}, len={len(embeddings)}")
    centroids, labels = find_centroid_kmeans(embeddings, num_classes)
    distances = calculate_distances(embeddings, labels, centroids)

    distances.sort(key=lambda x: x[1], reverse=True)
    # create a data_score_path if it does not exist
    print(f'Saving data score at {data_score_path}, length: {len(distances)}')
    import os 
    os.makedirs(os.path.dirname(data_score_path), exist_ok=True)
    with open(data_score_path, 'wb') as f:
        pickle.dump(distances, f)
