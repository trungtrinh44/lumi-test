import torch
import torchvision
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import os
import json
from checkpointer import Checkpointer
import models
import argparse
from datasets import CorruptDataset
import optax

def open_json(path):
    with open(path) as inp:
        obj = json.load(inp)
    return obj

corruption_types = ['saturate',
                    'shot_noise',
                    'gaussian_noise',
                    'zoom_blur',
                    'glass_blur',
                    'brightness',
                    'contrast',
                    'motion_blur',
                    'pixelate',
                    'snow',
                    'speckle_noise',
                    'spatter',
                    'gaussian_blur',
                    'frost',
                    'defocus_blur',
                    'elastic_transform',
                    'impulse_noise',
                    'jpeg_compression',
                    'fog']

NORM_STAT = {
    'cifar10': ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    'cifar100': ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    'svhn': ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
    'tinyimagenet': ((122.4786, 114.2755, 101.3963), (70.4924, 68.5679, 71.8127))
}
NUM_CLASSES = {
    'cifar10': 10, 'cifar100': 100, 'tinyimagenet': 200
}
def ece(softmax_logits, labels, n_bins=15):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidences, predictions = np.max(softmax_logits, -1), np.argmax(softmax_logits, -1)
    accuracies = predictions == labels

    ece = 0.0
    avg_confs_in_bins = []
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            delta = avg_confidence_in_bin - accuracy_in_bin
            avg_confs_in_bins.append(delta)
            ece += np.abs(delta) * prop_in_bin
        else:
            avg_confs_in_bins.append(None)

    return ece
def test_ensemble(model_fn, params, state, dataloader, ece_bins):
    tnll = 0
    acc = [0, 0, 0]
    nll_miss = 0
    y_prob = []
    y_true = []
    y_prob_all = []
    @jax.jit
    def eval_batch(bx, by):
        logits, _ = model_fn(params, state, None, bx, False)
        bnll = -optax.softmax_cross_entropy_with_integer_labels(
            logits, by[None, :]
        )
        bnll = jax.scipy.special.logsumexp(bnll, axis=0) - jnp.log(jnp.array(bnll.shape[0], dtype=jnp.float32))
        prob = jax.nn.softmax(logits, axis=2)
        vote = prob.mean(axis=0)
        top3 = jax.lax.top_k(vote, k=3)[1]
        return bnll, prob, vote, top3

    for bx, by in dataloader:
        bx = bx.permute(0, 2, 3, 1).numpy()
        by = by.numpy()
        bnll, prob, vote, top3 = eval_batch(bx, by)
        tnll -= bnll.sum()
        y_prob_all.append(prob)
        y_prob.append(vote)
        y_true.append(by)
        y_miss = top3[:, 0] != by
        if y_miss.sum() > 0:
            nll_miss -= bnll[y_miss].sum()
        for k in range(3):
            acc[k] += (top3[:, k] == by).sum()
    nll_miss /= len(dataloader.dataset) - acc[0]
    tnll /= len(dataloader.dataset)
    for k in range(3):
        acc[k] /= len(dataloader.dataset)
    acc = jnp.cumsum(jnp.array(acc))
    y_prob = jnp.concatenate(y_prob, axis=0)
    y_true = jnp.concatenate(y_true, axis=0)
    y_prob_all = jnp.concatenate(y_prob_all, axis=1)
    total_entropy = jax.scipy.special.entr(y_prob).sum(1)
    aleatoric = jax.scipy.special.entr(y_prob_all).sum(axis=2).mean(axis=0)
    epistemic = total_entropy - aleatoric
    ece_val = ece(y_prob, y_true, ece_bins)
    result = {
        'nll': float(tnll),
        'nll_miss': float(nll_miss),
        'ece': float(ece_val),
        'predictive_entropy': {
            'total': (float(total_entropy.mean()), float(total_entropy.std())),
            'aleatoric': (float(aleatoric.mean()), float(aleatoric.std())),
            'epistemic': (float(epistemic.mean()), float(epistemic.std()))
        },
        **{
            f"top-{k}": float(a) for k, a in enumerate(acc, 1)
        }
    }
    return result
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=str)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--ece_bins', type=int, default=15)
    parser.add_argument('--num_workers', type=int, default=2)
    args = parser.parse_args()
    
    config = open_json(os.path.join(args.root, 'config.json'))
    
    dataset = config['dataset']
    transform = torchvision.transforms.Compose([
        *([torchvision.transforms.ToTensor()] if dataset in ('cifar10', 'cifar100') else []),
         torchvision.transforms.Normalize(*(NORM_STAT[dataset]))
    ])
    
    model_fn = getattr(models, config['model_name'])
    def _forward(x, is_training):
        model = model_fn(NUM_CLASSES[dataset])
        return model(x, is_training)
    forward = hk.transform_with_state(_forward)
    parallel_apply_fn = jax.vmap(forward.apply, (0, 0, None, None, None), 0)
    
    checkpointer = Checkpointer(os.path.join(args.root, f'checkpoint.pkl'))
    param_state = checkpointer.load()
    params = param_state['params']
    state = param_state['state']
    
    if dataset == 'cifar10':
        datasetname = 'CIFAR-10-C'
    elif dataset == 'cifar100':
        datasetname = 'CIFAR-100-C'
        
    for corruption_type in corruption_types:
        os.makedirs(os.path.join(args.root, dataset, 'corruption', corruption_type), exist_ok=True)
        for i in range(5):
            dataloader = torch.utils.data.DataLoader(
                CorruptDataset(os.path.join('data', datasetname), [corruption_type], i, transform),
                batch_size=args.batch_size, pin_memory=True, shuffle=False, num_workers=args.num_workers
            )
            result = test_ensemble(parallel_apply_fn, params, state, dataloader, args.ece_bins)
            with open(os.path.join(args.root, dataset, 'corruption', corruption_type, f'result_{i}.json'), 'w') as out:
                json.dump(result, out)