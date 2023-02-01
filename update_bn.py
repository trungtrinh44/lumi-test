import json
import logging
import os
from bisect import bisect
from itertools import chain
import argparse
import haiku as hk
import jax
import jax.numpy as jnp
import functools
import numpy as np
from scipy.stats import entropy
from checkpointer import Checkpointer
import models
from datasets import get_corrupt_data_loader, get_data_loader
from tqdm import tqdm
import optax

def open_json(path):
    with open(path) as inp:
        obj = json.load(inp)
    return obj

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

    for bx, by in tqdm(dataloader):
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
    parser.add_argument('--ece_bins', type=int, default=15)
    args = parser.parse_args()
    
    config = open_json(os.path.join(args.root, 'config.json'))
    
    dataset = config['dataset']
    
    model_fn = getattr(models, config['model_name'])
    def _forward(x, is_training):
        model = model_fn(config['num_classes'], bn_config={'decay_rate': 0.0})
        return model(x, is_training)
    forward = hk.transform_with_state(_forward)
    parallel_apply_fn = jax.vmap(forward.apply, (0, 0, None, None, None), 0)
    parallel_apply_fn_update = jax.jit(lambda p, s, r, i: parallel_apply_fn(p, s, r, i, True))
    checkpointer = Checkpointer(os.path.join(args.root, 'checkpoint.pkl'))
    param_state = checkpointer.load()
    params = param_state['params']
    state = param_state['state']
    avg_state = {
        name: {n1: jnp.ones_like(v1) if n1 == 'hidden' else v1 for n1, v1 in value.items()} if 'var_ema' in name else ({n1: jnp.zeros_like(v1) if n1 == 'hidden' else v1 for n1, v1 in value.items()} if 'mean_ema' in name else value) for name, value in state.items()
    }
    train_loader, valid_loader, test_loader = get_data_loader(config['dataset'], train_bs=config['batch_size'], test_bs=config['test_batch_size'], validation=config['validation'], validation_fraction=config['validation_fraction'],
                           augment = config['augment_data'], num_train_workers = config['num_train_workers'], num_test_workers = config['num_test_workers'])
    n = 0
    for bx, _ in tqdm(train_loader):
        bx = bx.permute(0, 2, 3, 1).numpy()
        b = bx.shape[0]
        _, new_state = parallel_apply_fn_update(params, state, None, bx)
        rate = n / (n + b)
        avg_state = jax.tree_util.tree_map(lambda x, y: x * rate + y * (1.0 - rate), avg_state, new_state)
        n += b
    state = avg_state
    checkpointer = Checkpointer(os.path.join(args.root, 'bn_checkpoint.pkl'))
    checkpointer.save({'params': params, 'state': state})
    os.makedirs(os.path.join(args.root, dataset, 'bn_updated'), exist_ok=True)
    test_result = test_ensemble(parallel_apply_fn, params, state, test_loader, args.ece_bins)
    with open(os.path.join(args.root, dataset, 'bn_updated', 'test_result.json'), 'w') as out:
        json.dump(test_result, out)
    if config['validation']:
        valid_result = test_ensemble(parallel_apply_fn, params, state, valid_loader, args.ece_bins)
        with open(os.path.join(args.root, dataset, 'bn_updated', 'valid_result.json'), 'w') as out:
            json.dump(valid_result, out)
    for i in range(5):
        dataloader = get_corrupt_data_loader(config['dataset'], i, batch_size=config['test_batch_size'], root_dir='data/', num_workers=config['num_test_workers'])
        result = test_ensemble(parallel_apply_fn, params, state, dataloader, args.ece_bins)
        os.makedirs(os.path.join(args.root, dataset, 'bn_updated', str(i)), exist_ok=True)
        with open(os.path.join(args.root, dataset, 'bn_updated', str(i), 'result.json'), 'w') as out:
            json.dump(result, out)

