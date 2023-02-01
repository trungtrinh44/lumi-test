import json
import logging
import os
from bisect import bisect
from itertools import chain

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from sacred import Experiment
from sacred.observers import FileStorageObserver, RunObserver
from scipy.stats import entropy
from checkpointer import Checkpointer
import models
from datasets import get_corrupt_data_loader, get_data_loader
# from trainer import ParVITrainer
import optax
import jax.example_libraries.optimizers as optimizers
from optimizers import nesterov_weight_decay
class SetID(RunObserver):
    priority = 50  # very high priority

    def started_event(self, ex_info, command, host_info, start_time, config, meta_info, _id):
        return f"{config['model_name']}_{config['seed']}_{config['dataset']}_{config['name']}"


EXPERIMENT = 'experiments'
BASE_DIR = EXPERIMENT
ex = Experiment(EXPERIMENT)
ex.observers.append(SetID())
ex.observers.append(FileStorageObserver(BASE_DIR))


@ex.config
def my_config():
    ece_bins = 15
    seed = 1  # Random seed
    name = 'name'  # Unique name for the folder of the experiment
    model_name = 'StoResNet18'  # Choose with model to train
    batch_size = 128  # Batch size
    test_batch_size = 512
    n_members = 2  # Number of components in the posterior
    # Options of the deterministic weights for the SGD
    weight_decay = 5e-4
    init_lr = 0.1
    # Universal options for the SGD
    sgd_params = {
        'momentum': 0.9,
        'nesterov': True
    }
    num_epochs = 300  # Number of training epoch
    validation = True  # Whether of not to use a validation set
    validation_fraction = 0.1  # Size of the validation set
    save_freq = 301  # Frequency of saving checkpoint
    num_test_sample = 1  # Number of samples drawn from each component during testing
    logging_freq = 1  # Logging frequency
    device = 'cuda'
    lr_ratio = 0.01  # For annealing the learning rate of the deterministic weights
    # First value chooses which epoch to start decreasing the learning rate and the second value chooses which epoch to stop. See the schedule function for more information.
    milestones = (0.5, 0.9)
    augment_data = True
    # if not torch.cuda.is_available():
    #     device = 'cpu'
    dataset = 'cifar100'  # Dataset of the experiment
    if dataset == 'cifar100' or dataset == 'vgg_cifar100':
        num_classes = 100
        input_size = (32, 32, 3)
    elif dataset == 'cifar10' or dataset == 'vgg_cifar10' or dataset == 'fmnist':
        num_classes = 10
        input_size = (32, 32, 3)
    elif dataset == 'tinyimagenet':
        num_classes = 200
        input_size = (64, 64, 3)

    num_train_workers = 4
    num_test_workers = 2
    data_norm_stat = None
    num_start_epochs = 5
    length_scale = {
        'start': 100.0, 'end': 1.0, 'n_epochs': 80
    }
    repulsive_ensemble = True
    mask_path = ""
    mask_power = 1.0
    norm_after = True
    eps = 1e-12
    multiply_input = False
    sep_channels = False

@ex.capture
def get_length_scale(length_scale, num_epochs):
    def get_scale(epoch, start, end, n_epochs):
        value = start + epoch * (end-start)/n_epochs
        return value if epoch < n_epochs else end
    values = jnp.array([
        get_scale(i, **length_scale) for i in range(num_epochs)
    ], dtype=jnp.float32)
    return values

class LrScheduler():
    def __init__(self, init_value, num_epochs, milestones, lr_ratio, num_start_epochs):
        self.init_value = init_value
        self.num_epochs = num_epochs
        self.milestones = milestones
        self.lr_ratio = lr_ratio
        self.num_start_epochs = num_start_epochs
        self.lrs = jnp.array([self.__lr(i) for i in range(num_epochs)], dtype=jnp.float32)
    
    def __call__(self, i):
        return self.lrs[i]

    def __lr(self, epoch):
        if epoch < self.num_start_epochs:
            return self.init_value * (1.0 - self.lr_ratio)/self.num_start_epochs * epoch + self.lr_ratio
        t = epoch / self.num_epochs
        m1, m2 = self.milestones
        if t <= m1:
            factor = 1.0
        elif t <= m2:
            factor = 1.0 - (1.0 - self.lr_ratio) * (t - m1) / (m2 - m1)
        else:
            factor = self.lr_ratio
        return self.init_value * factor

@ex.capture
def get_model(model_name, num_classes, input_size, seed, n_members):
    model_fn = getattr(models, model_name)
    def _forward(x, is_training):
        model = model_fn(num_classes)
        return model(x, is_training)
    forward = hk.transform_with_state(_forward)
    parallel_init_fn = jax.vmap(forward.init, (0, None, None), 0)
    # parallel_apply_fn = jax.vmap(forward.apply, (0, 0, None, None, None), 0)

    rng = jax.random.PRNGKey(seed)
    keys = jax.random.split(rng, n_members)
    params, state = parallel_init_fn(keys, jnp.ones((1, *input_size)), True)

    return (params, state), forward.apply

@ex.capture
def get_optimizer(init_lr, milestones, num_epochs, lr_ratio, num_start_epochs, sgd_params, weight_decay):
    scheduler = LrScheduler(init_lr, num_epochs, milestones, lr_ratio, num_start_epochs)
    opt_init, opt_update, get_params = nesterov_weight_decay(scheduler, mass=sgd_params['momentum'], weight_decay=weight_decay)
    return opt_init, opt_update, get_params, scheduler

def l2_loss(params):
    return 0.5 * sum(jnp.sum(jnp.square(p)) for p in params)

@ex.capture
def get_dataloader(batch_size, test_batch_size, validation, validation_fraction, dataset, augment_data, num_train_workers, num_test_workers):
    return get_data_loader(dataset, train_bs=batch_size, test_bs=test_batch_size, validation=validation, validation_fraction=validation_fraction,
                           augment = augment_data, num_train_workers = num_train_workers, num_test_workers = num_test_workers)

@ex.capture
def get_corruptdataloader(intensity, test_batch_size, dataset, num_test_workers):
    return get_corrupt_data_loader(dataset, intensity, batch_size=test_batch_size, root_dir='data/', num_workers=num_test_workers)

@ex.capture
def get_logger(_run, _log):
    fh=logging.FileHandler(os.path.join(BASE_DIR, _run._id, 'train.log'))
    formatter=logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    _log.addHandler(fh)
    _log.setLevel(logging.INFO)
    return _log

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

@ex.capture
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

@ex.automain
def main(_run, validation, num_epochs, batch_size, dataset, repulsive_ensemble, mask_path, mask_power, sep_channels, norm_after, eps, multiply_input, input_size):
    logger=get_logger()
    if validation:
        train_loader, valid_loader, test_loader = get_dataloader()
        logger.info(
            f"Train size: {len(train_loader.dataset)}, validation size: {len(valid_loader.dataset)}, test size: {len(test_loader.dataset)}")
    else:
        train_loader, test_loader= get_dataloader()
        logger.info(
            f"Train size: {len(train_loader.dataset)}, test size: {len(test_loader.dataset)}")
    n_batch= len(train_loader)
    (params, state), apply_fn = get_model()
    opt_init, opt_update, get_params, scheduler = get_optimizer()
    opt_state = opt_init(params)
    length_scales = get_length_scale()
    freq_mask = jnp.array(np.load(mask_path)**mask_power)
    def loss_fn(params, state, bx, by, epoch):
        logits, new_state = jax.vmap(apply_fn, (0, 0, None, None, None), 0)(params, state, None, bx, True)
        cross_ent_loss = optax.softmax_cross_entropy_with_integer_labels(
            logits, by[None, :]
        ).mean(1)
        if repulsive_ensemble:
            def __per_example_apply(x, y, params, state):
                logits, _ = apply_fn(params, state, None, x[None, ...], False)
                logits = jax.nn.log_softmax(logits, axis=-1)
                one_hot = jax.nn.one_hot(y, logits.shape[1])
                return -(logits * one_hot).sum()
            values, jacobian = jax.vmap(jax.vmap(jax.value_and_grad(__per_example_apply, 0), (0, 0, None, None), 0), (None, None, 0, 0), 0)(bx, by, params, new_state) # [n_members, batch_size, *x.size]
            values = jax.lax.stop_gradient(values)
            ppd = jnp.exp(jax.scipy.special.logsumexp(-values, axis=0) - jnp.log(values.shape[0]))
            if sep_channels:
                axes = (2, 3)
            else:
                axes = (2, 3, 4)
            if multiply_input:
                jacobian = jacobian * bx
            #if norm_after:
            #    jacobian = jax.scipy.fft.dctn(jacobian, axes=axes, norm="ortho")
            #    normed_jacobian = jacobian / jnp.sqrt(jnp.sum(jnp.square(jacobian), axis=axes, keepdims=True) + eps)
            #else:
            #    normed_jacobian = jacobian / jnp.sqrt(jnp.sum(jnp.square(jacobian), axis=axes, keepdims=True) + eps)
            #    normed_jacobian = jax.scipy.fft.dctn(normed_jacobian, axes=axes, norm="ortho")
            normed_jacobian = jacobian * ppd[:, None, None, None]
            jacobian = normed_jacobian * freq_mask
            jacobian = jnp.reshape(jacobian, (*jacobian.shape[:2], -1))
            normed_jacobian = jnp.reshape(jax.lax.stop_gradient(normed_jacobian), (*normed_jacobian.shape[:2], -1))
            dist = (jnp.square(jnp.expand_dims(jacobian, axis=1) - jnp.expand_dims(jax.lax.stop_gradient(jacobian), axis=0))).sum(axis=-1)
            norm_dist = (jnp.square(jnp.expand_dims(normed_jacobian, axis=1) - jnp.expand_dims(normed_jacobian, axis=0))).sum(axis=-1)
            median = jnp.median(norm_dist, (0, 1))
            length_scale = length_scales[epoch]
            bandwidth = median * length_scale / jnp.log(jnp.array([dist.shape[0]], dtype=jnp.float32))
            kernel_matrix = jnp.exp(-dist/jnp.clip(bandwidth, 1e-24)/2).mean(2)
            repulsion_term = jnp.sum(jnp.log(jnp.sum(kernel_matrix, axis=1)), axis=0)
            log_prior = 0.5 * jnp.square(normed_jacobian).sum()
            loss = cross_ent_loss.sum(0) + repulsion_term / batch_size #+ log_prior * gradient_decay
            return loss, (cross_ent_loss.mean(0), repulsion_term, length_scale, median.mean(0), log_prior, new_state)
        else:
            return cross_ent_loss.sum(0), (cross_ent_loss.mean(0), 0.0, 0.0, 0.0, 0.0, new_state)
    
    @jax.jit
    def train_step(epoch, params, state, opt_state, bx, by):
        grads, (cross_ent_loss, repulsion_term, repulsion_weight, median, log_prior, new_state) = jax.grad(loss_fn, has_aux=True)(params, state, bx, by, epoch)
        new_opt_state = opt_update(epoch, grads, opt_state)
        # params = optax.apply_updates(params, updates)
        return new_state, new_opt_state, cross_ent_loss, repulsion_term, repulsion_weight, median, log_prior

    # trainer = ParVITrainer(model, optimizer, n_members, torch.nn.functional.nll_loss)
    for i in range(num_epochs):
        total_loss = 0
        n_count = 0
        for bx, by in train_loader:
            bx = bx.permute(0, 2, 3, 1).numpy()
            by = by.numpy()
            state, opt_state, nll_loss, repulsion_term, length_scale, median, log_prior = train_step(i, params, state, opt_state, bx, by)
            params = get_params(opt_state)
            total_loss += nll_loss
            n_count += 1
            logger.info(f"Epoch {i}: neg_log_like {nll_loss:.4f}, repulsion term {repulsion_term:.4f}, median {median:.4f}, log_prior {log_prior:.4f}, length_scale {length_scale:.4f}, lr {scheduler(i).item():.4f}")
        ex.log_scalar("nll.train", total_loss / n_count, i)
        # ex.log_scalar("median", total_median / n_count, i)
    checkpointer = Checkpointer(os.path.join(BASE_DIR, _run._id, f'checkpoint.pkl'))
    checkpointer.save({'params': params, 'state': state})
    logger.info('Save checkpoint')
    param_state = checkpointer.load()
    params = param_state['params']
    state = param_state['state']
    parallel_apply_fn = jax.vmap(apply_fn, (0, 0, None, None, None), 0)
    test_result = test_ensemble(parallel_apply_fn, params, state, test_loader)
    os.makedirs(os.path.join(BASE_DIR, _run._id, dataset), exist_ok=True)
    with open(os.path.join(BASE_DIR, _run._id, dataset, 'test_result.json'), 'w') as out:
        json.dump(test_result, out)
    if validation:
        valid_result = test_ensemble(parallel_apply_fn, params, state, valid_loader)
        with open(os.path.join(BASE_DIR, _run._id, dataset, 'valid_result.json'), 'w') as out:
            json.dump(valid_result, out)
    for i in range(5):
        dataloader = get_corruptdataloader(intensity=i)
        result = test_ensemble(parallel_apply_fn, params, state, dataloader)
        os.makedirs(os.path.join(BASE_DIR, _run._id, dataset, str(i)), exist_ok=True)
        with open(os.path.join(BASE_DIR, _run._id, dataset, str(i), 'result.json'), 'w') as out:
            json.dump(result, out)

