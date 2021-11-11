#!/usr/bin/env python3

from functools import partial
from time import time

from jax import grad, jit, lax
from jax import numpy as jnp
from jax import random as jrand
from jax.example_libraries import optimizers

from args import args
from expect import expect
from net import get_net, prev_index_2d
from utils import (clear_log, get_last_ckpt_step, init_out_dir, load_ckpt,
                   my_log, print_args, save_ckpt)

args.log_filename = args.full_out_dir + 'train.log'


# spins: (batch, L, L, 1), values in {-1, 1}
def energy_fun(spins):
    if args.lattice == 'ising':
        # 2D classical Ising model, square lattice, periodic boundary
        env = jnp.roll(spins, 1, axis=1) + jnp.roll(spins, 1, axis=2)
        energy = (spins * env).sum(axis=(1, 2, 3))
        return energy
    elif args.lattice == 'fpm':
        # Frustrated plaquette model
        sx = jnp.roll(spins, 1, axis=1)
        sy = jnp.roll(spins, 1, axis=2)
        sxy = jnp.roll(spins, 1, axis=(1, 2))
        sx2 = jnp.roll(spins, 2, axis=1)
        sy2 = jnp.roll(spins, 2, axis=2)
        env = -sx - sy - sx2 - sy2 + 2 * sx * sy * sxy
        energy = (spins * env).sum(axis=(1, 2, 3))
        return energy
    else:
        raise ValueError(f'Unknown lattice: {args.lattice}')


# We never take the gradient through the sampling procedure
def get_sample_fun(net_apply, cache_init):
    use_fast = (cache_init is not None)

    indices = [(i, j) for i in range(args.L) for j in range(args.L)]
    indices = jnp.asarray(indices)

    @partial(jit, static_argnums=0)
    def sample_fun(batch_size, params, rng_init):
        def scan_fun(carry, _args):
            spins, cache = carry
            (i, j), rng = _args

            if use_fast:
                i_in, j_in = prev_index_2d(i, j, args.L)
                spins_slice = spins[:, i_in, j_in, :]
                spins_slice = jnp.expand_dims(spins_slice, axis=(1, 2))
                s_hat, cache = net_apply(params, spins_slice, cache, (i, j))
                s_hat = s_hat.squeeze(axis=(1, 2))
            else:
                s_hat = net_apply(params, spins)
                s_hat = s_hat[:, i, j, :]

            # s_hat are parameters of Bernoulli distributions
            spins_new = jrand.bernoulli(rng, s_hat).astype(jnp.float32) * 2 - 1
            spins = spins.at[:, i, j, :].set(spins_new)

            return (spins, cache), None

        rngs = jrand.split(rng_init, args.L**2)
        spins_init = jnp.zeros((batch_size, args.L, args.L, 1))
        (spins, _), _ = lax.scan(scan_fun, (spins_init, cache_init),
                                 (indices, rngs))
        return spins

    return sample_fun


def get_log_q_fun(net_apply):
    def log_q_fun(params, spins):
        mask = (spins + 1) / 2
        s_hat = net_apply(params, spins)
        log_q = jnp.log(mask * s_hat + (1 - mask) * (1 - s_hat) + args.eps)
        log_q = log_q.sum(axis=(1, 2, 3))
        return log_q

    return log_q_fun


def main():
    start_time = time()
    init_out_dir()
    last_step = get_last_ckpt_step()
    if last_step >= 0:
        my_log(f'\nCheckpoint found: {last_step}\n')
    else:
        clear_log()
    print_args()

    net_init, net_apply, net_init_cache, net_apply_fast = get_net()

    rng, rng_net = jrand.split(jrand.PRNGKey(args.seed))
    in_shape = (args.batch_size, args.L, args.L, 1)
    out_shape, params_init = net_init(rng_net, in_shape)

    _, cache_init = net_init_cache(params_init, jnp.zeros(in_shape), (-1, -1))

    # sample_fun = get_sample_fun(net_apply, None)
    sample_fun = get_sample_fun(net_apply_fast, cache_init)
    log_q_fun = get_log_q_fun(net_apply)

    need_beta_anneal = args.beta_anneal_step > 0

    opt_init, opt_update, get_params = optimizers.adam(args.lr)

    @jit
    def update(step, opt_state, rng):
        params = get_params(opt_state)
        rng, rng_sample = jrand.split(rng)
        spins = sample_fun(args.batch_size, params, rng_sample)
        log_q = log_q_fun(params, spins) / args.L**2
        energy = energy_fun(spins) / args.L**2

        def neg_log_Z_fun(params, spins):
            log_q = log_q_fun(params, spins) / args.L**2
            energy = energy_fun(spins) / args.L**2
            beta = args.beta
            if need_beta_anneal:
                beta *= jnp.minimum(step / args.beta_anneal_step, 1)
            neg_log_Z = log_q + beta * energy
            return neg_log_Z

        loss_fun = partial(expect,
                           log_q_fun,
                           neg_log_Z_fun,
                           mean_grad_expected_is_zero=True)
        grads = grad(loss_fun)(params, spins, spins)
        opt_state = opt_update(step, grads, opt_state)

        return spins, log_q, energy, opt_state, rng

    if last_step >= 0:
        params_init = load_ckpt(last_step)

    opt_state = opt_init(params_init)

    my_log('Training...')
    for step in range(last_step + 1, args.max_step + 1):
        spins, log_q, energy, opt_state, rng = update(step, opt_state, rng)

        if args.print_step and step % args.print_step == 0:
            # Use the final beta, not the annealed beta
            free_energy = log_q / args.beta + energy
            my_log(', '.join([
                f'step = {step}',
                f'F = {free_energy.mean():.8g}',
                f'F_std = {free_energy.std():.8g}',
                f'S = {-log_q.mean():.8g}',
                f'E = {energy.mean():.8g}',
                f'time = {time() - start_time:.3f}',
            ]))

        if args.save_step and step % args.save_step == 0:
            params = get_params(opt_state)
            save_ckpt(params, step)


if __name__ == '__main__':
    main()
