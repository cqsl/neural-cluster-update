#!/usr/bin/env python3

from functools import partial
from time import time

from jax import grad, jit, lax
from jax import numpy as jnp
from jax import random as jrand
from jax.experimental import optimizers

from args import args
from expect import expect
from net import get_net
from utils import (clear_log, get_last_ckpt_step, init_out_dir, load_ckpt,
                   my_log, print_args, save_ckpt)

args.log_filename = args.full_out_dir + 'train.log'


# spins: (batch_size, L, L, 1), values in {-1, 1}
@jit
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
def get_sample_fun(net_apply):
    indices = [(i, j) for i in range(args.L) for j in range(args.L)]
    indices = jnp.array(indices)

    @partial(jit, static_argnums=0)
    def sample_fun(batch_size, params, rng_init):
        def scan_fun(carry, index):
            spins, rng = carry
            i, j = index
            rng, rng_now = jrand.split(rng)
            # s_hat are parameters of Bernoulli distributions
            s_hat = net_apply(params, spins)
            spins_now = jrand.bernoulli(rng_now, s_hat[:, i, j, :]).astype(
                jnp.float32) * 2 - 1
            spins = spins.at[:, i, j, :].set(spins_now)
            return (spins, rng), None

        spins_init = jnp.zeros((batch_size, args.L, args.L, 1))
        (spins, _), _ = lax.scan(scan_fun, (spins_init, rng_init), indices)

        return spins

    return sample_fun


def get_log_q_fun(net_apply):
    @jit
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

    net_init, net_apply = get_net()
    sample_fun = get_sample_fun(net_apply)
    log_q_fun = get_log_q_fun(net_apply)

    need_beta_anneal = args.beta_anneal_step > 0

    opt_init, opt_update, get_params = optimizers.adam(args.lr)

    @jit
    def update(step, opt_state, rng):
        params = get_params(opt_state)
        rng, rng_now = jrand.split(rng)
        spins = sample_fun(args.batch_size, params, rng_now)
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

        loss_fun = partial(expect, log_q_fun, neg_log_Z_fun)
        grads = grad(loss_fun)(params, spins, spins)
        opt_state = opt_update(step, grads, opt_state)

        return spins, log_q, energy, opt_state, rng

    rng, rng_net = jrand.split(jrand.PRNGKey(args.seed))
    in_shape = (args.batch_size, args.L, args.L, 1)
    out_shape, init_params = net_init(rng_net, in_shape)

    if last_step >= 0:
        init_params = load_ckpt(last_step)

    opt_state = opt_init(init_params)

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
