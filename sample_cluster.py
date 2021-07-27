#!/usr/bin/env python3
#
# Neural cluster update with symmetries (NCUS)

from functools import partial
from time import time

import numpy as np
from jax import jit, lax
from jax import numpy as jnp
from jax import random as jrand

from args import args
from chunked_data import ChunkedDataWriter
from net import get_net
from train import energy_fun, get_log_q_fun, get_sample_fun
from utils import ensure_dir, get_last_ckpt_step, load_ckpt, my_log, print_args

args.log_filename = '{full_out_dir}sample_cluster_{k_type}_{k_param:g}.log'
args.log_filename = args.log_filename.format(**vars(args))


@jit
def get_k(rng):
    if args.k_type == 'const':
        return int(args.k_param), rng

    idx = jnp.arange(1, args.L**2 + 1)

    if args.k_type == 'exp':
        logits = -idx * jnp.log(args.k_param)
    elif args.k_type == 'power':
        logits = -args.k_param * jnp.log(idx)
    else:
        raise ValueError('Unknown k_type: {}'.format(args.k_type))

    rng, rng_now = jrand.split(rng)
    k = jrand.categorical(rng, logits) + 1
    return k, rng


def get_sample_k_fun(net_apply):
    indices = [(i, j) for i in range(args.L) for j in range(args.L)]
    indices = jnp.array(indices)

    @partial(jit, static_argnums=0)
    def sample_k_fun(k, params, spins_init, rng_init):
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

        (spins, _), _ = lax.scan(scan_fun, (spins_init, rng_init),
                                 indices[-k:])

        return spins

    return sample_k_fun


def main():
    start_time = time()
    last_step = get_last_ckpt_step()
    assert last_step >= 0
    my_log(f'Checkpoint found: {last_step}\n')
    print_args()

    net_init, net_apply = get_net()
    params = load_ckpt(last_step)
    sample_raw_fun = get_sample_fun(net_apply)
    sample_k_fun = get_sample_k_fun(net_apply)
    log_q_fun = get_log_q_fun(net_apply)

    @partial(jit, static_argnums=0)
    def update(k, spins_old, log_q_old, energy_old, step, accept_count,
               energy_mean, energy_sqr_mean, rng):
        rng, rng_sample, rng_accept, rng_trans, rng_refl = jrand.split(rng, 5)
        spins = sample_k_fun(k, params, spins_old, rng_sample)
        log_q = log_q_fun(params, spins)
        energy = energy_fun(spins)

        log_uniform = jnp.log(jrand.uniform(rng_accept, (args.batch_size, )))
        accept = log_uniform < (log_q_old - log_q + args.beta *
                                (energy_old - energy))

        spins = jnp.where(jnp.expand_dims(accept, axis=(1, 2, 3)), spins,
                          spins_old)
        energy = jnp.where(accept, energy, energy_old)

        # Apply a random translation on the batch
        i, j = jrand.randint(rng_trans, (2, ), 0, args.L)
        spins = jnp.roll(spins, (i, j), axis=(1, 2))

        # Apply random reflections on the batch
        refl_x, refl_y, refl_d, refl_z = jrand.uniform(rng_refl, (4, ))
        spins = jnp.where(refl_x > 0.5, jnp.flip(spins, axis=1), spins)
        spins = jnp.where(refl_y > 0.5, jnp.flip(spins, axis=2), spins)
        spins = jnp.where(refl_d > 0.5, spins.transpose((0, 2, 1, 3)), spins)
        spins = jnp.where(refl_z > 0.5, -spins, spins)

        log_q = log_q_fun(params, spins)
        mag = spins.mean(axis=(1, 2, 3))

        step += 1
        accept_count += accept.sum()

        # TODO: if max_step is large, we need Kahan summation to reduce
        # the floating point error
        energy_per_spin = energy / args.L**2
        energy_mean += (energy_per_spin.mean() - energy_mean) / step
        energy_sqr_mean += (
            (energy_per_spin**2).mean() - energy_sqr_mean) / step

        return (spins, log_q, energy, mag, accept, step, accept_count,
                energy_mean, energy_sqr_mean, rng)

    rng, rng_init = jrand.split(jrand.PRNGKey(args.seed))
    # Sample initial configurations from the network
    spins = sample_raw_fun(args.batch_size, params, rng_init)
    log_q = log_q_fun(params, spins)
    energy = energy_fun(spins)

    step = 0
    accept_count = 0
    energy_mean = 0
    energy_sqr_mean = 0

    data_filename = '{full_out_dir}sample_cluster_{k_type}_{k_param:g}.hdf5'
    data_filename = data_filename.format(**vars(args))
    writer_proto = [
        # Uncomment to record all the sampled spins
        # ('spins', bool, (args.batch_size, args.L, args.L)),
        ('log_q', np.float32, (args.batch_size, )),
        ('energy', np.int32, (args.batch_size, )),
        ('mag', np.float32, (args.batch_size, )),
        ('accept', bool, (args.batch_size, )),
        ('k', np.int32, None),
    ]
    ensure_dir(data_filename)
    with ChunkedDataWriter(data_filename, writer_proto,
                           args.save_step) as writer:
        my_log('Sampling...')
        while step < args.max_step:
            k, rng = get_k(rng)
            # TODO: can we put get_k() into update()?
            k = k.item()
            (spins, log_q, energy, mag, accept, step, accept_count,
             energy_mean, energy_sqr_mean,
             rng) = update(k, spins, log_q, energy, step, accept_count,
                           energy_mean, energy_sqr_mean, rng)
            # writer.write(spins[:, :, :, 0] > 0, log_q, energy, mag, accept, k)
            writer.write(log_q, energy, mag, accept, k)

            if args.print_step and step % args.print_step == 0:
                accept_rate = accept_count / (step * args.batch_size)
                energy_std = jnp.sqrt(energy_sqr_mean - energy_mean**2)
                my_log(', '.join([
                    f'step = {step}',
                    f'P = {accept_rate:.8g}',
                    f'E = {energy_mean:.8g}',
                    f'E_std = {energy_std:.8g}',
                    f'time = {time() - start_time:.3f}',
                ]))


if __name__ == '__main__':
    main()
