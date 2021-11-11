#!/usr/bin/env python3
#
# Neural cluster update (NCU)

from time import time

import numpy as np
from jax import jit
from jax import numpy as jnp
from jax import random as jrand

from args import args
from chunked_data import ChunkedDataWriter
from net import get_net
from sample_ncus import get_k, get_sample_k_fun
from sample_raw import welford_update
from train import energy_fun, get_log_q_fun, get_sample_fun
from utils import ensure_dir, get_last_ckpt_step, load_ckpt, my_log, print_args

args.log_filename = '{full_out_dir}sample_ncu_{k_type}_{k_param:g}.log'
args.log_filename = args.log_filename.format(**vars(args))


def main():
    start_time = time()
    last_step = get_last_ckpt_step()
    assert last_step >= 0
    my_log(f'Checkpoint found: {last_step}\n')
    print_args()

    net_init, net_apply, net_init_cache, net_apply_fast = get_net()

    params = load_ckpt(last_step)
    in_shape = (args.batch_size, args.L, args.L, 1)
    _, cache_init = net_init_cache(params, jnp.zeros(in_shape), (-1, -1))

    # sample_raw_fun = get_sample_fun(net_apply, None)
    sample_raw_fun = get_sample_fun(net_apply_fast, cache_init)
    # sample_k_fun = get_sample_k_fun(net_apply, None)
    sample_k_fun = get_sample_k_fun(net_apply_fast, net_init_cache)
    log_q_fun = get_log_q_fun(net_apply)

    @jit
    def update(spins_old, log_q_old, energy_old, step, accept_count,
               energy_mean, energy_var_sum, rng):
        rng, rng_k, rng_sample, rng_accept = jrand.split(rng, 4)
        k = get_k(rng_k)
        spins = sample_k_fun(k, params, spins_old, rng_sample)
        log_q = log_q_fun(params, spins)
        energy = energy_fun(spins)

        log_uniform = jnp.log(jrand.uniform(rng_accept, (args.batch_size, )))
        accept = log_uniform < (log_q_old - log_q + args.beta *
                                (energy_old - energy))

        spins = jnp.where(jnp.expand_dims(accept, axis=(1, 2, 3)), spins,
                          spins_old)
        log_q = jnp.where(accept, log_q, log_q_old)
        energy = jnp.where(accept, energy, energy_old)
        mag = spins.mean(axis=(1, 2, 3))

        step += 1
        accept_count += accept.sum()
        energy_per_spin = energy / args.L**2
        energy_mean, energy_var_sum = welford_update(energy_per_spin.mean(),
                                                     step, energy_mean,
                                                     energy_var_sum)

        return (spins, log_q, energy, mag, accept, k, step, accept_count,
                energy_mean, energy_var_sum, rng)

    rng, rng_init = jrand.split(jrand.PRNGKey(args.seed))
    # Sample initial configurations from the network
    spins = sample_raw_fun(args.batch_size, params, rng_init)
    log_q = log_q_fun(params, spins)
    energy = energy_fun(spins)

    step = 0
    accept_count = 0
    energy_mean = 0
    energy_var_sum = 0

    data_filename = args.log_filename.replace('.log', '.hdf5')
    writer_proto = [
        # Uncomment to save all the sampled spins
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
            (spins, log_q, energy, mag, accept, k, step, accept_count,
             energy_mean, energy_var_sum,
             rng) = update(spins, log_q, energy, step, accept_count,
                           energy_mean, energy_var_sum, rng)
            # Uncomment to save all the sampled spins
            # writer.write(spins[:, :, :, 0] > 0, log_q, energy, mag, accept, k)
            writer.write(log_q, energy, mag, accept, k)

            if args.print_step and step % args.print_step == 0:
                accept_rate = accept_count / (step * args.batch_size)
                energy_std = jnp.sqrt(energy_var_sum / step)
                my_log(', '.join([
                    f'step = {step}',
                    f'P = {accept_rate:.8g}',
                    f'E = {energy_mean:.8g}',
                    f'E_std = {energy_std:.8g}',
                    f'time = {time() - start_time:.3f}',
                ]))


if __name__ == '__main__':
    main()
