#!/usr/bin/env python3
#
# Generate raw samples without rejection

from time import time

import numpy as np
from jax import jit
from jax import numpy as jnp
from jax import random as jrand

from args import args
from chunked_data import ChunkedDataWriter
from net import get_net
from train import energy_fun, get_log_q_fun, get_sample_fun
from utils import ensure_dir, get_last_ckpt_step, load_ckpt, my_log, print_args

args.log_filename = args.full_out_dir + 'sample_raw.log'


# TODO: if max_step is large, we need Kahan summation to reduce
# the floating point error
def welford_update(curr, step, mean, var_sum):
    diff = curr - mean
    mean_new = mean + diff / step
    var_sum_new = var_sum + diff * (curr - mean_new)
    return mean_new, var_sum_new


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

    # sample_fun = get_sample_fun(net_apply, None)
    sample_fun = get_sample_fun(net_apply_fast, cache_init)
    log_q_fun = get_log_q_fun(net_apply)

    def sample_energy_fun(rng):
        spins = sample_fun(args.batch_size, params, rng)
        log_q = log_q_fun(params, spins)
        energy = energy_fun(spins)
        return spins, log_q, energy

    @jit
    def update(spins_old, log_q_old, energy_old, step, energy_mean,
               energy_var_sum, rng):
        rng, rng_sample = jrand.split(rng)
        spins, log_q, energy = sample_energy_fun(rng_sample)
        mag = spins.mean(axis=(1, 2, 3))

        step += 1
        energy_per_spin = energy / args.L**2
        energy_mean, energy_var_sum = welford_update(energy_per_spin.mean(),
                                                     step, energy_mean,
                                                     energy_var_sum)

        return (spins, log_q, energy, mag, step, energy_mean, energy_var_sum,
                rng)

    rng, rng_init = jrand.split(jrand.PRNGKey(args.seed))
    spins, log_q, energy = sample_energy_fun(rng_init)

    step = 0
    energy_mean = 0
    energy_var_sum = 0

    data_filename = args.log_filename.replace('.log', '.hdf5')
    writer_proto = [
        # Uncomment to save all the sampled spins
        # ('spins', bool, (args.L, args.L)),
        ('log_q', np.float32, None),
        ('energy', np.int32, None),
        ('mag', np.float32, None),
    ]
    ensure_dir(data_filename)
    with ChunkedDataWriter(data_filename, writer_proto,
                           args.save_step * args.batch_size) as writer:
        my_log('Sampling...')
        while step < args.max_step:
            (spins, log_q, energy, mag, step, energy_mean, energy_var_sum,
             rng) = update(spins, log_q, energy, step, energy_mean,
                           energy_var_sum, rng)
            # Uncomment to save all the sampled spins
            # writer.write_batch(spins[:, :, :, 0] > 0, log_q, energy, mag)
            writer.write_batch(log_q, energy, mag)

            if args.print_step and step % args.print_step == 0:
                energy_std = jnp.sqrt(energy_var_sum / step)
                my_log(', '.join([
                    f'step = {step}',
                    f'E = {energy_mean:.8g}',
                    f'E_std = {energy_std:.8g}',
                    f'time = {time() - start_time:.3f}',
                ]))


if __name__ == '__main__':
    main()
