#!/usr/bin/env python3
#
# Generate raw samples without rejection

from time import time

import numpy as np
from jax import jit
from jax import random as jrand

from args import args
from chunked_data import ChunkedDataWriter
from net import get_net
from train import energy_fun, get_log_q_fun, get_sample_fun
from utils import ensure_dir, get_last_ckpt_step, load_ckpt, my_log, print_args

args.log_filename = args.full_out_dir + 'sample_raw.log'


def main():
    start_time = time()
    last_step = get_last_ckpt_step()
    assert last_step >= 0
    my_log(f'Checkpoint found: {last_step}\n')
    print_args()

    net_init, net_apply = get_net()
    params = load_ckpt(last_step)
    sample_fun = get_sample_fun(net_apply)
    log_q_fun = get_log_q_fun(net_apply)

    @jit
    def sample_energy_fun(rng):
        spins = sample_fun(args.batch_size, params, rng)
        log_q = log_q_fun(params, spins)
        energy = energy_fun(spins)
        return spins, log_q, energy

    @jit
    def update(spins_old, log_q_old, energy_old, step, rng):
        rng, rng_sample = jrand.split(rng)
        spins, log_q, energy = sample_energy_fun(rng_sample)
        step += 1
        return spins, log_q, energy, step, rng

    rng, rng_init = jrand.split(jrand.PRNGKey(args.seed))
    spins, log_q, energy = sample_energy_fun(rng_init)
    step = 0

    data_filename = args.full_out_dir + 'sample_raw.hdf5'
    writer_proto = [
        # Uncomment to record all the sampled spins
        # ('spins', bool, (args.L, args.L)),
        ('log_q', np.float32, None),
        ('energy', np.int32, None),
    ]
    ensure_dir(data_filename)
    with ChunkedDataWriter(data_filename, writer_proto,
                           args.save_step * args.batch_size) as writer:
        my_log('Sampling...')
        while step < args.max_step:
            spins, log_q, energy, step, rng = update(spins, log_q, energy,
                                                     step, rng)
            # writer.write_batch(spins[:, :, :, 0] > 0, log_q, energy)
            writer.write_batch(log_q, energy)

            if args.print_step and step % args.print_step == 0:
                my_log(', '.join([
                    f'step = {step}',
                    f'time = {time() - start_time:.3f}',
                ]))


if __name__ == '__main__':
    main()
