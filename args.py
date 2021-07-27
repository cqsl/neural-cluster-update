import argparse
import os
from datetime import datetime

parser = argparse.ArgumentParser()

group = parser.add_argument_group('physics parameters')

group.add_argument(
    '--lattice',
    type=str,
    default='ising',
    choices=['ising', 'fpm'],
    help='lattice type',
)
group.add_argument(
    '--L',
    type=int,
    default=16,
    help='edge length of the lattice',
)
group.add_argument(
    '--beta',
    type=float,
    default=0.44,
    help='inverse temperature',
)

group = parser.add_argument_group('network parameters')
group.add_argument(
    '--net_depth',
    type=int,
    default=3,
    help='number of conv layers in the network',
)
group.add_argument(
    '--net_width',
    type=int,
    default=16,
    help='number of channels in each conv layer',
)
group.add_argument(
    '--kernel_size',
    type=int,
    default=5,
    help='conv kernel size',
)
group.add_argument(
    '--dilation_step',
    type=int,
    default=2,
    help='increment of conv kernel dilation in each layer',
)

group = parser.add_argument_group('optimizer parameters')
group.add_argument(
    '--seed',
    type=int,
    default=0,
    help='random seed, 0 for randomized',
)
group.add_argument(
    '--batch_size',
    type=int,
    default=64,
    help='batch size',
)
group.add_argument(
    '--lr',
    type=float,
    default=1e-3,
    help='learning rate',
)
group.add_argument(
    '--max_step',
    type=int,
    default=2 * 10**4,
    help='number of training/sampling steps',
)
group.add_argument(
    '--beta_anneal_step',
    type=int,
    default=10**4,
    help='number of steps to gradually increase beta from 0 in training, 0 for disabled',
)
group.add_argument(
    '--eps',
    type=float,
    default=1e-7,
    help='a small number to avoid numerical instability',
)

group = parser.add_argument_group('sampling parameters')
group.add_argument(
    '--k_type',
    type=str,
    default='exp',
    choices=['exp', 'power', 'const'],
    help='type of the distribution of k',
)
group.add_argument(
    '--k_param',
    type=float,
    default=1,
    help='parameter of the distribution of k',
)

group = parser.add_argument_group('system parameters')
group.add_argument(
    '--no_stdout',
    action='store_true',
    help='do not print log to stdout, for better performance',
)
group.add_argument(
    '--print_step',
    type=int,
    default=10,
    help='print log every how many steps, 0 for disabled',
)
group.add_argument(
    '--save_step',
    type=int,
    default=10**2,
    help='save network weights every how many steps, 0 for disabled',
)
group.add_argument(
    '--keep_step',
    type=int,
    default=10**3,
    help='keep network weights every how many steps, 0 for keeping all',
)
group.add_argument(
    '--cuda',
    type=str,
    default='0',
    help='GPU ID, empty string for disabled, multi-GPU parallelism is not supported yet',
)
group.add_argument(
    '--run_name',
    type=str,
    default='',
    help='output subdirectory to keep repeated runs, empty string for disabled',
)
group.add_argument(
    '-o',
    '--out_dir',
    type=str,
    default='./out',
    help='output directory, empty string for disabled',
)

args = parser.parse_args()

if args.seed == 0:
    # The seed depends on time and PID
    args.seed = hash((datetime.now(), os.getpid()))

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'


def get_ham_net_name():
    ham_name = '{lattice}_L{L}_beta{beta:g}'
    ham_name = ham_name.format(**vars(args))

    net_name = 'nd{net_depth}_nw{net_width}_ks{kernel_size}'
    if args.dilation_step:
        net_name += '_ds{dilation_step}'
    if args.beta_anneal_step:
        net_name += '_ba{beta_anneal_step}'
    net_name = net_name.format(**vars(args))

    return ham_name, net_name


args.ham_name, args.net_name = get_ham_net_name()

if args.out_dir:
    args.full_out_dir = '{out_dir}/{ham_name}/{net_name}/'.format(**vars(args))
    if args.run_name:
        args.full_out_dir = '{full_out_dir}{run_name}/'.format(**vars(args))
    args.log_filename = args.full_out_dir + 'out.log'
    if args.save_step:
        args.ckpt_dir = args.full_out_dir + 'ckpt/'
    else:
        args.ckpt_dir = None
else:
    args.full_out_dir = None
    args.log_filename = None
    args.ckpt_dir = None
