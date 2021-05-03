import os
import pickle
from glob import glob

from args import args


def ensure_dir(filename):
    dirname = os.path.dirname(filename)
    if dirname:
        try:
            os.makedirs(dirname)
        except OSError:
            pass


def init_out_dir():
    if not args.full_out_dir:
        return
    ensure_dir(args.full_out_dir)
    if args.ckpt_dir:
        ensure_dir(args.ckpt_dir)


def clear_log():
    if args.log_filename:
        with open(args.log_filename, 'w'):
            pass


def my_log(s):
    if args.log_filename:
        with open(args.log_filename, 'a', newline='\n') as f:
            f.write(s + '\n')
    if args.no_stdout:
        return
    print(s)


def print_args(print_fn=my_log):
    for k, v in args._get_kwargs():
        print_fn(f'{k} = {v}')
    print_fn('')


def parse_ckpt_name(filename):
    filename = os.path.basename(filename)
    filename = filename.replace('.pickle', '')
    step = int(filename)
    return step


def get_last_ckpt_step():
    if not args.ckpt_dir:
        return -1
    filename_list = glob(args.ckpt_dir + '*.pickle')
    if not filename_list:
        return -1
    step = max(parse_ckpt_name(x) for x in filename_list)
    return step


def load_ckpt(step):
    save_filename = f'{args.ckpt_dir}{step}.pickle'
    with open(save_filename, 'rb') as f:
        params = pickle.load(f)
    return params


def save_ckpt(params, step):
    save_filename = f'{args.ckpt_dir}{step}.pickle'
    with open(save_filename, 'wb') as f:
        pickle.dump(params, f)

    last_step = step - args.save_step
    if last_step > 0 and args.keep_step and last_step % args.keep_step != 0:
        last_save_filename = f'{args.ckpt_dir}{last_step}.pickle'
        os.remove(last_save_filename)
