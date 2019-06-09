import os
import random
import mxnet as mx
import numpy as np



#########
# Setup #
#########


__all__ = ['mkdir_p', 'set_log_dir', 'save_arg_dict', 'format_value', 'ImageNetSetting', 'set_random_seed']
def mkdir_p(path):
    import errno
    try:
        os.makedirs(path)
        print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            print('Directory {} already exists.'.format(path))
        else:
            raise

def set_log_dir(args, run):
    dir = args['base_dir'] + '/{:d}'.format(run)
    mkdir_p(dir)
    return dir

def save_arg_dict(args, filename='settings.txt'):
    path_to_file = os.path.join(args['log_dir'], filename)
    with open(path_to_file, 'w') as f:
        for key, value in args.items():
            if isinstance(value, (str, int, float, list)):
                f.write('{}\t{}\n'.format(key, format_value(value)))
    print('Saved settings to {}'.format(path_to_file))

def format_value(v):
    if isinstance(v, float):
        return '{:.4f}'.format(v)
    elif isinstance(v, bool):
        return '{}'.format(v)
    elif isinstance(v, int):
        return '{:d}'.format(v)
    else:
        return '{}'.format(v)

def ImageNetSetting(args):
    if args['dataset'] == 'imagenet':
        new_dict = {'optimizer': 'RMSProp',
                    'start_lr': 0.256}
    args.update(new_dict)

def set_random_seed(ctx, seed):
    if seed is None:
        seed = random.randint(1, 10000)
    random.seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed, ctx)
    return seed
