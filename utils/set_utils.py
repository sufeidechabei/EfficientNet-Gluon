import os



#########
# Setup #
#########
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