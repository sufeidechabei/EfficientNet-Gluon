import argparse, time, logging, os, math

import numpy as np
import mxnet as mx
from mxnet import gluon, nd
from mxnet import autograd as ag
from mxnet.gluon.data.vision import transforms

from gluoncv.data import imagenet
from model import get_efficientnet
from gluoncv.utils import makedirs, LRSequential, LRScheduler
from utils import *


# CLI
def parse_args():
    parser = argparse.ArgumentParser(description='Train a model for image classification.')
    parser.add_argument('--data-dir', type=str, default='~/.mxnet/datasets/imagenet',
                        help='training and validation pictures to use.')
    parser.add_argument('--rec-train', type=str, default='~/.mxnet/datasets/imagenet/rec/train.rec',
                        help='the training data')
    parser.add_argument('--rec-train-idx', type=str, default='~/.mxnet/datasets/imagenet/rec/train.idx',
                        help='the index of training data')
    parser.add_argument('--rec-val', type=str, default='~/.mxnet/datasets/imagenet/rec/val.rec',
                        help='the validation data')
    parser.add_argument('--rec-val-idx', type=str, default='~/.mxnet/datasets/imagenet/rec/val.idx',
                        help='the index of validation data')
    parser.add_argument('--use-rec', action='store_true',
                        help='use image record iter for data input. default is false.')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='training batch size per device (CPU/GPU).')
    parser.add_argument('--dtype', type=str, default='float32',
                        help='data type for training. default is float32')
    parser.add_argument('--num-gpus', type=int, default=8,
                        help='number of gpus to use.')
    parser.add_argument('-j', '--num-data-workers', dest='num_workers', default=4, type=int,
                        help='number of preprocessing workers')
    parser.add_argument('--num-epochs', type=int, default=3,
                        help='number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate. default is 0.1.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum value for optimizer, default is 0.9.')
    parser.add_argument('--wd', type=float, default=0.0001,
                        help='weight decay rate. default is 0.0001.')
    parser.add_argument('--lr-mode', type=str, default='step',
                        help='learning rate scheduler mode. options are step, poly and cosine.')
    parser.add_argument('--lr-decay', type=float, default=0.1,
                        help='decay rate of learning rate. default is 0.1.')
    parser.add_argument('--lr-decay-period', type=int, default=0,
                        help='interval for periodic learning rate decays. default is 0 to disable.')
    parser.add_argument('--lr-decay-epoch', type=str, default='40,60',
                        help='epochs at which learning rate decays. default is 40,60.')
    parser.add_argument('--warmup-lr', type=float, default=0.0,
                        help='starting warmup learning rate. default is 0.0.')
    parser.add_argument('--warmup-epochs', type=int, default=0,
                        help='number of warmup epochs.')
    parser.add_argument('--last-gamma', action='store_true',
                        help='whether to init gamma of the last BN layer in each bottleneck to 0.')
    parser.add_argument('--mode', type=str,
                        help='mode in which to train the model. options are symbolic, imperative, hybrid')
    parser.add_argument('--model', type=str, choices=['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7'],
                        help='type of model to use. see vision_model for options.')
    parser.add_argument('--input-size', type=int, default=224,
                        help='size of the input image size. default is 224')
    parser.add_argument('--crop-ratio', type=float, default=0.875,
                        help='Crop ratio during validation. default is 0.875')
    parser.add_argument('--use-pretrained', action='store_true',
                        help='enable using pretrained model from gluon.')
    parser.add_argument('--use_se', action='store_true',
                        help='use SE layers or not in resnext. default is false.')
    parser.add_argument('--mixup', action='store_true',
                        help='whether train the model with mix-up. default is false.')
    parser.add_argument('--mixup-alpha', type=float, default=0.2,
                        help='beta distribution parameter for mixup sampling, default is 0.2.')
    parser.add_argument('--mixup-off-epoch', type=int, default=0,
                        help='how many last epochs to train without mixup, default is 0.')
    parser.add_argument('--label-smoothing', action='store_true',
                        help='use label smoothing or not in training. default is false.')
    parser.add_argument('--no-wd', action='store_true',
                        help='whether to remove weight decay on bias, and beta/gamma for batchnorm layers.')
    parser.add_argument('--temperature', type=float, default=20,
                        help='temperature parameter for distillation teacher model')
    parser.add_argument('--hard-weight', type=float, default=0.5,
                        help='weight for the loss of one-hot label for distillation training')
    parser.add_argument('--batch-norm', action='store_true',
                        help='enable batch normalization or not in vgg. default is false.')
    parser.add_argument('--save-frequency', type=int, default=10,
                        help='frequency of model saving.')
    parser.add_argument('--resume-epoch', type=int, default=0,
                        help='epoch to resume training from.')
    parser.add_argument('--resume-params', type=str, default='',
                        help='path of parameters to load from.')
    parser.add_argument('--resume-states', type=str, default='',
                        help='path of trainer state to load from.')
    parser.add_argument('--log-interval', type=int, default=50,
                        help='Number of batches to wait before logging.')
    parser.add_argument('--logging-file', type=str, default='train_imagenet.log',
                        help='name of training log file')
    parser.add_argument('--use-gn', action='store_true',
                        help='whether to use group norm.')
    parser.add_argument('--n_run', default=0, type=int, help="time for running")
    parser.add_argument('--save_model', type=bool, default=True, help='whether save model')
    args = parser.parse_args()
    return args


def main(args):
    filehandler = logging.FileHandler(args['log_dir'] + '/train.log')
    streamhandler = logging.StreamHandler()

    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)

    batch_size = args['batch_size']
    classes = 1000
    num_training_samples = 1281167

    num_gpus = args['num_gpus']
    batch_size *= max(1, num_gpus)
    context = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
    num_workers = args['num_workers']
    model_name = 'efficientnet-' + args['model']
    lr_decay = args['lr_decay']
    lr_decay_period = args['lr_decay_period']
    if args['lr_decay_period'] > 0:
        lr_decay_epoch = list(range(lr_decay_period, args['num_epochs'], lr_decay_period))
    else:
        lr_decay_epoch = [int(i) for i in args['lr_decay_epoch'].split(',')]
    lr_decay_epoch = [e - args['warmup_epochs'] for e in lr_decay_epoch]
    num_batches = num_training_samples // batch_size

    lr_scheduler = LRSequential([
        LRScheduler('linear', base_lr=0, target_lr=args['lr'],
                    nepochs=args['warmup_epochs'], iters_per_epoch=num_batches),
        LRScheduler(args['lr_mode'], base_lr=args['lr'], target_lr=0,
                    nepochs=args['num_epochs'] - args['warmup_epochs'],
                    iters_per_epoch=num_batches,
                    step_epoch=lr_decay_epoch,
                    step_factor=lr_decay, power=2)
    ])

    optimizer = 'nag'
    optimizer_params = {'wd': args['wd'], 'momentum': args['momentum'], 'lr_scheduler': lr_scheduler}
    if args['dtype'] != 'float32':
        optimizer_params['multi_precision'] = True

    net, input_size = get_efficientnet(model_name)
    net.cast(args['dtype'])
    if args['resume_params'] is not '':
        net.load_parameters(args['resume_params'], ctx=context)

    # Two functions for reading data from record file or raw images
    def get_data_rec(rec_train, rec_train_idx, rec_val, rec_val_idx, batch_size, num_workers):
        rec_train = os.path.expanduser(rec_train)
        rec_train_idx = os.path.expanduser(rec_train_idx)
        rec_val = os.path.expanduser(rec_val)
        rec_val_idx = os.path.expanduser(rec_val_idx)
        jitter_param = 0.4
        lighting_param = 0.1
        crop_ratio = args['crop_ratio'] if args['crop_ratio'] > 0 else 0.875
        resize = int(math.ceil(input_size / crop_ratio))
        mean_rgb = [123.68, 116.779, 103.939]
        std_rgb = [58.393, 57.12, 57.375]

        def batch_fn(batch, ctx):
            data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
            return data, label

        train_data = mx.io.ImageRecordIter(
            path_imgrec=rec_train,
            path_imgidx=rec_train_idx,
            preprocess_threads=num_workers,
            shuffle=True,
            batch_size=batch_size,

            data_shape=(3, input_size, input_size),
            mean_r=mean_rgb[0],
            mean_g=mean_rgb[1],
            mean_b=mean_rgb[2],
            std_r=std_rgb[0],
            std_g=std_rgb[1],
            std_b=std_rgb[2],
            rand_mirror=True,
            random_resized_crop=True,
            max_aspect_ratio=4. / 3.,
            min_aspect_ratio=3. / 4.,
            max_random_area=1,
            min_random_area=0.08,
            brightness=jitter_param,
            saturation=jitter_param,
            contrast=jitter_param,
            pca_noise=lighting_param,
        )
        val_data = mx.io.ImageRecordIter(
            path_imgrec=rec_val,
            path_imgidx=rec_val_idx,
            preprocess_threads=num_workers,
            shuffle=False,
            batch_size=batch_size,

            resize=resize,
            data_shape=(3, input_size, input_size),
            mean_r=mean_rgb[0],
            mean_g=mean_rgb[1],
            mean_b=mean_rgb[2],
            std_r=std_rgb[0],
            std_g=std_rgb[1],
            std_b=std_rgb[2],
        )
        return train_data, val_data, batch_fn

    def get_data_loader(data_dir, batch_size, num_workers):
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        jitter_param = 0.4
        lighting_param = 0.1
        crop_ratio = args['crop_ratio'] if args['crop_ratio'] > 0 else 0.875
        resize = int(math.ceil(input_size / crop_ratio))

        def batch_fn(batch, ctx):
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
            return data, label

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomFlipLeftRight(),
            transforms.RandomColorJitter(brightness=jitter_param, contrast=jitter_param,
                                         saturation=jitter_param),
            transforms.RandomLighting(lighting_param),
            transforms.ToTensor(),
            normalize
        ])
        transform_test = transforms.Compose([
            transforms.Resize(resize, keep_ratio=True),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize
        ])

        train_data = gluon.data.DataLoader(
            imagenet.classification.ImageNet(data_dir, train=True).transform_first(transform_train),
            batch_size=batch_size, shuffle=True, last_batch='discard', num_workers=num_workers)
        val_data = gluon.data.DataLoader(
            imagenet.classification.ImageNet(data_dir, train=False).transform_first(transform_test),
            batch_size=batch_size, shuffle=False, num_workers=num_workers)

        return train_data, val_data, batch_fn

    if args['use_rec']:
        train_data, val_data, batch_fn = get_data_rec(args['rec_train'], args['rec_train_idx'],
                                                      args['rec_val'], args['rec_val_idx'],
                                                      batch_size, num_workers)
    else:
        train_data, val_data, batch_fn = get_data_loader(args['data_dir'], batch_size, num_workers)

    if args['mixup']:
        train_metric = mx.metric.RMSE()
    else:
        train_metric = mx.metric.Accuracy()
    acc_top1 = mx.metric.Accuracy()
    acc_top5 = mx.metric.TopKAccuracy(5)

    save_frequency = args['save_frequency']
    if args['save_model'] and save_frequency:
        save_dir = args['log_dir']

    else:
        save_dir = ''
        save_frequency = 0

    def mixup_transform(label, classes, lam=1, eta=0.0):
        if isinstance(label, nd.NDArray):
            label = [label]
        res = []
        for l in label:
            y1 = l.one_hot(classes, on_value=1 - eta + eta / classes, off_value=eta / classes)
            y2 = l[::-1].one_hot(classes, on_value=1 - eta + eta / classes, off_value=eta / classes)
            res.append(lam * y1 + (1 - lam) * y2)
        return res

    def smooth(label, classes, eta=0.1):
        if isinstance(label, nd.NDArray):
            label = [label]
        smoothed = []
        for l in label:
            res = l.one_hot(classes, on_value=1 - eta + eta / classes, off_value=eta / classes)
            smoothed.append(res)
        return smoothed

    def test(ctx, val_data):
        if args['use_rec']:
            val_data.reset()
        acc_top1.reset()
        acc_top5.reset()
        for i, batch in enumerate(val_data):
            data, label = batch_fn(batch, ctx)
            outputs = [net(X.astype(args['dtype'], copy=False)) for X in data]
            acc_top1.update(label, outputs)
            acc_top5.update(label, outputs)

        _, top1 = acc_top1.get()
        _, top5 = acc_top5.get()
        return (1 - top1, 1 - top5)

    def train(ctx):
        if isinstance(ctx, mx.Context):
            ctx = [ctx]
        if args['resume_params'] is '':
            net.initialize(mx.init.MSRAPrelu(), ctx=ctx)

        if args['no_wd']:
            for k, v in net.collect_params('.*beta|.*gamma|.*bias').items():
                v.wd_mult = 0.0

        trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params)
        if args['resume_states'] is not '':
            trainer.load_states(args['resume_states'])

        if args['label_smoothing'] or args['mixup']:
            sparse_label_loss = False
        else:
            sparse_label_loss = True

        L = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=sparse_label_loss)
        best_val_score = 1

        for epoch in range(args['resume_epoch'], args['num_epochs']):
            tic = time.time()
            if args['use_rec']:
                train_data.reset()
            train_metric.reset()
            btic = time.time()

            for i, batch in enumerate(train_data):
                data, label = batch_fn(batch, ctx)

                if args['mixup']:
                    lam = np.random.beta(args['mixup_alpha'], args['mixup_alpha'])
                    if epoch >= args['num_epochs'] - args['mixup_off_epoch']:
                        lam = 1
                    data = [lam * X + (1 - lam) * X[::-1] for X in data]

                    if args['label_smoothing']:
                        eta = 0.1
                    else:
                        eta = 0.0
                    label = mixup_transform(label, classes, lam, eta)

                elif args['label_smoothing']:
                    hard_label = label
                    label = smooth(label, classes)

                with ag.record():
                    outputs = [net(X.astype(args['dtype'], copy=False)) for X in data]
                    loss = [L(yhat, y.astype(args['dtype'], copy=False)) for yhat, y in zip(outputs, label)]
                for l in loss:
                    l.backward()
                trainer.step(batch_size)

                if args['mixup']:
                    output_softmax = [nd.SoftmaxActivation(out.astype('float32', copy=False)) \
                                      for out in outputs]
                    train_metric.update(label, output_softmax)
                else:
                    if args['label_smoothing']:
                        train_metric.update(hard_label, outputs)
                    else:
                        train_metric.update(label, outputs)

                if args['log_interval'] and not (i + 1) % args['log_interval']:
                    train_metric_name, train_metric_score = train_metric.get()
                    logger.info('Epoch[%d] Batch [%d]\tSpeed: %f samples/sec\t%s=%f\tlr=%f' % (
                        epoch, i, batch_size * args['log_interval'] / (time.time() - btic),
                        train_metric_name, train_metric_score, trainer.learning_rate))
                    btic = time.time()

            train_metric_name, train_metric_score = train_metric.get()
            throughput = int(batch_size * i / (time.time() - tic))

            err_top1_val, err_top5_val = test(ctx, val_data)

            logger.info('[Epoch %d] training: %s=%f' % (epoch, train_metric_name, train_metric_score))
            logger.info('[Epoch %d] speed: %d samples/sec\ttime cost: %f' % (epoch, throughput, time.time() - tic))
            logger.info('[Epoch %d] validation: err-top1=%f err-top5=%f' % (epoch, err_top1_val, err_top5_val))

            if err_top1_val < best_val_score:
                best_val_score = err_top1_val
                net.save_parameters(
                    '%s/%.4f-imagenet-%s-%d-best.params' % (save_dir, best_val_score, model_name, epoch))
                trainer.save_states(
                    '%s/%.4f-imagenet-%s-%d-best.states' % (save_dir, best_val_score, model_name, epoch))

            if save_frequency and save_dir and (epoch + 1) % save_frequency == 0:
                net.save_parameters('%s/imagenet-%s-%d.params' % (save_dir, model_name, epoch))
                trainer.save_states('%s/imagenet-%s-%d.states' % (save_dir, model_name, epoch))

        if save_frequency and save_dir:
            net.save_parameters('%s/imagenet-%s-%d.params' % (save_dir, model_name, args['num_epochs'] - 1))
            trainer.save_states('%s/imagenet-%s-%d.states' % (save_dir, model_name, args['num_epochs'] - 1))

    if args['mode'] == 'hybrid':
        net.hybridize(static_alloc=True, static_shape=True)
    train(context)


if __name__ == '__main__':
    args = parse_args()
    args = args.__dict__
    args['base_dir'] = './efficientnet_' + args['model']
    mkdir_p(args['base_dir'])
    args['log_dir'] = set_log_dir(args, 1)
    save_arg_dict(args)
    main(args)
