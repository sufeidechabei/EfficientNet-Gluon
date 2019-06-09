"""
Implement EfficientNet By Using Gluon
paper: https://arxiv.org/abs/1905.11946
"""
import argparse
import warnings
parser = argparse.ArgumentParser(description='EfficientNet')
parser.add_argument('-p', '--path', type=str, help='path of dataset')
parser.add_argument('-c', '--cuda', action='store_true', help='whether use GPU')
parser.add_argument('--epoch', type=int, default=90, help = 'number of epochs')
parser.add_argument('--num_gpu', type=int, default=1, help='indicate the number of GPU used for training')
parser.add_argument('-b', '--batch_size', default=256)
parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate ')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='momentum')
parser.add_argument('-wd', '--weight_decay', type=float, default=1e-4)
parser.add_argument('--pretrain', type=bool, default=False, help='whether use pretrained model')
parser.add_argument('--resume', type=str, help='path of pretrained model')
parser.add_argument('--set_seed', default=None, type=bool,
                    help='seed for initializing training. ')
parser.add_argument('--model', type=str, choices=['b0, b1, b2, b3, b4, b5, b6, b7'], help='choose the'
                                                                                          ' version of EfficientNet')
args = parser.parse_args().__dict__




















