"""
Implement EfficientNet By Using Gluon
paper: https://arxiv.org/abs/1905.11946
"""
import argparse
from utils import *
parser = argparse.ArgumentParser(description='EfficientNet')
parser.add_argument('-p', '--path', help='path of dataset')
parser.add_argument('-c', 'cuda', action='store_true', help='whether use GPU')
parser.add_argument('--epoch', type=int, help = 'number of epochs')


