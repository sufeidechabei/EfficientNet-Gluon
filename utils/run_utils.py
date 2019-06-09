"""
Some running functions.
"""
import math
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
from math import ceil
class SamePadding(HybridBlock):
    def __init__(self, **kwargs):
        super(SamePadding, self).__init__(**kwargs)
    def hybrid_forward(self, F, x, kernel_size, stride, dilation):
        ih, iw = x.shape[-2:]
        kh, kw = kernel_size
        sh, sw = stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * stride[0] + (kh - 1) * dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * stride[1] + (kw - 1) * dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, mode='constant', pad_width=(pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2))
            return x
        return x



















