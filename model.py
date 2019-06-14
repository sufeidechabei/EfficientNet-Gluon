import math
import mxnet as mx
import collections
import re
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn

# Parameters for the entire model (stem, all blocks, and head)
GlobalParams = collections.namedtuple('GlobalParams', [
    'dropout_rate','num_classes', 'width_coefficient', 'depth_coefficient',
    'depth_divisor', 'min_depth', 'drop_connect_rate',])


# Parameters for an individual model block
BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'stride', 'se_ratio'])


def round_repeats(repeats, global_params):
    """ Round number of filters based on depth multiplier. """
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


def round_filters(filters, global_params):
    """ Calculate and round number of filters based on depth multiplier. """
    multiplier = global_params.width_coefficient
    if not multiplier:
        return filters
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
        new_filters += divisor
    return int(new_filters)


class BlockDecoder(object):
    """ Block Decoder for readability, straight from the official TensorFlow repository """

    @staticmethod
    def _decode_block_string(block_string):
        """ Gets a block through a string notation of arguments. """
        assert isinstance(block_string, str)

        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        # Check stride
        assert (('s' in options and len(options['s']) == 1) or
                (len(options['s']) == 2 and options['s'][0] == options['s'][1]))

        return BlockArgs(
            kernel_size=int(options['k']),
            num_repeat=int(options['r']),
            input_filters= int(options['i']),
            output_filters=int(options['o']),
            expand_ratio=int(options['e']),
            id_skip=('noskip' not in block_string),
            se_ratio=float(options['se']) if 'se' in options else None,
            stride=int(options['s'][0]))

    @staticmethod
    def _encode_block_string(block):
        """Encodes a block to a string."""
        args = [
            'r%d' % block.num_repeat,
            'k%d' % block.kernel_size,
            's%d%d' % (block.strides[0], block.strides[1]),
            'e%s' % block.expand_ratio,
            'i%d' % block.input_filters,
            'o%d' % block.output_filters
        ]
        if 0 < block.se_ratio <= 1:
            args.append('se%s' % block.se_ratio)
        if block.id_skip is False:
            args.append('noskip')
        return '_'.join(args)

    @staticmethod
    def decode(string_list):
        """
        Decodes a list of string notations to specify blocks inside the network.

        :param string_list: a list of strings, each string is a notation of block
        :return: a list of BlockArgs namedtuples of block args
        """
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(BlockDecoder._decode_block_string(block_string))
        return blocks_args

    @staticmethod
    def encode(blocks_args):
        """
        Encodes a list of BlockArgs to a list of strings.

        :param blocks_args: a list of BlockArgs namedtuples of block args
        :return: a list of strings, each string is a notation of block
        """
        block_strings = []
        for block in blocks_args:
            block_strings.append(BlockDecoder._encode_block_string(block))
        return block_strings

def efficientnet(width_coefficient=None, depth_coefficient=None,
                 dropout_rate=0.2, drop_connect_rate=0.2):
    """ Creates a efficientnet model. """

    blocks_args = [
        'r1_k3_s11_e1_i32_o16_se0.25', 'r2_k3_s22_e6_i16_o24_se0.25',
        'r2_k5_s22_e6_i24_o40_se0.25', 'r3_k3_s22_e6_i40_o80_se0.25',
        'r3_k5_s11_e6_i80_o112_se0.25', 'r4_k5_s22_e6_i112_o192_se0.25',
        'r1_k3_s11_e6_i192_o320_se0.25',
    ]
    blocks_args = BlockDecoder.decode(blocks_args)

    global_params = GlobalParams(
        dropout_rate=dropout_rate,
        drop_connect_rate=drop_connect_rate,
        # data_format='channels_last',  # removed, this is always true in PyTorch
        num_classes=1000,
        width_coefficient=width_coefficient,
        depth_coefficient=depth_coefficient,
        depth_divisor=8,
        min_depth=None
    )
    return blocks_args, global_params

def drop_connect(x, p, training):
    if not training: return x
    batch_size = x.shape[0]
    keep_prob = 1 - p
    random_tensor = keep_prob
    random_tensor += mx.nd.random.uniform(shape = [batch_size, 1, 1, 1]).astype(x.dtype).as_in_context(x.context)  # uniform [0,1)
    binary_tensor = mx.nd.floor(random_tensor)
    output = x / keep_prob * binary_tensor
    return output

class SamePadding(HybridBlock):
    def __init__(self, kernel_size, stride, dilation, **kwargs):
        super(SamePadding, self).__init__(**kwargs)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, ) * 2
        if isinstance(stride, int):
            stride = (stride, ) * 2
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
    def hybrid_forward(self, F, x):
        ih, iw = x.shape[-2:]
        kh, kw = self.kernel_size
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, mode='constant', pad_width=(0, 0, 0, 0, pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2))
            return x
        return x

def _add_conv(out, channels=1, kernel=1, stride=1, pad=0,
              num_group=1, active=True, batchnorm=True):
    out.add(SamePadding(kernel, stride,  dilation=(1, 1)))
    out.add(nn.Conv2D(channels, kernel, stride, pad, groups=num_group, use_bias=False))
    if batchnorm:
       out.add(nn.BatchNorm(scale=True, momentum=0.99, epsilon=1e-3))
    if active:
        out.add(nn.Swish())

class MBConv(nn.HybridBlock):
    def __init__(self, in_channels, channels, t, kernel, stride, se_ratio = 0, drop_connect_rate=0,**kwargs):
        super(MBConv, self).__init__(**kwargs)
        self.use_shortcut = stride == 1 and in_channels == channels
        self.se_ratio = se_ratio
        self.drop_connect_rate = drop_connect_rate
        with self.name_scope():
            self.out = nn.HybridSequential(prefix = "out_")
            with self.out.name_scope():
                if t != 1:
                    _add_conv(self.out, in_channels * t, active=True, batchnorm=True)
                _add_conv(self.out, in_channels * t, kernel=kernel, stride=stride, num_group=in_channels * t,
                          active=True, batchnorm=True)
            if se_ratio:
                num_squeezed_channels = max(1, int(in_channels * se_ratio))
                self._se_reduce = nn.HybridSequential(prefix = "se_reduce_")
                self._se_expand = nn.HybridSequential(prefix = "se_expand_")
                with self._se_reduce.name_scope():
                     _add_conv(self._se_reduce, num_squeezed_channels, active=False, batchnorm=False)
                with self._se_expand.name_scope():
                     _add_conv(self._se_expand, in_channels * t, active=False, batchnorm=False)
            self.project_layer = nn.HybridSequential(prefix = "project_layer_")
            with self.project_layer.name_scope():
                 _add_conv(self.project_layer,channels, active=False, batchnorm=True)
    def hybrid_forward(self, F, input):
        x = input
        x = self.out(x)
        if self.se_ratio:
            out = mx.nd.contrib.AdaptiveAvgPooling2D(x, 1)
            out = self._se_expand(self._se_reduce(out))
            out = mx.ndarray.sigmoid(out) * x
        out = self.project_layer(out)
        if self.use_shortcut:
            if self.drop_connect_rate:
                out = drop_connect(out, p=self.drop_connect_rate, training=self.training)
            out = F.elemwise_add(out, input)
        return out
    def get_mode(self, training):
        self.training = training

class EfficientNet(nn.HybridBlock):
    r"""
    Parameters
    ----------
    block_args : : list, hyperparamter of every block.
    global_param: collection.namedtuple, hyperparameter of every layer.
    """

    def __init__(self, blocks_args=None, global_params=None, **kwargs):
        super(EfficientNet, self).__init__(**kwargs)
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='features_')
            with self.features.name_scope():
                # stem conv
                out_channels  = round_filters(32, self._global_params)
                _add_conv(self.features, out_channels, kernel=3, stride=2, active=True, batchnorm=True)
            self._blocks = nn.HybridSequential(prefix='blocks_')
            with self._blocks.name_scope():
                for block_arg in self._blocks_args:
                    # Update block input and output filters based on depth multiplier.
                    block_arg = block_arg._replace(
                        input_filters=round_filters(block_arg.input_filters, self._global_params),
                        output_filters=round_filters(block_arg.output_filters, self._global_params),
                        num_repeat=round_repeats(block_arg.num_repeat, self._global_params)
                    )
                    self._blocks.add(MBConv(block_arg.input_filters,
                                            block_arg.output_filters,
                                            block_arg.expand_ratio,
                                            block_arg.kernel_size,
                                            block_arg.stride,
                                            block_arg.se_ratio,
                                            global_params.drop_connect_rate)
                                     )
                    if block_arg.num_repeat > 1:
                        block_arg = block_arg._replace(input_filters=block_arg.output_filters, stride=1)
                    for _ in range(block_arg.num_repeat - 1):
                        self._blocks.add(MBConv(block_arg.input_filters,
                                                block_arg.output_filters,
                                                block_arg.expand_ratio,
                                                block_arg.kernel_size,
                                                block_arg.stride,
                                                block_arg.se_ratio,
                                                global_params.drop_connect_rate))

           # Head
            out_channels = round_filters(1280, self._global_params)
            self._conv_head = nn.HybridSequential(prefix='conv_head_')
            with self._conv_head.name_scope():
                _add_conv(self._conv_head, out_channels, active=True, batchnorm=True)
            # Final linear layer
            self._dropout = self._global_params.dropout_rate
            self._fc = nn.Dense(self._global_params.num_classes)


    def hybrid_forward(self, F, x, training):
        x = self.features(x)
        for block in self._blocks:
            block.get_mode(training=training)
            x = block(x)
        x = self._conv_head(x)
        x = F.squeeze(F.squeeze(mx.nd.contrib.AdaptiveAvgPooling2D(x, 1), axis=-1), axis=-1)
        if self._dropout:
            x = F.Dropout(x, self._dropout)
        x = self._fc(x)
        return x

def get_efficientnet(model_name):
    params_dict = {  # (width_coefficient, depth_coefficient, input_resolution, dropout_rate)
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5)
    }
    width_coeff, depth_coeff, input_resolution, dropout_rate = params_dict[model_name]
    blocks_args, global_params = efficientnet(width_coeff, depth_coeff)
    model = EfficientNet(blocks_args, global_params)
    return model, input_resolution
    blocks_args, global_params = efficientnet(width_coeff, depth_coeff)
    model = EfficientNet(blocks_args, global_params)
    return model, input_resolution
