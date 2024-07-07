import torch.nn as nn
import collections

# different relu function config
_activations = {'relu': nn.ReLU, 'relu6': nn.ReLU6, 'leaky_relu': nn.LeakyReLU}


class BaseBlock(nn.Module):
    """
    base layer
    """

    def __init__(self):
        super(BaseBlock, self).__init__()
        self._layer: nn.Sequential

    def forward(self, x):
        return self._layer(x)


class Conv2DBlock(BaseBlock):
    def __init__(self, shape, stride, padding='same', **params):
        """
        convolutional layer
        shape (np.array): [h, w, input_channel, output_channel], eg: [5, 5, 1, 16]
        """
        super(Conv2DBlock, self).__init__()

        (
            h,
            w,
            in_channels,
            out_channels,
        ) = shape
        _seq = collections.OrderedDict(
            [
                (
                    'conv',
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=(h, w),
                        stride=stride,
                        padding=padding,
                    ),
                )
            ]
        )
        # batch normalization
        _bn = params.get('batch_norm')
        if _bn:
            _seq.update({'bn': nn.BatchNorm2d(out_channels)})
        # activation function
        _act_name = params.get('activation')
        if _act_name:
            _seq.update({_act_name: _activations[_act_name](inplace=True)})
        # whether use pooling layer
        _max_pool = params.get('max_pool')
        if _max_pool:
            _kernel_size = params.get('max_pool_size', 2)
            _stride = params.get('max_pool_stride', _kernel_size)
            _seq.update(
                {'max_pool': nn.MaxPool2d(kernel_size=_kernel_size, stride=_stride)}
            )

        self._layer = nn.Sequential(_seq)
        # weight initialization
        w_init = params.get('w_init', None)
        idx = list(dict(self._layer.named_children()).keys()).index('conv')
        if w_init:
            w_init(self._layer[idx].weight)
        # bias initialization
        b_init = params.get('b_init', None)
        if b_init:
            b_init(self._layer[idx].bias)


class DenseBlock(BaseBlock):
    def __init__(self, shape, **params):
        """
        full connection layer
        shape (np.array): the input and output data shape, eg: [64, 128]
        """
        super(DenseBlock, self).__init__()

        in_dims, out_dims = shape
        _seq = collections.OrderedDict(
            [
                ('dense', nn.Linear(in_dims, out_dims)),
            ]
        )

        # activation function 
        _act_name = params.get('activation')
        if _act_name:
            _seq.update({_act_name: _activations[_act_name](inplace=True)})

        self._layer = nn.Sequential(_seq)

        # weight initialization
        w_init = params.get('w_init', None)
        idx = list(dict(self._layer.named_children()).keys()).index('dense')
        if w_init:
            w_init(self._layer[idx].weight)
        # bias initialization
        b_init = params.get('b_init', None)
        if b_init:
            b_init(self._layer[idx].bias)
