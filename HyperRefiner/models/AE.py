# -*- coding：UTF-8 -*-
'''
@Project : HyperRefiner
@File ：AE.py
@Author : Zerbo
@Date : 2022/10/31 20:28
'''

import torch
import torch.nn as nn
from math import floor


class Reshape(nn.Module):
    """ Helper module which performs a reshape operation."""

    def __init__(self, new_shape):
        super().__init__()
        self.new_shape = new_shape

    def forward(self, x):
        return x.reshape(self.new_shape)


class ConvEncoder(nn.Module):
    """ A convolutional encoder, intended as a drop-in replacement for Encoder.
    """

    def __init__(self, input_size, conv_kernel_sizes, channels, pool_kernel_sizes, conv_strides=None, pool_strides=None, pool_op=nn.MaxPool1d, input_channels=1,
                 dropout=0.0, activation=nn.ReLU, ):
        """ Initializer. The overall structure of this module is:
        Input -> Conv -> Conv -> Pool -> <repeat> -> Linear
        Where the number of [Conv -> Conv -> Pool] units is equal to
        len(conv_kernel_sizes).
        conv_kernel_sizes, channels, pool_kernel_sizes, conv_strides, and
        pool_strides must all be the same length (if provided).
        Parameters
        ----------
        input_size : int
            The spatial size of the input.  输入的空间大小
        conv_kernel_sizes : List[int]       列表， 每个阶段的卷积和大小
            The kernel sizes of the convolutions at each stage.
            Padding at each conv is of size (kernel_size - 1) // 2, to maintain
            spatial resolution if conv_stride = 1.
        channels : List[int]                列表， 每个阶段的通道数
            The number of channels at each stage.
        pool_kernel_sizes : List[int]       列表， 每个阶段池化核大小
            The kernel sizes of the pooling operations at each stage. There is
            no padding done.
        latent_space_size : int
            The size of the final latent space.   隐空间维度
        conv_strides : Optional[List[int]]  列表， 每个阶段卷积步长，默认1
            The stride of the convolutions at each stage. If not provided, is 1
            for all stages.
        pool_strides : Optional[List[int]]  列表， 每个阶段池化步长，默认2
            The stride of the pooling operations at each stage. If not
            provided, is 2 for all stages.
        pool_op : Callable                  池化操作
            Constructor for a nn.Module to perform the pooling operation. By
            default, nn.MaxPool1d.
        input_channels : int                输入通道
            The number of channels the input has.
        dropout : float
            Probability of dropping internal values while training.
        activation : Callable
            Constructor for a nn.Module defining the activation function.
        """

        super().__init__()

        # 构造相应的卷积、池化参数
        if not len(conv_kernel_sizes) == len(channels) == len(pool_kernel_sizes):
            raise ValueError("Specification lists must all be the same length.")

        if conv_strides is None:
            conv_strides = [1] * len(conv_kernel_sizes)
        elif len(conv_strides) != len(conv_kernel_sizes):
            raise ValueError("len(conv_strides) must equal len(conv_kernel_sizes).")

        if pool_strides is None:
            pool_strides = [2] * len(pool_kernel_sizes)
        elif len(pool_strides) != len(pool_kernel_sizes):
            raise ValueError("len(pool_strides) must equal len(pool_kernel_sizes).")

        dropout_module = nn.Dropout(p=dropout, inplace=True)
        activation_module = activation(inplace=True)

        modules = []
        curr_size = input_size

        for i in range(len(conv_kernel_sizes)):
            # Padding chosen such that, for odd kernel sizes and stride = 1,
            #  the result after convolution is the same length as the input.
            conv_kernel_size = conv_kernel_sizes[i]
            conv_padding = (conv_kernel_sizes[i] - 1) // 2
            conv_stride = conv_strides[i]
            prev_channels = input_channels if i == 0 else channels[i - 1]
            new_channels = channels[i]

            # The overall structure is Conv -> Conv -> MaxPool, to give two
            # convolutions at each spatial resolution to allow the receptive
            # field to be large enough before pooling.
            modules.append(nn.Conv1d(prev_channels, new_channels, kernel_size=conv_kernel_size, padding=conv_padding, stride=conv_stride))
            curr_size = floor((curr_size + 2 * conv_padding - conv_kernel_size) / conv_stride + 1)
            modules.append(activation_module)
            modules.append(nn.BatchNorm1d(new_channels))

            if dropout != 0.0:
                modules.append(dropout_module)

            modules.append(nn.Conv1d(new_channels, new_channels, kernel_size=conv_kernel_size, padding=conv_padding))
            curr_size = floor((curr_size + 2 * conv_padding - conv_kernel_size) / conv_stride + 1)

            pool_kernel_size = pool_kernel_sizes[i]
            pool_stride = pool_strides[i]
            modules.append(pool_op(pool_kernel_size, stride=pool_stride))
            curr_size = floor((curr_size - pool_kernel_size) / pool_stride + 1)
            modules.append(activation_module)
            modules.append(nn.BatchNorm1d(channels[i]))

            if dropout != 0.0:
                modules.append(dropout_module)

        self.module = nn.Sequential(*modules)

    def forward(self, x):
        """Forward propagation.
        out = []
        for i in range(len(self.module)):
            x = self.module[i](x)
            if i%9==8:
                out.append(x)
        """
        out = []
        for i in range(len(self.module)):
            x = self.module[i](x)
            if i % 9 == 8:
                out.append(x)
        return out

class ConvDecoder(nn.Module):
    """A convolutional decoder, intended as a drop-in replacement for Decoder
    """

    def __init__(self, conv_kernel_sizes, channels, pool_kernel_sizes, output_size, conv_strides=None, pool_strides=None, output_channels=1, dropout=0.0,
                 activation=nn.ReLU):
        """ Initializer. The overall structure of this module is:
        Linear -> ConvTranspose -> Conv -> <repeat> -> Output
        Where the number of [ConvTranspose -> Conv] units is equal to
        len(conv_kernel_sizes).
        conv_kernel_sizes, channels, pool_kernel_sizes, conv_strides, and
        pool_strides must all be the same length (if provided).
        The spatial size of the first hidden layer (following the nn.Linear)
        is calculated as if this was a ConvEncoder, running in reverse. [i.e.
        if the arguments to both ConvEncoder, ConvDecoder are symmetric, this
        size should match on either side of the nn.Linear.
        Parameters
        ----------
        latent_space_size : int
            The size of the input latent space.
        conv_kernel_sizes : List[int]
            The kernel sizes of the convolutions at each stage. Padding at
            each ConvTranspose is 0, and at each Conv is of size
            (kernel_size - 1) // 2, to maintain spatial resolution if
            conv_stride = 1.
        channels : List[int]
            The number of channels at each stage.
        pool_kernel_sizes : List[int]
            The kernel sizes of the "pooling" operations at each stage that the
            decoder must undo.
        output_size : int
            The spatial size of the output.
        conv_strides : Optional[List[int]]
            The stride of the convolutions at each stage. If not provided, is 1
            for all stages.
        pool_strides : Optional[List[int]]
            The stride of the "pooling" operations at each stage the decoder
            must undo. If not provided, is 2 for all stages.
        output_channels : int
            The number of channels the output has.
        dropout : float
            Probability of dropping internal values while training.
        activation : Callable
            Constructor for a nn.Module defining the activation function.
        """

        super().__init__()

        if not len(conv_kernel_sizes) == len(channels) == len(pool_kernel_sizes):
            raise ValueError("Specification lists must all be the same length.")

        if conv_strides is None:
            conv_strides = [1] * len(conv_kernel_sizes)
        elif len(conv_strides) != len(conv_kernel_sizes):
            raise ValueError("len(conv_strides) must equal len(conv_kernel_sizes).")

        if pool_strides is None:
            pool_strides = [2] * len(pool_kernel_sizes)
        elif len(pool_strides) != len(pool_kernel_sizes):
            raise ValueError("len(pool_strides) must equal len(pool_kernel_sizes).")

        activation_module = activation(inplace=True)
        dropout_module = nn.Dropout(p=dropout, inplace=True)

        # Do some calculations to figure out what the spatial sizes should be
        #  to match up with a symmetric ConvEncoder
        sizes = [output_size]
        reversed_layers = zip(reversed(conv_kernel_sizes), reversed(conv_strides), reversed(pool_kernel_sizes), reversed(pool_strides))
        cur_size = output_size
        for c_ksize, c_str, p_ksize, p_str in reversed_layers:
            c_pad = (c_ksize - 1) // 2
            # Conv -> Conv
            cur_size = floor((cur_size + 2 * c_pad - c_ksize) / c_str + 1)
            # Pool
            cur_size = floor((cur_size - p_ksize) / p_str + 1)
            sizes.append(cur_size)
        sizes.reverse()

        modules = []

        # Initial dense layer, then reshape to convolutional shape

        for i in range(len(conv_kernel_sizes)):
            # Padding chosen such that, for odd kernel sizes and stride = 1,
            #  the result after convolution is the same length as the input.
            conv_kernel_size = conv_kernel_sizes[i]
            conv_padding = (conv_kernel_sizes[i] - 1) // 2
            conv_stride = conv_strides[i]
            pool_kernel_size = pool_kernel_sizes[i]
            pool_stride = pool_strides[i]
            prev_channels = channels[i]
            new_channels = output_channels if i == len(conv_kernel_sizes) - 1 else channels[i + 1]
            prev_size = sizes[i]
            new_size = sizes[i + 1]

            modules.append(activation_module)
            modules.append(nn.BatchNorm1d(prev_channels))
            modules.append(dropout_module)

            # The overall structure is ConvTranspose -> Conv
            convt_size = (prev_size - 1) * pool_stride + pool_kernel_size
            modules.append(nn.ConvTranspose1d(prev_channels, prev_channels, kernel_size=pool_kernel_size, stride=pool_stride,

                                              # Resolves ambiguity about resulting size
                                              output_padding=new_size - convt_size))
            modules.append(activation_module)
            modules.append(nn.BatchNorm1d(prev_channels))
            if dropout != 0.0:
                modules.append(dropout_module)

            modules.append(nn.Conv1d(prev_channels, new_channels, kernel_size=conv_kernel_size, stride=conv_stride, padding=conv_padding))

        self.module = nn.Sequential(*modules)

    def forward(self, x):
        """Forward propagation"""

        return self.module(x)


class ConvAutoencoder(nn.Module):
    """ Convolutional autoencoder.
    """

    def __init__(self, config):
        super().__init__()
        self.input_size = config[config["train_dataset"]]["LR_size"] ** 2
        self.in_channels = config[config["train_dataset"]]["spectral_bands"]
        self.out_channels = config[config["train_dataset"]]["spectral_bands"]
        self.conv_kernel_sizes = config["AE"]["conv_kernel_sizes"]
        self.pool_kernel_sizes = config["AE"]["pool_kernel_sizes"]
        self.conv_strides = config["AE"]["conv_strides"]
        self.pool_strides = config["AE"]["pool_strides"]
        self.dropout = config["AE"]["dropout"]
        self.channels = config["AE"]["channels"]
        self.pool_op = nn.MaxPool1d
        self.activation = nn.ReLU
        # Multiply the encoder output by 2 if using the VAE
        self.encoder = ConvEncoder(input_size=self.input_size, conv_kernel_sizes=self.conv_kernel_sizes, channels=self.channels,
                                   pool_kernel_sizes=self.pool_kernel_sizes, conv_strides=self.conv_strides, pool_strides=self.pool_strides,
                                   pool_op=self.pool_op, dropout=self.dropout, activation=self.activation, input_channels=self.in_channels)
        self.decoder = ConvDecoder(conv_kernel_sizes=list(reversed(self.conv_kernel_sizes)), channels=list(reversed(self.channels)),
                                   pool_kernel_sizes=list(reversed(self.pool_kernel_sizes)), output_size=self.input_size, conv_strides=self.conv_strides,
                                   pool_strides=self.pool_strides, dropout=self.dropout, activation=self.activation, output_channels=self.out_channels)
        self.softplus = nn.Softplus()
        self.flag=config["train"]

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.contiguous().view(b, c, -1)
        out = self.encoder(x)
        z = out[2]
        #z = self.encoder(x)
        b1, c1, _ = z.size()
        if self.flag:
            return out, self.softplus(self.decoder(z)).view(b, c, h, w)
        else:
            return out
