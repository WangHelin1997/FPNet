import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def num_flat_features(x):
    # (32L, 50L, 11L, 14L), 32 is batch_size
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.contiguous()
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1) - 1,
                                                                 -1, -1), ('cpu', 'cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)


# fast SincConv layer.
# Use this instead of Sincconv.
class SincConv_fast(nn.Module):
    """Sinc-based convolution
    Parameters
    ----------
    in_channels : `int`
        Number of input channels. Must be 1.
    out_channels : `int`
        Number of filters.
    kernel_size : `int`
        Filter length.
    sample_rate : `int`, optional
        Sample rate. Defaults to 16000.
    Usage
    -----
    See `torch.nn.Conv1d`
    Reference
    ---------
    Mirco Ravanelli, Yoshua Bengio,
    "Speaker Recognition from raw waveform with SincNet".
    https://arxiv.org/abs/1808.00158
    """

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, out_channels, kernel_size, sample_rate, in_channels=1,
                 stride=1, padding=0, dilation=1, bias=False, groups=1, min_low_hz=1, min_band_hz=50, hz_low=1,
                 hz_high=8000):

        super(SincConv_fast, self).__init__()

        if in_channels != 1:
            # msg = (f'SincConv only support one input channel '
            #       f'(here, in_channels = {in_channels:d}).')
            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')

        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # initialize filterbanks such that they are equally spaced in Mel scale
        low_hz = hz_low
        high_hz = hz_high - (self.min_low_hz + self.min_band_hz)

        mel = np.linspace(self.to_mel(low_hz),
                          self.to_mel(high_hz),
                          self.out_channels + 1)
        hz = self.to_hz(mel)

        # filter lower frequency (out_channels, 1)
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))

        # filter frequency band (out_channels, 1)
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))
        # self.hz_high = hz_high
        self.hz_high = (self.low_hz_[out_channels - 1][-1] + self.band_hz_[out_channels - 1][-1]).detach().numpy()
        # Hamming window
        # self.window_ = torch.hamming_window(self.kernel_size)
        n_lin = torch.linspace(0, (self.kernel_size / 2) - 1,
                               steps=int((self.kernel_size / 2)))  # computing only half of the window
        self.window_ = 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / self.kernel_size);

        # (kernel_size, 1)
        n = (self.kernel_size - 1) / 2.0
        self.n_ = 2 * math.pi * torch.arange(-n, 0).view(1,
                                                         -1) / self.sample_rate  # Due to symmetry, I only need half of the time axes

    def forward(self, waveforms):
        """
        Parameters
        ----------
        waveforms : `torch.Tensor` (batch_size, 1, n_samples)
            Batch of waveforms.
        Returns
        -------
        features : `torch.Tensor` (batch_size, out_channels, n_samples_out)
            Batch of sinc filters activations.
        """

        self.n_ = self.n_.to(waveforms.device)

        self.window_ = self.window_.to(waveforms.device)

        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = low + self.min_band_hz + torch.abs(self.band_hz_)
        band = (high - low)[:, 0]

        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        band_pass_left = ((torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (
                self.n_ / 2)) * self.window_  # Equivalent of Eq.4 of the reference paper (SPEAKER RECOGNITION FROM RAW WAVEFORM WITH SINCNET). I just have expanded the sinc and simplified the terms. This way I avoid several useless computations.
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])

        band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)

        band_pass = band_pass / (2 * band[:, None])

        self.filters = (band_pass).view(
            self.out_channels, 1, self.kernel_size)

        return F.conv1d(waveforms, self.filters, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                        bias=None, groups=1)


def act_fun(act_type):
    if act_type == "relu":
        return nn.ReLU()

    if act_type == "tanh":
        return nn.Tanh()

    if act_type == "sigmoid":
        return nn.Sigmoid()

    if act_type == "leaky_relu":
        return nn.LeakyReLU(0.2)

    if act_type == "elu":
        return nn.ELU()

    if act_type == "softmax":
        return nn.LogSoftmax(dim=1)

    if act_type == "linear":
        return nn.LeakyReLU(1)  # initializzed like this, but not used in forward!


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


def init_layer(layer, nonlinearity='leaky_relu'):
    """Initialize a Linear or Convolutional layer. """
    nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer. """

    bn.bias.data.fill_(0.)
    bn.running_mean.data.fill_(0.)
    bn.weight.data.fill_(1.)
    bn.running_var.data.fill_(1.)


class WaveMsNet(nn.Module):
    def __init__(self):
        super(WaveMsNet, self).__init__()
        # Sincnet_layer
        self.bn = LayerNorm(24000)
        self.sincnet_1 = SincConv_fast(out_channels=80, kernel_size=1001, sample_rate=16000, in_channels=1,
                                       stride=1, padding=500, dilation=1, bias=False, groups=1, min_low_hz=1,
                                       min_band_hz=50, hz_high=8000, hz_low=1)
        self.bn1 = LayerNorm([80, 24000])
        self.conv2_1 = nn.Conv1d(in_channels=80, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2_3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2_4 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        # self.conv2_3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=11, stride=1, padding=5)
        self.bn2_1 = nn.BatchNorm1d(64)
        self.bn2_2 = nn.BatchNorm1d(64)
        self.bn2_3 = nn.BatchNorm1d(128)
        self.bn2_4 = nn.BatchNorm1d(128)
        # self.bn2_3 = nn.BatchNorm1d(64)
        self.pool2_1 = nn.MaxPool1d(kernel_size=5, stride=5)
        self.pool2_2 = nn.MaxPool1d(kernel_size=5, stride=5)
        self.pool2_3 = nn.MaxPool1d(kernel_size=5, stride=5)
        self.pool2_4 = nn.MaxPool1d(kernel_size=3, stride=3)
        # self.pool2_3 = nn.MaxPool1d(kernel_size=15, stride=15)

        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(256)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm1d(512)
        self.pool5 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv6 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm1d(512)
        self.pool6 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(2048, 2048)
        self.fc2 = nn.Linear(2048, 50)
        # self.fc3 = nn.Linear(4096, 50)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.init_weights()

    def init_weights(self):
        init_layer(self.conv2_1)
        init_layer(self.conv2_2)
        init_layer(self.conv2_3)
        init_layer(self.conv2_4)
        init_layer(self.conv3)
        init_layer(self.conv4)
        init_layer(self.conv5)
        init_layer(self.conv6)
        init_layer(self.fc1)
        init_layer(self.fc2)

        init_bn(self.bn2_1)
        init_bn(self.bn2_2)
        init_bn(self.bn2_3)
        init_bn(self.bn2_4)
        init_bn(self.bn3)
        init_bn(self.bn4)
        init_bn(self.bn5)
        init_bn(self.bn6)

    def forward(self, x):
        # Sincnet_layer
        h = self.bn(x)
        h = self.leaky_relu(self.bn1(torch.abs(self.sincnet_1(h))))
        # h = self.relu(self.bn1(h))
        h = self.leaky_relu(self.bn2_1(self.conv2_1(h)))
        h = self.pool2_1(h)
        h = self.leaky_relu(self.bn2_2(self.conv2_2(h)))
        h = self.pool2_2(h)
        h = self.leaky_relu(self.bn2_3(self.conv2_3(h)))
        h = self.pool2_3(h)
        h = self.leaky_relu(self.bn2_4(self.conv2_4(h)))
        h = self.pool2_4(h)

        h = self.conv3(h)
        h = self.bn3(h)
        h = self.leaky_relu(h)
        h = self.pool3(h)  # (bs, 64L, 32L, 40L)

        h = self.conv4(h)
        h = self.bn4(h)
        h = self.leaky_relu(h)
        h = self.pool4(h)  # (bs, 128L, 16L, 20L)

        h = self.conv5(h)
        h = self.bn5(h)
        h = self.leaky_relu(h)
        h = self.pool5(h)  # (bs, 256L, 8L, 10L)

        h = self.conv6(h)
        h = self.bn6(h)
        h = self.leaky_relu(h)
        h = self.pool6(h)  # (bs, 256L, 4L, 5L)

        h = h.view(-1, num_flat_features(h))  # (batchSize, 1024L)
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        h = self.fc2(h)
        return h


def init_layer(layer, nonlinearity='leaky_relu'):
    """Initialize a Linear or Convolutional layer. """
    nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
    
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    
    bn.bias.data.fill_(0.)
    bn.running_mean.data.fill_(0.)
    bn.weight.data.fill_(1.)
    bn.running_var.data.fill_(1.)

    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.init_weights()
        
    def init_weights(self):
        
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)
        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        else:
            raise Exception('Incorrect argument!')
        
        return x

class ConvBlock_mix(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(ConvBlock_mix, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(in_channels+out_channels)
        
        self.init_weights()
        
    def init_weights(self):
        
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)
        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        x = torch.cat((x,input), dim=1)
#         x = self.bn3(x)
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        else:
            raise Exception('Incorrect argument!')
        
        return x

class Cnn_9layers_AvgPooling_1D(nn.Module):
    
    def __init__(self, classes_num, activation):
        super(Cnn_9layers_AvgPooling_1D, self).__init__()
        # Sincnet_layer
        self.bn = LayerNorm(80000)
        self.sincnet_1 = SincConv_fast(out_channels=40, kernel_size=251, sample_rate=16000, in_channels=1,
                                       stride=2, padding=125, dilation=1, bias=False, groups=1, min_low_hz=1,
                                       min_band_hz=50, hz_high=8000, hz_low=1)
        self.bn1 = LayerNorm([40, 40000])
        self.conv2_1 = nn.Conv1d(in_channels=40, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2_3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2_4 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        # self.conv2_3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=11, stride=1, padding=5)
        self.bn2_1 = nn.BatchNorm1d(64)
        self.bn2_2 = nn.BatchNorm1d(64)
        self.bn2_3 = nn.BatchNorm1d(128)
        self.bn2_4 = nn.BatchNorm1d(128)
        # self.bn2_3 = nn.BatchNorm1d(64)
        self.pool2_1 = nn.AvgPool1d(kernel_size=5, stride=5)
        self.pool2_2 = nn.AvgPool1d(kernel_size=5, stride=5)
        self.pool2_3 = nn.AvgPool1d(kernel_size=5, stride=5)
        self.pool2_4 = nn.AvgPool1d(kernel_size=5, stride=5)
        # self.pool2_3 = nn.MaxPool1d(kernel_size=15, stride=15)

        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.AvgPool1d(kernel_size=4, stride=4)

        self.conv4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(256)
        self.pool4 = nn.AvgPool1d(kernel_size=4, stride=4)

        self.conv5 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm1d(512)
        self.pool5 = nn.AvgPool1d(kernel_size=2, stride=2)

        self.conv6 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm1d(512)
        self.pool6 = nn.AvgPool1d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(512, 64)
        self.fc2 = nn.Linear(64, 50)
        # self.fc3 = nn.Linear(4096, 50)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.2)

        # self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=51, stride=5, padding=25)
        # self.bn1 = nn.BatchNorm1d(32)
        # self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=11, stride=1, padding=5)
        # self.bn2 = nn.BatchNorm1d(32)
        # self.pool2 = nn.MaxPool1d(kernel_size=30, stride=30)
        self.activation = activation
        #
        # self.conv_block1 = ConvBlock_mix(in_channels=1, out_channels=64)
        # self.conv_block2 = ConvBlock_mix(in_channels=65, out_channels=128)
        # self.conv_block3 = ConvBlock_mix(in_channels=193, out_channels=256)
        # self.conv_block4 = ConvBlock_mix(in_channels=449, out_channels=512)
        # self.fc2 = nn.Linear(961, 1024, bias=True)
        # self.dropout = nn.Dropout(p=0.5)
        # self.fc = nn.Linear(1024, classes_num, bias=True)
        #
        # self.init_weights()

    def init_weights(self):

        init_layer(self.fc)

    def forward(self, input):
        '''
        Input: (batch_size, times_steps, freq_bins)'''
        x = input[:, None, :]
        h = self.bn(x)
        h = self.leaky_relu(self.bn1(torch.abs(self.sincnet_1(h))))
        # h = self.relu(self.bn1(h))
        h = self.leaky_relu(self.bn2_1(self.conv2_1(h)))
        h = self.pool2_1(h)
        h = self.leaky_relu(self.bn2_2(self.conv2_2(h)))
        h = self.pool2_2(h)
        h = self.leaky_relu(self.bn2_3(self.conv2_3(h)))
        h = self.pool2_3(h)
        h = self.leaky_relu(self.bn2_4(self.conv2_4(h)))
        h = self.pool2_4(h)

        h = self.conv3(h)
        h = self.bn3(h)
        h = self.leaky_relu(h)
        h = self.pool3(h)  # (bs, 64L, 32L, 40L)

        h = self.conv4(h)
        h = self.bn4(h)
        h = self.leaky_relu(h)
        h = self.pool4(h)  # (bs, 128L, 16L, 20L)

        h = self.conv5(h)
        h = self.bn5(h)
        h = self.leaky_relu(h)
        h = self.pool5(h)  # (bs, 256L, 8L, 10L)

        h = self.conv6(h)
        h = self.bn6(h)
        h = self.leaky_relu(h)
        h = self.pool6(h)  # (bs, 256L, 4L, 5L)

        h = h.view(-1, num_flat_features(h))  # (batchSize, 1024L)
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        h = self.fc2(h)
        if self.activation == 'logsoftmax':
            output = F.log_softmax(h, dim=-1)

        elif self.activation == 'sigmoid':
            output = torch.sigmoid(h)

        return output



    #     x = F.relu_(self.bn1(self.conv1(x)))
    #     x = F.relu_(self.bn2(self.conv2(x)))
    #     x = self.pool2(x)
    #     x = x[:, None, :, :]
    #     '''(batch_size, 1, times_steps, freq_bins)'''
    #
    #     x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')#（64，64，215，20）
    #     x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')#（64，193，107，10）
    #     x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')#（64，449，53，5）
    #     x = self.conv_block4(x, pool_size=(1, 1), pool_type='avg')#（64，961，53，5）
    #     '''(batch_size, feature_maps, time_steps, freq_bins)'''
    #     (x, _) = torch.max(x, dim=3)
    #     x = torch.mean(x, dim=2)
    #     x = F.relu_(self.fc2(x))
    #     x = self.dropout(x)
    #     x = self.fc(x)
    #
    #     if self.activation == 'logsoftmax':
    #         output = F.log_softmax(x, dim=-1)
    #
    #     elif self.activation == 'sigmoid':
    #         output = torch.sigmoid(x)
    #
    #     return output
    #