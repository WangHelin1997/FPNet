import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

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
    
    
class Cnn_5layers_AvgPooling(nn.Module):
    
    def __init__(self, classes_num, activation):
        super(Cnn_5layers_AvgPooling, self).__init__()

        self.activation = activation

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64,
                              kernel_size=(5, 5), stride=(1, 1),
                              padding=(2, 2), bias=False)
        
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128,
                              kernel_size=(5, 5), stride=(1, 1),
                              padding=(2, 2), bias=False)
                              
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256,
                              kernel_size=(5, 5), stride=(1, 1),
                              padding=(2, 2), bias=False)
                              
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512,
                              kernel_size=(5, 5), stride=(1, 1),
                              padding=(2, 2), bias=False)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)

        self.fc = nn.Linear(512, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_layer(self.conv3)
        init_layer(self.conv4)
        init_layer(self.fc)
        
        init_bn(self.bn1)
        init_bn(self.bn2)
        init_bn(self.bn3)
        init_bn(self.bn4)

    def forward(self, input):
        '''
        Input: (batch_size, times_steps, freq_bins)'''
        x = input[:, None, :, :]
        '''(batch_size, 1, times_steps, freq_bins)'''
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.avg_pool2d(x, kernel_size=(2, 2))
        
        x = F.relu_(self.bn2(self.conv2(x)))
        x = F.avg_pool2d(x, kernel_size=(2, 2))
        
        x = F.relu_(self.bn3(self.conv3(x)))
        x = F.avg_pool2d(x, kernel_size=(2, 2))
        
        x = F.relu_(self.bn4(self.conv4(x)))
        x = F.avg_pool2d(x, kernel_size=(1, 1))
        '''(batch_size, feature_maps, time_steps, freq_bins)'''
        x = torch.mean(x, dim=3)        # (batch_size, feature_maps, time_stpes)
        (x, _) = torch.max(x, dim=2)    # (batch_size, feature_maps)
        x = self.fc(x)
        if self.activation == 'logsoftmax':
            output = F.log_softmax(x, dim=-1)
            
        elif self.activation == 'sigmoid':
            output = F.sigmoid(x)
        return output
    
    
# class ConvBlock(nn.Module):
#     def __init__(self, in_channels, mid_channels, out_channels, sub_bins=20, sub_stride=10, mel_bins=40):
        
#         super(ConvBlock, self).__init__()
#         self.bn1 = nn.BatchNorm2d(in_channels)
#         self.n_max_pool = int(sub_bins / 10)
#         self.sub_bins, self.sub_stride, self.mel_bins = sub_bins, sub_stride, mel_bins
#         self.sub_num = int(self.mel_bins/self.sub_stride) - 1
        
#         self.conv1 = nn.ModuleList( 
#             [nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=5, stride=1, padding=2) for _ in
#              range(self.sub_num)])
#         self.conv1_bn = nn.ModuleList([nn.BatchNorm2d(64) for _ in range(self.sub_num)])
#         self.mp1 = nn.ModuleList([nn.MaxPool2d((1, 3)) for _ in range(self.sub_num)])
        
#         self.conv2 = nn.Conv2d(in_channels=mid_channels, 
#                               out_channels=out_channels,
#                               kernel_size=(3, 3), stride=(1, 1),
#                               padding=(1, 1), bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.mp2 = nn.AvgPool2d((3, 1)
        
        
#     def init_weights(self):
        
#         init_layer(self.conv2)
#         init_bn(self.bn1)
#         init_bn(self.bn2)
        
#     def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        
#         x = input
#         x = x.transpose(2, 3)
#         for i in range(self.sub_num):
#             x = input_var[:, :, i*self.sub_stride:(i+2)*self.sub_stride, :]
#             x = self.conv1[i](x)
#             x = self.conv1_bn[i](x)
#             x = F.relu(x)
#             x = self.conv2[i](x)
#             x = self.conv2_bn[i](x)
#             x = F.relu(x)
#             x = self.mp2[i](x)
#             x = self.conv3[i](x)
#             x = self.conv3_bn[i](x)
#             x = F.relu(x)
#             x = self.conv4[i](x)
#             x = self.conv4_bn[i](x)
#             x = F.relu(x)
#             x = self.mp4[i](x)
#             x = self.conv5[i](x)
#             x = self.conv5_bn[i](x)
#             x = F.relu(x)
#             x = self.conv6[i](x)
#             x = self.conv6_bn[i](x)
#             x = F.relu(x)
#             x = self.mp6[i](x)
                 
#             (x, _) = torch.max(x, dim=3)    
#             x = torch.mean(x, dim=2) 
#             x = self.fc1[i](x)
#             x = F.relu(x)
#             intermediate.append(x)

#         x = F.relu_(self.bn1(self.conv1(x)))
#         x = F.relu_(self.bn2(self.conv2(x)))
#         if pool_type == 'max':
#             x = F.max_pool2d(x, kernel_size=pool_size)
#         elif pool_type == 'avg':
#             x = F.avg_pool2d(x, kernel_size=pool_size)
#         else:
#             raise Exception('Incorrect argument!')
        
#         return x

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
#         x = self.bn3(x)
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        else:
            raise Exception('Incorrect argument!')
        
        return x

class Cnn_9layers_AvgPooling_mix(nn.Module):
    
    def __init__(self, classes_num, activation):
        super(Cnn_9layers_AvgPooling_mix, self).__init__()

        self.activation = activation

        self.conv_block1 = ConvBlock_mix(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock_mix(in_channels=65, out_channels=128)
        self.conv_block3 = ConvBlock_mix(in_channels=193, out_channels=256)
        self.conv_block4 = ConvBlock_mix(in_channels=449, out_channels=512)
        self.fc2 = nn.Linear(961, 1024, bias=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(1024, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):

        init_layer(self.fc)

    def forward(self, input):
        '''
        Input: (batch_size, times_steps, freq_bins)'''
        
        x = input[:, None, :, :]
        '''(batch_size, 1, times_steps, freq_bins)'''
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')#（64，64，215，20）
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')#（64，193，107，10）
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')#（64，449，53，5）
        x = self.conv_block4(x, pool_size=(1, 1), pool_type='avg')#（64，961，53，5）
        '''(batch_size, feature_maps, time_steps, freq_bins)'''
        x = torch.mean(x, dim=3)        # (batch_size, feature_maps, time_stpes)
        (x, _) = torch.max(x, dim=2)    # (batch_size, feature_maps)
        x = F.relu_(self.fc2(x))
        x = self.dropout(x)
        x = self.fc(x)
        
        if self.activation == 'logsoftmax':
            output = F.log_softmax(x, dim=-1)
            
        elif self.activation == 'sigmoid':
            output = torch.sigmoid(x)
        
        return output 


    
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ConvBlock_mix2(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(ConvBlock_mix2, self).__init__()
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

        if out_channels == 64:
            self.globalAvgPool = nn.AvgPool2d((100,40), stride=1)
            self.globalAvgPool2 = nn.AvgPool2d((100,64), stride=1)
            self.fc1_2 = nn.Linear(in_features=40, out_features=40)
            self.fc2_2 = nn.Linear(in_features=40, out_features=40)
        elif out_channels == 128:
            self.globalAvgPool = nn.AvgPool2d((50,20), stride=1)
            self.globalAvgPool2 = nn.AvgPool2d((50,128), stride=1)
            self.fc1_2 = nn.Linear(in_features=20, out_features=20)
            self.fc2_2 = nn.Linear(in_features=20, out_features=20)
        elif out_channels == 256:
            self.globalAvgPool = nn.AvgPool2d((25,10), stride=1)
            self.globalAvgPool2 = nn.AvgPool2d((25,256), stride=1)
            self.fc1_2 = nn.Linear(in_features=10, out_features=10)
            self.fc2_2 = nn.Linear(in_features=10, out_features=10)
        elif out_channels == 512:
            self.globalAvgPool = nn.AvgPool2d((12,5), stride=1)
            self.globalAvgPool2 = nn.AvgPool2d((12,512), stride=1)
            self.fc1_2 = nn.Linear(in_features=5, out_features=5)
            self.fc2_2 = nn.Linear(in_features=5, out_features=5)
        self.fc1 = nn.Linear(in_features=out_channels, out_features=round(out_channels / 16))
        self.fc2 = nn.Linear(in_features=round(out_channels / 16), out_features=out_channels)
        self.sigmoid = nn.Sigmoid()
        self.sigmoid2 = nn.Sigmoid()
        self.downsample = conv1x1(in_channels, out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.init_weights()
        
        
    def init_weights(self):
        
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)
        init_bn(self.bn)
        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        res = x
        y = x
        res_2 = x
        x = self.globalAvgPool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu_(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = x.view(x.size(0), x.size(1), 1, 1)
        x = x * res
        h = self.downsample(input)
        h = self.bn(h)
        x += h
        res_2 = res_2.transpose(1,3)
        y = y.transpose(1,3)
        y = self.globalAvgPool2(y)
        y = y.view(y.size(0), -1)
        y = self.fc1_2(y)
        y = F.relu_(y)
        y = self.fc2_2(y)
        y = self.sigmoid2(y)
        y = y.view(y.size(0), y.size(1), 1, 1)
        y = y * res_2
        y = y.transpose(1,3)
        x += y
        x = F.relu_(x)
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        else:
            raise Exception('Incorrect argument!')
        
        return x
class Cnns(nn.Module):
    
    def __init__(self, classes_num=50, activation='logsoftmax'):
        super(Cnns, self).__init__()

        self.activation = activation

        self.conv_block1 = ConvBlock_mix2(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock_mix2(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock_mix2(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock_mix2(in_channels=256, out_channels=512)
        self.fc2 = nn.Linear(512, 512, bias=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(512, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):

        init_layer(self.fc)

    def forward(self, input):
        '''
        Input: (batch_size, times_steps, freq_bins)'''
        
        x = input[:, None, :, :]
        '''(batch_size, 1, times_steps, freq_bins)'''
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')#（64，64，215，20）
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')#（64，193，107，10）
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')#（64，449，53，5）
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')#（64，961，53，5）
        '''(batch_size, feature_maps, time_steps, freq_bins)'''
        x = torch.mean(x, dim=3)        # (batch_size, feature_maps, time_stpes)
        (x, _) = torch.max(x, dim=2)    # (batch_size, feature_maps)
        x = F.relu_(self.fc2(x))
        x = self.dropout(x)
        x = self.fc(x)
        
        if self.activation == 'logsoftmax':
            output = F.log_softmax(x, dim=-1)
            
        elif self.activation == 'sigmoid':
            output = torch.sigmoid(x)
        
        return output  

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        if planes == 64:
            self.globalAvgPool = nn.AvgPool2d(40, stride=1)
        elif planes == 128:
            self.globalAvgPool = nn.AvgPool2d(28, stride=1)
        elif planes == 256:
            self.globalAvgPool = nn.AvgPool2d(14, stride=1)
        elif planes == 512:
            self.globalAvgPool = nn.AvgPool2d(7, stride=1)
        self.fc1 = nn.Linear(in_features=planes, out_features=round(planes / 16))
        self.fc2 = nn.Linear(in_features=round(planes / 16), out_features=planes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        print(x.shape)
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        print(out.shape)
        if self.downsample is not None:
            residual = self.downsample(x)
            print(residual.shape)

        original_out = out
        out = self.globalAvgPool(out)
        print(out.shape)
        out = out.view(out.size(0), -1)
        print(out.shape)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.view(out.size(0), out.size(1), 1, 1)
        out = out * original_out

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        if planes == 64:
            self.globalAvgPool = nn.AvgPool2d(56, stride=1)
        elif planes == 128:
            self.globalAvgPool = nn.AvgPool2d(28, stride=1)
        elif planes == 256:
            self.globalAvgPool = nn.AvgPool2d(14, stride=1)
        elif planes == 512:
            self.globalAvgPool = nn.AvgPool2d(7, stride=1)
        self.fc1 = nn.Linear(in_features=planes * 4, out_features=round(planes / 4))
        self.fc2 = nn.Linear(in_features=round(planes / 4), out_features=planes * 4)
        self.sigmoid = nn.Sigmoid()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        original_out = out
        out = self.globalAvgPool(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.view(out.size(0),out.size(1),1,1)
        out = out * original_out

        out += residual
        out = self.relu(out)

        return out

# class Cnns(nn.Module):

#     def __init__(self, block=BasicBlock, layers=[1,1,1,1], classes_num=50, activation = 'logsoftmax'):
#         self.inplanes = 64
#         super(Cnns, self).__init__()
#         self.activation = activation
#         self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3,
#                                bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
#         self.layer1 = self._make_layer(BasicBlock, 64, 1)
#         self.layer2 = self._make_layer(BasicBlock, 128, 1, stride=2)
#         self.layer3 = self._make_layer(BasicBlock, 256, 1, stride=2)
#         self.layer4 = self._make_layer(BasicBlock, 512, 1, stride=2)
#         self.avgpool = nn.AvgPool2d(7, stride=1)
#         self.fc = nn.Linear(512, classes_num)

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()

#     def _make_layer(self, block, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(planes),
#             )

#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes))

#         return nn.Sequential(*layers)

#     def forward(self, x):
#         print(x.shape)
#         x = x[:, None, :, :]
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)

#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)

#         if self.activation == 'logsoftmax':
#             output = F.log_softmax(x, dim=-1)
            
#         elif self.activation == 'sigmoid':
#             output = torch.sigmoid(x)
        
#         return output
    
    
    
# class Cnns(nn.Module):

#     def __init__(self,  sub_bins=20, sub_stride=10, mel_bins=200, classes_num=50, activation='logsoftmax'):
#         super(Cnns, self).__init__()
        
#         self.activation = activation
#         self.bn1 = nn.BatchNorm2d(1)
#         self.n_max_pool = int(sub_bins / 10)
#         self.sub_bins, self.sub_stride, self.mel_bins = sub_bins, sub_stride, mel_bins
#         self.sub_num = int(self.mel_bins/self.sub_stride) - 1

#         self.conv1 = nn.ModuleList(
#             [nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1) for _ in
#              range(self.sub_num)])
#         self.conv1_bn = nn.ModuleList([nn.BatchNorm2d(64) for _ in range(self.sub_num)])

#         self.conv2 = nn.ModuleList(
#             [nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1) for _ in
#              range(self.sub_num)])
#         self.conv2_bn = nn.ModuleList([nn.BatchNorm2d(64) for _ in range(self.sub_num)])
#         self.mp2 = nn.ModuleList([nn.AvgPool2d((2, 2)) for _ in range(self.sub_num)])
        
#         self.conv3 = nn.ModuleList(
#             [nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1) for _ in
#              range(self.sub_num)])
#         self.conv3_bn = nn.ModuleList([nn.BatchNorm2d(128) for _ in range(self.sub_num)])
 

#         self.conv4 = nn.ModuleList(
#             [nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1) for _ in
#              range(self.sub_num)])
#         self.conv4_bn = nn.ModuleList([nn.BatchNorm2d(128) for _ in range(self.sub_num)])
#         self.mp4 = nn.ModuleList([nn.AvgPool2d((2, 2)) for _ in range(self.sub_num)])

        
#         self.conv5 = nn.ModuleList(
#             [nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1) for _ in
#              range(self.sub_num)])
#         self.conv5_bn = nn.ModuleList([nn.BatchNorm2d(256) for _ in range(self.sub_num)])

#         self.conv6 = nn.ModuleList(
#             [nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1) for _ in
#              range(self.sub_num)])
#         self.conv6_bn = nn.ModuleList([nn.BatchNorm2d(256) for _ in range(self.sub_num)])
#         self.mp6 = nn.ModuleList([nn.AvgPool2d((1, 2)) for _ in range(self.sub_num)])


#         self.fc1 = nn.ModuleList([nn.Linear(256, 32) for _ in range(self.sub_num)])
#         self.drop3 = nn.ModuleList([nn.Dropout(0.3) for _ in range(self.sub_num)])

#         numFCs = int(math.log(self.sub_num * 32, 2))
#         neurons = int(math.pow(2, numFCs))
#         self.fcGlobal = []
#         tempNeurons = int(32 * self.sub_num)
#         while (neurons >= 128):
#             self.fcGlobal.append(nn.Linear(tempNeurons, neurons))
#             self.fcGlobal.append(nn.ReLU(0.3))
#             self.fcGlobal.append(nn.Dropout(0.3))
#             tempNeurons = neurons
#             neurons = int(neurons / 2)
#         self.fcGlobal.append(nn.Linear(tempNeurons, 50))
#         self.fcGlobal = nn.ModuleList(self.fcGlobal)
        
#         self.conv_block1 = ConvBlock_mix(in_channels=1, out_channels=64)
#         self.conv_block2 = ConvBlock_mix(in_channels=64, out_channels=128)
#         self.conv_block3 = ConvBlock_mix(in_channels=128, out_channels=256)
#         self.conv_block4 = ConvBlock_mix(in_channels=256, out_channels=512)
#         self.fc2 = nn.Linear(512, 512)
#         self.dropout = nn.Dropout(p=0.5)
# #         self.fc = nn.Linear(tempNeurons+ 512, 512)
#         self.fc = nn.Linear(tempNeurons, 128)
#         self.dropouth = nn.Dropout(p=0.3)
# #         self.fch = nn.Linear(512, 50)
#         self.fch = nn.Linear(128,50)

#     def forward(self, x):
#         '''shape: (batch_size, 1, freq_bins, time_steps)'''

#         '''first step: get sub spectrogram'''
#         y = x[:,None,:,:]
#         x = x.transpose(1,2)
#         x = x[:,None,:,:]
#         intermediate = []
#         input_var = x

#         # for every sub-spectrogram
#         for i in range(self.sub_num):
#             x = input_var[:, :, i*self.sub_stride:(i+2)*self.sub_stride, :]
#             x = self.conv1[i](x)
#             x = self.conv1_bn[i](x)
#             x = F.relu(x)
#             x = self.conv2[i](x)
#             x = self.conv2_bn[i](x)
#             x = F.relu(x)
#             x = self.mp2[i](x)
#             x = self.conv3[i](x)
#             x = self.conv3_bn[i](x)
#             x = F.relu(x)
#             x = self.conv4[i](x)
#             x = self.conv4_bn[i](x)
#             x = F.relu(x)
#             x = self.mp4[i](x)
#             x = self.conv5[i](x)
#             x = self.conv5_bn[i](x)
#             x = F.relu(x)
#             x = self.conv6[i](x)
#             x = self.conv6_bn[i](x)
#             x = F.relu(x)
#             x = self.mp6[i](x)
                 
#             (x, _) = torch.max(x, dim=3)    
#             x = torch.mean(x, dim=2) 
#             x = self.fc1[i](x)
#             x = F.relu(x)
#             intermediate.append(x)

        
#         # extracted intermediate layers
#         x = torch.cat((intermediate), 1)
#         # global classification
#         for i in range(len(self.fcGlobal)):
#             x = self.fcGlobal[i](x)
        
# #         y = self.conv_block1(y, pool_size=(2, 2), pool_type='avg')#（64，64，215，20）
# #         y = self.conv_block2(y, pool_size=(2, 2), pool_type='avg')#（64，193，107，10）
# #         y = self.conv_block3(y, pool_size=(2, 2), pool_type='avg')#（64，449，53，5）
# #         y = self.conv_block4(y, pool_size=(2, 2), pool_type='avg')#（64，961，53，5）
# #         '''(batch_size, feature_maps, time_steps, freq_bins)'''
# #         y = torch.mean(y, dim=3)        # (batch_size, feature_maps, time_stpes)
# #         (y, _) = torch.max(y, dim=2)    # (batch_size, feature_maps)
# #         y = F.relu_(self.fc2(y))
        
# #         h = torch.cat((x, y), 1)
#         h= x
# #         h = self.fc(h)
# #         h = self.dropouth(h)
# #         h = F.relu_(self.fch(h))
        
#         if self.activation == 'logsoftmax':
#             output = F.log_softmax(h, dim=-1)
            
#         elif self.activation == 'sigmoid':
#             output = torch.sigmoid(h)
        
#         return output
