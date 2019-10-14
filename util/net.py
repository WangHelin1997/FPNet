import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils import weight_norm

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
        
    def show(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        x = torch.mean(x, dim=1)
        return x
        
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


class Conv2dSame(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, padding_layer=torch.nn.ReplicationPad2d):
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = torch.nn.Sequential(
            padding_layer((ka,kb,ka,kb)),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias)
        )
    def forward(self, x):
        return self.net(x)
    
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
#         self.conv1 = nn.Conv2d(in_channels=in_channels, 
#                               out_channels=out_channels,
#                               kernel_size=(3, 3), stride=(1, 1),
#                               padding=(1, 1), bias=False)
                              
#         self.conv2 = nn.Conv2d(in_channels=out_channels, 
#                              out_channels=out_channels,
#                                  kernel_size=(3, 3), stride=(1, 1),
#                              padding=(1, 1), bias=False)
        self.conv1 = Conv2dSame(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=3, bias=False)
                              
        self.conv2 = Conv2dSame(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=3, bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if out_channels == 64:
#             self.globalAvgPool = nn.AvgPool2d((100,40), stride=1)
            self.globalAvgPool2 = nn.AvgPool2d((100,64), stride=1)
            self.globalAvgPool3 = nn.AvgPool2d((64,40), stride=1)
            self.fc1_2 = nn.Linear(in_features=40, out_features=40)
            self.fc2_2 = nn.Linear(in_features=40, out_features=40)
        elif out_channels == 128:
#             self.globalAvgPool = nn.AvgPool2d((50,20), stride=1)
            self.globalAvgPool2 = nn.AvgPool2d((50,128), stride=1)
            self.globalAvgPool3 = nn.AvgPool2d((128,20), stride=1)
            self.fc1_2 = nn.Linear(in_features=20, out_features=20)
            self.fc2_2 = nn.Linear(in_features=20, out_features=20)
        elif out_channels == 256:
#             self.globalAvgPool = nn.AvgPool2d((25,10), stride=1)
            self.globalAvgPool2 = nn.AvgPool2d((25,256), stride=1)
            self.globalAvgPool3 = nn.AvgPool2d((256,10), stride=1)
            self.fc1_2 = nn.Linear(in_features=10, out_features=10)
            self.fc2_2 = nn.Linear(in_features=10, out_features=10)
        elif out_channels == 512:
#             self.globalAvgPool = nn.AvgPool2d((12,5), stride=1)
            self.globalAvgPool2 = nn.AvgPool2d((12,512), stride=1)
            self.globalAvgPool3 = nn.AvgPool2d((512,5), stride=1)
            self.fc1_2 = nn.Linear(in_features=5, out_features=5)
            self.fc2_2 = nn.Linear(in_features=5, out_features=5)
#         self.fc1 = nn.Linear(in_features=out_channels, out_features=round(out_channels / 16))
#         self.fc2 = nn.Linear(in_features=round(out_channels / 16), out_features=out_channels)
        self.lstm = nn.LSTM(input_size=1, hidden_size=1, num_layers=1, batch_first=True, bidirectional=False)
        self.sigmoid = nn.Sigmoid()
        self.sigmoid2 = nn.Sigmoid()
        self.downsample = conv1x1(in_channels, out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.init_weights()
        
        
    def init_weights(self):
        
        #init_layer(self.conv1)
        #init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)
        init_bn(self.bn)
    def show(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        res = x
        y = x
        res_2 = x
        z = x
        res_3 = x
#         x = self.globalAvgPool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc1(x)
#         x = F.relu_(x)
#         x = self.fc2(x)
#         x = self.sigmoid(x)
#         x = x.view(x.size(0), x.size(1), 1, 1)
#         x = x * res
#         x1= x
#         x1 = torch.mean(x1, dim=1)
        h = self.downsample(input)
        h = self.bn(h)
#         x += h
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
        x2 = y
        x2 = torch.mean(x2, dim=1)
        x = y + h
        res_3 = res_3.transpose(1,2)
        z = z.transpose(1,2)
        z = self.globalAvgPool3(z)
        z = z.view(z.size(0), -1)
        z = z[:,:,None]
        out, hidden = self.lstm(z, None)
        z = out.view(z.size(0), z.size(1), z.size(2), 1)
        z = z * res_3
        z = z.transpose(1,2)
        x += z
        x3 = z
        x3 = torch.mean(x3, dim=1)
        x4 = h
        x4 = torch.mean(x4, dim=1)
        x5 = x
        x5 = torch.mean(x5, dim=1)
        x6 = res
        x6 = torch.mean(x6, dim=1)
        return x2, x3, x4, x5, x6
        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        res = x
        y = x
        res_2 = x
        z = x
        res_3 = x
#         x = self.globalAvgPool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc1(x)
#         x = F.relu_(x)
#         x = self.fc2(x)
#         x = self.sigmoid(x)
#         x = x.view(x.size(0), x.size(1), 1, 1)
#         x = x * res
#         x1= x
#         x1 = torch.mean(x1, dim=1)
        h = self.downsample(input)
        h = self.bn(h)
#         x += h
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
        x = y + h
        res_3 = res_3.transpose(1,2)
        z = z.transpose(1,2)
        z = self.globalAvgPool3(z)
        z = z.view(z.size(0), -1)
        z = z[:,:,None]
        print(z.shape)
        out, hidden = self.lstm(z, None)
        print(out.shape)
        z = out.view(z.size(0), z.size(1), z.size(2), 1)
        z = z * res_3
        z = z.transpose(1,2)
        x += z
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
        
    def show(self, input):
       
        x = input[:, None, :, :]
        '''(batch_size, 1, times_steps, freq_bins)'''
        b ,c, d, e, f = self.conv_block1.show(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')#（64，64，215，20）
        x1 = torch.mean(x, dim=1)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')#（64，193，107，10）
        x2 = torch.mean(x, dim=1)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')#（64，449，53，5）
        x3 = torch.mean(x, dim=1)
        x = self.conv_block4(x, pool_size=(1, 1), pool_type='avg')#（64，961，53，5）
        x4 = torch.mean(x, dim=1)
        return x1, x2, x3, x4, b, c, d, e, f
    
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
class ConvBlock_mix3(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(ConvBlock_mix3, self).__init__()
        self.conv1 = Conv2dSame(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=3, bias=False)
                              
        self.conv2 = Conv2dSame(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=3, bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.init_weights()
        
    def init_weights(self):
        
        init_bn(self.bn1)
        init_bn(self.bn2)
        
    def show(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        x = torch.mean(x, dim=1)
        return x
        
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
    
class Predict_Depth(nn.Module):
    def __init__(self, in_channels=1, time=100, frequency=40):
        
        super(Predict_Depth, self).__init__()
        self.globalAvgpooling = nn.AvgPool2d((time,in_channels), stride=1)
        self.fc1 = nn.Linear(frequency, frequency, bias=True)
        self.fc2 = nn.Linear(frequency, frequency, bias=True) 
        self.sigmoid = nn.Sigmoid()
        
        
    def forward(self, input):
        
        x = input
        x = x.transpose(1, 3)
        x = self.globalAvgpooling(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu_(x)
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = x.view(x.size(0), x.size(1), 1, 1)
        return x
    
class Cnns2(nn.Module):
    
    def __init__(self, classes_num=50, activation='logsoftmax'):
        super(Cnns2, self).__init__()

        self.activation = activation
#         self.depth = Predict_Depth(in_channels=1, time=100, frequency=40)
        self.conv_block1 = ConvBlock_mix(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock_mix(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock_mix(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock_mix(in_channels=256, out_channels=512)
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
#         res = x
#         x = self.depth(x)
#         res = res.transpose(1, 3)
#         x = x * res
#         x = x.transpose(1, 3)
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

# class Chomp1d(nn.Module):
#     def __init__(self, chomp_size):
#         super(Chomp1d, self).__init__()
#         self.chomp_size = chomp_size

#     def forward(self, x):
#         return x[:, :, :-self.chomp_size].contiguous()
    
# class AttentionBlock(nn.Module):
#     def __init__(self, dims=40, k_size=40, v_size=40, seq_len=None):
#         super(AttentionBlock, self).__init__()
#         self.key_layer = nn.Linear(dims, k_size)
#         self.query_layer = nn.Linear(dims, k_size)
#         self.value_layer = nn.Linear(dims, v_size)
#         self.sqrt_k = math.sqrt(k_size)

#     def forward(self, minibatch):
#         keys = self.key_layer(minibatch)
#         queries = self.query_layer(minibatch)
#         values = self.value_layer(minibatch)
#         logits = torch.bmm(queries, keys.transpose(2,1))
#         # Use numpy triu because you can't do 3D triu with PyTorch
#         # TODO: using float32 here might break for non FloatTensor inputs.
#         # Should update this later to use numpy/PyTorch types of the input.
#         mask = np.triu(np.ones(logits.size()), k=1).astype('uint8')
#         mask = torch.from_numpy(mask).cuda()
#         # do masked_fill_ on data rather than Variable because PyTorch doesn't
#         # support masked_fill_ w/-inf directly on Variables for some reason.
#         logits.data.masked_fill_(mask, float('-inf'))
#         probs = F.softmax(logits, dim=1) / self.sqrt_k
#         read = torch.bmm(probs, values)
#         return minibatch + read

# class TemporalBlock(nn.Module):
#     def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
#         super(TemporalBlock, self).__init__()
#         self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
#                                            stride=stride, padding=padding, dilation=dilation))
#         self.chomp1 = Chomp1d(padding)
#         self.relu1 = nn.ReLU()
#         self.dropout1 = nn.Dropout(dropout)

#         self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
#                                            stride=stride, padding=padding, dilation=dilation))
#         self.chomp2 = Chomp1d(padding)
#         self.relu2 = nn.ReLU()
#         self.dropout2 = nn.Dropout(dropout)

#         self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
#                                  self.conv2, self.chomp2, self.relu2, self.dropout2)
#         self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
#         self.relu = nn.ReLU()
#         self.init_weights()
#         self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

#     def init_weights(self):
#         self.conv1.weight.data.normal_(0, 0.01)
#         self.conv2.weight.data.normal_(0, 0.01)
#         if self.downsample is not None:
#             self.downsample.weight.data.normal_(0, 0.01)

#     def forward(self, x):
#         out = self.net(x)
#         res = x if self.downsample is None else self.downsample(x)
#         return self.relu(out + res)


# class TemporalConvNet(nn.Module):
#     def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, attention=False):
#         super(TemporalConvNet, self).__init__()
#         layers = []
#         num_levels = len(num_channels)
#         for i in range(num_levels):
#             dilation_size = 2 ** i
#             in_channels = num_inputs if i == 0 else num_channels[i-1]
#             out_channels = num_channels[i]
#             layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
#                                      padding=(kernel_size-1) * dilation_size, dropout=dropout)]
#             if attention == True:
#                 layers += [AttentionBlock(out_channels, out_channels, out_channels)]

#         self.network = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.network(x)
