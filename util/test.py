import torch
import numpy as np
from net import Cnns,Cnns2
from feature import *
from matplotlib import pyplot as plt
    
def data_pre(audio, audio_length, fs, audio_skip):
    stride = int(audio_skip * fs /2)
    loop =  int((audio_length * fs) // stride - 1)
    area = 0
    maxamp = 0.
    i = 0
    out = audio
    while i < loop:
        win_data = out[i*stride: (i+2)*stride]
        maxamp = np.max(np.abs(win_data))
        if maxamp < 0.005:
            loop = loop - 2
            out[i*stride: (loop+1)*stride] = out[(i+2)*stride: (loop+3)*stride]
        else:
            i = i + 1
    length = (audio_length * fs) // stride - loop - 1
    if length == 0:
        return out
    else:
        out[(loop + 1) * stride:(audio_length * fs // stride) * stride] = out[0:length * stride]
        if length < (audio_length * fs//stride)/2:
            out[(loop+1)*stride:(audio_length * fs//stride)*stride] = out[0:length*stride]
            return out
        else:
            out[(loop + 1) * stride:(loop + 1)*2  * stride] = out[0:(loop + 1) * stride]
            return data_pre(out, audio_length, fs, audio_skip)


if __name__ == '__main__':
    Model = eval('Cnns2')
    model = Model(50, activation='logsoftmax')
    checkpoint_path = '2700_iterations.pth'
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    for name, param in model.named_parameters():
        print(name)
        
    audio_path = '1-34094-A-5.wav'
    (audio, _) = read_audio(audio_path=audio_path, target_fs=44100)
    audio = pad_truncate_sequence(audio, 44100*5)
    audio = data_pre(audio=audio, audio_length=5, fs=44100, audio_skip=0.1)
    feature_extractor = LogMelExtractor(
        sample_rate=44100, 
        window_size=1764, 
        hop_size=882, 
        mel_bins=40, 
        fmin=50, 
        fmax=22050)
    feature = feature_extractor.transform(audio)
    feature = feature[0 : 100]
    x = np.transpose(feature, (1, 0))
    plt.imshow(x, cmap = plt.cm.jet)
    plt.savefig("original.png")
    plt.show()
    feature = torch.from_numpy(feature[None, :, :])
#     x1, x2, x3, x4, b, c, d, e, f = model.show(feature)
    x1, x2, x3, x4, a= model.show(feature)
    
#     feature = torch.squeeze(x1)
#     x1 = np.transpose(feature.detach().numpy(), (1, 0))
#     plt.imshow(x1, cmap = plt.cm.jet)
#     plt.savefig("conv1.png")
#     plt.show()
    
#     feature = torch.squeeze(x2)
#     x2 = np.transpose(feature.detach().numpy(), (1, 0))
#     plt.imshow(x2)
#     plt.savefig("conv2.png")
#     plt.show()
    
#     feature = torch.squeeze(x3)
#     x3 = np.transpose(feature.detach().numpy(), (1, 0))
#     plt.imshow(x3)
#     plt.savefig("conv3.png")
#     plt.show()
    
#     feature = torch.squeeze(x4)
#     x4 = np.transpose(feature.detach().numpy(), (1, 0))
#     plt.imshow(x4)
#     plt.savefig("conv4.png")
#     plt.show()
    
    feature = torch.squeeze(a)
    x1 = np.transpose(feature.detach().numpy(), (1, 0))
    plt.imshow(x1, cmap = plt.cm.jet)
    plt.savefig("conv1_global.png")
    plt.show()
    
#     feature = torch.squeeze(b)
#     x2 = np.transpose(feature.detach().numpy(), (1, 0))
#     plt.imshow(x2)
#     plt.savefig("conv1_fre.png")
#     plt.show()
    
#     feature = torch.squeeze(c)
#     x3 = np.transpose(feature.detach().numpy(), (1, 0))
#     plt.imshow(x3)
#     plt.savefig("conv1_time.png")
#     plt.show()
    
#     feature = torch.squeeze(d)
#     x4 = np.transpose(feature.detach().numpy(), (1, 0))
#     plt.imshow(x4)
#     plt.savefig("conv1_ins.png")
#     plt.show()
    
#     feature = torch.squeeze(e)
#     x5 = np.transpose(feature.detach().numpy(), (1, 0))
#     plt.imshow(x5)
#     plt.savefig("conv1_concat.png")
#     plt.show()
    
#     feature = torch.squeeze(f)
#     x6 = np.transpose(feature.detach().numpy(), (1, 0))
#     plt.imshow(x6)
#     plt.savefig("conv1_res.png")
#     plt.show()