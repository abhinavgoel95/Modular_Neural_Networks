import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
import time
import numpy as np
import shutil
import os
import argparse
from utils import progress_bar
from torchsummary import summary
from utils1 import load_state_dict_from_url
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
from utils1 import load_state_dict_from_url
import torch.nn.init as init


__all__ = ['SqueezeNet', 'squeezenet1_0', 'squeezenet1_1']

model_urls = {
    'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
    'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth',
}


class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


class SqueezeNet(nn.Module):

    def __init__(self, version='1_0', num_classes=1000):
        super(SqueezeNet, self).__init__()
        self.num_classes = num_classes
        if version == '1_0':
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=7, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256),
            )
        else:
            # FIXME: Is this needed? SqueezeNet should only be called from the
            # FIXME: squeezenet1_x() functions
            # FIXME: This checking is not done for the other models
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1_0 or 1_1 expected".format(version=version))

        # Final convolution is initialized differently from the rest
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)


def _squeezenet(version, pretrained, progress, **kwargs):
    model = SqueezeNet(version, **kwargs)
    if pretrained:
        arch = 'squeezenet' + version
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def squeezenet1_0(pretrained=False, progress=True, **kwargs):
    r"""SqueezeNet model architecture from the `"SqueezeNet: AlexNet-level
    accuracy with 50x fewer parameters and <0.5MB model size"
    <https://arxiv.org/abs/1602.07360>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _squeezenet('1_0', pretrained, progress, **kwargs)


def train(model, optimizer, loss_fn, trainloader, valloader, device):
       max = 0.62
       for epoch in range(30):
           model.train()
           count = 0
           total = 0
           for batch_num, (data, target) in enumerate(trainloader):
               data = data.to(device)
               target = target.to(device)
               target1 = target.clone().to(device)
               optimizer.zero_grad()
               net_out = model(data)
               loss = loss_fn(net_out, target1)
               loss.backward()
               optimizer.step()
               pred = net_out.max(1, keepdim=True)[1]
               count += pred.eq(target1.view_as(pred)).sum().item()
               total += target.size(0)
               progress_bar(batch_num, len(trainloader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                % (loss / (batch_num + 1), 100. * count / total, count, total))

           print("Epoch: ",epoch+1,"\nTrain: ", count/len(trainloader.sampler.indices))
           if count/len(trainloader.sampler.indices) > max:
               print("model saved")
               max = count/len(trainloader.sampler.indices)
               torch.save(model.state_dict(), "model_vgg.pth")

           del data, target

           model.eval()
           count1 = 0
           for data, target in valloader:
               data = data.to(device)
               target = target.to(device)
               net_out = model(data)

               pred = net_out.max(1, keepdim=True)[1]
               count1 += pred.eq(target.view_as(pred)).sum().item()

           print("Val: ", count1/len(valloader.sampler.indices))

def main():
       train_root = '/path/to/dataset/'
       normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

       train_data = datasets.ImageFolder(train_root, transform=transforms.Compose([
                   transforms.RandomResizedCrop(224),
                   transforms.RandomHorizontalFlip(),
                   transforms.ToTensor(),
                   normalize
       ]))

       shuffle_dataset = True
       val_split = 0.1

       dataset_size = len(train_data)
       indices = list(range(dataset_size))
       split = int(np.floor(val_split * dataset_size))

       if shuffle_dataset:
               np.random.seed(2)
               np.random.shuffle(indices)

       train_indices, val_indices = indices[split:], indices[:split]

       train_sampler = SubsetRandomSampler(train_indices)
       val_sampler = SubsetRandomSampler(val_indices)

       train_loader = torch.utils.data.DataLoader(train_data,
       sampler = train_sampler,
       batch_size = 100)

       val_loader = torch.utils.data.DataLoader(train_data,
       sampler = val_sampler,
       batch_size = 100)

       model = squeezenet1_0(pretrained=True).to(torch.device("cuda"))
       
       for param in model.parameters():
           param.requires_grad = False

       model.classifier[-3] = nn.Conv2d(512, 20, kernel_size=1).cuda()

       for param in model.classifier[-3].parameters():
           param.requires_grad = True

       learning_rate = 0.001
       optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
       loss_fn = nn.CrossEntropyLoss()
       torch.save(model.state_dict(), "model_squeeze.pth")
       train(model, optimizer, loss_fn, train_loader, val_loader, torch.device("cuda"))
       
       for i in range(3):
           learning_rate /= 10
           optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 5e-4)
           loss_fn = nn.CrossEntropyLoss()
           train(model, optimizer, loss_fn, train_loader, val_loader, torch.device("cuda"))

if __name__== "__main__":
   main()
