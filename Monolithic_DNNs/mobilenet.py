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
from thop import profile


__all__ = ['MobileNetV2', 'mobilenet_v2']


model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
        """
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x


def mobilenet_v2(pretrained=False, progress=True, **kwargs):
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV2(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def train(model, valloader, device):
           model.eval()
           count1 = 0
           for data, target in valloader:
               data = data.to(device)
               target = target.to(device)
               net_out = model(data)

               pred = net_out.max(1, keepdim=True)[1]
               count1 += pred.eq(target.view_as(pred)).sum().item()

           print("Val: ", count1/len(valloader.sampler.indices))


def train1(model, optimizer, loss_fn, trainloader, valloader, device):
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
                torch.save(model.state_dict(), "model_mobile.pth")

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

       model = mobilenet_v2(pretrained=True).to(torch.device("cuda"))
       
       for param in model.parameters():
           param.requires_grad = False

       model.classifier[-1] = nn.Linear(1280, 20).cuda()
       #torch.save(model.state_dict(), "model_mobile.pth")
       input = torch.randn(1, 3, 224, 224).cuda()
       
       learning_rate = 0.0001
       optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
       loss_fn = nn.CrossEntropyLoss()
       train1(model, optimizer, loss_fn, train_loader, val_loader, torch.device("cuda"))

if __name__== "__main__":
   main()
