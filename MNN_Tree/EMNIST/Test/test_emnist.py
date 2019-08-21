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
from torchsummary import summary
from thop import profile

m = nn.Softmax()

cfg = {
    '0_0': [32, 48, 'M'],
    '0_3': [32, 32, 'M'],
    '0_5': [32, 32, 'M'],
    '0': [16, 32, 'M', 'D'],
    '1': [16, 32, 'M', 'D'],
    '1_0': [32, 64, 'M'],
    'root': [16, 'M', 16, 'D'],
    '3': [16, 32, 'M'],
    '3_0': [32, 32, 'M'],
    '4': [16, 32,'M'],
    '6': [16, 32, 'M'],
    '8': [16, 32, 'M'],
    '8_0': [32, 32, 'M'],
    '9': [16, 32, 'M'],
    '10': [16, 32, 'M'],
}

class model(nn.Module):
    def __init__(self, size):
        super(model, self).__init__()
        self.features = self._make_layers(cfg[size])
        self.classifier = nn.Sequential(
                        nn.Linear(16*14*14, 14),
                )

    def forward(self, x):
        y = self.features(x)
        x = y.view(y.size(0), -1)
        out = self.classifier(x)
        return y,out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 1
        for x in cfg:
            if x == 'D':
                layers += [nn.Dropout()]
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1), nn.BatchNorm2d(x), nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

def model_root():
    return model('root')

class mod_zero(nn.Module):
    def __init__(self, size):
        super(mod_zero, self).__init__()
        self.features = self._make_layers(cfg[size], 16)
        self.classifier = nn.Sequential(
                        nn.Linear(32*7*7, 8),
                )

    def forward(self, x):
        y = self.features(x)
        x = y.view(y.size(0), -1)
        out = self.classifier(x)
        return y,out

    def _make_layers(self, cfg, channels = 3):
        layers = []
        in_channels = channels
        for x in cfg:
            if x == 'D':
                layers += [nn.Dropout()]
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1), nn.BatchNorm2d(x), nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

def model_0():
    return mod_zero('0')


class mod_zero_three(nn.Module):
    def __init__(self, size):
        super(mod_zero_three, self).__init__()
        self.features = self._make_layers(cfg[size], 32)
        self.classifier = nn.Sequential(
                        nn.Linear(32*3*3, 2),
                )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return x,out

    def _make_layers(self, cfg, channels = 3):
        layers = []
        in_channels = channels
        for x in cfg:
            if x == 'D':
                layers += [nn.Dropout()]
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1), nn.BatchNorm2d(x), nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

def model_0_3():
    return mod_zero_three('0_3')

class mod_zero_zero(nn.Module):
    def __init__(self, size):
        super(mod_zero_zero, self).__init__()
        self.features = self._make_layers(cfg[size], 32)
        self.classifier = nn.Sequential(
                        nn.Linear(48*3*3, 2),
                )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return x,out

    def _make_layers(self, cfg, channels = 3):
        layers = []
        in_channels = channels
        for x in cfg:
            if x == 'D':
                layers += [nn.Dropout()]
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1), nn.BatchNorm2d(x), nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

def model_0_0():
    return mod_zero_zero('0_0')

class mod_zero_five(nn.Module):
    def __init__(self, size):
        super(mod_zero_five, self).__init__()
        self.features = self._make_layers(cfg[size], 32)
        self.classifier = nn.Sequential(
                        nn.Linear(32*3*3, 3),
                )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return x,out

    def _make_layers(self, cfg, channels = 3):
        layers = []
        in_channels = channels
        for x in cfg:
            if x == 'D':
                layers += [nn.Dropout()]
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1), nn.BatchNorm2d(x), nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

def model_0_5():
    return mod_zero_five('0_5')

class mod_one(nn.Module):
    def __init__(self, size):
        super(mod_one, self).__init__()
        self.features = self._make_layers(cfg[size], 16)
        #self.features_down = self._make_layers(cfg_down[size], 32)
        self.classifier = nn.Sequential(
                        nn.Linear(32*7*7, 2),
                )

    def forward(self, x):
        y = self.features(x)
        #x  = self.features_down(y)
        x = y.view(y.size(0), -1)
        out = self.classifier(x)
        return y,out

    def _make_layers(self, cfg, channels = 3):
        layers = []
        in_channels = channels
        for x in cfg:
            if x == 'D':
                layers += [nn.Dropout()]
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1), nn.BatchNorm2d(x), nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

def model_1():
    return mod_one('1')


class mod_one_zero(nn.Module):
    def __init__(self, size):
        super(mod_one_zero, self).__init__()
        self.features = self._make_layers(cfg[size], 32)
        self.classifier = nn.Sequential(
                        nn.Linear(64*3*3, 3),
                )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return x,out

    def _make_layers(self, cfg, channels = 3):
        layers = []
        in_channels = channels
        for x in cfg:
            if x == 'D':
                layers += [nn.Dropout()]
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1), nn.BatchNorm2d(x), nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

def model_1_0():
    return mod_one_zero('1_0')

class mod_three(nn.Module):
    def __init__(self, size):
        super(mod_three, self).__init__()
        self.features = self._make_layers(cfg[size], 16)
        #self.features_down = self._make_layers(cfg_down[size], 32)
        self.classifier = nn.Sequential(
                        nn.Linear(32*7*7, 12),
                )

    def forward(self, x):
        y = self.features(x)
        #x  = self.features_down(y)
        x = y.view(y.size(0), -1)
        out = self.classifier(x)
        return y,out

    def _make_layers(self, cfg, channels = 3):
        layers = []
        in_channels = channels
        for x in cfg:
            if x == 'D':
                layers += [nn.Dropout()]
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1), nn.BatchNorm2d(x), nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

def model_3():
    return mod_three('3')


class mod_three_zero(nn.Module):
    def __init__(self, size):
        super(mod_three_zero, self).__init__()
        self.features = self._make_layers(cfg[size], 32)
        self.classifier = nn.Sequential(
                        nn.Linear(32*3*3, 2),
                )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return x,out

    def _make_layers(self, cfg, channels = 3):
        layers = []
        in_channels = channels
        for x in cfg:
            if x == 'D':
                layers += [nn.Dropout()]
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1), nn.BatchNorm2d(x), nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

def model_3_0():
    return mod_three_zero('3_0')

class mod_four(nn.Module):
    def __init__(self, size):
        super(mod_four, self).__init__()
        self.features = self._make_layers(cfg[size], 16)
        #self.features_down = self._make_layers(cfg_down[size], 32)
        self.classifier = nn.Sequential(
                        nn.Linear(32*7*7, 2),
                )

    def forward(self, x):
        x = self.features(x)
        #x  = self.features_down(y)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return x,out

    def _make_layers(self, cfg, channels = 3):
        layers = []
        in_channels = channels
        for x in cfg:
            if x == 'D':
                layers += [nn.Dropout()]
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1), nn.BatchNorm2d(x), nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

def model_4():
    return mod_four('4')

class mod_six(nn.Module):
    def __init__(self, size):
        super(mod_six, self).__init__()
        self.features = self._make_layers(cfg[size], 16)
        #self.features_down = self._make_layers(cfg_down[size], 32)
        self.classifier = nn.Sequential(
                        nn.Linear(32*7*7, 2),
                )

    def forward(self, x):
        x = self.features(x)
        #x  = self.features_down(y)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return x,out

    def _make_layers(self, cfg, channels = 3):
        layers = []
        in_channels = channels
        for x in cfg:
            if x == 'D':
                layers += [nn.Dropout()]
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1), nn.BatchNorm2d(x), nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

def model_6():
    return mod_six('6')

class mod_eight(nn.Module):
    def __init__(self, size):
        super(mod_eight, self).__init__()
        self.features = self._make_layers(cfg[size], 16)
        #self.features_down = self._make_layers(cfg_down[size], 32)
        self.classifier = nn.Sequential(
                        nn.Linear(32*7*7, 3),
                )

    def forward(self, x):
        y = self.features(x)
        #x  = self.features_down(y)
        x = y.view(y.size(0), -1)
        out = self.classifier(x)
        return y,out

    def _make_layers(self, cfg, channels = 3):
        layers = []
        in_channels = channels
        for x in cfg:
            if x == 'D':
                layers += [nn.Dropout()]
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1), nn.BatchNorm2d(x), nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

def model_8():
    return mod_eight('8')

class mod_eight_zero(nn.Module):
    def __init__(self, size):
        super(mod_eight_zero, self).__init__()
        self.features = self._make_layers(cfg[size], 32)
        self.classifier = nn.Sequential(
                        nn.Linear(32*3*3, 2),
                )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return x,out

    def _make_layers(self, cfg, channels = 3):
        layers = []
        in_channels = channels
        for x in cfg:
            if x == 'D':
                layers += [nn.Dropout()]
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1), nn.BatchNorm2d(x), nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

def model_8_0():
    return mod_eight_zero('8_0')

class mod_nine(nn.Module):
    def __init__(self, size):
        super(mod_nine, self).__init__()
        self.features = self._make_layers(cfg[size], 16)
        #self.features_down = self._make_layers(cfg_down[size], 32)
        self.classifier = nn.Sequential(
                        nn.Linear(32*7*7, 2),
                )

    def forward(self, x):
        x = self.features(x)
        #x  = self.features_down(y)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return x,out

    def _make_layers(self, cfg, channels = 3):
        layers = []
        in_channels = channels
        for x in cfg:
            if x == 'D':
                layers += [nn.Dropout()]
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1), nn.BatchNorm2d(x), nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

def model_9():
    return mod_nine('9')

class mod_ten(nn.Module):
    def __init__(self, size):
        super(mod_ten, self).__init__()
        self.features = self._make_layers(cfg[size], 16)
        #self.features_down = self._make_layers(cfg_down[size], 32)
        self.classifier = nn.Sequential(
                        nn.Linear(32*7*7, 2),
                )

    def forward(self, x):
        x = self.features(x)
        #x  = self.features_down(y)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return x,out

    def _make_layers(self, cfg, channels = 3):
        layers = []
        in_channels = channels
        for x in cfg:
            if x == 'D':
                layers += [nn.Dropout()]
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1), nn.BatchNorm2d(x), nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

def model_10():
    return mod_ten('10')


def test(model, model_zero, model_one, model_three, model_four, model_six, model_eight, model_nine, model_ten, model_zero_zero, model_zero_three, model_zero_five, model_one_zero, model_three_zero, model_eight_zero, valloader, device):
        model.eval()
        model_zero.eval()
        model_one.eval()
        model_three.eval()
        model_four.eval()
        model_six.eval()
        model_eight.eval()
        model_nine.eval()
        model_ten.eval()
        model_zero_zero.eval()
        model_zero_three.eval()
        model_zero_five.eval()
        model_one_zero.eval()
        model_eight_zero.eval()
        count = 0
        root = {2:3, 5:7, 7:14, 11:25, 12:38, 13:45}
        zero = {1:13,2:36,4:26,6:8,7:11}
        zero_zero = {0:0,1:24}
        zero_three = {0:2,1:35}
        zero_five = {0:44,1:41,2:9}
        one = {1:19}
        one_zero = {0:1,1:18,2:21}
        eight = {1:46,2:29}
        eight_zero = {0:15,1:40}
        three = dict(zip(range(1,12), [17, 30, 10, 31, 33, 20, 27, 42, 37, 6, 16]))
        three_zero = {0:4,1:34}
        four = {0:5, 1:28}
        six = {0:12, 1:39}
        nine = {0:22, 1:43}
        ten = {0:23, 1:32}
        for data, target in valloader:
            data = data.to(device)
            target = target.to(device)
            next_data, root_out = model(data)
            root_out = root_out.max(1, keepdim=True)[1]

            for i in root:
                root_out1 = root_out.clone().to(device)
                indices = (root_out == i).nonzero()[:,0]
                out_target = target[indices]
                root_out1[(root_out1 == i).nonzero()] = root[i]
                count += root_out1[indices].eq(out_target.view_as(root_out1[indices])).sum().item()
                del root_out1

            indices = (root_out == 0).nonzero()[:,0]
            zero_data = next_data[indices]
            zero_target = target[indices]
            next_zero, zero_out = model_zero(zero_data)
            zero_out = zero_out.max(1, keepdim=True)[1]
            for i in zero:
                zero_out1 = zero_out.clone().to(device)
                indices = (zero_out == i).nonzero()[:,0]
                out_target = zero_target[indices]
                zero_out1[(zero_out1 == i).nonzero()] = zero[i]
                count += zero_out1[indices].eq(out_target.view_as(zero_out1[indices])).sum().item()

            if (zero_out == 0).nonzero().shape[0] != 0:
                indices = (zero_out == 0).nonzero()[:,0]
                zero_data = next_zero[indices]
                zero_zero_target = zero_target[indices]
                _, zero_zero_out = model_zero_zero(zero_data)
                zero_zero_out = zero_zero_out.max(1, keepdim=True)[1]
                for i in zero_zero:
                    zero_out1 = zero_zero_out.clone().to(device)
                    indices = (zero_zero_out == i).nonzero()[:,0]
                    out_target = zero_zero_target[indices]
                    zero_out1[(zero_out1 == i).nonzero()] = zero_zero[i]
                    count += zero_out1[indices].eq(out_target.view_as(zero_out1[indices])).sum().item()

            if (zero_out == 3).nonzero().shape[0] != 0:
                indices3 = (zero_out == 3).nonzero()[:,0]
                zero_data = next_zero[indices]
                zero_three_target = zero_target[indices]
                _, zero_three_out = model_zero_three(zero_data)
                zero_three_out = zero_three_out.max(1, keepdim=True)[1]
                for i in zero_three:
                    zero_out1 = zero_three_out.clone().to(device)
                    indices = (zero_three_out == i).nonzero()[:,0]
                    out_target = zero_three_target[indices]
                    zero_out1[(zero_out1 == i).nonzero()] = zero_three[i]
                    count += zero_out1[indices].eq(out_target.view_as(zero_out1[indices])).sum().item()

            if (zero_out == 5).nonzero().shape[0] != 0:
                indices = (zero_out == 5).nonzero()[:,0]
                zero_data = next_zero[indices]
                zero_five_target = zero_target[indices]
                _, zero_five_out = model_zero_five(zero_data)
                zero_five_out = zero_five_out.max(1, keepdim=True)[1]
                for i in zero_five:
                    zero_out1 = zero_five_out.clone().to(device)
                    indices = (zero_five_out == i).nonzero()[:,0]
                    out_target = zero_five_target[indices]
                    zero_out1[(zero_out1 == i).nonzero()] = zero_five[i]
                    count += zero_out1[indices].eq(out_target.view_as(zero_out1[indices])).sum().item()

            indices = (root_out == 1).nonzero()[:,0]
            one_data = next_data[indices]
            one_target = target[indices]
            next_one, one_out = model_one(one_data)
            one_out = one_out.max(1, keepdim=True)[1]
            for i in one:
                one_out1 = one_out.clone().to(device)
                indices = (one_out == i).nonzero()[:,0]
                out_target = one_target[indices]
                one_out1[(one_out1 == i).nonzero()] = one[i]
                count += one_out1[indices].eq(out_target.view_as(one_out1[indices])).sum().item()
                del one_out1

            if (one_out == 0).nonzero().shape[0] != 0:
                indices = (one_out == 0).nonzero()[:,0]
                zero_data = next_one[indices]
                zero_target = one_target[indices]
                _, zero_out = model_one_zero(zero_data)
                zero_out = zero_out.max(1, keepdim=True)[1]
                for i in one_zero:
                    zero_out1 = zero_out.clone().to(device)
                    indices = (zero_out == i).nonzero()[:,0]
                    out_target = zero_target[indices]
                    zero_out1[(zero_out1 == i).nonzero()] = one_zero[i]
                    count += zero_out1[indices].eq(out_target.view_as(zero_out1[indices])).sum().item()
            
            indices = (root_out == 3).nonzero()[:,0]
            three_data = next_data[indices]
            three_target = target[indices]
            next_three, three_out = model_three(three_data)
            three_out = three_out.max(1, keepdim=True)[1]
            for i in three:
                three_out1 = three_out.clone().to(device)
                indices = (three_out == i).nonzero()[:,0]
                out_target = three_target[indices]
                three_out1[(three_out1 == i).nonzero()] = three[i]
                count += three_out1[indices].eq(out_target.view_as(three_out1[indices])).sum().item()
            if (three_out == 0).nonzero().shape[0] != 0:
                indices = (three_out == 0).nonzero()[:,0]
                zero_data = next_three[indices]
                zero_target = three_target[indices]
                _, zero_out = model_three_zero(zero_data)
                zero_out = zero_out.max(1, keepdim=True)[1]
                for i in three_zero:
                    zero_out1 = zero_out.clone().to(device)
                    indices = (zero_out == i).nonzero()[:,0]
                    out_target = zero_target[indices]
                    zero_out1[(zero_out1 == i).nonzero()] = three_zero[i]
                    count += zero_out1[indices].eq(out_target.view_as(zero_out1[indices])).sum().item()
            indices = (root_out == 4).nonzero()[:,0]
            four_data = next_data[indices]
            four_target = target[indices]
            __, four_out = model_four(four_data)
            four_out = four_out.max(1, keepdim=True)[1]
            for i in four:
                four_out1 = four_out.clone().to(device)
                indices = (four_out == i).nonzero()[:,0]
                out_target = four_target[indices]
                four_out1[(four_out1 == i).nonzero()] = four[i]
                count += four_out1[indices].eq(out_target.view_as(four_out1[indices])).sum().item()

            indices = (root_out == 6).nonzero()[:,0]
            six_data = next_data[indices]
            six_target = target[indices]
            __, six_out = model_six(six_data)
            six_out = six_out.max(1, keepdim=True)[1]
            for i in six:
                six_out1 = six_out.clone().to(device)
                indices = (six_out == i).nonzero()[:,0]
                out_target = six_target[indices]
                six_out1[(six_out1 == i).nonzero()] = six[i]
                count += six_out1[indices].eq(out_target.view_as(six_out1[indices])).sum().item()

            indices = (root_out == 8).nonzero()[:,0]
            eight_data = next_data[indices]
            eight_target = target[indices]
            next_eight, eight_out = model_eight(eight_data)
            eight_out = eight_out.max(1, keepdim=True)[1]
            for i in eight:
                eight_out1 = eight_out.clone().to(device)
                indices = (eight_out == i).nonzero()[:,0]
                out_target = eight_target[indices]
                eight_out1[(eight_out1 == i).nonzero()] = eight[i]
                count += eight_out1[indices].eq(out_target.view_as(eight_out1[indices])).sum().item()

            if (eight_out == 0).nonzero().shape[0] != 0:
                indices = (eight_out == 0).nonzero()[:,0]
                zero_data = next_eight[indices]
                zero_target = eight_target[indices]
                _, zero_out = model_eight_zero(zero_data)
                zero_out = zero_out.max(1, keepdim=True)[1]
                for i in eight_zero:
                    zero_out1 = zero_out.clone().to(device)
                    indices = (zero_out == i).nonzero()[:,0]
                    out_target = zero_target[indices]
                    zero_out1[(zero_out1 == i).nonzero()] = eight_zero[i]
                    count += zero_out1[indices].eq(out_target.view_as(zero_out1[indices])).sum().item()

            indices = (root_out == 9).nonzero()[:,0]
            nine_data = next_data[indices]
            nine_target = target[indices]
            __, nine_out = model_nine(nine_data)
            nine_out = nine_out.max(1, keepdim=True)[1]
            for i in nine:
                nine_out1 = nine_out.clone().to(device)
                indices = (nine_out == i).nonzero()[:,0]
                out_target = nine_target[indices]
                nine_out1[(nine_out1 == i).nonzero()] = nine[i]
                count += nine_out1[indices].eq(out_target.view_as(nine_out1[indices])).sum().item()

            indices = (root_out == 10).nonzero()[:,0]
            ten_data = next_data[indices]
            ten_target = target[indices]
            __, ten_out = model_ten(ten_data)
            ten_out = ten_out.max(1, keepdim=True)[1]
            for i in ten:
                ten_out1 = ten_out.clone().to(device)
                indices = (ten_out == i).nonzero()[:,0]
                out_target = ten_target[indices]
                ten_out1[(ten_out1 == i).nonzero()] = ten[i]
                count += ten_out1[indices].eq(out_target.view_as(ten_out1[indices])).sum().item()



        print("Val: ", (count)/len(valloader.sampler.indices))


def main():
        train_batch_size = 100

        dataset = datasets.EMNIST(
                        '../dataEMNIST',
                        split = 'balanced',
                        train = True,
                        download = True,
                        transform=transforms.ToTensor()
                )

        shuffle_dataset =  True
        val_split = 0.1
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(val_split * dataset_size))

        if shuffle_dataset :
            np.random.seed(2)
            np.random.shuffle(indices)


        train_indices, val_indices = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)

        train_loader = torch.utils.data.DataLoader(
                                        dataset,
                                        sampler = train_sampler,
                                        batch_size = 5000
                                    )

        val_loader = torch.utils.data.DataLoader(
                                        dataset,
                                        sampler = val_sampler,
                                        batch_size = 5000
                                    )

        model = model_root().to(torch.device("cuda"))
        model.load_state_dict(torch.load('../Models/emnist_root.pth'))

        model_zero = model_0().to(torch.device("cuda"))
        model_zero.load_state_dict(torch.load('../Models/emnist_0.pth'))

        model_one = model_1().to(torch.device("cuda"))
        model_one.load_state_dict(torch.load('../Models/emnist_1.pth'))

        model_three = model_3().to(torch.device("cuda"))
        model_three.load_state_dict(torch.load('../Models/emnist_3.pth'))

        model_four = model_4().to(torch.device("cuda"))
        model_four.load_state_dict(torch.load('../Models/emnist_4.pth'))

        model_six = model_6().to(torch.device("cuda"))
        model_six.load_state_dict(torch.load('../Models/emnist_6.pth'))

        model_eight = model_8().to(torch.device("cuda"))
        model_eight.load_state_dict(torch.load('../Models/emnist_8.pth'))

        model_nine = model_9().to(torch.device("cuda"))
        model_nine.load_state_dict(torch.load('../Models/emnist_9.pth'))

        model_ten = model_10().to(torch.device("cuda"))
        model_ten.load_state_dict(torch.load('../Models/emnist_10.pth'))

        model_zero_zero = model_0_0().to(torch.device("cuda"))
        model_zero_zero.load_state_dict(torch.load('../Models/emnist_0_0.pth'))

        model_zero_three = model_0_3().to(torch.device("cuda"))
        model_zero_three.load_state_dict(torch.load('../Models/emnist_0_3.pth'))

        model_zero_five = model_0_5().to(torch.device("cuda"))
        model_zero_five.load_state_dict(torch.load('../Models/emnist_0_5.pth'))

        model_one_zero = model_1_0().to(torch.device("cuda"))
        model_one_zero.load_state_dict(torch.load('../Models/emnist_1_0.pth'))

        model_three_zero = model_3_0().to(torch.device("cuda"))
        model_three_zero.load_state_dict(torch.load('../Models/emnist_3_0.pth'))

        model_eight_zero = model_8_0().to(torch.device("cuda"))
        model_eight_zero.load_state_dict(torch.load('../Models/emnist_8_0.pth'))

        test(model, model_zero, model_one, model_three, model_four, model_six, model_eight, model_nine, model_ten, model_zero_zero, model_zero_three, model_zero_five, model_one_zero, model_three_zero, model_eight_zero, val_loader, torch.device("cuda"))

if __name__== "__main__":
        main()
