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
    'root': [16, 32, 32,'M'],
    '2': [16, 'M', 32, 'M', 'D'],
    '3': [32, 64, 'M', 'D'],
    '1': [32, 32, 'M', 'D'],
    '5': [16, 32, 'M', 32, 32, 'M', 64,'D'],
    '6': [16, 32, 32, 'M', 64, 64, 128, 'M', 'D'],
}

class model(nn.Module):
    def __init__(self, size):
        super(model, self).__init__()
        self.features = self._make_layers(cfg[size])
        self.classifier = nn.Sequential(
                        nn.Linear(32*16*16, 6),
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

def model_root():
    return model('root')

class mod_one(nn.Module):
    def __init__(self, size):
        super(mod_one, self).__init__()
        self.features = self._make_layers(cfg[size], 32)
        self.classifier = nn.Sequential(
                        nn.Linear(32*8*8, 2),
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

def model_1():
    return mod_one('1')

class mod_three(nn.Module):
    def __init__(self, size):
        super(mod_three, self).__init__()
        self.features = self._make_layers(cfg[size], 32)
        self.classifier = nn.Sequential(
                        nn.Linear(64*8*8, 4),
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

def model_3():
    return mod_three('3')


def test(model, model_one, model_three, valloader, device):
        model.eval()
        model_one.eval()
        model_three.eval()
        count1 = 0
        count_root0 = 0
        count_root2 = 0
        count_root4 = 0
        count_root5 = 0
        count3 = 0
        check_root = 0
        for data, target in valloader:
            data = data.to(device)
            target = target.to(device)

            next_data, root_out = model(data)
            root_out = root_out.max(1, keepdim=True)[1]

            indices_three = (root_out == 3).nonzero()[:,0]
            indices_one = (root_out == 1).nonzero()[:,0]
            indices_zero = (root_out == 0).nonzero()[:,0]
            indices_two = (root_out == 2).nonzero()[:,0]
            indices_four = (root_out == 4).nonzero()[:,0]
            indices_five = (root_out == 5).nonzero()[:,0]

            one_target = target[indices_one]
            root0_target = target[indices_zero]
            root2_target = target[indices_two]
            root4_target = target[indices_four]
            root5_target = target[indices_five]
            three_target = target[indices_three]


            one_data = next_data[indices_one]
            three_data = next_data[indices_three]

            __, one_out = model_one(one_data)
            one_out = one_out.max(1, keepdim=True)[1]
            __, three_out = model_three(three_data)
            three_out = three_out.max(1, keepdim=True)[1]

            root_out[(root_out == 0).nonzero()] = 0
            root_out[(root_out == 2).nonzero()] = 2
            root_out[(root_out == 4).nonzero()] = 4
            root_out[(root_out == 5).nonzero()] = 9

            one_out[(one_out == 1).nonzero()] = 7
            one_out[(one_out == 0).nonzero()] = 1

            three_out[(three_out == 1).nonzero()] = 5
            three_out[(three_out == 2).nonzero()] = 8
            three_out[(three_out == 3).nonzero()] = 6
            three_out[(three_out == 0).nonzero()] = 3

            count_root0 += root_out[indices_zero].eq(root0_target.view_as(root_out[indices_zero])).sum().item()
            count_root2 += root_out[indices_two].eq(root2_target.view_as(root_out[indices_two])).sum().item()
            count_root4 += root_out[indices_four].eq(root4_target.view_as(root_out[indices_four])).sum().item()
            count_root5 += root_out[indices_five].eq(root5_target.view_as(root_out[indices_five])).sum().item()
            count1 += one_out.eq(one_target.view_as(one_out)).sum().item()
            count3 += three_out.eq(three_target.view_as(three_out)).sum().item()

        print("Val: ", (count1+count3+count_root5+count_root2+count_root4+count_root0)/len(valloader.sampler.indices))


def main():
        train_batch_size = 1000
        svhn_e = datasets.SVHN(root='../dataSVHN', split = 'extra', download=True, transform = transforms.ToTensor())
        svhn_t = datasets.SVHN(root='../dataSVHN', split = 'train', download=True, transform = transforms.ToTensor())
        svhn = torch.utils.data.ConcatDataset([svhn_e, svhn_t])
        shuffle_dataset = True
        val_split = 0.1
        dataset_size = len(svhn)
        indices = list(range(dataset_size))
        split = int(np.floor(val_split * dataset_size))

        if shuffle_dataset:
                np.random.seed(8080)
                np.random.shuffle(indices)

        train_indices, val_indices = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)

        train_loader = torch.utils.data.DataLoader(svhn,
        sampler = train_sampler,
        batch_size = 500)

        val_loader = torch.utils.data.DataLoader(svhn,
        sampler = val_sampler,
        batch_size = 500)

        model = model_root().to(torch.device("cuda"))
        model.load_state_dict(torch.load('../Models/svhn_root.pth'))

        model_three = model_3().to(torch.device("cuda"))
        model_three.load_state_dict(torch.load('../Models/svhn_3.pth'))

        model_one = model_1().to(torch.device("cuda"))
        model_one.load_state_dict(torch.load('../Models/svhn_1.pth'))

        test(model, model_one, model_three, val_loader, torch.device("cuda"))

if __name__== "__main__":
        main()
