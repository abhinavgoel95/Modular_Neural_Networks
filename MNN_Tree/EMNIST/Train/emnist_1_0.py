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

m = nn.Softmax()

cfg = {
    '1_0': [32, 64, 'M'],
    '0_0': [32, 32, 'M'],
    '0_3': [32, 32, 'M'],
    '1': [16, 32, 'M'],
    '0': [16, 32, 'M', 'D'],
    'root': [16, 'M', 16, 'D'],
    '3': [16,'M', 16, 'M', 32, 'D'],
    '4': [16, 32,'M', 32, 48, 'M', 'D'],
    '5': [16, 32, 'M', 32, 32, 'M', 64,'D'],
    '6': [16, 32, 32, 'M', 64, 64, 92, 'M', 'D'],
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

def train_mod(model, optimizer, loss_fn, data, target, device):
    data = data.to(device)
    target = target.to(device)
    optimizer.zero_grad()
    _,net_out = model(data)
    loss = loss_fn(net_out, target)
    loss.backward()
    optimizer.step()
    pred = net_out.max(1, keepdim=True)[1]
    model.acc += pred.eq(target.view_as(pred)).sum().item()
    return loss

def train(model, model_one, model_one_zero, optimizer, loss_fn, trainloader, valloader, device, maxi):
        max = maxi
        for epoch in range(10):
                model.eval()
                model_one.eval()
                model_one_zero.train()
                count = 0
                num_one_zero = 0
                total = 0
                model_one_zero.acc = 0
                for batch_num, (data, target) in enumerate(trainloader):
                        data = data.to(device)
                        target = target.to(device)
                        target1 = target.clone().to(device)
                        target1[(target1 == 0).nonzero()] = 100
                        target1[(target1 == 13).nonzero()] = 100
                        target1[(target1 == 24).nonzero()] = 100
                        target1[(target1 == 36).nonzero()] = 100
                        target1[(target1 == 2).nonzero()] = 100
                        target1[(target1 == 35).nonzero()] = 100
                        target1[(target1 == 26).nonzero()] = 100
                        target1[(target1 == 44).nonzero()] = 100
                        target1[(target1 == 8).nonzero()] = 100
                        target1[(target1 == 11).nonzero()] = 100
                        target1[(target1 == 41).nonzero()] = 100
                        target1[(target1 == 9).nonzero()] = 100
                        target1[(target1 == 1).nonzero()] = 101
                        target1[(target1 == 18).nonzero()] = 101
                        target1[(target1 == 21).nonzero()] = 101
                        target1[(target1 == 19).nonzero()] = 101
                        target1[(target1 == 3).nonzero()] = 102
                        target1[(target1 == 4).nonzero()] = 103
                        target1[(target1 == 17).nonzero()] = 103
                        target1[(target1 == 30).nonzero()] = 103
                        target1[(target1 == 34).nonzero()] = 103
                        target1[(target1 == 10).nonzero()] = 103
                        target1[(target1 == 31).nonzero()] = 103
                        target1[(target1 == 33).nonzero()] = 103
                        target1[(target1 == 20).nonzero()] = 103
                        target1[(target1 == 27).nonzero()] = 103
                        target1[(target1 == 42).nonzero()] = 103
                        target1[(target1 == 37).nonzero()] = 103
                        target1[(target1 == 6).nonzero()] = 103
                        target1[(target1 == 16).nonzero()] = 103
                        target1[(target1 == 5).nonzero()] = 104
                        target1[(target1 == 28).nonzero()] = 104
                        target1[(target1 == 7).nonzero()] = 105
                        target1[(target1 == 12).nonzero()] = 106
                        target1[(target1 == 39).nonzero()] = 106
                        target1[(target1 == 14).nonzero()] = 107
                        target1[(target1 == 15).nonzero()] = 108
                        target1[(target1 == 40).nonzero()] = 108
                        target1[(target1 == 46).nonzero()] = 108
                        target1[(target1 == 29).nonzero()] = 108
                        target1[(target1 == 22).nonzero()] = 109
                        target1[(target1 == 43).nonzero()] = 109
                        target1[(target1 == 23).nonzero()] = 110
                        target1[(target1 == 32).nonzero()] = 110
                        target1[(target1 == 25).nonzero()] = 111
                        target1[(target1 == 38).nonzero()] = 112
                        target1[(target1 == 45).nonzero()] = 113
                        target1 -= 100

                        next_data ,net_out = model(data)
                        if (target1 == 1).nonzero().shape[0] != 0:
                            indices = (target1 == 1).nonzero()[:,0]

                            one_data = next_data[indices]
                            one_target = target[indices]

                            one_target1 = one_target.clone().to(device)
                            one_target1[(one_target1 == 1).nonzero()] = 100
                            one_target1[(one_target1 == 18).nonzero()] = 100
                            one_target1[(one_target1 == 21).nonzero()] = 100
                            one_target1[(one_target1 == 19).nonzero()] = 101
                            one_target1 -= 100


                            next_data ,net_out = model_one(one_data)
                            if (one_target1 == 0).nonzero().shape[0] != 0:
                                indices = (one_target1 == 0).nonzero()[:,0]

                                one_zero_data = next_data[indices]
                                one_zero_target = one_target[indices]

                                one_zero_target1 = one_zero_target.clone().to(device)
                                one_zero_target1[(one_zero_target1 == 1).nonzero()] = 100
                                one_zero_target1[(one_zero_target1 == 18).nonzero()] = 101
                                one_zero_target1[(one_zero_target1 == 21).nonzero()] = 102
                                one_zero_target1 -= 100

                                loss = train_mod(model_one_zero, optimizer, loss_fn, one_zero_data, one_zero_target1, device)
                                num_one_zero += one_zero_data.shape[0]

                                progress_bar(batch_num, len(trainloader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                                 % (loss / (batch_num + 1), 100. * model_one_zero.acc / num_one_zero, model_one_zero.acc, num_one_zero))
                print("Epoch: ",epoch+1,"\nTrain: ", model_one_zero.acc / num_one_zero)


                model.eval()
                count1 = 0
                model_one.eval()
                model_one_zero.train()
                model_one_zero.acc = 0
                num_one_zero = 0
                count_one_zero = 0
                for data, target in valloader:
                        data = data.to(device)
                        target = target.to(device)
                        target1 = target.clone().to(device)
                        target1[(target1 == 0).nonzero()] = 100
                        target1[(target1 == 13).nonzero()] = 100
                        target1[(target1 == 24).nonzero()] = 100
                        target1[(target1 == 36).nonzero()] = 100
                        target1[(target1 == 2).nonzero()] = 100
                        target1[(target1 == 35).nonzero()] = 100
                        target1[(target1 == 26).nonzero()] = 100
                        target1[(target1 == 44).nonzero()] = 100
                        target1[(target1 == 8).nonzero()] = 100
                        target1[(target1 == 11).nonzero()] = 100
                        target1[(target1 == 41).nonzero()] = 100
                        target1[(target1 == 9).nonzero()] = 100
                        target1[(target1 == 1).nonzero()] = 101
                        target1[(target1 == 18).nonzero()] = 101
                        target1[(target1 == 21).nonzero()] = 101
                        target1[(target1 == 19).nonzero()] = 101
                        target1[(target1 == 3).nonzero()] = 102
                        target1[(target1 == 4).nonzero()] = 103
                        target1[(target1 == 17).nonzero()] = 103
                        target1[(target1 == 30).nonzero()] = 103
                        target1[(target1 == 34).nonzero()] = 103
                        target1[(target1 == 10).nonzero()] = 103
                        target1[(target1 == 31).nonzero()] = 103
                        target1[(target1 == 33).nonzero()] = 103
                        target1[(target1 == 20).nonzero()] = 103
                        target1[(target1 == 27).nonzero()] = 103
                        target1[(target1 == 42).nonzero()] = 103
                        target1[(target1 == 37).nonzero()] = 103
                        target1[(target1 == 6).nonzero()] = 103
                        target1[(target1 == 16).nonzero()] = 103
                        target1[(target1 == 5).nonzero()] = 104
                        target1[(target1 == 28).nonzero()] = 104
                        target1[(target1 == 7).nonzero()] = 105
                        target1[(target1 == 12).nonzero()] = 106
                        target1[(target1 == 39).nonzero()] = 106
                        target1[(target1 == 14).nonzero()] = 107
                        target1[(target1 == 15).nonzero()] = 108
                        target1[(target1 == 40).nonzero()] = 108
                        target1[(target1 == 46).nonzero()] = 108
                        target1[(target1 == 29).nonzero()] = 108
                        target1[(target1 == 22).nonzero()] = 109
                        target1[(target1 == 43).nonzero()] = 109
                        target1[(target1 == 23).nonzero()] = 110
                        target1[(target1 == 32).nonzero()] = 110
                        target1[(target1 == 25).nonzero()] = 111
                        target1[(target1 == 38).nonzero()] = 112
                        target1[(target1 == 45).nonzero()] = 113
                        target1 -= 100
                        next_data,net_out = model(data)

                        if (target1 == 1).nonzero().shape[0] != 0:
                            indices = (target1 == 1).nonzero()[:,0]

                            one_data = next_data[indices]
                            one_target = target[indices]

                            one_target1 = one_target.clone().to(device)
                            one_target1[(one_target1 == 1).nonzero()] = 100
                            one_target1[(one_target1 == 18).nonzero()] = 100
                            one_target1[(one_target1 == 21).nonzero()] = 100
                            one_target1[(one_target1 == 19).nonzero()] = 101
                            one_target1 -= 100


                            next_data ,net_out = model_one(one_data)
                            if (one_target1 == 0).nonzero().shape[0] != 0:
                                indices = (one_target1 == 0).nonzero()[:,0]

                                one_zero_data = next_data[indices]
                                one_zero_target = one_target[indices]

                                one_zero_target1 = one_zero_target.clone().to(device)
                                one_zero_target1[(one_zero_target1 == 1).nonzero()] = 100
                                one_zero_target1[(one_zero_target1 == 18).nonzero()] = 101
                                one_zero_target1[(one_zero_target1 == 21).nonzero()] = 102
                                one_zero_target1 -= 100

                                _, one_zero_out = model_one_zero(one_zero_data)
                                pred_one_zero = one_zero_out.max(1, keepdim=True)[1]
                                count_one_zero += pred_one_zero.eq(one_zero_target1.view_as(pred_one_zero)).sum().item()
                                num_one_zero += one_zero_data.shape[0]
                print("Val: ", count_one_zero/num_one_zero)
                if count_one_zero/num_one_zero > max:
                    print("checkpoint saved")
                    max = count_one_zero/num_one_zero
                    torch.save(model_one_zero.state_dict(), "../Models/emnist_1_0.pth")
        return max

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
        val_split = 0.002
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
                                        batch_size = 100
                                    )

        val_loader = torch.utils.data.DataLoader(
                                        dataset,
                                        sampler = val_sampler,
                                        batch_size = 100
                                    )

        model = model_root().to(torch.device("cuda"))
        model.load_state_dict(torch.load('root_emnist.pth'))
        model_one = model_1().to(torch.device("cuda"))
        model_one.load_state_dict(torch.load('emnist_1.pth'))
        model_one_zero = model_1_0().to(torch.device("cuda"))
        model_one_zero.load_state_dict(torch.load('emnist_1_0.pth'))

        learning_rate = 0.0001
        optimizer = torch.optim.Adam(model_one_zero.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()
        max = train(model, model_one, model_one_zero, optimizer, loss_fn, train_loader, val_loader, torch.device("cuda"), 0)
        for i in range(2):
            model_one_zero.load_state_dict(torch.load('emnist_1_0.pth'))
            learning_rate /= 10
            optimizer = torch.optim.Adam(model_one_zero.parameters(), lr=learning_rate, weight_decay = 5e-4)
            loss_fn = nn.CrossEntropyLoss()
            max = train(model, model_one, model_one_zero, optimizer, loss_fn, train_loader, val_loader, torch.device("cuda"), max)

if __name__== "__main__":
        main()
