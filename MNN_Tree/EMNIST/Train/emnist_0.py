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

class mod_zero(nn.Module):
    def __init__(self, size):
        super(mod_zero, self).__init__()
        self.features = self._make_layers(cfg[size], 16)
        #self.features_down = self._make_layers(cfg_down[size], 32)
        self.classifier = nn.Sequential(
                        nn.Linear(32*7*7, 8),
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

def model_0():
    return mod_zero('0')

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

def train(model, model_zero, optimizer, loss_fn, trainloader, valloader, device, maxi):
        max = maxi
        for epoch in range(10):
                model.eval()
                model_zero.train()
                count = 0
                num_zero = 0
                total = 0
                model_zero.acc = 0
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
                        indices = (target1 == 0).nonzero()[:,0]

                        zero_data = next_data[indices]
                        zero_target = target[indices]

                        zero_target1 = zero_target.clone().to(device)
                        zero_target1[(zero_target1 == 0).nonzero()] = 100
                        zero_target1[(zero_target1 == 24).nonzero()] = 100
                        zero_target1[(zero_target1 == 13).nonzero()] = 101
                        zero_target1[(zero_target1 == 36).nonzero()] = 102
                        zero_target1[(zero_target1 == 2).nonzero()] = 103
                        zero_target1[(zero_target1 == 35).nonzero()] = 103
                        zero_target1[(zero_target1 == 26).nonzero()] = 104
                        zero_target1[(zero_target1 == 44).nonzero()] = 105
                        zero_target1[(zero_target1 == 41).nonzero()] = 105
                        zero_target1[(zero_target1 == 9).nonzero()] = 105
                        zero_target1[(zero_target1 == 8).nonzero()] = 106
                        zero_target1[(zero_target1 == 11).nonzero()] = 107
                        zero_target1 -= 100

                        loss = train_mod(model_zero, optimizer, loss_fn, zero_data, zero_target1, device)
                        num_zero += zero_data.shape[0]

                        progress_bar(batch_num, len(trainloader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                         % (loss / (batch_num + 1), 100. * model_zero.acc / num_zero, model_zero.acc, num_zero))
                print("Epoch: ",epoch+1,"\nTrain: ", model_zero.acc / num_zero)

                model.eval()
                count1 = 0
                model_zero.eval()
                model_zero.acc = 0
                num_zero = 0
                count_zero = 0
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

                        indices = (target1 == 0).nonzero()[:,0]

                        zero_data = next_data[indices]
                        zero_target = target[indices]
                        zero_target1 = zero_target.clone().to(device)
                        zero_target1[(zero_target1 == 0).nonzero()] = 100
                        zero_target1[(zero_target1 == 24).nonzero()] = 100
                        zero_target1[(zero_target1 == 13).nonzero()] = 101
                        zero_target1[(zero_target1 == 36).nonzero()] = 102
                        zero_target1[(zero_target1 == 2).nonzero()] = 103
                        zero_target1[(zero_target1 == 35).nonzero()] = 103
                        zero_target1[(zero_target1 == 26).nonzero()] = 104
                        zero_target1[(zero_target1 == 44).nonzero()] = 105
                        zero_target1[(zero_target1 == 41).nonzero()] = 105
                        zero_target1[(zero_target1 == 9).nonzero()] = 105
                        zero_target1[(zero_target1 == 8).nonzero()] = 106
                        zero_target1[(zero_target1 == 11).nonzero()] = 107
                        zero_target1 -= 100

                        _, zero_out = model_zero(zero_data)
                        pred_zero = zero_out.max(1, keepdim=True)[1]
                        count_zero += pred_zero.eq(zero_target1.view_as(pred_zero)).sum().item()
                        num_zero += zero_data.shape[0]

                print("Val: ", count_zero/num_zero)
                if count_zero/num_zero > max:
                    print("checkpoint saved")
                    max = count_zero/num_zero
                    torch.save(model_zero.state_dict(), "../Models/emnist_0.pth")
        return max

def average_softmax(model, model_zero, trainloader, valloader, device):
    nb_classes = 12
    out_classes = 12
    counts = [0 for i in range(nb_classes)]
    soft_out = torch.zeros((nb_classes, 1, nb_classes)).cuda()
    with torch.no_grad():
        for i, (data, target) in enumerate(trainloader):
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

            indices = (target1 == 0).nonzero()[:,0]

            zero_data = next_data[indices]
            zero_target = target[indices]
            zero_target1 = zero_target.clone().to(device)
            zero_target1 = zero_target.clone().to(device)
            zero_target1[(zero_target1 == 0).nonzero()] = 100
            zero_target1[(zero_target1 == 13).nonzero()] = 101
            zero_target1[(zero_target1 == 24).nonzero()] = 102
            zero_target1[(zero_target1 == 36).nonzero()] = 103
            zero_target1[(zero_target1 == 2).nonzero()] = 104
            zero_target1[(zero_target1 == 35).nonzero()] = 105
            zero_target1[(zero_target1 == 26).nonzero()] = 106
            zero_target1[(zero_target1 == 44).nonzero()] = 107
            zero_target1[(zero_target1 == 8).nonzero()] = 108
            zero_target1[(zero_target1 == 11).nonzero()] = 109
            zero_target1[(zero_target1 == 41).nonzero()] = 110
            zero_target1[(zero_target1 == 9).nonzero()] = 111
            zero_target1 -= 100

            _, zero_out = model_zero(zero_data)
            outputs = m(zero_out)
            for categ in range(nb_classes):
                indices = (zero_target1 == categ).nonzero()[:,0]
                hold = outputs[indices]
                soft_out[categ] += hold.sum(dim=0)
                counts[categ]+= hold.shape[0]
    for i in range(nb_classes):
        soft_out[i] = soft_out[i]/counts[i]
    for i in range(12):
        print(soft_out[i])
    torch.save(soft_out, 'emnist_0_softmax_output.pth')

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
        model_zero = model_0().to(torch.device("cuda"))
        model_zero.load_state_dict(torch.load('emnist_0.pth'))

        learning_rate = 0.0001
        optimizer = torch.optim.Adam(model_zero.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()
        max = train(model, model_zero, optimizer, loss_fn, train_loader, val_loader, torch.device("cuda"), .965)
        for i in range(2):
            model_zero.load_state_dict(torch.load('emnist_0.pth'))
            learning_rate /= 10
            optimizer = torch.optim.Adam(model_zero.parameters(), lr=learning_rate, weight_decay = 5e-4)
            loss_fn = nn.CrossEntropyLoss()
            max = train(model, model_zero, optimizer, loss_fn, train_loader, val_loader, torch.device("cuda"), max)
        model_zero.load_state_dict(torch.load('emnist_0.pth'))
        #average_softmax(model, model_zero, train_loader, val_loader, torch.device("cuda"))
if __name__== "__main__":
        main()
