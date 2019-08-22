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
    'root': [16, 32, 32,'M'],
    '2': [16, 'M', 32, 'M', 'D'],
    '3': [32, 64, 'M', 'D'],
    '4': [16, 32,'M', 32, 32, 'M', 'D'],
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

def model_one():
    return model('root')

class mod_three(nn.Module):
    def __init__(self, size):
        super(mod_three, self).__init__()
        self.features = self._make_layers(cfg[size], 32)
        self.classifier = nn.Sequential(
                        nn.Linear(64*8*8, 4),
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

def model_3():
    return mod_three('3')


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


def train(model, model_three, optimizer, loss_fn, trainloader, valloader, device, maxi):
        max = maxi
        for epoch in range(30):
                model.eval()
                model_three.train()
                count = 0
                num_three = 0
                total = 0
                model_three.acc = 0
                for batch_num, (data, target) in enumerate(trainloader):
                        data = data.to(device)
                        target = target.to(device)
                        target1 = target.clone().to(device)
                        target1[(target1 == 0).nonzero()] = 100
                        target1[(target1 == 1).nonzero()] = 101
                        target1[(target1 == 7).nonzero()] = 101
                        target1[(target1 == 2).nonzero()] = 102
                        target1[(target1 == 3).nonzero()] = 103
                        target1[(target1 == 5).nonzero()] = 103
                        target1[(target1 == 8).nonzero()] = 103
                        target1[(target1 == 6).nonzero()] = 103
                        target1[(target1 == 4).nonzero()] = 104
                        target1[(target1 == 9).nonzero()] = 105
                        target1 -= 100

                        optimizer.zero_grad()
                        next_data ,net_out = model(data)

                        indices = (target1 == 3).nonzero()[:,0]

                        three_data = next_data[indices]
                        three_target = target[indices]

                        three_target1 = three_target.clone().to(device)
                        three_target1[(three_target1 == 3).nonzero()] = 10
                        three_target1[(three_target1 == 5).nonzero()] = 11
                        three_target1[(three_target1 == 8).nonzero()] = 12
                        three_target1[(three_target1 == 6).nonzero()] = 13
                        three_target1 -= 10

                        loss = train_mod(model_three, optimizer, loss_fn, three_data, three_target1, device)
                        num_three += three_data.shape[0]

                        progress_bar(batch_num, len(trainloader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                         % (loss / (batch_num + 1), 100. * model_three.acc / num_three, model_three.acc, num_three))
                print("Epoch: ",epoch+1,"\nTrain: ", model_three.acc / num_three)

                if model_three.acc / num_three > max:
                    print("checkpoint saved")
                    max = model_three.acc / num_three
                    torch.save(model_three.state_dict(), "svhn_3.pth")

                model.eval()
                count1 = 0
                model_three.eval()
                model_three.acc = 0
                num_three = 0
                count_three = 0
                for data, target in valloader:
                        data = data.to(device)
                        target = target.to(device)
                        target1 = target.clone().to(device)
                        target1[(target1 == 0).nonzero()] = 100
                        target1[(target1 == 1).nonzero()] = 101
                        target1[(target1 == 7).nonzero()] = 101
                        target1[(target1 == 2).nonzero()] = 102
                        target1[(target1 == 3).nonzero()] = 103
                        target1[(target1 == 5).nonzero()] = 103
                        target1[(target1 == 8).nonzero()] = 103
                        target1[(target1 == 6).nonzero()] = 103
                        target1[(target1 == 4).nonzero()] = 104
                        target1[(target1 == 9).nonzero()] = 105
                        target1 -= 100
                        next_data ,net_out = model(data)

                        indices = (target1 == 3).nonzero()[:,0]

                        three_data = next_data[indices]
                        three_target = target[indices]

                        three_target1 = three_target.clone().to(device)
                        three_target1[(three_target1 == 3).nonzero()] = 10
                        three_target1[(three_target1 == 5).nonzero()] = 11
                        three_target1[(three_target1 == 8).nonzero()] = 12
                        three_target1[(three_target1 == 6).nonzero()] = 13
                        three_target1 -= 10

                        _, three_out = model_three(three_data)
                        pred_three = three_out.max(1, keepdim=True)[1]
                        count_three += pred_three.eq(three_target1.view_as(pred_three)).sum().item()
                        num_three += three_data.shape[0]

                print("Val: ", count_three/num_three)
        return max

def average_softmax(model, trainloader, valloader, device):
    nb_classes = 4
    out_classes = 4
    counts = [0 for i in range(nb_classes)]
    soft_out = torch.zeros((nb_classes, 1, nb_classes)).cuda()
    with torch.no_grad():
        for i, (inputs, classes) in enumerate(valloader):
            inputs = inputs.to(device)
            classes = classes.to(device)
            _,outputs = model(inputs)
            outputs = m(outputs)
            for categ in range(nb_classes):
                indices = (classes == categ).nonzero()[:,0]
                hold = outputs[indices]
                soft_out[categ] += hold.sum(dim=0)
                counts[categ]+= hold.shape[0]
    for i in range(nb_classes):
        soft_out[i] = soft_out[i]/counts[i]
    print(soft_out)

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
                np.random.seed(2)
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

        model = model_one().to(torch.device("cuda"))
        model.load_state_dict(torch.load('../Models/svhn_root.pth'))
        model_three = model_3().to(torch.device("cuda"))
        model_three.load_state_dict(torch.load('../Models/svhn_3.pth'))
        print(summary(model,(3,32,32)))
        learning_rate = 0.001
        optimizer = torch.optim.Adam(model_three.parameters(), lr=learning_rate, weight_decay = 5e-4)
        loss_fn = nn.CrossEntropyLoss()

        max = train(model, model_three, optimizer, loss_fn, train_loader, val_loader, torch.device("cuda"), 0)

        for i in range(3):
            model_three.load_state_dict(torch.load('../Models/svhn_3.pth'))
            learning_rate /= 10
            optimizer = torch.optim.Adam(model_three.parameters(), lr=learning_rate, weight_decay = 5e-4)
            loss_fn = nn.CrossEntropyLoss()
            max = train(model, model_three, optimizer, loss_fn, train_loader, val_loader, torch.device("cuda"), max)

        model_three.load_state_dict(torch.load('../Models/svhn_3.pth'))
        average_softmax(model, train_loader, val_loader, torch.device("cuda"))

if __name__== "__main__":
        main()
