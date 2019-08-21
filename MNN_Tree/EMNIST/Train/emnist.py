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
    '1': [8, 'M', 'D'],
    '2': [16, 'M', 16, 'M', 'D'],
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
                        nn.Linear(92*7*7, 47),
                )

    def forward(self, x):
        x = self.features(x)
        y = x.view(x.size(0), -1)
        out = self.classifier(y)
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

def model_one():
    return model('6')

def train(model, optimizer, loss_fn, trainloader, valloader, device):
        max = 0.0
        for epoch in range(20):
                model.train()
                count = 0
                total = 0
                for batch_num, (data, target) in enumerate(trainloader):
                        data = data.to(device)
                        target = target.to(device)
                        optimizer.zero_grad()
                        _,net_out = model(data)
                        loss = loss_fn(net_out, target)
                        loss.backward()
                        optimizer.step()
                        pred = net_out.max(1, keepdim=True)[1]
                        count += pred.eq(target.view_as(pred)).sum().item()
                        total += target.size(0)
                        progress_bar(batch_num, len(trainloader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                         % (loss / (batch_num + 1), 100. * count / total, count, total))
                print("Epoch: ",epoch+1,"\nTrain: ", count/len(trainloader.sampler.indices))

                model.eval()
                count1 = 0
                for data, target in valloader:
                        data = data.to(device)
                        target = target.to(device)
                        _,net_out = model(data)
                        pred = net_out.max(1, keepdim=True)[1]
                        count1 += pred.eq(target.view_as(pred)).sum().item()

                print("Val: ", count1/len(valloader.sampler.indices))
                if count1/len(valloader.sampler.indices) > max:
                    print("checkpoint saved")
                    max = count1/len(valloader.sampler.indices)
                    torch.save(model.state_dict(), "model_tester.pth")

def average_softmax(model, trainloader, valloader, device):
    nb_classes = 47
    out_classes = 47
    counts = [0 for i in range(nb_classes)]
    soft_out = torch.zeros((nb_classes, 1, nb_classes)).cuda()
    with torch.no_grad():
        for i, (inputs, classes) in enumerate(trainloader):
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
    for i in range(47):
        print(soft_out[i])
    torch.save(soft_out, 'saved.pth')

def main():
        train_batch_size = 100

        dataset = datasets.EMNIST(
                        'dataEMNIST',
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

        mod = model_one().to(torch.device("cuda"))
        #mod.load_state_dict(torch.load('model_emnist.pth'))
        learning_rate = 0.001
        optimizer = torch.optim.Adam(mod.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()
        train(mod, optimizer, loss_fn, train_loader, val_loader, torch.device("cuda"))
        average_softmax(mod, train_loader, val_loader, torch.device("cuda"))


if __name__== "__main__":
        main()
