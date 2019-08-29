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


__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model



def vgg16(pretrained=False, progress=True, **kwargs):
    return _vgg('vgg16', 'D', False, pretrained, progress, **kwargs)

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

        model = vgg16(pretrained=True).to(torch.device("cuda"))
        
        for param in model.parameters():
            param.requires_grad = False

        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 20).cuda()

        for param in model.classifier[6].parameters():
            param.requires_grad = True
        
        torch.save(model.state_dict(), "model_vgg.pth")

        learning_rate = 0.0001
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()
        train(model, optimizer, loss_fn, train_loader, val_loader, torch.device("cuda"))
        
        for i in range(3):
            learning_rate /= 10
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 5e-4)
            loss_fn = nn.CrossEntropyLoss()
            train(model, optimizer, loss_fn, train_loader, val_loader, torch.device("cuda"))

if __name__== "__main__":
    main()

