import os
import numpy as np
from glob import glob
import utils
import torch
import torchvision
from torchvision import datasets, transforms


def load_fashion_mnist():
    path = 'data_f'
    torch.cuda.manual_seed(1)
    kwargs = {'num_workers': 1, 'pin_memory': True, 'drop_last': True}
    train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(path, train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    ])),
                batch_size=32, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(path, train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    ])),
                batch_size=32, shuffle=True, **kwargs)
    return train_loader, test_loader


def load_cifar(args):
    path = './data_c'
    kwargs = {'num_workers': 1, 'pin_memory': True, 'drop_last':True}
    transform_train = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5)),
        ])  
    transform_test = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])  
    trainset = torchvision.datasets.CIFAR10(root=path, train=True,
            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
            shuffle=True, **kwargs)
    testset = torchvision.datasets.CIFAR10(root=path, train=False,
            download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
            shuffle=False, **kwargs)
    return trainloader, testloader


def load_mnist(args):
    torch.cuda.manual_seed(1)
    kwargs = {'num_workers': 1, 'pin_memory': True, 'drop_last':True}
    path = 'data_m/'
    train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(path, train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    ])),
                batch_size=32, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(path, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                ])),
            batch_size=32, shuffle=True, **kwargs)
    return train_loader, test_loader
