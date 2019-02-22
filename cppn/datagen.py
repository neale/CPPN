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
    path = '/scratch/eecs-share/ratzlafn/data_c'
    kwargs = {'num_workers': 1, 'pin_memory': True, 'drop_last':True}
    transform_train = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5)),
        ])  
    transform_test = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])  
    trainset = torchvision.datasets.CIFAR10(root=path, train=True,
            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
            shuffle=True, **kwargs)
    testset = torchvision.datasets.CIFAR10(root=path, train=False,
            download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
            shuffle=False, **kwargs)
    return trainloader, testloader


def load_mnist(args):
    torch.cuda.manual_seed(1)
    kwargs = {'num_workers': 1, 'pin_memory': True, 'drop_last':True}
    path = '/scratch/eecs-share/ratzlafn/data_m'
    train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(path, train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    ])),
                batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(path, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
    return train_loader, test_loader


def load_emnist(args):
    torch.cuda.manual_seed(1)
    path = '/scratch/eecs-share/ratzlafn/data_e'
    kwargs = {'num_workers': 1, 'pin_memory': True, 'drop_last':True}
    train_loader = torch.utils.data.DataLoader(
            datasets.EMNIST(path, split='letters', train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    ])),
                batch_size=32, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
            datasets.EMNIST(path, split='letters', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                ])),
            batch_size=32, shuffle=True, **kwargs)
    return train_loader, test_loader


def load_cifar_hidden(args, c_idx):
    path = '/scratch/eecs-share/ratzlafn/data_c'
    kwargs = {'num_workers': 2, 'pin_memory': True, 'drop_last': True}
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])  
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])  
    def get_classes(target, labels):
        label_indices = []
        for i in range(len(target)):
            if target[i][1] in labels:
                label_indices.append(i)
        return label_indices

    trainset = torchvision.datasets.CIFAR10(root=path, train=True,
            download=False, transform=transform_train)
    train_hidden = torch.utils.data.Subset(trainset, get_classes(trainset, c_idx))
    trainloader = torch.utils.data.DataLoader(train_hidden, batch_size=args.batch_size,
            shuffle=True, **kwargs)
    testset = torchvision.datasets.CIFAR10(root=path, train=False,
            download=False, transform=transform_test)
    test_hidden = torch.utils.data.Subset(testset, get_classes(testset, c_idx))
    testloader = torch.utils.data.DataLoader(test_hidden, batch_size=args.batch_size,
            shuffle=False, **kwargs)
    return trainloader, testloader


def load_latin(args):
    torch.cuda.manual_seed(1)
    kwargs = {'num_workers': 1, 'pin_memory': True, 'drop_last':True}
    loader = torch.utils.data.DataLoader(
            datasets.ImageFolder('Latin', 
                transform=transforms.Compose([
                    transforms.Grayscale(1),
                    transforms.ToTensor()
                    ])),
                batch_size=32, shuffle=True, **kwargs)
    return loader


def load_small_celeba(args):
    torch.cuda.manual_seed(1)
    kwargs = {'num_workers': 1, 'pin_memory': True, 'drop_last':True}
    loader = torch.utils.data.DataLoader(
            datasets.ImageFolder('/scratch/eecs-share/ratzlafn/celeba/val', 
                transform=transforms.Compose([
                    transforms.CenterCrop((108, 108)),
                    transforms.Resize((64, 64)),
                    transforms.ToTensor()
                    ])),
                batch_size=args.batch_size, shuffle=True, **kwargs)
    return loader


def load_mini_celeba(args):
    torch.cuda.manual_seed(1)
    kwargs = {'num_workers': 1, 'pin_memory': True, 'drop_last':True}
    loader = torch.utils.data.DataLoader(
            datasets.ImageFolder('/scratch/eecs-share/ratzlafn/celeba/transformed', 
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((.5, .5, .5), (.5, .5, .5)),
                    ])),
                batch_size=args.batch_size, shuffle=True, **kwargs)
    return loader


def load_celeba(args):
    torch.cuda.manual_seed(1)
    kwargs = {'num_workers': 0, 'pin_memory': True, 'drop_last':True}
    loader = torch.utils.data.DataLoader(
            datasets.ImageFolder('/scratch/eecs-share/ratzlafn/celeba', 
                transform=transforms.Compose([
                    transforms.CenterCrop((108, 108)),
                    transforms.Resize((64, 64)),
                    transforms.ToTensor()
                    ])),
                batch_size=args.batch_size, shuffle=True, **kwargs)
    return loader
