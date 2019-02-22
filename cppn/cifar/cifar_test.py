import os
import sys
import argparse
import numpy as np
import torch
import torchvision

from torch import nn
from torch import optim
from torch.nn import functional as F

import ops
import utils
import datagen


def load_args():

    parser = argparse.ArgumentParser(description='param-wgan')
    parser.add_argument('--n', default=1, type=int, help='n images')
    parser.add_argument('--x_dim', default=32, type=int, help='out image width')
    parser.add_argument('--y_dim', default=32, type=int, help='out image height')
    parser.add_argument('--scale', default=10, type=int, help='mutiplier on z')
    parser.add_argument('--c_dim', default=3, type=int, help='channels')
    parser.add_argument('--net', default=128, type=int, help='net width')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--walk', default=False, type=bool)
    parser.add_argument('--walk_steps', default=10, type=int)
    parser.add_argument('--sample', default=False, type=bool)
    parser.add_argument('--z', default=64, type=int, help='latent space width')
    parser.add_argument('--l', default=10, type=int, help='gp penalty')
    parser.add_argument('--disc_iters', default=5, type=int)
    parser.add_argument('--epochs', default=500000, type=int)
    parser.add_argument('--exp', default='0', type=str)
    parser.add_argument('--dataset', default='cifar', type=str)
    parser.add_argument('--ngpu', default=2, type=int)

    args = parser.parse_args()
    return args


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.conv1 = nn.Conv2d(3, self.z, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(self.z, 2*self.z, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(2*self.z, 4*self.z, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(4*self.z, 8*self.z, 3, stride=2, padding=1)
        self.linear = nn.Linear(2*4*4*self.z, self.z)
        self.relu = nn.LeakyReLU(True)
        self.dropout = nn.Dropout2d(.3)


    def forward(self, input):
        input = input.view(-1, 3, 32, 32)
        x = self.dropout(self.relu(self.conv1(input)))
        x = self.dropout(self.relu(self.conv2(x)))
        x = self.dropout(self.relu(self.conv3(x)))
        x = self.dropout(self.relu(self.conv4(x)))
        x = x.view(self.batch_gpu, -1)
        x = self.linear(x)
        return x.view(-1, self.z)


class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'Generator'
        self.linear1 = nn.Linear(self.z, 4*4*4*self.z)
        self.conv1 = nn.ConvTranspose2d(4*self.z, 2*self.z, 2, stride=2)
        self.conv2 = nn.ConvTranspose2d(2*self.z, self.z, 2, stride=2)
        self.conv3 = nn.ConvTranspose2d(self.z, 3, 2, stride=2)
        self.bn0 = nn.BatchNorm1d(4*4*4*self.z)
        self.bn1 = nn.BatchNorm2d(2*self.z)
        self.bn2 = nn.BatchNorm2d(self.z)
        self.relu = nn.ELU(inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.bn0(self.linear1(x)))
        x = x.view(-1, 4*self.z, 4, 4)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        x = self.tanh(x)
        x = x.view(-1, 3, 32, 32)
        return x


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'Discriminator'
        self.conv1 = nn.Conv2d(3, self.z, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(self.z, 2*self.z, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(2*self.z, 4*self.z, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(4*self.z, 8*self.z, 3, stride=2, padding=1)
        self.relu = nn.LeakyReLU(inplace=True)
        self.linear1 = nn.Linear(4*4*2*self.z, 1)
        self.bn1 = nn.BatchNorm2d(self.z)
        self.bn2 = nn.BatchNorm2d(self.z*2)
        self.bn3 = nn.BatchNorm2d(self.z*4)
        self.bn4 = nn.BatchNorm2d(self.z*8)
        self.bn5 = nn.BatchNorm2d(self.z*8)

    def forward(self, x):
        x = x.view(-1, 3, 32, 32)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = x.view(self.batch_gpu, -1)
        x = self.linear1(x)
        x = x.view(-1)
        return x


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.1)
    if classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.1)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def inf_gen(data_gen):
    while True:
        for images, targets in data_gen:
            images.requires_grad_(True)
            images = images.cuda()
            yield (images, targets)


def init(model):
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight.data)
    return model


def train(args):
    
    torch.manual_seed(8734)
    exp_dir = 'cifar_wgan_test' 
    args.batch_size *= args.ngpu
    args.batch_gpu = args.batch_size//args.ngpu
    netE = torch.nn.DataParallel(Encoder(args), device_ids=list(range(args.ngpu))).cuda()
    netG = torch.nn.DataParallel(Generator(args), device_ids=list(range(args.ngpu))).cuda()
    netD = torch.nn.DataParallel(Discriminator(args), device_ids=list(range(args.ngpu))).cuda()
    print (netE, netG, netD)

    #netE.apply(weight_init)
    #netG.apply(weight_init)
    #netD.apply(weight_init)

    optimE = optim.Adam(netE.parameters(), lr=1e-4, betas=(0.5, 0.9), weight_decay=1e-4)
    optimG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9), weight_decay=1e-4)
    optimD = optim.Adam(netD.parameters(), lr=1e-5, betas=(0.5, 0.9), weight_decay=1e-4)
    
    cifar_train, cifar_test = datagen.load_cifar_hidden(args, [5])
    train = inf_gen(cifar_train)
    one = torch.tensor(1.).cuda()
    mone = (one * -1)
    
    print ('==> Begin Training')
    for iter in range(args.epochs):
        ops.batch_zero_grad([netG, netD, netE])
        for p in netD.parameters():
            p.requires_grad = False
        data, targets = next(train)
        data = data.cuda()
        fake = netG(netE(data))
        ae_loss = F.mse_loss(fake, data)
        ae_loss.backward(one)
        optimE.step()
        optimG.step()

        for p in netD.parameters():
            p.requires_grad = True
        for _ in range(args.disc_iters):
            data, targets = next(train)
            data = data.view(args.batch_size, 3, 32, 32).cuda()
            netD.zero_grad()
            d_real = netD(data).mean()
            d_real.backward(mone, retain_graph=True)
            noise = torch.randn(args.batch_size, args.z, requires_grad=True).cuda()
            with torch.no_grad():
                fake = netG(noise).view(args.batch_size, -1)
            fake.requires_grad_(True)
            d_fake = netD(fake)
            d_fake = d_fake.mean()
            d_fake.backward(one, retain_graph=True)
            gp = ops.grad_penalty_3dim(args, netD, data, fake)
            gp.backward()
            d_cost = d_fake - d_real + gp
            wasserstein_d = d_real - d_fake
            optimD.step()

        for p in netD.parameters():
            p.requires_grad=False
        netG.zero_grad()
        noise = torch.randn(args.batch_size, args.z, requires_grad=True).cuda()
        fake = netG(noise).view(args.batch_size, -1)
        G = netD(fake)
        G = G.mean()
        G.backward(mone)
        g_cost = -G
        optimG.step()
       
        if iter % 100 == 0:
            print('iter: ', iter, 'train D cost', d_cost.cpu().item())
            print('iter: ', iter, 'train G cost', g_cost.cpu().item())
        if iter % 200 == 0:
            with torch.no_grad():
                ae_samples = netG(netE(data)).mul(.5).add(.5)
                ae_samples = ae_samples.view(-1, 3, 32, 32).cpu().data.numpy()
                noise = torch.randn(args.batch_size, args.z, requires_grad=True).cuda()
                r_samples = netG(noise).mul(.5).add(.5)
                r_samples = r_samples.view(-1, 3, 32, 32).cpu().data.numpy()
                rpath = exp_dir+'_results/gan_sample_{}.png'.format(iter)
                aepath = exp_dir+'_results/ae-gan_sample_{}.png'.format(iter)
                if not os.path.exists(exp_dir+'_results'):
                    os.makedirs(exp_dir+'_results')
                print ('saving ae-gan sample: ', rpath)
                utils.save_images(r_samples, rpath)
                utils.save_images(ae_samples, aepath)
        if iter % 5000 == 0:
            utils.save_model(exp_dir+'_results/netG_{}_{}'.format(iter, exp_dir), netG, optimG)
            utils.save_model(exp_dir+'_results/netD_{}_{}'.format(iter, exp_dir), netD, optimD)
            utils.save_model(exp_dir+'_results/netE_{}_{}'.format(iter, exp_dir), netE, optimE)

if __name__ == '__main__':

    args = load_args()
    train(args)
