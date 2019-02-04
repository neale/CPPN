import os
import sys
import argparse
import numpy as np
import torch

import utils, datagen, ops

from torch import nn
from torch import optim
from torch.nn import functional as F
from imageio import imwrite


def load_args():

    parser = argparse.ArgumentParser(description='cppn-wgan-gp')
    parser.add_argument('--z', default=128, type=int, help='latent space width')
    parser.add_argument('--n', default=1, type=int, help='n images')
    parser.add_argument('--l', default=10, type=int, help='gp weight')
    parser.add_argument('--epochs', default=10000, type=int, help='gan training rounds')
    parser.add_argument('--x_dim', default=2048, type=int, help='out image width')
    parser.add_argument('--y_dim', default=2048, type=int, help='out image height')
    parser.add_argument('--scale', default=10, type=int, help='mutiplier on z')
    parser.add_argument('--c_dim', default=1, type=int, help='channels')
    parser.add_argument('--net', default=32, type=int, help='net width')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--exp', default='0', type=str)
    parser.add_argument('--walk', default=False, type=bool)
    parser.add_argument('--walk_steps', default=10, type=int)
    parser.add_argument('--sample', default=False, type=bool)

    args = parser.parse_args()
    return args


class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'Generator'
        self.linear_z = nn.Linear(self.z, self.net)
        self.linear_x = nn.Linear(1, self.net, bias=False)
        self.linear_y = nn.Linear(1, self.net, bias=False)
        self.linear_r = nn.Linear(1, self.net, bias=False)
        self.linear_h = nn.Linear(self.net, self.net)
        self.linear_out = nn.Linear(self.net, self.c_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, args, inputs):
        x, y, z, r = inputs
        n_points = args.x_dim * args.y_dim
        ones = torch.ones(n_points, 1, dtype=torch.float)
        if args.gan is True: ones = ones.cuda(); z = z.cuda()
        else: ones = ones.cpu(); z = z.cpu()
        z_scaled = z.view(args.batch_size, 1, self.z) * ones * args.scale
        z_pt = self.linear_z(z_scaled.view(args.batch_size*n_points, self.z))
        x_pt = self.linear_x(x.view(args.batch_size*n_points, -1))
        y_pt = self.linear_y(y.view(args.batch_size*n_points, -1))
        r_pt = self.linear_r(r.view(args.batch_size*n_points, -1))
        U = z_pt + x_pt + y_pt + r_pt
        H = torch.tanh(U)
        H = F.tanh(self.linear_h(H))
        H = F.tanh(self.linear_h(H))
        H = F.tanh(self.linear_h(H))
        #x = self.sigmoid(self.linear_out(H))
        x = torch.sigmoid(self.linear_out(H))
        x = x.view(args.batch_size, self.c_dim, args.y_dim, args.x_dim)
        #print ('G out: ', x.shape)
        return x


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'Discriminator'
        self.conv1 = nn.Conv2d(1, self.z, 5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(self.z, 2*self.z, 5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(2*self.z, 4*self.z, 5, stride=2, padding=2)
        self.relu = nn.ELU(inplace=True)
        self.linear1 = nn.Linear(4*4*4*self.z, 1)

    def forward(self, x):
        # print ('D in: ', x.shape)
        x = x.view(-1, 1, 28, 28)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(-1, 4*4*4*self.z)
        x = self.linear1(x)
        x = x.view(-1)
        # print ('D out: ', x.shape)
        return x


def coordinates(args):
    x_dim, y_dim, scale = args.x_dim, args.y_dim, args.scale
    n_points = x_dim * y_dim
    x_range = scale*(np.arange(x_dim)-(x_dim-1)/2.0)/(x_dim-1)/0.5
    y_range = scale*(np.arange(y_dim)-(y_dim-1)/2.0)/(y_dim-1)/0.5
    x_mat = np.matmul(np.ones((y_dim, 1)), x_range.reshape((1, x_dim)))
    y_mat = np.matmul(y_range.reshape((y_dim, 1)), np.ones((1, x_dim)))
    r_mat = np.sqrt(x_mat*x_mat + y_mat*y_mat)
    x_mat = np.tile(x_mat.flatten(), args.batch_size).reshape(args.batch_size, n_points, 1)
    y_mat = np.tile(y_mat.flatten(), args.batch_size).reshape(args.batch_size, n_points, 1)
    r_mat = np.tile(r_mat.flatten(), args.batch_size).reshape(args.batch_size, n_points, 1)
    x_mat = torch.from_numpy(x_mat).float()#.cuda()
    y_mat = torch.from_numpy(y_mat).float()#.cuda()
    r_mat = torch.from_numpy(r_mat).float()#.cuda()
    return x_mat, y_mat, r_mat


def sample(args, netG, z, gan=False):
    if args.gan is True:
        args.x_dim = 28
        args.y_dim = 28
        args.scale = 1
    else:
        args.x_dim = 1024
        args.y_dim = 1024
        args.scale = 10
    x_vec, y_vec, r_vec = coordinates(args)
    if args.gan is True:
        x_vec = x_vec.cuda()
        y_vec = y_vec.cuda()
        r_vec = r_vec.cuda()
    image = netG(args, (x_vec, y_vec, z, r_vec))
    return image


def init(model):
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight.data)
    return model


def latent_walk(args, z1, z2, n_frames, netG, k):
    delta = (z2 - z1) / (n_frames + 1)
    total_frames = n_frames + 2
    states = []
    for i in range(total_frames):
        z = z1 + delta * float(i)
        if args.c_dim == 1:
            img = sample(args, netG, z)[0][0]*255
        else:
            img = sample(args, netG, z)[0].view(
                args.x_dim, args.y_dim, args.c_dim)*255

        img = img.detach().numpy()
        imwrite('{}_{}.jpg'.format(args.exp, k), img)
        k += 1
    return k


def cppn(args, netG, iter, zs):
    n_images = args.n
    n_walk = args.walk_steps
    netG = netG.cpu()
    if args.walk:
        k = 0
        for i in range(n_images):
            if i+1 not in range(n_images):
                print (k, ' images so far')
                k = latent_walk(args, zs[i], zs[0], n_walk, netG, k)
                break
            else:
                print (k, ' images so far')
                k = latent_walk(args, zs[i], zs[i+1], n_walk, netG, k)
            print ('walked {}/{}'.format(i+1, n_images))

    if args.sample:
        for i, z in enumerate(zs):
            img = sample(args, netG, z).cpu().detach().numpy()
            if args.c_dim == 1:
                img = img[0][0]
            else:
                img = img[0].reshape((args.x_dim, args.y_dim, args.c_dim))
            try:
                path = 'gan_training/cppn_sample_{}.png'.format(iter)
                #imwrite('cppn_gan_{}_{}-{}.png'.format(args.exp, i, iter), img*255)
                imwrite(path, (img*255).astype(np.int))
            except ValueError:
                continue
    netG = netG.cuda()

def inf_gen(data_gen):
    while True:
        for images, targets in data_gen:
            images.requires_grad_(True)
            images = images.cuda()
            yield (images, targets)


def train_gan(args):
    netG = init(Generator(args)).cuda()
    netD = Discriminator(args).cuda()
    optimG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9), weight_decay=1e-4)
    optimD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9), weight_decay=1e-4)
    mnist_train, mnist_test = datagen.load_mnist(args)
    train = inf_gen(mnist_train)
    print ('saving reals')
    reals, _ = next(train)
    utils.save_images(reals.detach().cpu().numpy(), 'gan_training/reals.png')

    one = torch.tensor(1.).cuda()
    mone = one * -1
    args.batch_size = 32
    args.gan = True
    print ('==> Begin Training')
    for iter in range(args.epochs):
        netG.zero_grad(); netD.zero_grad()
        for p in netD.parameters():
            p.requires_grad = True
        for _ in range(5):
            data, targets = next(train)
            data = data.view(args.batch_size, 28*28).cuda()
            netD.zero_grad()
            d_real = netD(data).mean()
            d_real.backward(mone, retain_graph=True)
            noise = torch.randn(args.batch_size, args.z, requires_grad=True).cuda()
            fake = []
            with torch.no_grad():
                fake = sample(args, netG, noise, gan=True).view(32, -1)
            fake.requires_grad_(True)
            d_fake = netD(fake)
            d_fake = d_fake.mean()
            d_fake.backward(one, retain_graph=True)
            gp = ops.grad_penalty_1dim(args, netD, data, fake)
            gp.backward()
            d_cost = d_fake - d_real + gp
            wasserstein_d = d_real - d_fake
            optimD.step()

        for p in netD.parameters():
            p.requires_grad=False
        netG.zero_grad()
        noise = torch.randn(args.batch_size, args.z, requires_grad=True).cuda()
        fake = []
        for z in noise:
            fake.append(sample(args, netG, noise, gan=True))
        fake = torch.stack(fake)
        G = netD(fake)
        G = G.mean()
        G.backward(mone)
        g_cost = -G
        optimG.step()
        if iter % 100 == 0:
            with torch.no_grad():
                noise = torch.randn(args.batch_size, args.z, requires_grad=True).cuda()
                samples = sample(args, netG, noise, gan=True)
                samples = samples.view(-1, 28, 28).cpu().data.numpy()
                path = 'gan_training/gan_sample_{}.png'.format(iter)
                print ('saving gan sample: ', path)
                utils.save_images(samples, path)
            args.gan = False; args.batch_size = 1 # accomodate large images
            cppn(args, netG, iter, noise[:args.n])
            args.batch_size = 32; args.gan = True
            print('iter: ', iter, 'G cost', g_cost.cpu().item())
            print('iter: ', iter, 'D cost', d_cost.cpu().item())
if __name__ == '__main__':

    args = load_args()
    train_gan(args)
