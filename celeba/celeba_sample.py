import os
import sys
import argparse
import numpy as np
import torch
import torchvision
from PIL import Image
from imageio import imwrite

from torch import nn
from torch import optim
from torch.nn import functional as F
from torchvision.utils import save_image

import ops
import utils
import datagen
from celeba_ae import Encoder

def load_args():

    parser = argparse.ArgumentParser(description='param-wgan')
    parser.add_argument('--n', default=1, type=int, help='n images')
    parser.add_argument('--x_dim', default=512, type=int, help='out image width')
    parser.add_argument('--y_dim', default=512, type=int, help='out image height')
    parser.add_argument('--scale', default=10, type=int, help='mutiplier on z')
    parser.add_argument('--c_dim', default=3, type=int, help='channels')
    parser.add_argument('--net', default=512, type=int, help='net width')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--walk', default=False, type=bool)
    parser.add_argument('--walk_steps', default=10, type=int)
    parser.add_argument('--sample', default=False, type=bool)
    parser.add_argument('--z', default=100, type=int, help='latent space width')
    parser.add_argument('--dim', default=100, type=int, help='latent space width')
    parser.add_argument('--l', default=10, type=int, help='latent space width')
    parser.add_argument('--epochs', default=200000, type=int)
    parser.add_argument('--exp', default='0', type=str)
    parser.add_argument('--load_iter', default='5000', type=str)
    parser.add_argument('--dataset', default='celeba', type=str)
    parser.add_argument('--ae', default=False, type=bool)
    parser.add_argument('--user', default='./', type=str)


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
        self.linear_h1 = nn.Linear(self.net, self.net)
        self.linear_h2 = nn.Linear(self.net, self.net)
        self.linear_h3 = nn.Linear(self.net, self.net)
        self.linear_h4 = nn.Linear(self.net, self.net)
        self.linear_h5 = nn.Linear(self.net, self.net)
        self.linear_h6 = nn.Linear(self.net, self.net)
        self.linear_out = nn.Linear(self.net, self.c_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, args, inputs):
        x, y, z, r = inputs
        n_points = args.x_dim * args.y_dim
        ones = torch.ones(n_points, 1, dtype=torch.float)
        z_scaled = z.view(args.batch_size, 1, self.z) * ones * args.scale
        z_pt = self.linear_z(z_scaled.view(args.batch_size*n_points, self.z))
        x_pt = self.linear_x(x.view(args.batch_size*n_points, -1))
        y_pt = self.linear_y(y.view(args.batch_size*n_points, -1))
        r_pt = self.linear_r(r.view(args.batch_size*n_points, -1))
        U = z_pt + x_pt + y_pt + r_pt
        H = F.softplus(U)
        H = torch.tanh(self.linear_h1(H))
        H = torch.tanh(self.linear_h2(H))
        H = torch.tanh(self.linear_h3(H))
        H = torch.tanh(self.linear_h4(H))
        H = torch.tanh(self.linear_h5(H))
        H = F.elu(self.linear_h6(H))
        x = torch.sigmoid(self.linear_out(H))
        x = x.view(args.batch_size, self.c_dim, args.y_dim, args.x_dim)
        return x


def inf_gen(data_gen):
    while True:
        for images, targets in data_gen:
            images.requires_grad_(True)
            images = images.cuda()
            yield (images, targets)


def load_networks(args):
    exp = args.exp
    netG = Generator(args)
    netG, _ = utils.load_model(exp+'_results/netG_{}_'.format(args.load_iter)+exp, netG, None)
    if args.ae:
        netE = Encoder(args)
        netE, _ = utils.load_model(exp+'_results/netE_{}_'.format(args.load_iter)+exp, netE, None)
        return netG, netE

    return netG, netD


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
    x_mat = torch.from_numpy(x_mat).float()
    y_mat = torch.from_numpy(y_mat).float()
    r_mat = torch.from_numpy(r_mat).float()
    return x_mat, y_mat, r_mat


def sample(args, netG, z):
    x_vec, y_vec, r_vec = coordinates(args)
    image = netG(args, (x_vec, y_vec, z, r_vec))
    return image


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

        img = img.cpu().detach().numpy().astype(np.uint8)
        #img = np.invert(img)
        imwrite('{}_{}.jpg'.format(args.exp, k), img)
        k += 1
    return k


def cppn(args):
    netG, _ = load_networks(args)
    n_images = args.n
    n_walk = args.walk_steps
    netG = netG.eval()
    celeba_gen = datagen.load_mini_celeba(args)
    zs = []
    for _ in range(n_images):
        zs.append(torch.randn((1, args.z)))
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
            img = (img*255).astype(np.uint8)
            img = np.invert(img)
            imwrite('cppn_gan_{}_{}.png'.format(args.exp, i), img)


def cppn_ae(args):
    netG, netE = load_networks(args)
    n_images = args.n
    n_walk = args.walk_steps
    netG.eval()
    netE.eval()
    print ('loaded models')
    train_gen = datagen.load_small_celeba(args)
    print ('loaded data')
    if args.walk:
        run = 0
        zs = torch.zeros(10, 32)
        for i, (data, targets) in enumerate(train_gen):
            for img, y in zip(data, targets):
                if y == run:
                    zs[y] = netE(img)
                    run += 1
            if run > 9:
                break
    else:
        zs = []
        for i, (data, _) in enumerate(train_gen):
            print (data.shape)
            zs.append(netE(data))
            if i*len(data) >= args.n:
                zs = torch.stack(zs).view(-1, args.z)[:args.n]
                break
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
        if args.user:
            path = args.user
            zs = netE(torchvision.transforms.ToTensor()(Image.open(path)))
        for i, z in enumerate(zs):
            img = sample(args, netG, z)
            save_image(img[0], 'celeba_ae-gan_{}_{}.png'.format(args.exp, i))


if __name__ == '__main__':

    args = load_args()
    if args.ae == True:
        cppn_ae(args)
    else:
        cppn(args)
