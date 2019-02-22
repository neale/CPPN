import os
import sys
import argparse
import numpy as np
import torch
import torchvision
from imageio import imwrite

from torch import nn
from torch import optim
from torch.nn import functional as F
from torchvision.utils import save_image

import ops
import utils
import datagen
from mnist_gan import Discriminator
from mnist_ae import Encoder

def load_args():

    parser = argparse.ArgumentParser(description='param-wgan')
    parser.add_argument('--n', default=1, type=int, help='n images')
    parser.add_argument('--x_dim', default=512, type=int, help='out image width')
    parser.add_argument('--y_dim', default=512, type=int, help='out image height')
    parser.add_argument('--scale', default=10, type=int, help='mutiplier on z')
    parser.add_argument('--c_dim', default=1, type=int, help='channels')
    parser.add_argument('--net', default=128, type=int, help='net width')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--walk', default=False, type=bool)
    parser.add_argument('--walk_steps', default=10, type=int)
    parser.add_argument('--sample', default=False, type=bool)
    parser.add_argument('--z', default=32, type=int, help='latent space width')
    parser.add_argument('--dim', default=32, type=int, help='latent space width')
    parser.add_argument('--l', default=10, type=int, help='latent space width')
    parser.add_argument('--disc_iters', default=5, type=int)
    parser.add_argument('--epochs', default=200000, type=int)
    parser.add_argument('--exp', default='0', type=str)
    parser.add_argument('--output', default=784, type=int)
    parser.add_argument('--load_iter', default='5000', type=str)
    parser.add_argument('--dataset', default='mnist', type=str)
    parser.add_argument('--ae', default=False, type=bool)

    args = parser.parse_args()
    return args


def load_networks(args):
    exp = args.exp
    netG = Generator(args)
    netG, _ = utils.load_model(exp+'_results/netG_{}_'.format(args.load_iter)+exp, netG, None)
    netD = Discriminator(args)
    netD, _ = utils.load_model(exp+'_results/netD_{}_'.format(args.load_iter)+exp, netD, None)
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
    n_points = args.x_dim * args.y_dim
    ones = torch.ones(n_points, 1, dtype=torch.float)
    z_scaled = z.view(args.batch_size, 1, args.z) * ones * args.scale
    image = netG((x_vec, y_vec, z_scaled, r_vec))
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


def cppn(args, networks):
    netG = networks
    n_images = args.n
    n_walk = args.walk_steps
    netG = netG.eval()
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
            img = sample(args, netG, z).cpu()#.detach().numpy()
            if args.c_dim == 1:
                img = img[0][0]
            else:
                img = img[0]
            #img = (img*255).astype(np.uint8)
            #img = np.invert(img)
            #imwrite('cppn_gan_{}_{}.png'.format(args.exp, i), img)
            img = 1 - img
            save_image(img, 'cppn_gan_{}_{}.png'.format(args.exp, i))


def cppn_ae(args, networks):
    netG, netE = networks
    n_images = args.n
    n_walk = args.walk_steps
    netG.eval()
    netE.eval()
    train_gen, test_gen = datagen.load_mnist(args)
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
        for i, z in enumerate(zs):
            img = sample(args, netG, z).cpu().detach().numpy()
            if args.c_dim == 1:
                img = img[0][0]
            else:
                img = img[0].reshape((args.x_dim, args.y_dim, args.c_dim))
            img = (img*255).astype(np.uint8)
            img = np.invert(img)
            imwrite('cppn_ae-gan_{}_{}.png'.format(args.exp, i), img)


if __name__ == '__main__':

    args = load_args()
    networks = load_networks(args)
    if args.ae == True:
        cppn_ae(args, networks)
    else:
        cppn(args, networks)
