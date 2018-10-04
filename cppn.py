import os
import sys
import argparse
import numpy as np
import torch

from torch import nn
from torch import optim
from torch.nn import functional as F
from imageio import imwrite


def load_args():

    parser = argparse.ArgumentParser(description='cppn-pytorch')
    parser.add_argument('--z', default=8, type=int, help='latent space width')
    parser.add_argument('--n', default=1, type=int, help='images to generate')
    parser.add_argument('--x_dim', default=2048, type=int, help='out image width')
    parser.add_argument('--y_dim', default=2048, type=int, help='out image height')
    parser.add_argument('--scale', default=10, type=int, help='mutiplier on z')
    parser.add_argument('--c_dim', default=1, type=int, help='channels')
    parser.add_argument('--net', default=32, type=int, help='net width')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--exp', default='0', type=str, help'output fn')

    parser.add_argument('--walk', default=False, type=bool, help='interpolate')
    parser.add_argument('--sample', default=False, type=bool, help='sample n images')

    args = parser.parse_args()
    return args


class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'Generator'
        dim = self.x_dim * self.y_dim * self.batch_size
        self.linear_z = nn.Linear(self.z, self.net)
        self.linear_x = nn.Linear(1, self.net, bias=False)
        self.linear_y = nn.Linear(1, self.net, bias=False)
        self.linear_r = nn.Linear(1, self.net, bias=False)
        self.linear_h = nn.Linear(self.net, self.net)
        self.linear_out = nn.Linear(self.net, self.c_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        #print ('G in: ', x.shape)
        x, y, z, r = inputs
        n_points = self.x_dim * self.y_dim
        ones = torch.ones(n_points, 1, dtype=torch.float)#.cuda()
        z_scaled = z.view(self.batch_size, 1, self.z) * ones * self.scale
        z_pt = self.linear_z(z_scaled.view(self.batch_size*n_points, self.z))
        x_pt = self.linear_x(x.view(self.batch_size*n_points, -1))
        y_pt = self.linear_y(y.view(self.batch_size*n_points, -1))
        r_pt = self.linear_r(r.view(self.batch_size*n_points, -1))
        U = z_pt + x_pt + y_pt + r_pt
        H = F.tanh(U)
        H = F.elu(self.linear_h(H))
        H = F.softplus(self.linear_h(H))
        H = F.tanh(self.linear_h(H))
        #x = self.sigmoid(self.linear_out(H))
        x = .5 * torch.sin(self.linear_out(H)) + .5
        x = x.view(self.batch_size, self.c_dim, self.y_dim, self.x_dim)
        #print ('G out: ', x.shape)
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


def sample(args, netG, z):
    x_vec, y_vec, r_vec = coordinates(args)
    image = netG((x_vec, y_vec, z, r_vec))
    return image


def init(model):
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight.data)
    return model


def latent_walk(args, z1, z2, n_frames, netG):
    delta = (z2 - z1) / (n_frames + 1)
    total_frames = n_frames + 2
    states = []
    for i in range(total_frames):
        z = z1 + delta * float(i)
        if args.c_dim == 1:
            states.append(sample(args, netG, z)[0][0]*255)
        else:
            states.append(sample(args, netG, z)[0].view(
                args.x_dim, args.y_dim, args.c_dim)*255)
    states = torch.stack(states).detach().numpy()
    return states

        
def cppn(args):
    netG = init(Generator(args))
    print (netG)
    n_images = args.n
    zs = []
    for _ in range(n_images):
        zs.append(torch.zeros(1, args.z).uniform_(-1.0, 1.0))

    if args.walk:
        k = 0
        for i in range(n_images):
            if i+1 not in range(n_images):
                images = latent_walk(args, zs[i], zs[0], 50, netG)
                break
            images = latent_walk(args, zs[i], zs[i+1], 50, netG)
            for img in images:
                imwrite('{}_{}.jpg'.format(args.exp, k), img)
                k += 1
            print ('walked {}/{}'.format(i+1, n_images))

    if args.sample:
        zs, _ = torch.stack(zs).sort()
        print (zs.shape)
        for i, z in enumerate(zs):
            img = sample(args, netG, z).cpu().detach().numpy()
            if args.c_dim == 1:
                img = img[0][0]
            else:
                img = img[0].reshape((args.x_dim, args.y_dim, args.c_dim))
            imwrite('{}_{}.png'.format(args.exp, i), img*255)
    
if __name__ == '__main__':

    args = load_args()
    cppn(args)
