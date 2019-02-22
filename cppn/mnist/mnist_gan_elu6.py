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
import samplers.mnist_sampler as sampler


def load_args():

    parser = argparse.ArgumentParser(description='param-wgan')
    parser.add_argument('--n', default=1, type=int, help='n images')
    parser.add_argument('--x_dim', default=28, type=int, help='out image width')
    parser.add_argument('--y_dim', default=28, type=int, help='out image height')
    parser.add_argument('--scale', default=10, type=int, help='mutiplier on z')
    parser.add_argument('--c_dim', default=1, type=int, help='channels')
    parser.add_argument('--net', default=128, type=int, help='net width')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--walk', default=False, type=bool)
    parser.add_argument('--walk_steps', default=10, type=int)
    parser.add_argument('--sample', default=False, type=bool)
    parser.add_argument('--z', default=32, type=int, help='latent space width')
    parser.add_argument('--dim', default=24, type=int, help='disc units')
    parser.add_argument('--l', default=10, type=int, help='gp')
    parser.add_argument('--disc_iters', default=5, type=int)
    parser.add_argument('--epochs', default=200000, type=int)
    parser.add_argument('--exp', default='0', type=str)
    parser.add_argument('--output', default=784, type=int)
    parser.add_argument('--dataset', default='mnist', type=str)
    parser.add_argument('--inference', default=False, type=bool)
    parser.add_argument('--load_iter', default='5000', type=str)
    parser.add_argument('--ae', default=False, type=bool)

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
        self.linear1 = nn.Linear(self.net, self.net)
        self.linear2 = nn.Linear(self.net, self.net)
        self.linear3 = nn.Linear(self.net, self.net)
        self.linear4 = nn.Linear(self.net, self.net)
        self.linear5 = nn.Linear(self.net, self.net)
        self.linear6 = nn.Linear(self.net, self.net)
        self.linear_out = nn.Linear(self.net, self.c_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        x, y, z, r = inputs
        n_points = self.x_dim * self.y_dim
        z_pt = self.linear_z(z.view(self.batch_size*n_points, self.z))
        x_pt = self.linear_x(x.view(self.batch_size*n_points, -1))
        y_pt = self.linear_y(y.view(self.batch_size*n_points, -1))
        r_pt = self.linear_r(r.view(self.batch_size*n_points, -1))
        x = z_pt + x_pt + y_pt + r_pt
        x = F.elu(x)
        x = F.elu(self.linear1(x))
        x = F.elu(self.linear2(x))
        x = F.elu(self.linear3(x))
        x = F.elu(self.linear4(x))
        x = F.elu(self.linear5(x))
        x = F.elu(self.linear6(x))
        x = torch.sigmoid(self.linear_out(x))
        x = x.view(self.batch_size, self.c_dim, self.y_dim, self.x_dim)
        return x


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'Discriminator'
        self.conv1 = nn.Conv2d(1, self.dim, 3, stride=2, padding=2)
        self.conv2 = nn.Conv2d(self.dim, 2*self.dim, 3, stride=2, padding=2)
        self.conv3 = nn.Conv2d(2*self.dim, 4*self.dim, 3, stride=2, padding=2)
        self.conv4 = nn.Conv2d(4*self.dim, 8*self.dim, 3, stride=2, padding=2)
        self.relu = nn.ReLU(inplace=True)
        self.linear1 = nn.Linear(8*4*4*self.dim, 1)
        self.ln1 = nn.LayerNorm([self.dim, 15, 15])
        self.ln2 = nn.LayerNorm([2*self.dim, 9, 9])
        self.ln3 = nn.LayerNorm([4*self.dim, 6, 6])
        self.ln4 = nn.LayerNorm([8*self.dim, 4, 4])


    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = x.view(-1, 8*4*4*self.dim)
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
    x_mat = torch.from_numpy(x_mat).float().cuda()
    y_mat = torch.from_numpy(y_mat).float().cuda()
    r_mat = torch.from_numpy(r_mat).float().cuda()
    return x_mat, y_mat, r_mat


def sample(args, netG, z):
    x_vec, y_vec, r_vec = coordinates(args)
    n_points = args.x_dim * args.y_dim
    ones = torch.ones(n_points, 1, dtype=torch.float).cuda()
    z_scaled = z.view(args.batch_size, 1, args.z) * ones * args.scale
    image = netG((x_vec, y_vec, z_scaled, r_vec))
    return image


def inf_gen(data_gen):
    while True:
        for images, targets in data_gen:
            images.requires_grad_(True)
            images = images.cuda()
            yield (images, targets)


def init(model):
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight.data, 0, 0.01)
    return model


def load_networks(args):
    exp = args.exp
    netG = Generator(args)
    path = 'mnist/'+exp+'_results/net'
    netG, _ = utils.load_model(path+'G_{}_'.format(args.load_iter)+exp, netG, None)
    netD = Discriminator(args)
    netD, _ = utils.load_model(path+'D_{}_'.format(args.load_iter)+exp, netD, None)
    if args.ae:
        netE = Encoder(args)
        netE, _ = utils.load_model(path+'E_{}_'.format(args.load_iter)+exp, netE, None)
        return netG, netE

    return netG#, netD


def run_sampler(args):
    networks = load_networks(args)
    if args.ae:
        sampler.cppn_ae(args, networks)
    else:
        sampler.cppn(args, networks)


def train(args):
    
    torch.manual_seed(8734)
    exp_dir = 'mnist/mnist_lr6_z'+str(args.z)+'n'+str(args.net)+'s'+str(args.scale) 
    netG = Generator(args).cuda()
    netD = Discriminator(args).cuda()
    print (netG, netD)

    optimG = optim.Adam(netG.parameters(), lr=1e-3, betas=(0.5, 0.9), weight_decay=1e-4)
    optimD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9), weight_decay=1e-4)
    
    mnist_train, mnist_test = datagen.load_mnist(args)
    train = inf_gen(mnist_train)
    reals, _ = next(train)
    
    one = torch.tensor(1.).cuda()
    mone = (one * -1)
    
    print ('==> Begin Training')
    for iter in range(args.epochs):
        ops.batch_zero_grad([netG, netD])
        for p in netD.parameters():
            p.requires_grad = True
        for _ in range(args.disc_iters):
            data, targets = next(train)
            data = data.view(args.batch_size, 28*28).cuda()
            netD.zero_grad()
            d_real = netD(data).mean()
            d_real.backward(mone, retain_graph=True)
            noise = torch.randn(args.batch_size, args.z, requires_grad=True).cuda()
            with torch.no_grad():
                fake = sample(args, netG, noise).view(args.batch_size, -1)
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
        fake = sample(args, netG, noise).view(args.batch_size, -1)
        G = netD(fake)
        G = G.mean()
        G.backward(mone)
        g_cost = -G
        optimG.step()
       
        if iter % 100 == 0:
            print('iter: ', iter, 'train D cost', d_cost.cpu().item())
            print('iter: ', iter, 'train G cost', g_cost.cpu().item())
        if iter % 200 == 0:
            val_d_costs = []
            with torch.no_grad():
                for i, (data, target) in enumerate(mnist_test):
                    data = data.cuda()
                    d = netD(data)
                    val_d_cost = -d.mean().item()
                    val_d_costs.append(val_d_cost)
                noise = torch.randn(args.batch_size, args.z, requires_grad=True).cuda()
                samples = sample(args, netG, noise)
                samples = samples.view(-1, 28, 28).cpu().data.numpy()
                path = exp_dir+'_results/gan_sample_{}.png'.format(iter)
                if not os.path.exists(exp_dir+'_results'):
                    os.makedirs(exp_dir+'_results')
                print ('saving sample: ', path)
                utils.save_images(samples, path)
        if iter % 5000 == 0:
            utils.save_model(exp_dir+'_results/netG_{}_{}'.format(iter, exp_dir), netG, optimG)
            utils.save_model(exp_dir+'_results/netD_{}_{}'.format(iter, exp_dir), netD, optimD)

if __name__ == '__main__':

    args = load_args()
    if args.inference:
        run_sampler(args)
    else:
        train(args)
