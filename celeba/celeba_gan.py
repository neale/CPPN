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
    parser.add_argument('--x_dim', default=64, type=int, help='out image width')
    parser.add_argument('--y_dim', default=64, type=int, help='out image height')
    parser.add_argument('--scale', default=10, type=int, help='mutiplier on z')
    parser.add_argument('--c_dim', default=3, type=int, help='channels')
    parser.add_argument('--net', default=512, type=int, help='net width')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--walk', default=False, type=bool)
    parser.add_argument('--walk_steps', default=10, type=int)
    parser.add_argument('--sample', default=False, type=bool)
    parser.add_argument('--z', default=128, type=int, help='latent space width')
    parser.add_argument('--dim', default=128, type=int, help='latent space width')
    parser.add_argument('--l', default=10, type=int, help='latent space width')
    parser.add_argument('--disc_iters', default=5, type=int)
    parser.add_argument('--epochs', default=500000, type=int)
    parser.add_argument('--exp', default='0', type=str)
    parser.add_argument('--dataset', default='celeba', type=str)
    parser.add_argument('--ngpu', default=1, type=int)

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
        self.linear_h7 = nn.Linear(self.net, self.net)
        self.linear_out = nn.Linear(self.net, self.c_dim)
        self.sigmoid = nn.Sigmoid()
        """
        self.in1 = nn.InstanceNorm1d(self.net)
        self.in2 = nn.InstanceNorm1d(self.net)
        self.in3 = nn.InstanceNorm1d(self.net)
        self.in4 = nn.InstanceNorm1d(self.net)
        self.in5 = nn.InstanceNorm1d(self.net)
        self.in6 = nn.InstanceNorm1d(self.net)
        #self.in7 = nn.InstanceNorm1d(self.net)
        """
    def forward(self, args, inputs):
        x, y, z, r = inputs
        n_points = args.x_dim * args.y_dim
        ones = torch.ones(n_points, 1, dtype=torch.float).cuda()
        z_scaled = z.view(args.batch_gpu, 1, self.z) * ones * args.scale
        z_pt = self.linear_z(z_scaled.view(args.batch_gpu*n_points, self.z))
        x_pt = self.linear_x(x.view(args.batch_gpu*n_points, -1))
        y_pt = self.linear_y(y.view(args.batch_gpu*n_points, -1))
        r_pt = self.linear_r(r.view(args.batch_gpu*n_points, -1))
        U = z_pt + x_pt + y_pt + r_pt
        H = F.softplus(U)
        """
        H = F.leaky_relu(self.in1(self.linear_h1(H)))
        H = torch.tanh(self.in2(self.linear_h2(H)))
        H = F.leaky_relu(self.in3(self.linear_h3(H)))
        H = torch.tanh(self.in4(self.linear_h4(H)))
        H = F.leaky_relu(self.in5(self.linear_h5(H)))
        H = torch.tanh(self.in6(self.linear_h6(H)))
        #H = torch.tanh(self.in7(self.linear_h7(H)))
        """
        H = F.leaky_relu(self.linear_h1(H))
        H = F.leaky_relu(self.linear_h2(H))
        H = F.leaky_relu(self.linear_h3(H))
        H = F.leaky_relu(self.linear_h4(H))
        H = F.leaky_relu(self.linear_h5(H))
        H = F.leaky_relu(self.linear_h6(H))
        H = torch.tanh(self.linear_h7(H))
        x = torch.tanh(self.linear_out(H))
        x = x.view(self.batch_gpu, self.c_dim, args.y_dim, args.x_dim)
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
        self.conv5 = nn.Conv2d(8*self.z, 8*self.z, 3, stride=2, padding=1)
        self.relu = nn.LeakyReLU(inplace=True)
        self.linear1 = nn.Linear(4*4*2*self.dim, 1)
        self.bn1 = nn.BatchNorm2d(self.z)
        self.bn2 = nn.BatchNorm2d(2*self.z)
        self.bn3 = nn.BatchNorm2d(4*self.z)
        self.bn4 = nn.BatchNorm2d(8*self.z)
        self.bn5 = nn.BatchNorm2d(8*self.z)

    def forward(self, x):
        x = x.view(-1, 3, 64, 64)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        x = x.view(self.batch_gpu, -1)
        x = self.linear1(x)
        x = x.view(-1)
        return x

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.3)
    if classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.3)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


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


def sample(args, netG, z, gan=False):
    x_vec, y_vec, r_vec = coordinates(args)
    image = netG(args, (x_vec, y_vec, z, r_vec))
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
            nn.init.normal_(layer.weight.data)
    return model


def train(args):
    
    torch.manual_seed(8734)
    exp_dir = 'celeba_gan_fixed_z'+str(args.z)+'n'+str(args.net)+'s'+str(args.scale) 
    args.batch_size *= args.ngpu
    args.batch_gpu = args.batch_size//args.ngpu
    netG = torch.nn.DataParallel(Generator(args), device_ids=list(range(args.ngpu))).cuda()
    netD = torch.nn.DataParallel(Discriminator(args), device_ids=list(range(args.ngpu))).cuda()
    netG.apply(weight_init)
    netD.apply(weight_init)
    print (netG, netD)

    optimG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9), weight_decay=1e-4)
    optimD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9), weight_decay=1e-4)
    
    celeba_data = datagen.load_mini_celeba(args)
    train = inf_gen(celeba_data)
    one = torch.tensor(1.).cuda()
    mone = (one * -1)
    
    print ('==> Begin Training')
    for iter in range(args.epochs):
        ops.batch_zero_grad([netG, netD])
    
        for p in netD.parameters():
            p.requires_grad = True
        for _ in range(args.disc_iters):
            data, targets = next(train)
            data = data.view(args.batch_size, 3, 64, 64).cuda()
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
            gp = ops.grad_penalty_3dim(args, netD, data, fake)
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
        if iter % 1000 == 0:
            with torch.no_grad():
                noise = torch.randn(args.batch_size, args.z, requires_grad=True).cuda()
                r_samples = sample(args, netG, noise).mul(.5).add(.5)
                r_samples = r_samples.view(-1, 3, 64, 64).cpu().data.numpy()
                rpath = exp_dir+'_results/gan_sample_{}.png'.format(iter)
                if not os.path.exists(exp_dir+'_results'):
                    os.makedirs(exp_dir+'_results')
                print ('saving ae-gan sample: ', rpath)
                utils.save_images(r_samples, rpath)
        if iter % 5000 == 0:
            utils.save_model(exp_dir+'_results/netG_{}_{}'.format(iter, exp_dir), netG, optimG)
            utils.save_model(exp_dir+'_results/netD_{}_{}'.format(iter, exp_dir), netD, optimD)

if __name__ == '__main__':

    args = load_args()
    train(args)
