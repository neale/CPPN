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
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--walk', default=False, type=bool)
    parser.add_argument('--walk_steps', default=10, type=int)
    parser.add_argument('--sample', default=False, type=bool)
    parser.add_argument('--z', default=100, type=int, help='latent space width')
    parser.add_argument('--dim', default=100, type=int, help='latent space width')
    parser.add_argument('--l', default=10, type=int, help='latent space width')
    parser.add_argument('--disc_iters', default=5, type=int)
    parser.add_argument('--epochs', default=200000, type=int)
    parser.add_argument('--exp', default='0', type=str)
    parser.add_argument('--output', default=12288, type=int)
    parser.add_argument('--dataset', default='celeba', type=str)

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
        self.conv5 = nn.Conv2d(8*self.z, 16*self.z, 3, stride=2, padding=1)
        self.linear = nn.Linear(4*4*4*self.z, self.z)
        self.relu = nn.LeakyReLU(True)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, input):
        input = input.view(-1, 3, 64, 64)
        x = self.relu(self.dropout(self.conv1(input)))
        x = self.relu(self.dropout(self.conv2(x)))
        x = self.relu(self.dropout(self.conv3(x)))
        x = self.relu(self.dropout(self.conv4(x)))
        x = self.relu(self.dropout(self.conv5(x)))
        x = x.view(-1, 4*4*4*self.z)
        x = self.linear(x)
        return x.view(-1, self.z)


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
        self.conv_h1 = nn.Conv2d(2, 16, 3, stride=1, padding=0)
        self.conv_h2 = nn.Conv2d(16, 32, 3, stride=1, padding=0)
        self.conv_h3 = nn.Conv2d(32, 32, 3, stride=1, padding=0)
        self.conv_h4 = nn.Conv2d(32, 32, 3, stride=1, padding=0)
        self.conv_h5 = nn.Conv2d(32, 32, 3, stride=1, padding=0)
        self.linear_h6 = nn.Linear(32*6*6, self.net)
        self.linear_out = nn.Linear(self.net, self.c_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, args, inputs):
        x, y, z, r = inputs
        n_points = args.x_dim * args.y_dim
        ones = torch.ones(n_points, 1, dtype=torch.float).to(gpu2)
        z_scaled = (z.view(args.batch_size, 1, self.z) * ones * args.scale).to(gpu2)
        z_pt = self.linear_z(z_scaled.view(args.batch_size*n_points, self.z))
        x_pt = self.linear_x(x.view(args.batch_size*n_points, -1))
        y_pt = self.linear_y(y.view(args.batch_size*n_points, -1))
        r_pt = self.linear_r(r.view(args.batch_size*n_points, -1))
        U = z_pt + x_pt + y_pt + r_pt
        H = F.softplus(U)
        H = H.view(-1, 2, 16, 16)
        H = F.elu(self.conv_h1(H))
        H = F.elu(self.conv_h2(H))
        H = F.elu(self.conv_h3(H))
        H = F.elu(self.conv_h4(H))
        H = F.elu(self.conv_h5(H))
        print (H.shape)
        H = H.view(-1, 32*6*6)
        print (H.shape)
        H = torch.tanh(self.linear_h6(H))
        print (H.shape)
        x = torch.sigmoid(self.linear_out(H))
        print (x.shape)
        x = x.view(args.batch_size, self.c_dim, args.y_dim, args.x_dim)
        return x.to(gpu1)


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'Discriminator'
        self.conv1 = nn.Conv2d(3, self.z, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(self.z, 2*self.z, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(2*self.z, 2*self.z, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(2*self.z, 4*self.z, 3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(4*self.z, 4*self.z, 3, stride=2, padding=1)
        self.relu = nn.ELU(inplace=True)
        self.linear1 = nn.Linear(4*4*4*self.dim, 1)

    def forward(self, x):
        x = x.view(-1, 3, 64, 64)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = x.view(-1, 4*4*4*self.dim)
        x = self.linear1(x)
        x = x.view(-1)
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
    x_mat = torch.from_numpy(x_mat).float().to(gpu2)
    y_mat = torch.from_numpy(y_mat).float().to(gpu2)
    r_mat = torch.from_numpy(r_mat).float().to(gpu2)
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
    exp_dir = 'celeba_lin_mixed_z'+str(args.z)+'n'+str(args.net)+'s'+str(args.scale) 
    netE = Encoder(args).cuda()
    netG = Generator(args).to(gpu2)
    netD = Discriminator(args).cuda()
    print (netE, netG, netD)

    optimE = optim.Adam(netE.parameters(), lr=2e-4, betas=(0.5, 0.9), weight_decay=1e-4)
    optimG = optim.Adam(netG.parameters(), lr=2e-4, betas=(0.5, 0.9), weight_decay=1e-4)
    optimD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9), weight_decay=1e-4)
    
    celeba_data = datagen.load_celeba(args)
    train = inf_gen(celeba_data)
    one = torch.tensor(1.).cuda()
    mone = (one * -1)
    
    print ('==> Begin Training')
    for iter in range(args.epochs):
        ops.batch_zero_grad([netG, netD, netE])
        for p in netD.parameters():
            p.requires_grad = False
        data, targets = next(train)
        data = data.cuda()
        code = netE(data)
        fake = sample(args, netG, code.to(gpu2))
        ae_loss = F.mse_loss(fake.to(gpu1), data)
        ae_loss.backward(one)
        optimE.step()
        optimG.step()


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
                fake = sample(args, netG, noise.to(gpu2)).view(args.batch_size, -1)
            fake.requires_grad_(True)
            d_fake = netD(fake.to(gpu1))
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
        fake = sample(args, netG, noise.to(gpu2)).view(args.batch_size, -1)
        G = netD(fake.to(gpu1))
        G = G.mean()
        G.backward(mone)
        g_cost = -G
        optimG.step()
       
        if iter % 100 == 0:
            print('iter: ', iter, 'train D cost', d_cost.cpu().item())
            print('iter: ', iter, 'train G cost', g_cost.cpu().item())
        if iter % 1000 == 0:
            val_d_costs = []
            with torch.no_grad():
                for i, (data, target) in enumerate(celeba_data):
                    data = data.cuda()
                    d = netD(data)
                    val_d_cost = -d.mean().item()
                    val_d_costs.append(val_d_cost)
                    if i > 50:
                        break
                ae_samples = sample(args, netG, netE(data.to(gpu2))).to(gpu1)
                ae_samples = ae_samples.view(-1, 3, 64, 64).cpu().data.numpy()
                noise = torch.randn(args.batch_size, args.z, requires_grad=True).cuda()
                r_samples = sample(args, netG, noise.to(gpu2)).to(gpu1)
                r_samples = r_samples.view(-1, 3, 64, 64).cpu().data.numpy()
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

    gpu1 = torch.device('cuda:0')
    gpu2 = torch.device('cuda:1')
    args = load_args()
    train(args)
