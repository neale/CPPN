import sys
import time
import torch
import glob
import itertools
import numpy as np

from imageio import imwrite
import torch.nn as nn
import torch.nn.init as init
import torch.distributions.multivariate_normal as N


def create_d(shape):
    mean = torch.zeros(shape)
    cov = torch.eye(shape)
    D = N.MultivariateNormal(mean, cov)
    return D


def sample_d(D, shape, scale=1., grad=True):
    z = scale * D.sample((shape,)).cuda()
    z.requires_grad = grad
    return z



def save_model(model, optim, path):
    torch.save({
        'state_dict': model.state_dict(),
        'optimizer': optim.state_dict(),
        }, path)


def load_model(model, optim, path):
    ckpt = torch.load(path)
    model.load_state_dict(ckpt['state_dict'])
    if optim is not None:
        optim.load_state_dict(ckpt['optimizer'])
    return model, optim


def get_net_only(model):
    net_dict = {
            'state_dict': model.state_dict(),
    }
    return net_dict


def load_net_only(model, d):
    model.load_state_dict(d['state_dict'])
    return model



def save_images(X, save_path):
    # [0, 1] -> [0,255]
    if isinstance(X.flatten()[0], np.floating):
        X = (255.99*X).astype('uint8')

    n_samples = X.shape[0]
    rows = int(np.sqrt(n_samples))
    while n_samples % rows != 0:
        rows -= 1
    nh, nw = rows, n_samples//rows
    if X.ndim == 2:
        X = np.reshape(X, (X.shape[0], int(np.sqrt(X.shape[1])),
            int(np.sqrt(X.shape[1]))))
    if X.ndim == 4:
        # BCHW -> BHWC
        X = X.transpose(0,2,3,1)
        h, w = X[0].shape[:2]
        img = np.zeros((h*nh, w*nw, 3))
    elif X.ndim == 3:
        h, w = X[0].shape[:2]
        img = np.zeros((h*nh, w*nw))

    for n, x in enumerate(X):
        j = n//nw
        i = n%nw
        img[j*h:j*h+h, i*w:i*w+w] = x

    print (save_path)
    imwrite(save_path, img)


def generate_ae_image(args, iter, netG, netE, real_data):
    encoding = netE(real_data)
    samples = netG(encoding)
    if samples.dim() < 4:
        channels = 1
        out_size = int(np.sqrt(args.output))
        samples = samples.view(-1, out_size, out_size)
    else:
        channels = samples.shape[1]
        out_size = int(np.sqrt(args.output//3))
        samples = samples.view(-1, channels, out_size, out_size)
        samples = samples.mul(0.5).add(0.5) 
    samples = samples.cpu().data.numpy()
    path = 'results/{}/ae_samples{}.png'.format(args.dataset, iter)
    print ('saving sample: ', path)
    save_images(samples, path)


def generate_image(args, iter, netG):
    with torch.no_grad():
        noise = torch.randn(args.batch_size, args.z, requires_grad=True).cuda()
        samples = netG(noise)
    if samples.dim() < 4:
        channels = 1
        out_size = int(np.sqrt(args.output))
        samples = samples.view(-1, out_size, out_size)
    else:
        channels = samples.shape[1]
        out_size = int(np.sqrt(args.output//3))
        samples = samples.view(-1, channels, out_size, out_size)
       	samples = samples.mul(0.5).add(0.5) 
    samples = samples.cpu().data.numpy()
    path = 'gan_training/gan_sample_{}.png'.format(iter)
    print ('saving sample: ', path)
    save_images(samples, path)
