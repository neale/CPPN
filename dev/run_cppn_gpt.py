import os
import gc
import sys
import glob
import shutil
import argparse

import cv2
import numpy as np
import scipy.stats as st
from sklearn.datasets import make_blobs

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Subset
from torchvision import transforms, datasets
from torchvision.utils import save_image
import matplotlib.pyplot as plt

# Because imageio uses the root logger instead of warnings package
import logging
import tifffile

import utils
from primitives import p_gmm, p_squares_right, p_squares_left, p_grad_img
from maps import MapRandomGraph, MapRandomAct, Map, plot_graph


logging.getLogger().setLevel(logging.ERROR)


class CPPN(object):
    """initializes a CPPN"""
    def __init__(self,
                 noise_dim=4,
                 noise_scale=10,
                 n_samples=6,
                 x_dim=512,
                 y_dim=512,
                 c_dim=3,
                 layer_width=4,
                 batch_size=1,
                 weight_init_mean=0.,
                 weight_init_std=1.0,
                 output_dir='.',
                 graph_nodes=10,
                 seed_gen=987654321, # 123456789,
                 seed=None,
                 graph=None,
                 device='cpu',
                 use_extra_primitives=False):

        self.noise_dim = noise_dim
        self.n_samples = n_samples
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.c_dim = c_dim 
        self.noise_scale = noise_scale
        self.weight_init_mean = weight_init_mean
        self.weight_init_std = weight_init_std
        self.layer_width = layer_width
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.device = device

        self.graph_nodes = graph_nodes
        self.graph = graph
        self.use_extra_primitives = use_extra_primitives
        
        self.map_fn = None
        self.logger = logging.getLogger('CPPNlogger')
        self.logger.setLevel(logging.INFO)

        self.seed = seed
        self.seed_gen = seed_gen
        self.init_random_seed(seed=seed)
        self._init_paths()
        
    def init_random_seed(self, seed=None):
        """ 
        initializes random seed for torch. Random seed needs
            to be a stored value so that we can save the right metadata. 
            This is not to be confused with the uid that is not a seed, 
            rather how we associate user sessions with data
        """
        if seed == None:
            self.logger.debug(f'initing with seed gen: {self.seed_gen}')
            self.seed = np.random.randint(self.seed_gen)
            torch.manual_seed(self.seed)
        else:
            torch.manual_seed(self.seed)

    def _init_paths(self):
        os.makedirs(self.output_dir, exist_ok=True)

    def _init_weights(self, model, layer_i, mul):
        for i, layer in enumerate(model.modules()):
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight.data,
                                self.weight_init_mean,
                                self.weight_init_std)

        return model

    def init_map_fn(self,
                    seed=None,
                    activations='fixed',
                    graph_topology='fixed',
                    layer_i=0,
                    mul=1.0,
                    graph=None,
                    activation_set='large'):
        if graph_topology == 'fixed' and activations == 'fixed':
            map_fn = Map(
                self.noise_dim, self.c_dim, self.layer_width, self.noise_scale)
        elif graph_topology == 'WS':
            map_fn = MapRandomGraph(
                self.noise_dim, self.c_dim, self.layer_width, self.noise_scale, self.graph_nodes,
                graph, activation_set, activations=activations)
        elif graph_topology == 'fixed' and activations == 'permute':
            map_fn = MapRandomAct(
                self.noise_dim, self.c_dim, self.layer_width, self.noise_scale)
        else:
            raise NotImplementedError

        self.map_fn = self._init_weights(map_fn, layer_i, mul)
        self.map_fn.eval()

    def init_inputs(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        return torch.ones(batch_size, 1, self.noise_dim).uniform_(-2., 2.)


    def _coordinates(self, x_dim, y_dim, batch_size, zoom, pan, noise_scale, device):
        xpan, ypan = pan
        xzoom, yzoom = zoom
        n_pts = x_dim * y_dim
        
        x_range = noise_scale * (torch.arange(x_dim) - (x_dim - 1) / xpan) / (x_dim - 1) / xzoom
        y_range = noise_scale * (torch.arange(y_dim) - (y_dim - 1) / ypan) / (y_dim - 1) / yzoom
        
        x_mat, y_mat = torch.meshgrid(x_range, y_range)
        r_mat = torch.sqrt(x_mat * x_mat + y_mat * y_mat)
        
        x_vec = x_mat.flatten().repeat(batch_size).view(batch_size * n_pts, -1)
        y_vec = y_mat.flatten().repeat(batch_size).view(batch_size * n_pts, -1)
        r_vec = r_mat.flatten().repeat(batch_size).view(batch_size * n_pts, -1)
        
        x_vec = x_vec.float().to(device)
        y_vec = y_vec.float().to(device)
        r_vec = r_vec.float().to(device)
        
        return x_vec, y_vec, r_vec, n_pts
    

    def _coordinates_3d(self, x_dim, y_dim, z_dim, batch_size, zoom, pan, noise_scale, device):
        xpan, ypan, zpan = pan
        xzoom, yzoom, zzoom = zoom
        n_pts = x_dim * y_dim * z_dim
        
        x_range = noise_scale * (torch.arange(x_dim) - (x_dim - 1) / xpan) / (x_dim - 1) / xzoom
        y_range = noise_scale * (torch.arange(y_dim) - (y_dim - 1) / ypan) / (y_dim - 1) / yzoom
        z_range = noise_scale * (torch.arange(z_dim) - (z_dim - 1) / zpan) / (z_dim - 1) / zzoom
        
        x_mat, y_mat, z_mat = torch.meshgrid(x_range, y_range, z_range)
        r_mat = torch.sqrt(x_mat * x_mat + y_mat * y_mat + z_mat * z_mat)
        
        x_vec = x_mat.flatten().repeat(batch_size).view(batch_size * n_pts, -1)
        y_vec = y_mat.flatten().repeat(batch_size).view(batch_size * n_pts, -1)
        z_vec = z_mat.flatten().repeat(batch_size).view(batch_size * n_pts, -1)
        r_vec = r_mat.flatten().repeat(batch_size).view(batch_size * n_pts, -1)
        
        x_vec = x_vec.float().to(device)
        y_vec = y_vec.float().to(device)
        z_vec = z_vec.float().to(device)
        r_vec = r_vec.float().to(device)
        
        return x_vec, y_vec, z_vec, r_vec, n_pts
    
    def generate_3d_frame(self, noise, x_dim, y_dim, z_dim, batch_size, zoom=(.5, .5, .5), pan=(2, 2, 2), splits=1):
        # generates a 3d tensor of size x_dim, y_dim, z_dim'
        x_vec, y_vec, z_vec, r_vec, n_pts = self._coordinates_3d(x_dim, y_dim, z_dim, batch_size, zoom, pan, self.noise_scale, self.device)
        one_vec = torch.ones(n_pts, 1, dtype=torch.float)
        if splits == 1:
            noise_scale = noise.view(batch_size, 1, self.noise_dim) * one_vec * self.noise_scale
            noise_scale = noise_scale.view(batch_size * n_pts, self.noise_dim)
            frame = self.map_fn(x_vec, y_vec, z_vec, r_vec, noise_scale)
        elif splits > 1:
            frame = torch.zeros(batch_size, n_pts, dtype=torch.float)
            for i in range(splits):
                noise_scale = noise.view(batch_size, 1, self.noise_dim) * one_vec * self.noise_scale
                noise_scale = noise_scale.view(batch_size * n_pts, self.noise_dim)
                frame += self.map_fn(x_vec, y_vec, z_vec, r_vec, noise_scale)
            frame /= splits 
        else:
            raise ValueError('splits must be >= 1')
        frame = frame.view(batch_size, x_dim, y_dim, z_dim)
        return frame
    


    @torch.no_grad()
    def sample_frame(self, noise, x_dim, y_dim, batch_size,
                 zoom=(.5, .5), pan=(2, 2), splits=1):
        x_vec, y_vec, r_vec, n_pts = self._coordinates(
            x_dim,
            y_dim,
            batch_size,
            zoom,
            pan,
            self.noise_scale,
            self.device)
        one_vec = torch.ones(n_pts, 1, dtype=torch.float)
        if splits == 1:
            noise_scale = noise.view(batch_size, 1, self.noise_dim) * one_vec * self.noise_scale
            noise_scale = noise_scale.view(batch_size * n_pts, self.noise_dim)
            frame = self.map_fn(x_vec, y_vec, r_vec, noise_scale)
        elif splits > 1:
            n_pts_split = n_pts // splits
            one_vecs = torch.split(one_vec, len(one_vec) // splits, dim=0)
            x = torch.split(x_vec, len(x_vec) // splits, dim=0)
            y = torch.split(y_vec, len(y_vec) // splits, dim=0)
            r = torch.split(r_vec, len(r_vec) // splits, dim=0)

            frame = torch.empty(batch_size * n_pts, self.c_dim, device=self.device)
            start_index = 0
            for i, one_vec in enumerate(one_vecs):
                noise_reshape = noise.view(batch_size, 1, self.noise_dim)
                noise_one_vec_i = noise_reshape * one_vec * self.noise_scale
                noise_scale_i = noise_one_vec_i.view(batch_size * n_pts_split, self.noise_dim)
                # forward split through map_fn
                f = self.map_fn(x[i], y[i], r[i], noise_scale_i)
                end_index = start_index + len(f)
                frame[start_index:end_index] = f
                start_index = end_index
        else:
            raise ValueError

        self.logger.debug(f'Output Frame Shape: {frame.shape}')
        frame = frame.reshape(batch_size, y_dim, x_dim, self.c_dim)
        return frame


def run_cppn(cppn, autosave=False, noise=None, debug=False, no_tiff=False):
    suff = f'z-{cppn.noise_dim}_scale-{cppn.noise_scale}_cdim-{cppn.c_dim}' \
           f'_net-{cppn.layer_width}_wmean-{cppn.weight_init_mean}_wstd-{cppn.weight_init_std}'
    zs = cppn.init_inputs()
    zs, _ = zs.sort()
    batch_samples = [cppn.sample_frame(
                        z_i,
                        cppn.x_dim,
                        cppn.y_dim,
                        batch_size=1,
                        splits=1)[0].numpy() * 255. for z_i in zs]

    n = np.random.randint(99999999)

    if 'MapRand' in cppn.map_fn.name:
        randgen = 1
    else:
        randgen = 0

    if 'Graph' in cppn.map_fn.name:
        graph = cppn.map_fn.get_graph_str()
    else:
        graph = ''

    if debug:
        print (graph)
        print ('data')
        print('noise', cppn.noise_dim)
        print('net', cppn.layer_width)
        print('noise scale', cppn.noise_scale)
        print('sample', str(zs[0].numpy().reshape(-1).tolist()))
        print ('seed', cppn.seed)
        print ('w_mean', cppn.weight_init_mean)
        print ('w_std', cppn.weight_init_std)

    for i, (img, z_j) in enumerate(zip([*batch_samples], [*zs])):
        
        save_fn = '{}/{}{}_{}'.format(cppn.output_dir, n, suff, i)
        print ('saving TIFF/PNG image pair at: {}'.format(save_fn))
        utils.write_image(path=save_fn, img=img, suffix='jpg')
        if not no_tiff:
            metadata = dict(seed=str(cppn.seed),
                seed_gen=str(cppn.seed_gen),
                noise_sample=str(z_j.numpy().reshape(-1).tolist()),
                noise=str(cppn.noise_dim), 
                c_dim=str(cppn.c_dim),
                scale=str(cppn.noise_scale),
                net=str(cppn.layer_width),
                graph=graph,
                weight_init_mean=str(cppn.weight_init_mean),
                weight_init_std=str(cppn.weight_init_std),
                randgen=str(randgen))

            print ('saving TIFF/PNG image pair at: {}'.format(save_fn))
            utils.write_image(path=save_fn, img=img.astype('u1'), suffix='tif',
                metadata=metadata)
    
    if debug:
        if 'Graph' in cppn.map_fn.name:
            graph_save = cppn.map_fn.graph
            plot_graph(graph_save,
                       path='{}/{}graph_{}.png'.format(cppn.output_dir, n, suff))

    py_files = glob.glob('*.py')
    assert len(py_files) > 0
    for fn in py_files:
        shutil.copy(fn, os.path.join(args.output_dir, fn))


def load_args():

    parser = argparse.ArgumentParser(description='cppn-pytorch')
    parser.add_argument('--noise_dim', default=8, type=int, help='latent space width')
    parser.add_argument('--n_samples', default=1, type=int, help='images to generate')
    parser.add_argument('--x_dim', default=512, type=int, help='out image width')
    parser.add_argument('--y_dim', default=512, type=int, help='out image height')
    parser.add_argument('--noise_scale', default=10, type=float, help='mutiplier on z')
    parser.add_argument('--c_dim', default=3, type=int, help='channels')
    parser.add_argument('--layer_width', default=32, type=int, help='net width')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--graph_nodes', default=10, type=int, help='number of graph_nodes in graph')
    parser.add_argument('--output_dir', default='trial', type=str, help='output fn')
    parser.add_argument('--no_tiff', action='store_true', help='save tiff metadata')
    parser.add_argument('--sweep_settings', action='store_true', help='sweep hps')
    parser.add_argument('--activations', default='fixed', type=str, help='')
    parser.add_argument('--graph_topology', default='fixed', type=str, help='')
    parser.add_argument('--use_extra_primitives', action='store_true', help='')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = load_args()

    if args.sweep_settings:
        for noise_dim in [8, 16, 34, 32, 48, 64]:
            print ('moving to noise_dim: ', noise_dim)
            for scale in [.1, .5, 1, 2, 4, 10, 32, 64]: # [.5, 1, 2, 4, 10, 32, 64]:
                print ('moving to noise scale: ', scale)
                for w in [4, 16, 32, 64]:
                    print ('moving to width: ', w)
                    for c_dim in [3,]:
                        for w_mean in [-3, -2, -1, 0, 1, 2, 3]:
                            print ('moving to w_mean: ', w_mean)
                            for w_std in [.85, 1.0, 1.15]:
                                print ('moving to w_std: ', w_std)
                                graph_nodes=30
                                name = args.output_dir+f'/z{noise_dim}_scale{scale}'
                                name += f'_width{w}_cdim{c_dim}_wm{w_mean}_wstd{w_std}'
                                if args.graph_topology != 'fixed':
                                    name += f'_graph_nodes{graph_nodes}'
                                cppn = CPPN(noise_dim=noise_dim,
                                            n_samples=1,
                                            x_dim=args.x_dim,
                                            y_dim=args.y_dim,
                                            c_dim=c_dim,
                                            noise_scale=scale,
                                            layer_width=w,
                                            batch_size=args.batch_size,
                                            output_dir=name,
                                            weight_init_mean=w_mean,
                                            weight_init_std=w_std,
                                            graph_nodes=graph_nodes,
                                            use_extra_primitives=args.use_extra_primitives)
                                cppn.init_map_fn(activations=args.activations,
                                                 graph_topology=args.graph_topology)
                                breakpoint()
                                run_cppn(cppn, debug=False, no_tiff=args.no_tiff)
    else: 
        for _ in range(args.n_samples):
            cppn = CPPN(noise_dim=args.noise_dim,
                        n_samples=args.n_samples,
                        x_dim=args.x_dim,
                        y_dim=args.y_dim,
                        c_dim=args.c_dim,
                        noise_scale=args.noise_scale,
                        layer_width=args.layer_width,
                        batch_size=args.batch_size,
                        output_dir=args.output_dir,
                        graph_nodes=args.graph_nodes,
                        use_extra_primitives=args.use_extra_primitives)
            cppn.init_map_fn(activations=args.activations,
                             graph_topology=args.graph_topology)
            run_cppn(cppn, debug=False, no_tiff=args.no_tiff)
