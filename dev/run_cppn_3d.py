import os
import gc
import sys
import glob
import time
import tqdm
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import cv2

import tifffile
# Because imageio uses the root logger instead of warnings package...
import logging
import matplotlib.pyplot as plt

from torch.utils.data import Subset
from torchvision import transforms, datasets
from torchvision.utils import save_image

import utils
from maps_3d import MapRandomGraph3D, plot_graph, MapRandomAct3D, Map3D


logging.getLogger().setLevel(logging.ERROR)


class CPPN(object):
    """initializes a CPPN"""
    def __init__(self,
                 noise_dim=4,
                 noise_scale=10,
                 n_samples=6,
                 x_dim=128,
                 y_dim=128,
                 z_dim=128,
                 c_dim=3,
                 layer_width=4,
                 batch_size=16,
                 weight_init_mean=0.,
                 weight_init_std=1.0,
                 output_dir='.',
                 graph_nodes=10,
                 seed_gen=987654321, # 123456789,
                 seed=None,
                 graph=None):

        self.noise_dim = noise_dim
        self.n_samples = n_samples
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.c_dim = c_dim 
        self.noise_scale = noise_scale
        self.weight_init_mean = weight_init_mean
        self.weight_init_std = weight_init_std
        self.layer_width = layer_width
        self.batch_size = batch_size
        self.output_dir = output_dir

        self.graph_nodes = graph_nodes
        self.graph = graph
        
        self.map_fn = None
        self.logger = logging.getLogger('CPPN3axLogger')
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
            map_fn = Map3D(
                self.noise_dim, self.c_dim, self.layer_width, self.noise_scale)
        elif graph_topology == 'WS':
            map_fn = MapRandomGraph3D(
                self.noise_dim, self.c_dim, self.layer_width, self.noise_scale,
                self.graph_nodes, graph, activation_set)
        elif graph_topology == 'fixed' and activations == 'permute':
            map_fn = MapRandomAct3D(
                self.noise_dim, self.c_dim, self.layer_width, self.noise_scale)
        else:
            raise NotImplementedError

        self.map_fn = self._init_weights(map_fn, layer_i, mul)
        self.map_fn.eval()

    def init_inputs(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        return torch.ones(batch_size, 1, self.noise_dim).uniform_(-1., 1.)
    
    def _coordinates(self, zoom, pan):
        x_dim = self.x_dim
        y_dim = self.y_dim
        z_dim = self.z_dim
        batch_size = self.batch_size
        device = 'cpu'
        noise_scale = self.noise_scale
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
    """
    def _coordinates(self, zoom, pan):
        xpan, ypan, zpan = pan
        xzoom, yzoom, zzoom = zoom
        x_dim = self.x_dim
        y_dim = self.y_dim
        z_dim = self.z_dim
        batch_size = self.batch_size
        n_pts = x_dim * y_dim * z_dim
        # range is values in the range [(-`dim`//2)-1, (`dim`//2)-1]
        # for `noise_dim`=10, we have values [-4, 4] of length `noise_dim`
        x_range = (self.noise_scale*(np.arange(x_dim)-(x_dim-1)/xpan)/(x_dim-1)/xzoom).astype(np.float32)
        y_range = (self.noise_scale*(np.arange(y_dim)-(y_dim-1)/ypan)/(y_dim-1)/yzoom).astype(np.float32)
        z_range = (self.noise_scale*(np.arange(z_dim)-(z_dim-1)/zpan)/(z_dim-1)/zzoom).astype(np.float32)

        x_mat = np.matmul(np.ones((y_dim, 1)), x_range.reshape((1, x_dim)))
        y_mat = np.matmul(y_range.reshape((y_dim, 1)), np.ones((1, x_dim)))
        
        x_mat3 = (np.tile(x_mat[:, :, None], [1, 1, self.z_dim])).astype(np.float32)
        y_mat3 = (np.tile(y_mat[:, :, None], [1, 1, self.z_dim])).astype(np.float32)
        z_mat3 = (np.rot90(x_mat3, 1, axes=[1, 2])).astype(np.float32)
        r_mat2 = np.sqrt(x_mat*x_mat + y_mat*y_mat)
        r_mat3 = (np.sqrt(x_mat3**2 + y_mat3**2 + z_mat3**2)).astype(np.float32)
        # FIX RMAT3
        #for i in range(r_mat3.shape[-1]):
        #    plt.imshow(r_mat3[:, :, i])
        #    plt.show()
        assert batch_size == 1, "BS > 1 exceeds compute for now"
        x_mat3 = x_mat3.ravel().reshape(n_pts, -1)
        y_mat3 = y_mat3.ravel().reshape(n_pts, -1)
        z_mat3 = z_mat3.ravel().reshape(n_pts, -1)
        r_mat3 = r_mat3.ravel().reshape(n_pts, -1)

        x_mat3 = torch.from_numpy(x_mat3).float()
        y_mat3 = torch.from_numpy(y_mat3).float()
        z_mat3 = torch.from_numpy(z_mat3).float()
        r_mat3 = torch.from_numpy(r_mat3).float()

        return x_mat3, y_mat3, z_mat3, r_mat3, n_pts
    """
    @torch.no_grad()
    def sample_frame(self, noise, zoom=(.5,.5,.5), pan=(2,2,2), splits=1):
        x_vec, y_vec, z_vec, r_vec, n_pts = self._coordinates(zoom, pan)
        one_vec = torch.ones(n_pts, 1, dtype=torch.float)
        if splits == 1:
            noise_scaled = noise.view(self.batch_size, 1, self.noise_dim)
            noise_scaled = noise_scaled * one_vec * self.noise_scale
            frame = self.map_fn(x_vec, y_vec, z_vec, r_vec, noise_scaled)
        elif splits > 1:
            n_pts_split = n_pts // splits
            one_vecs = torch.split(one_vec, len(one_vec)//splits, dim=0)
            x = torch.split(x_vec, len(x_vec)//splits, dim=0)
            y = torch.split(y_vec, len(y_vec)//splits, dim=0)
            z = torch.split(z_vec, len(z_vec)//splits, dim=0)
            r = torch.split(r_vec, len(r_vec)//splits, dim=0)
            self.logger.info('Processing Splits...')
            for split, one_vec in enumerate(tqdm.tqdm(one_vecs)):
                noise_reshape = noise.view(self.batch_size, 1, self.noise_dim)
                noise_one_vec_i = noise_reshape * one_vec * self.noise_scale
                noise_scale_i = noise_one_vec_i.view(self.batch_size*n_pts_split, self.noise_dim)
                f = self.map_fn(x[split], y[split], z[split], r[split], noise_scale_i)
                torch.save(f, f'f_temp_gen{split}.pt')
            return 
        else:
            raise ValueError
        self.logger.debug(f'Output Frame Shape: {frame.shape}')
        frame = frame.reshape(self.batch_size, self.y_dim, self.x_dim, self.z_dim, self.c_dim)
        return frame


def run_cppn(cppn, autosave=False, z=None, debug=False):
    suff = 'z-{}_scale-{}_cdim-{}_net-{}_wmean-{}_wstd-{}'.format(
        cppn.noise_dim, cppn.noise_scale, cppn.c_dim, cppn.layer_width,
        cppn.weight_init_mean, cppn.weight_init_std)
    zs = []
    for k in range(cppn.n_samples):
        zs.append(cppn.init_inputs())
    zs, _ = torch.stack(zs).sort()

    if 'GeneratorRand' in cppn.map_fn.name:
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
        print ('z', cppn.noise_dim)
        print ('net', cppn.layer_width)
        print ('scale', cppn.noise_scale)
        print ('sample', str(zs[0].numpy().reshape(-1).tolist()))
        print ('seed', cppn.seed)
        print ('w_mean', cppn.weight_init_mean)
        print ('w_std', cppn.weight_init_std)

    randID = np.random.randint(99999999)
   
    # Hueristics based on 32GB of system memory
    if cppn.x_dim > 256 and cppn.x_dim <= 512:
        splits = 16 - (cppn.x_dim % 16)
    elif cppn.x_dim <= 1024:
        splits = 128 - (cppn.x_dim % 512)
    else:
        splits = 1
    
    print(f'Calculated {splits} splits')
    batch_samples = []
    for i, z_i in enumerate(zs):
        cppn.sample_frame(z_i, splits=splits)
        gc.collect()
        frame = torch.load(f'f_temp_gen0.pt')
        print ('composing..')
        for split in tqdm.tqdm(range(1, splits)):
            frame_next = torch.load(f'f_temp_gen{split}.pt')
            frame = torch.cat([frame, frame_next], dim=0)
            os.remove(f'f_temp_gen{split}.pt')
        frame = frame.reshape(cppn.batch_size, cppn.y_dim, cppn.x_dim, cppn.z_dim, cppn.c_dim)
        frame = frame[0].numpy() * 255.
        is_blank = False
        if frame.all() == frame[0][0][0][0]:
            print ('blank image')
            continue

        metadata = dict(seed=str(cppn.seed),
                seed_gen=str(cppn.seed_gen),
                z_sample=str(z_i.numpy().reshape(-1).tolist()),
                z=str(cppn.noise_dim), 
                c_dim=str(cppn.c_dim),
                scale=str(cppn.noise_scale),
                net=str(cppn.layer_width),
                graph=graph,
                weight_init_mean=str(cppn.weight_init_mean),
                weight_init_std=str(cppn.weight_init_std),
                randgen=str(randgen))

        save_fn = '{}/{}{}_{}'.format(cppn.output_dir, randID, suff, i)
        print ('saving TIFF/PNG image pair at: {} of size: {}'.format(save_fn, frame.shape))
        utils.write_image(path=save_fn+'_first_frame', img=frame[0, :, :, :], suffix='jpg')
        utils.write_image(path=save_fn, img=frame.astype('u1'), suffix='tif',
            metadata=metadata)
    

def load_args():

    parser = argparse.ArgumentParser(description='cppn-pytorch')
    parser.add_argument('--noise_dim', default=32, type=int, help='latent space width')
    parser.add_argument('--noise_scale', default=4, type=float, help='mutiplier on z')
    parser.add_argument('--n_samples', default=1, type=int, help='images to generate')
    parser.add_argument('--x_dim', default=1024, type=int, help='out image width')
    parser.add_argument('--y_dim', default=1024, type=int, help='out image height')
    parser.add_argument('--z_dim', default=1024, type=int, help='out image height')
    parser.add_argument('--c_dim', default=4, type=int, help='channels')
    parser.add_argument('--layer_width', default=32, type=int, help='net width')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--graph_nodes', default=10, type=int, help='number of nodes in graph')
    parser.add_argument('--output_dir', default='trial', type=str, help='output fn')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = load_args()
    noise = np.array([4, 8, 16, 34, 32, 48, 64, 128, 256])
    scale = np.array([.001, .01, .1, .5, 1, 2, 4, 10, 32, 64, 128])
    w = np.array([4, 16, 32, 64, 128, 256])
    c_dim = np.array([4])
    w_mean = np.array([-3, -2, -1, 0, 1, 2, 3])
    w_std = np.array([.5, .85, 1.0, 1.15, 1.5])
    graph_nodes = np.array([4, 6, 10, 20, 30, 50])

    choice = lambda x: np.random.choice(x)

    noise = choice(noise)
    scale = choice(scale)
    w = choice(w)
    c_dim = 4
    w_mean = choice(w_mean)
    w_std = choice(w_std)
    nodes = choice(graph_nodes)
    print (noise, scale, w, c_dim, w_mean, w_std, nodes) 

    cppn = CPPN(noise_dim=noise,
                n_samples=args.n_samples,
                x_dim=args.x_dim,
                y_dim=args.y_dim,
                z_dim=args.z_dim,
                c_dim=args.c_dim,
                noise_scale=scale,
                layer_width=w,
                weight_init_mean=w_mean,
                weight_init_std=w_std,
                batch_size=args.batch_size,
                output_dir=args.output_dir,
                seed_gen=np.random.randint(999999),
                graph_nodes=nodes)
    cppn.init_map_fn(activations='permute',
                     graph_topology='WS')
    run_cppn(cppn, debug=False)
