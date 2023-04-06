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
from maps import MapRandomGraph, MapRandomAct, Map, plot_graph, ConvDecoder, LinDecoder, MapConv
from mlp_mixer import MLPMixer
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

from lpips_reduced import LPIPS
from discriminator import NLayerDiscriminator

from clip_utils import load_clip, MakeCutouts, spherical_dist_loss, tv_loss, parse_prompt, fetch, range_loss
import clip


logging.getLogger().setLevel(logging.ERROR)


class ZSTransform(nn.Module):
    def __init__(self, z_dim):
        super(ZSTransform, self).__init__()
        self.linear1 = nn.Linear(z_dim, z_dim)
        self.linear2 = nn.Linear(z_dim, z_dim)
        self.linear_mean = nn.Linear(z_dim, 1)
        self.linear_std = nn.Linear(z_dim, 1)

    def forward(self, z):
        z = F.silu(self.linear1(z))
        z = F.silu(self.linear2(z))
        mean = self.linear_mean(z)
        std = self.linear_std(z)
        return mean, std


class ZDTransform(nn.Module):
    def __init__(self, z_dim):
        super(ZDTransform, self).__init__()
        self.linear1 = nn.Linear(z_dim, z_dim)
        self.linear2 = nn.Linear(z_dim, z_dim)
        self.linear3 = nn.Linear(z_dim, z_dim)

    def forward(self, z):
        z = F.silu(self.linear1(z))
        z = F.silu(self.linear2(z))
        z = self.linear3(z)
        return z


        
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
                 patch_size=4,
                 batch_size=1,
                 weight_init_mean=0.,
                 weight_init_std=1.0,
                 output_dir='.',
                 graph_nodes=10,
                 seed_gen=987654321, # 123456789,
                 seed=None,
                 use_conv=False,
                 use_linear=False,
                 use_mixer=False,
                 clip_loss=False,
                 device='cpu',
                 graph=None):

        self.noise_dim = noise_dim
        self.n_samples = n_samples
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.c_dim = c_dim 
        self.noise_scale = noise_scale
        self.weight_init_mean = weight_init_mean
        self.weight_init_std = weight_init_std
        self.layer_width = layer_width
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.output_dir = output_dir

        self.graph_nodes = graph_nodes
        self.graph = graph
        
        self.map_fn = None
        self.use_conv = use_conv
        self.conv_cppn = False
        self.use_linear = use_linear
        self.use_mixer = use_mixer
        self.clip_loss = clip_loss
        self.device = device
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
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight.data, 0, 1)

                if layer.out_channels == 3:
                    pass
                    #nn.init.zeros_(layer.weight.data)   

                else:
                    pass
                    #std = (1./layer.in_channels)**.5
                    #nn.init.normal_(layer.weight.data, 0, np.sqrt(1./layer.in_channels))
                #nn.init.zeros_(layer.bias.data)

       
        return model

    def init_map_fn(self,
                    seed=None,
                    activations='fixed',
                    graph_topology='fixed',
                    layer_i=0,
                    mul=1.0,
                    graph=None,
                    activation_set='large'):
        if self.use_conv:
            map_fn = ConvDecoder(self.noise_dim)
        elif self.use_linear:
            map_fn = LinDecoder(self.noise_dim)
        elif self.use_mixer:
            map_fn = MLPMixer(
                image_size=self.x_dim,
                channels=self.c_dim,
                patch_size=self.patch_size,
                dim=self.layer_width,
                depth=4,
                randomize_act=(activations!='fixed'),
                num_classes=self.x_dim*self.y_dim*self.c_dim)
        elif graph_topology == 'fixed' and activations == 'fixed':
            map_fn = Map(
                self.noise_dim, self.c_dim, self.layer_width, self.noise_scale)
        elif graph_topology == 'WS':
            map_fn = MapRandomGraph(
                self.noise_dim, self.c_dim, self.layer_width, self.noise_scale, self.graph_nodes,
                graph, activation_set, activations=activations)
        elif graph_topology == 'fixed' and activations == 'permute':
            map_fn = MapRandomAct(
                self.noise_dim, self.c_dim, self.layer_width, self.noise_scale)
        elif graph_topology == 'conv_fixed':
            self.conv_cppn = True
            map_fn = MapConv(
                self.noise_dim, self.c_dim, self.layer_width, self.noise_scale, clip_loss=self.clip_loss)
        else:
            raise NotImplementedError

        self.map_fn = self._init_weights(map_fn, layer_i, mul)
        self.map_fn = self.map_fn.to(self.device)

    def init_inputs(self, batch_size=None, x_dim=None, y_dim=None):
        if batch_size is None:
            batch_size = self.batch_size
        if x_dim is None:
            x_dim = self.x_dim
        if y_dim is None:
            y_dim = self.y_dim
        inputs = torch.ones(batch_size, 1, self.noise_dim).uniform_(-2., 2.)
        inputs = inputs.to(self.device)
        inputs = inputs.reshape(batch_size, 1, self.noise_dim)
        one_vec = torch.ones(x_dim*y_dim, 1).float().to(self.device)
        inputs_scaled = inputs * one_vec * self.noise_scale
        return inputs_scaled.unsqueeze(0).float()

    def _coordinates(self, x_dim, y_dim, batch_size, zoom, pan, as_mat=False):
        xpan, ypan = pan
        xzoom, yzoom = zoom
        x_dim, y_dim, scale = x_dim, y_dim, self.noise_scale
        n_pts = x_dim * y_dim
        x_range = scale*(np.arange(x_dim)-(x_dim-1)/xpan)/(x_dim-1)/xzoom
        y_range = scale*(np.arange(y_dim)-(y_dim-1)/ypan)/(y_dim-1)/yzoom
        x_mat = np.matmul(np.ones((y_dim, 1)), x_range.reshape((1, x_dim)))
        y_mat = np.matmul(y_range.reshape((y_dim, 1)), np.ones((1, x_dim)))
        r_mat = np.sqrt(x_mat*x_mat + y_mat*y_mat)
        if as_mat:
            return x_mat, y_mat, r_mat, n_pts
        x_vec = np.tile(x_mat.flatten(), batch_size).reshape(batch_size*n_pts, -1)
        y_vec = np.tile(y_mat.flatten(), batch_size).reshape(batch_size*n_pts, -1)
        r_vec = np.tile(r_mat.flatten(), batch_size).reshape(batch_size*n_pts, -1)

        x_vec = torch.from_numpy(x_vec).float().to(self.device)
        y_vec = torch.from_numpy(y_vec).float().to(self.device)
        r_vec = torch.from_numpy(r_vec).float().to(self.device)

        return x_vec, y_vec, r_vec, n_pts

    def sample_frame(self, noise, x_dim, y_dim, batch_size=1,
            zoom=(.5,.5), pan=(2,2), splits=1):
        x_vec, y_vec, r_vec, n_pts = self._coordinates(
            x_dim, y_dim, batch_size, zoom, pan)
        #one_vec = torch.ones(n_pts, 1, dtype=torch.float).to(self.device)
        #one_vec.requires_grad_(True)
        if splits == 1:
            noise = noise.reshape(batch_size*n_pts, self.noise_dim)
            frame = self.map_fn(x_vec, y_vec, r_vec, noise, extra=None)
        elif splits > 1:
            n_pts_split = n_pts // splits
            one_vecs = torch.split(one_vec, len(one_vec)//splits, dim=0)
            x = torch.split(x_vec, len(x_vec)//splits, dim=0)
            y = torch.split(y_vec, len(y_vec)//splits, dim=0)
            r = torch.split(r_vec, len(r_vec)//splits, dim=0)
            for i, one_vec in enumerate(one_vecs):
                noise_reshape = noise.view(batch_size, 1, self.noise_dim) 
                noise_one_vec_i = noise_reshape * one_vec * self.noise_scale
                noise_scale_i = noise_one_vec_i.view(batch_size*n_pts_split, self.noise_dim)
                # forward split through map_fn    
                f = self.map_fn(x[i], y[i], r[i], noise_scale_i, extra=None)
                torch.save(f, 'f_temp_gen{}.pt'.format(i))
            frames = [torch.load('f_temp_gen{}.pt'.format(j)) for j in range(splits)]
            frame = torch.cat(frames, dim=0)
            temp_files = glob.glob('f_temp_gen*')
            for temp in temp_files:
                os.remove(temp)
        else:
            raise ValueError

        self.logger.debug(f'Output Frame Shape: {frame.shape}')
        #frame = frame.reshape(batch_size, y_dim, x_dim, self.c_dim)
        return frame


def load_target_data(args, device):
    target = cv2.cvtColor(cv2.imread(args.target_img_path), cv2.COLOR_BGR2RGB)
    target = target / 255.
    target = torch.from_numpy(target).permute(2, 0, 1).unsqueeze(0).float().to(device)
    target_fn = f'{args.output_dir}/target'
    utils.write_image(path=target_fn,
        img=target.permute(0, 2, 3, 1)[0].cpu().numpy()*255, suffix='jpg')
    return target


def run_cppn(args, cppn, debug=False):
    suff = f'z-{cppn.noise_dim}_scale-{cppn.noise_scale}_cdim-{cppn.c_dim}' \
           f'_net-{cppn.layer_width}_wmean-{cppn.weight_init_mean}_wstd-{cppn.weight_init_std}'
    # load target data
    if args.target_img_path is not None:
        target = load_target_data(args, cppn.device)
        print ('Using: ', cppn.device, ' target: ', target.shape)
    else:
        target = None

    if args.optimize_deterministic_input:
        input_map = ZDTransform(cppn.noise_dim).to(cppn.device)  # deterministic
        input_optim = torch.optim.AdamW(input_map.parameters(), lr=1e-2, weight_decay=1e-3)
    elif args.optimize_stochastic_input:
        input_map = ZSTransform(cppn.noise_dim).to(cppn.device)  # stochastic
        input_optim = torch.optim.AdamW(input_map.parameters(), lr=1e-2, weight_decay=1e-3)
    else:
        input_map = None
        input_optim = None
    
    if args.optimize_map:
        optim_map = torch.optim.AdamW(cppn.map_fn.parameters(), 
                                      lr=0.001, weight_decay=1e-5,
                                      betas=(.9, .999), eps=1e-8)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optim_map, T_0=100, T_mult=2)
    else:
        optim_map = None

    # load LPIPS loss
    if args.perceptual_loss:
        pips_loss = LPIPS().eval().to(cppn.device)
    # load VQGAN discriminator
    if args.discriminator_loss:
        discriminator = NLayerDiscriminator(3, 64, 2)
        d_state = torch.load('./last.ckpt')
        d_state = {k[19:]: v for k, v in d_state['state_dict'].items() if 'discriminator' in k}
        discriminator.load_state_dict(d_state)
        discriminator = discriminator.to(cppn.device).eval()
    # Load CLIP loss
    if args.clip_loss:
        #assert args.perceptual_loss, 'CLIP loss requires perceptual loss'
        cutn = 16
        clip_model, clip_normalizer = load_clip(cppn.device)
        clip_size = clip_model.visual.input_resolution
        make_cutouts = MakeCutouts(clip_size, cutn=16)
        txt, weight = parse_prompt(args.clip_prompt)
        target_embeds = clip_model.encode_text(clip.tokenize(txt).to(cppn.device)).float()
        weights = torch.tensor([weight]).to(cppn.device)
        if weights.abs() < 1e-3:
            raise RuntimeError('The weights must not sum to 0.')
        weights /= weights.sum().abs()
        if target is not None:
            target = target.mul(2).sub(1)
            init = target
        else:
            init = None
        tv_scale = 150              # Controls the smoothness of the final output.
        range_scale = 50   
        init_scale = 0
        def clip_cond(x):
            clip_in = clip_normalizer(make_cutouts(x))
            image_embeds = clip_model.encode_image(clip_in).float()
            dists = spherical_dist_loss(image_embeds.unsqueeze(1), target_embeds.unsqueeze(0))
            dists = dists.view([cutn, x.shape[0], -1])
            losses = dists.mul(weights).sum(2).mean(0)
            tv_losses = tv_loss(x)
            range_losses = range_loss(x)
            loss = losses.sum() + tv_losses.sum() * tv_scale + range_losses.sum() * range_scale
            if init is not None and init_scale:
                init_losses = pips_loss(x, init)
                loss = loss + init_losses.sum() * init_scale
            return loss # -torch.autograd.grad(loss, x)[0]
        

    # load Z inputs to map function
    if cppn.use_conv or cppn.use_linear:
        z = torch.ones(cppn.batch_size, 1, cppn.noise_dim).uniform_(-2., 2.)
        if cppn.use_linear:
            z = z.squeeze(0)
    elif cppn.use_mixer:
        z = torch.ones_like(target).uniform_(-2., 2.)
    else:
        z = cppn.init_inputs()
    z = z.to(cppn.device)

    # load X, Y, R inputs to map function
    x_vec, y_vec, r_vec, n_pts = cppn._coordinates(
        cppn.x_dim, cppn.y_dim, 1, zoom=(.5,.5), pan=(2,2), as_mat=True)
    x_vec = torch.from_numpy(x_vec).to(cppn.device).float()
    y_vec = torch.from_numpy(y_vec).to(cppn.device).float() 
    r_vec = torch.from_numpy(r_vec).to(cppn.device).float()
    
    x2, y2, r2, _ = cppn._coordinates(
        1024, 1024, 1, zoom=(.25,.25), pan=(2,2), as_mat=True)
    x2 = torch.from_numpy(x2).to('cpu').float()
    y2 = torch.from_numpy(y2).to('cpu').float()
    r2 = torch.from_numpy(r2).to('cpu').float()

    if args.optimize_map or args.optimize_deterministic_input or args.optimize_stochastic_input:
        losses = []
        for iteration in range(int(1e6)):
            if args.optimize_map:
                optim_map.zero_grad()

            if args.optimize_deterministic_input:
                input_optim.zero_grad()
                z_i = input_map(z)
            elif args.optimize_stochastic_input: 
                input_optim.zero_grad()
                mu_z, log_sigma_z = input_map(z)
                dist = torch.distributions.Normal(mu_z, torch.exp(.5 * log_sigma_z))
                z_i = dist.rsample([cppn.noise_dim])
            else:
                z_i = z.clone().requires_grad_(True).to(cppn.device)

            if cppn.use_conv or cppn.use_linear or cppn.use_mixer:
                sample = cppn.map_fn(z_i)
            elif cppn.conv_cppn:
                z = cppn.init_inputs()
                z_i = z.to(cppn.device)
                sample = cppn.map_fn(x_vec, y_vec, r_vec, z_i)
            else:
                z = cppn.init_inputs()
                z_i = z.to(cppn.device)
                sample = cppn.sample_frame(z_i, cppn.x_dim, cppn.y_dim, 1, splits=1)

            loss = 0.
            if args.l1_loss:
                loss += F.l1_loss(sample, target) 
            
            if args.l2_loss:
                loss += F.mse_loss(sample, target) 
            
            if args.perceptual_loss:
                loss += pips_loss(sample, target)[0, 0, 0, 0]
            
            if args.ssim_loss:
                loss += ssim(sample.permute(0, 2, 3, 1),
                                target.permute(0, 2, 3, 1),
                                data_range=1, size_average=False)
            
            if args.embedding_loss:
                loss += (z_i**2).mean()
            
            if args.discriminator_loss:
                logits_fake = discriminator(sample)
                loss += .2 * logits_fake.mean()
            
            if args.clip_loss:
                if init is not None:
                    if iteration > 4000:
                        clip_grad = clip_cond(sample)
                        loss += (clip_grad * sample).sum()
                    else:
                        loss += F.mse_loss(sample, target) 
                else:
                    clip_grad = clip_cond(sample)
                    loss += clip_grad # (clip_grad * sample).sum()
            
            loss.backward()
            
            if optim_map is not None:
                optim_map.step()
            if input_optim is not None:
                input_optim.step()
            scheduler.step(iteration)
            #cppn.map_fn.order = torch.randint(0, 15, size=(9,))
            #cppn.map_fn.generate_act_list()

            if iteration % 1000 == 0:
                losses.append(loss)
                #if len(losses) > 4 and (loss - losses[-3]) < 1e-2 and loss > 0.8:
                save_fn = f'{cppn.output_dir}/iter{iteration}'
                print (sample[0].shape)
                sample = (sample + 1) / 2. 
                img = sample[0].permute(1, 2, 0).detach().cpu().numpy()*255
                img = img.astype(np.uint8)
                utils.write_image(path=save_fn, img=img, suffix='jpg')
                print (f'[iter {iteration}] Loss: {loss.detach().cpu().item()}, saving...')
                print (f'[iter {iteration}] Noise(mmm): {z_i.mean()}/{z_i.min()}/{z_i.max()}')
            """
            if iteration % 3000 == 0:
                save_fn = f'{cppn.output_dir}/iter{iteration}_big'
                z2 = cppn.init_inputs(x_dim=1024, y_dim=1024).to('cpu')
                cppn.map_fn = cppn.map_fn.to('cpu')
                sample = cppn.map_fn(x2, y2, r2, z2)[0] 
                sample = (sample + 1) / 2. 
                sample = sample.permute(1, 2, 0).detach().cpu().numpy()*255.
                utils.write_image(path=save_fn, img=sample, suffix='jpg')
                cppn.map_fn = cppn.map_fn.to(cppn.device)
            """
    else:
        zs = []
        batch_samples = []
        for _ in range(1):
            if cppn.use_conv or cppn.use_linear:
                z = torch.ones(cppn.batch_size, 1, cppn.noise_dim).uniform_(-2., 2.)
                if cppn.use_linear:
                    z = z.squeeze(0)
            elif cppn.use_mixer:
                z = torch.ones_like(target).uniform_(-2., 2.)
                #z = cppn.init_inputs()
                x_vec, y_vec, r_vec, n_pts = cppn._coordinates(
                    cppn.x_dim, cppn.y_dim, 1, zoom=(.5,.5), pan=(2,2))
            else:
                z = cppn.init_inputs()
            z = z.to(cppn.device)
         
            if cppn.use_conv or cppn.use_linear or cppn.use_mixer:
                sample = cppn.map_fn(z)#, x_vec, y_vec, r_vec)
            elif cppn.conv_cppn:
                sample = cppn.map_fn(x_vec, y_vec, r_vec, z)
                sample = sample.permute(0, 2, 3, 1)[0]
            else:
                sample = cppn.sample_frame(z, cppn.x_dim, cppn.y_dim, 1, splits=1)
            zs.append(z)
            batch_samples.append(sample.detach().cpu().numpy()*255)
    
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
        print (img.shape)
        save_fn = '{}/{}{}_{}'.format(cppn.output_dir, n, suff, i)
        print ('saving PNG image at: {}'.format(save_fn))
        utils.write_image(path=save_fn, img=img, suffix='jpg')
        if not args.no_tiff:
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
    parser.add_argument('--noise_dim', default=2, type=int, help='latent space width')
    parser.add_argument('--n_samples', default=1, type=int, help='images to generate')
    parser.add_argument('--x_dim', default=256, type=int, help='out image width')
    parser.add_argument('--y_dim', default=256, type=int, help='out image height')
    parser.add_argument('--c_dim', default=3, type=int, help='channels')
    parser.add_argument('--layer_width', default=16, type=int, help='net width')
    parser.add_argument('--noise_scale', default=10, type=float, help='mutiplier on z')
    parser.add_argument('--patch_size', default=4, type=int, help='net width')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--graph_nodes', default=10, type=int, help='number of graph_nodes in graph')
    parser.add_argument('--output_dir', default='trial', type=str, help='output fn')
    parser.add_argument('--target_img_path', default=None, type=str, help='image to match')
    parser.add_argument('--optimize_deterministic_input', action='store_true')
    parser.add_argument('--optimize_stochastic_input', action='store_true')
    parser.add_argument('--optimize_map', action='store_true')
    parser.add_argument('--perceptual_loss', action='store_true')
    parser.add_argument('--ssim_loss', action='store_true')
    parser.add_argument('--l1_loss', action='store_true')
    parser.add_argument('--l2_loss', action='store_true')
    parser.add_argument('--discriminator_loss', action='store_true')
    parser.add_argument('--clip_loss', action='store_true')
    parser.add_argument('--clip_prompt', type=str, default='')
    parser.add_argument('--embedding_loss', action='store_true')
    parser.add_argument('--no_tiff', action='store_true', help='save tiff metadata')
    parser.add_argument('--activations', default='fixed', type=str, help='')
    parser.add_argument('--graph_topology', default='fixed', type=str, help='')
    parser.add_argument('--use_gpu', action='store_true', help='use GPU')
    parser.add_argument('--use_conv', action='store_true', help='use conv generator')
    parser.add_argument('--use_linear', action='store_true', help='use linear generator')
    parser.add_argument('--use_mixer', action='store_true', help='use linear generator')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = load_args()
    if args.use_gpu:
        device = 'cuda'
    else:
        device = 'cpu'

    for _ in range(200):
        cppn = CPPN(noise_dim=args.noise_dim,
                    n_samples=args.n_samples,
                    x_dim=args.x_dim,
                    y_dim=args.y_dim,
                    c_dim=args.c_dim,
                    noise_scale=args.noise_scale,
                    layer_width=args.layer_width,
                    batch_size=args.batch_size,
                    patch_size=args.patch_size,
                    output_dir=args.output_dir,
                    device=device,
                    use_conv=args.use_conv,
                    use_linear=args.use_linear,
                    use_mixer=args.use_mixer,
                    clip_loss=args.clip_loss,
                    graph_nodes=args.graph_nodes)
        cppn.init_map_fn(activations=args.activations,
                        graph_topology=args.graph_topology)
        run_cppn(args, cppn, debug=False)
