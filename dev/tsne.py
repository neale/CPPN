import io
import gc
import os
import gc
import cv2
import glob
import logging
import argparse
import tifffile
from ast import literal_eval

import cv2
import torch
import torchviz
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import networkx
from networkx.drawing.nx_agraph import graphviz_layout, to_agraph
from sklearn.manifold import TSNE

import utils
from run_cppn import CPPN


logging.getLogger().setLevel(logging.ERROR)


def load_args():

    parser = argparse.ArgumentParser(description='cppn-pytorch')
    parser.add_argument('--x_dim', default=512, type=int, help='out image width')
    parser.add_argument('--y_dim', default=512, type=int, help='out image height')
    parser.add_argument('--c_dim', default=1, type=int, help='channels')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--save_dir', default='.', type=str, help='output fn')
    parser.add_argument('--name', default='.', type=str, help='output fn')
    parser.add_argument('--suffix', default='_reprojection', type=str)
    parser.add_argument('--file', default=None, type=str, help='choose file path to reproduce')
    parser.add_argument('--dir', default=None, type=str, help='input directory of images to reproduce')
    parser.add_argument('--draw_text', type=bool, default=False)
    parser.add_argument('--draw_graph', type=bool, default=False)
    parser.add_argument('--plot_graph', type=bool, default=False)

    parser.add_argument('--save_all_formats', type=bool, default=False)
    parser.add_argument('--generate_zoom_set', action='store_true')
    parser.add_argument('--generate_video_from_frames', action='store_true')
    args = parser.parse_args()
    return args


@torch.no_grad()
def main():
    args = load_args()
    if args.file is not None:
        files = [args.file]
    elif args.dir is not None:
        files = glob.glob(args.dir+'/*.tif')
    else:
        raise NotImplementedError

    frames_all, z_all, weights_all = [], [], []
    os.makedirs(args.save_dir, exist_ok=True)
    print ('Generating {} files'.format(len(files)))

    for idx in range(len(files)):
        gc.collect()
        path = files[idx]
        act_set = 'large'
        img, metadata = utils.load_tif_metadata(path)
        if img is None:
            print ('[FAIL] to load TIF file')
            continue
        print (f'[PASS] Loaded file {idx} from {path}')

        z = metadata['z_sample']
        if metadata['graph'] is not None:
            random = 'graph'
        else:
            random = 'act'
        print (metadata.items())
        cppn = CPPN(noise_dim=metadata['z_dim'],
                    x_dim=args.x_dim,
                    y_dim=args.y_dim,
                    c_dim=metadata['c_dim'],
                    noise_scale=metadata['z_scale'],
                    seed=metadata['seed'],
                    graph=metadata['graph'],
                    weight_init_mean=metadata['weight_init_mean'],
                    weight_init_std=metadata['weight_init_std'],
                    layer_width=metadata['layer_width'])
        if random == 'graph':
            activations = 'permute'
            graph_topology = 'WS'
        elif random == 'act':
            activations = 'permute'
            graph_topology= 'fixed'
        else:
            activations = 'fixed',
            graph_topology= 'fixed'
        
        cppn.init_map_fn(activations=activations,
                         graph_topology=graph_topology,
                         graph=metadata['graph'],
                         activation_set=act_set)
        
        original_params = path.split('/')[-1].split('.')[0]
        if metadata['graph'] is not None:
            random = 'graph'
            num_nodes = cppn.map_fn.graph.number_of_nodes()
            print (f'loaded {num_nodes} nodes')
        else:
            random = 'act'

        if metadata['layer_width'] == 8 and metadata['z_dim'] == 16:
            print ('[START] Generating frame')
            weights = torch.nn.utils.parameters_to_vector(cppn.map_fn.parameters())
            print (weights.shape)
            frame = cppn.sample_frame(z, args.x_dim, args.y_dim, 1, splits=1)
            frame = frame.cpu().detach().numpy()[0]*255
            frames_all.append(frame)
            z_all.append(z)
            weights_all.append(weights)


        save_name = args.name+original_params+args.suffix
    print ('-'*80)

    weights = torch.stack(weights_all).detach().numpy().astype(float)
    z = torch.stack(z_all).detach().numpy().astype(float)
    print (weights.shape)
    print (z.shape)

    tsne_z = TSNE(n_components=2, init='random')
    emb_z = tsne_z.fit_transform(z)
    tsne_w = TSNE(n_components=2, init='random')
    emb_w = tsne_w.fit_transform(weights)

    f, (ax0, ax1) = plt.subplots(1, 2)
    ax0.scatter(emb_z[:, 0], emb_z[:, 1], color='blueviolet')
    ax0.set_title('z densities')
    ax1.scatter(emb_z[:, 0], emb_w[:, 1], color='salmon')
    ax1.set_title('weight densities')
    plt.show()

    #for i, frame in enumerate(frames):
    #    ind = f'{i}'.zfill(5)
    #    save_name_i = f'{ind}_'+save_name
    #    cv2.imwrite(os.path.join(args.save_dir, save_name_i)+'ci.png', frame)

 

if __name__ == '__main__':
    main()
