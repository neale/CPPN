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
import moviepy.video.io.ImageSequenceClip
from networkx.drawing.nx_agraph import graphviz_layout, to_agraph

import utils
from cppn import CPPN
from cppn_generator_graph_shared_repr import GeneratorRandGraph, plot_graph, randact


logging.getLogger().setLevel(logging.ERROR)


def load_args():

    parser = argparse.ArgumentParser(description='cppn-pytorch')
    parser.add_argument('--x_dim', default=2000, type=int, help='out image width')
    parser.add_argument('--y_dim', default=2000, type=int, help='out image height')
    parser.add_argument('--c_dim', default=1, type=int, help='channels')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--save_dir', default='.', type=str, help='output fn')
    parser.add_argument('--name', default='vid', type=str, help='output fn')
    parser.add_argument('--suffix', default='_reprojection', type=str)
    parser.add_argument('--file', default=None, type=str, help='choose file path to reproduce')
    parser.add_argument('--dir', default=None, type=str, help='input directory of images to reproduce')
    args = parser.parse_args()
    return args


def main():
    args = load_args()
    if args.file is not None:
        files = [args.file]
    elif args.dir is not None:
        files = glob.glob(args.dir+'/*.tif')
    else:
        raise NotImplementedError

    os.makedirs(args.save_dir, exist_ok=True)
    print ('Generating {} files'.format(len(files)))
    f_keep = glob.glob('temp/**/*.tif')
    f_keep = [f.split('/')[-1] for f in f_keep]

    for idx in range(len(files)):
        gc.collect()
        path = files[idx]
        if path.split('/')[1] in f_keep:  # use smaller act set
            act_set = 'small'
        else:
            act_set = 'large'
        img, metadata = utils.load_tif_metadata(path)
        if img is None:
            print ('[FAIL] to load TIF file')
            continue
        print ('[PASS] Loaded {}th file from {}'.format(idx, path))

        z = metadata['z_sample']
        if metadata['graph'] is not None:
            random = 'graph'
        else:
            random = 'act'

        cppn = CPPN(z_dim=metadata['z_dim'],
                    x_dim=args.x_dim,
                    y_dim=args.y_dim,
                    c_dim=metadata['c_dim'],
                    z_scale=metadata['z_scale'],
                    seed=metadata['seed'],
                    graph=metadata['graph'],
                    weight_init_mean=metadata['weight_init_mean'],
                    weight_init_std=metadata['weight_init_std'],
                    layer_width=metadata['layer_width'])
        cppn.init_generator(random=random, graph=metadata['graph'], act_set=act_set)
        
        original_params = path.split('/')[-1].split('.')[0]
        if metadata['graph'] is not None:
            random = 'graph'
            num_nodes = cppn.generator.graph.number_of_nodes()
        else:
            random = 'act'
        save_name = args.name+original_params+args.suffix
        save_path = os.path.join(args.save_dir, save_name)
        print (f'[SAVING] saving MP4 clips at: {save_path}')
        # img \in [x, y, z, c]

        print ('Gen XY')
        fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
        (h, w) = img.shape[:2]
        out_xy = cv2.VideoWriter(f'{save_path}_xy.mp4', fourcc, 50, (w, h), isColor=True)
        for frame in img.transpose(2, 0, 1, 3):
            out_xy.write(frame)
        out_xy.release()
        print ('Gen XZ')
        fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
        out_xz = cv2.VideoWriter(f'{save_path}_xz.mp4', fourcc, 50, (w, h), isColor=True)
        img_t = np.ascontiguousarray(img.transpose(1, 0, 2, 3))
        for frame in img_t:
            out_xz.write(frame)
        out_xz.release()
        out_xz = None
        gc.collect()
        print ('Gen YZ')
        fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
        out_yz = cv2.VideoWriter(f'{save_path}_yz.mp4', fourcc, 50, (w, h), isColor=True)
        for frame in img:
            out_yz.write(frame)
        out_yz.release()

        print ('[DONE]')
        print ('-'*80)
 

if __name__ == '__main__':
    main()
