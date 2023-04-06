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

from cppn import CPPN
from cppn_generator_graph_shared_repr import GeneratorRandGraph, plot_graph, randact


logging.getLogger().setLevel(logging.ERROR)


def load_args():

    parser = argparse.ArgumentParser(description='cppn-pytorch')
    parser.add_argument('--x_dim', default=2000, type=int, help='out image width')
    parser.add_argument('--y_dim', default=2000, type=int, help='out image height')
    parser.add_argument('--c_dim', default=1, type=int, help='channels')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--interpolation', default=10, type=int)
    parser.add_argument('--name_style', default='params', type=str, help='output fn')
    parser.add_argument('--exp', default='.', type=str, help='output fn')
    parser.add_argument('--name', default='.', type=str, help='output fn')
    parser.add_argument('--file', action='store_true', help='choose file path to reproduce')
    parser.add_argument('--rarch', action='store_true', help='load arch graph')
    parser.add_argument('--dir', action='store_true', help='input directory of images to reproduce')
    parser.add_argument('--z_dim', default=8, type=int, help='latent space width')
    parser.add_argument('--z_scale', default=10., type=float, help='mutiplier on z')
    parser.add_argument('--net', default=32, type=int, help='net width')

    args = parser.parse_args()
    return args


def main():
    args = load_args()
    if args.file:
        fn = args.name
        files = [fn]
    if args.dir:
        dirname = args.name
        files = glob.glob(dirname+'/*.tif')
    if not os.path.exists('trials/'+args.exp):
        os.makedirs('trials/'+args.exp)
    print ('Generating {} files'.format(len(files)))
    f_keep = glob.glob('temp/**/*.tif')
    f_keep = [f.split('/')[-1] for f in f_keep]
    for idx in range(len(files)):
        if idx < 6:
            continue
        gc.collect()
        path = files[idx]
        try:
            tifffile.TiffFile(path)
        except:
            print ('[FAIL] to load file')
            continue
        print ('[PASS] Loaded {}th file from {}'.format(idx, path))
        with tifffile.TiffFile(path) as tif:
           img = tif.asarray()
           metadata = tif.shaped_metadata[0]

        seed = int(metadata['seed'])
        z_dim = int(metadata['z'])
        z_scale = float(metadata['scale'])
        net = int(metadata['net'])
        c_dim = int(metadata['c_dim'])

        if 'graph' in metadata:
            graph = metadata['graph']
        else:
            graph = None
        if 'weight_init_mean' in metadata:
            w_mean = float(metadata['weight_init_mean'])
        else:
            w_mean = 0.0
        if 'weight_init_std' in metadata:
            w_std = float(metadata['weight_init_std'])
        else:
            w_std = 1.0

        z = torch.tensor(literal_eval(metadata['z_sample']))
        print ('\tz', z_dim)
        print ('\tscale', z_scale)
        print ('\twidth', net)
        print ('\tsample', z)
        print ('\tw mean', w_mean)
        print ('\tw std', w_std)
        cppn = CPPN(z_dim=z_dim,
                    x_dim=args.x_dim,
                    y_dim=args.y_dim,
                    c_dim=c_dim,
                    z_scale=z_scale,
                    seed=seed,
                    graph=graph,
                    weight_init_mean=w_mean,
                    weight_init_std=w_std,
                    layer_width=net)
        
        suff_extracted = path.split('/')[-1].split('.')[0]
        if graph not in [None, '']:
            random = 'graph'
        else:
            random = 'act'

        if path.split('/')[1] in f_keep:  # use smaller act set
            act_set = 'small'
        else:
            act_set = 'large'

        cppn.init_generator(random=random, graph=graph, act_set=act_set)
        if random == 'graph' and False:
            plot_graph(cppn.generator.graph,
                path=None, 
                #f'temp/{args.exp}/{suff_extracted}_reprojected_graph.png',
                plot=True)

        if random == 'graph':
            n = cppn.generator.graph.number_of_nodes()
            if n == 75 and net == 128:
                splits = 50
            elif n >= 40 and net >= 64:
                splits = 20
            elif n >= 40 and net >= 32:
                splits = 10
            elif n >=20 and net >=64:
                splits = 10
            else:
                splits = 5
        splits = 1
        if net > 32 and splits < 100:
            splits = 10
        elif splits < 40:
            splits = 10
        img = cppn.sample_frame(z, args.x_dim, args.y_dim, 1, splits=splits)
        img = img.cpu().detach().numpy()[0]*255

        if hasattr(cppn.generator, "get_graph"):
            g = cppn.generator.get_graph()
        else:
            g = cppn.generator.graph

        g.dpi = 1000
        options = {
            'label': '',
	    "font_size": 36,
	    "node_size": 3000,
	    "node_color": "white",
	    "edgecolors": "black",
	    "linewidths": 3,
	    "width": 2,
            "with_labels": False,
        }
        if random == 'graph':
            n = cppn.generator.graph.number_of_nodes()
            if n > 40:
                size = 30
            elif n > 20:
                size = 90
            elif n > 10:
                size = 200
            else:
                size = 250
            options['node_size'] = size

        H_layout = networkx.nx_pydot.pydot_layout(g, prog='dot')
        networkx.draw_networkx(g, H_layout, **options)
        ax = plt.gca()
        ax.margins(0.20)
        plt.axis("off")
        text = '{} {} {} {}'.format(int(z_dim), z_scale, int(net), int(seed))
        plt.savefig('temp_net.png', dpi=700)
        x = cv2.imread('temp_net.png')

        if c_dim == 3:
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        else:
            x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        x = cv2.bitwise_not(x)
        x_s = cv2.resize(x, (100, 100), interpolation=cv2.INTER_AREA)
        if c_dim == 1:
            x_s = x_s.reshape((x_s.shape[0], x_s.shape[1], 1))
        img_trans = np.zeros_like(img)
        img_trans[-x_s.shape[0]:, -x_s.shape[1]:, :] = x_s
        plt.close('all')

        """
        img = cv2.putText(img, text,
                (img.shape[0]-50, img.shape[1]-50), 
                (img.shape[0]-20, img.shape[1]-200),
                fontFace=2,
                fontScale=.10,
                color=(255, 255, 255),
                lineType=2,
                thickness=1)
        """

        if img.shape[-1] == 3:
            img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            img_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
            img_luv = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
            
            img = cv2.addWeighted(img, 1.0, img_trans, 0.5, 0)
            img_lab = cv2.addWeighted(img_lab, 1.0, img_trans, 0.5, 0)
            img_hsv = cv2.addWeighted(img_hsv, 1.0, img_trans, 0.5, 0)
            img_hls = cv2.addWeighted(img_hls, 1.0, img_trans, 0.5, 0)
            img_luv = cv2.addWeighted(img_luv, 1.0, img_trans, 0.5, 0)
            img_gray = cv2.addWeighted(
                img_gray,
                1.0,
                cv2.cvtColor(img_trans, cv2.COLOR_RGB2GRAY),
                0.5,
                0)
        else:
            img2 = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            img_hsv = cv2.cvtColor(img2, cv2.COLOR_RGB2HSV)
            img_gray = img
            img_lab = cv2.cvtColor(img2, cv2.COLOR_RGB2LAB)
            img_hls = cv2.cvtColor(img2, cv2.COLOR_RGB2HLS)
            img_luv = cv2.cvtColor(img2, cv2.COLOR_RGB2LUV) 
            
            img = cv2.addWeighted(img, 1.0, img_trans, 0.5, 0)
            img_gray = cv2.addWeighted(img_gray, 1.0, img_trans, 0.5, 0)
            img_trans = cv2.cvtColor(img_trans, cv2.COLOR_GRAY2RGB)
            img_lab = cv2.addWeighted(img_lab, 1.0, img_trans, 0.5, 0)
            img_hsv = cv2.addWeighted(img_hsv, 1.0, img_trans, 0.5, 0)
            img_hls = cv2.addWeighted(img_hls, 1.0, img_trans, 0.5, 0)
            img_luv = cv2.addWeighted(img_luv, 1.0, img_trans, 0.5, 0)

        os.makedirs(args.exp, exist_ok=True)
        save_fn = f'{args.exp}/{suff_extracted}_reprojection'
        print ('[SAVING] saving PNG image at: ', save_fn)
        cv2.imwrite(save_fn+'ci.png', img)
        cv2.imwrite(save_fn+'lab_ci.png', img_lab)
        cv2.imwrite(save_fn+'hsv_ci.png', img_hsv)
        cv2.imwrite(save_fn+'hls_ci.png', img_hls)
        cv2.imwrite(save_fn+'luv_ci.png', img_luv)
        cv2.imwrite(save_fn+'gray_ci.png', img_gray)
        print ('-'*80)
 

if __name__ == '__main__':
    main()
