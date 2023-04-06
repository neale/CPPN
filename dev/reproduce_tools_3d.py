import io
import gc
import os
import gc
import cv2
import glob
import tqdm
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

import utils
from run_cppn_3d_dev import CPPN
from maps import plot_graph


logging.getLogger().setLevel(logging.ERROR)


def load_args():

    parser = argparse.ArgumentParser(description='cppn-pytorch')
    parser.add_argument('--x_dim', default=1100, type=int, help='out image width')
    parser.add_argument('--y_dim', default=1100, type=int, help='out image height')
    parser.add_argument('--z_dim', default=1100, type=int, help='out image height')
    parser.add_argument('--c_dim', default=3, type=int, help='channels')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--save_dir', default='.', type=str, help='output fn')
    parser.add_argument('--name', default='.', type=str, help='output fn')
    parser.add_argument('--suffix', default='_reprojection1100', type=str)
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
        activation_set = 'large'
        img, metadata = utils.load_tif_metadata(path)
        if img is None:
            print ('[FAIL] to load TIF file')
            continue
        print ('[PASS] Loaded {}th file from {}'.format(idx, path))

        noise = metadata['noise_sample']
        if metadata['graph'] is not None:
            random = 'graph'
        else:
            random = 'act'

        cppn = CPPN(noise_dim=metadata['noise_dim'],
                    batch_size=args.batch_size,
                    x_dim=args.x_dim,
                    y_dim=args.y_dim,
                    z_dim=args.z_dim,
                    c_dim=metadata['c_dim'],
                    noise_scale=metadata['noise_scale'],
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
            graph_topology = 'fixed'
        else:
            activations = 'fixed'
            graph_topology = 'fixed'
        cppn.init_map_fn(activations=activations,
                         graph_topology=graph_topology,
                         graph=metadata['graph'],
                         activation_set=activation_set)
        
        original_params = path.split('/')[-1].split('.')[0]
        if metadata['graph'] is not None:
            random = 'graph'
            num_nodes = cppn.map_fn.graph.number_of_nodes()
            print ('num nodes', num_nodes)
            splits = 1
            if args.plot_graph:
                name = os.path.join('temp', args.save_dir, original_params)
                plot_graph(cppn.map_fn.graph,
                    path=None, 
                    plot=True)
        else:
            random = 'act'
            splits = 1

        if args.generate_zoom_set:
            frames = []
            print ('[START] Generating zoom frames')
            zoom_set = torch.from_numpy(np.geomspace(0.000001, 1000, num=200))
            for i, zoom_z in enumerate(zoom_set):
                zoom = (zoom_z, zoom_z)
                frame = cppn.sample_frame(noise, args.x_dim, args.y_dim, 1, zoom=zoom, splits=splits)
                frame = frame.cpu().detach().numpy()[0]*255
                frames.append(frame)
        else:
            print ('[START] Generating frame')
            if cppn.x_dim < 256:
                splits = 1
            elif cppn.x_dim >= 256 and cppn.x_dim <= 512:
                splits = 16 - (cppn.x_dim % 16)
            elif cppn.x_dim <= 1024:
                splits = 16 - (cppn.x_dim % 512)
            print ('using ', splits)
            
            batch_samples = []
            frame = cppn.sample_frame(noise, splits=splits)
            frame = frame.reshape(cppn.y_dim, cppn.x_dim, cppn.z_dim, cppn.c_dim)
            #frame = frame.cpu().numpy() * 255.
            """ 
            gc.collect()
            frame = torch.load(f'f_temp_gen0.pt')
            print ('composing..')
            for split in tqdm.tqdm(range(1, splits)):
                frame_next = torch.load(f'f_temp_gen{split}.pt')
                frame = torch.cat([frame, frame_next], dim=0)
                os.remove(f'f_temp_gen{split}.pt')
            """

        if args.draw_graph:
            if hasattr(cppn.map_fn, "get_graph"):
                graph = cppn.map_fn.get_graph()
            else:
                graph = cppn.map_fn.graph
            annotation = utils.draw_graph(num_nodes, random=='graph', graph)

        if args.draw_text:
            text = "{data['seed']} {data['scale']}, {data['net']}"
            for i in range(len(frames)):
                img = cv2.putText(frames[i], text,
                        (img.shape[0]-50, gen_img.shape[1]-50), 
                        (img.shape[0]-20, gen_img.shape[1]-200),
                        fontFace=2,
                        fontScale=.10,
                        color=(255, 255, 255),
                        lineType=2,
                        thickness=1)

        save_name = args.name+original_params+args.suffix
        print (f'[SAVING] saving PNG image at: {args.save_dir}/{save_name}')
        if args.save_all_formats:
            for i, frame in enumerate(frames):
                ind = f'{i}'.zfill(5)
                save_name_i = f'{ind}_'+save_name
                utils.save_all_formats(frame, annotation, args.save_dir, save_name_i)
        else:
            save_name_i = f'0_'+save_name
            utils.write_image(os.path.join(args.save_dir, save_name_i)+'ci',
                img=frame, suffix='tif', metadata={'232': 232})


        if args.generate_video_from_frames:
            print (f'[SAVING] saving MP4 video at: {args.save_dir}/{save_name}')
            save_name = args.name+original_params+args.suffix
            utils.write_video(frames, args.save_dir, save_name)
        print ('[DONE]')
        print ('-'*80)
 

if __name__ == '__main__':
    main()
