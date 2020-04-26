import os
import argparse
import numpy as np
import torch
import tifffile
import glob
from imageio import imwrite
from cppn import sample, init, Generator
from ast import literal_eval


def load_args():

    parser = argparse.ArgumentParser(description='cppn-pytorch')
    parser.add_argument('--x_dim', default=2048, type=int, help='out image width')
    parser.add_argument('--y_dim', default=2048, type=int, help='out image height')
    parser.add_argument('--c_dim', default=1, type=int, help='channels')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--interpolation', default=10, type=int)
    parser.add_argument('--name_style', default='params', type=str, help='output fn')
    parser.add_argument('--exp', default='.', type=str, help='output fn')
    parser.add_argument('--name', default='.', type=str, help='output fn')
    parser.add_argument('--file', action='store_true', help='choose file path to reproduce')
    parser.add_argument('--dir', action='store_true', help='input directory of images to reproduce')
    parser.add_argument('--z', default=8, type=int, help='latent space width')
    parser.add_argument('--scale', default=10., type=float, help='mutiplier on z')
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
    if args.name_style == 'simple':
        suff = 'image'
    if args.name_style == 'params':
        suff = 'z-{}_scale-{}_cdim-{}_net-{}'.format(args.z, args.scale, args.c_dim, args.net)
    print ('Generating {} files'.format(len(files)))

    for idx, path in enumerate(files):
        print (path)
        with tifffile.TiffFile(path) as tif:
           img = tif.asarray()
           metadata = tif.shaped_metadata[0]

        np.random.seed(int(metadata['seed']))
        torch.manual_seed(int(metadata['seed']))
        args.z = int(metadata['z'])
        args.scale = float(metadata['scale'])
        args.net = int(metadata['net'])
        args.c_dim = int(metadata['c_dim'])

        netG = init(Generator(args))
        z = literal_eval(metadata['z_sample'])
        z = torch.tensor(z)
        img = sample(args, netG, z).cpu().detach().numpy()
        if args.c_dim == 1:
            img = img[0][0]
        elif args.c_dim == 3:
            if args.x_dim == args.y_dim:
                img = img[0].reshape((args.x_dim, args.y_dim, 3))
            else:
                img = img[0].reshape((args.y_dim, args.x_dim, 3))
        imwrite('trials/{}/{}_{}.png'.format(args.exp, suff, idx), img*255)


if __name__ == '__main__':
    main()
