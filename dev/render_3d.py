"""render_3d.py

The purpose of this file is to take a .tiff file that represents
volumetric data and render it in 3D. This is done by using using 
transfer functions to map the data to a color and opacity. This renderer
can handle RGB valued volumetric data with x, y, z dimensions.
"""
import torch   
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import argparse
import pyvista as pv


def load_file(path):
    """Load a tiff file into a numpy array"""
    with tifffile.TiffFile(path) as tif:
        data = tif.asarray()
    return data


def render_volumetric_data(data, args):
    """"
    render volumetric RGB valued data
    data is a 3D array of shape (x, y, z, 3)
    args is an argparse object with the following attributes:
    """
    # Generate synthetic data
    x, y, z = np.meshgrid(np.linspace(-1, 1, 20),
                        np.linspace(-1, 1, 20),
                        np.linspace(-1, 1, 20), indexing='ij')
    data = np.empty((*x.shape, 3), dtype=float)

    # Assign RGB values to the synthetic data
    data[..., 0] = x
    data[..., 1] = y
    data[..., 2] = z

    # Normalize the RGB values to [0, 1]
    data = (data - data.min()) / (data.max() - data.min())

    # Create a StructuredGrid with the synthetic data
    grid = pv.StructuredGrid(x, y, z)
    grid.cell_arrays['colors'] = (data * 255).astype(np.uint8).reshape((-1, 3))

    # Create a PyVista plotter
    plotter = pv.Plotter()

    # Add the volume to the plotter
    plotter.add_volume(grid, opacity='linear', shade=True, use_cell_data=True)

    # Show the interactive window
    plotter.show()

    

def load_args():
    parser = argparse.ArgumentParser(description='cppn-pytorch')
    parser.add_argument('--x_dim', default=1100, type=int, help='out image width')
    parser.add_argument('--y_dim', default=1100, type=int, help='out image height')
    parser.add_argument('--z_dim', default=1100, type=int, help='out image height')
    parser.add_argument('--c_dim', default=3, type=int, help='channels')
    parser.add_argument('--save_dir', default='.', type=str, help='output fn')
    parser.add_argument('--name', default='.', type=str, help='output fn')
    parser.add_argument('--suffix', default='_reprojection1100', type=str)
    parser.add_argument('--input_file', default=None, type=str, help='choose file path to reproduce')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = load_args()
    data = load_file(args.input_file)
    render_volumetric_data(data, args)