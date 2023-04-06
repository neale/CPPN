import gc
import cv2
import torch
import numpy as np
import tifffile

import utils


x = np.ones((1500, 1500, 1500, 3), dtype=np.uint8)
for i in range(1500):
    x[i] = (torch.load(f'f_temp_gen_{i}.pt').numpy() * 255.).astype('u1')
utils.write_image('final_1500', x.astype('u1'), 'tif', metadata={'this': 'that'})
gc.collect()
"""
x = np.ones((3000, 3000, 3000), dtype=np.uint8)
x[:1500] = np.load('frame_g0.npy')
x[1500:] = np.load('frame_g1.npy')
utils.write_image('final_g', x, 'tif', metadata={})
gc.collect()

x = np.ones((3000, 3000, 3000), dtype=np.uint8)
x[:1500] = np.load('frame_b0.npy')
x[1500:] = np.load('frame_b1.npy')
utils.write_image('final_b', x, 'tif', metadata={})
"""
