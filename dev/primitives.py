import os
import cv2
import numpy as np
import scipy.stats as st
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


def p_gmm(x_dim, y_dim):
    n_components = 3
    X, truth = make_blobs(
        n_samples=100,
        centers=n_components, 
        cluster_std=np.random.randint(1, 5, size=(n_components,)))
    x = X[:, 0]
    y = X[:, 1]# Define the borders
    deltaX = (max(x) - min(x))/10
    deltaY = (max(y) - min(y))/10
    xmin = min(x) - deltaX
    xmax = max(x) + deltaX
    ymin = min(y) - deltaY
    ymax = max(y) + deltaY
    xx, yy = np.mgrid[xmin:xmax:complex(0,x_dim), ymin:ymax:complex(0,y_dim)]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = st.gaussian_kde(values)
    canvas = np.reshape(kernel(positions).T, xx.shape)
    #plt.imshow(canvas, cmap='gray')
    #plt.show()
    return canvas


def p_squares_right(x_dim, y_dim, num_x=2):
    canvas = np.zeros((x_dim, y_dim))
    if y_dim % 128 == 0:
        num_y = 16
        size = stride = int(y_dim/num_y)
    elif y_dim % 100 == 0:
        num_y = 20
        size = stride = int(y_dim/num_y)
    for i in range(num_y):
        block_start = i * stride
        block_end = (i * stride) + stride
        if i % 2 == 0:
            canvas[-stride:, block_start:block_end] = 1.0
            canvas[-3*stride:-2*stride, block_start:block_end] = 1.0
        else:
            canvas[-2*stride:-stride, block_start:block_end] = 1.0
            canvas[-4*stride:-3*stride, block_start:block_end] = 1.0

    # blur
    canvas = canvas.T
    #canvas *= np.random.normal(1.0, 0.1, size=(512, 512))
    #plt.imshow(canvas, cmap='gray')
    #plt.show()
    return canvas


def p_squares_left(x_dim, y_dim, num_x=2):
    canvas = np.zeros((x_dim, y_dim))
    if y_dim % 128 == 0:
        num_y = 8
        size = stride = int(y_dim/num_y)
    elif y_dim % 100 == 0:
        num_y = 10
        size = stride = int(y_dim/num_y)
    for i in range(num_y):
        block_start = i * stride
        block_end = (i * stride) + stride
        if i % 2 == 0:
            canvas[:stride, block_start:block_end] = 255.
        else:
            canvas[stride:2*stride, block_start:block_end] = 255.

    # blur
    canvas = cv2.GaussianBlur(canvas.T, (21, 21), 0)
    #plt.imshow(canvas, cmap='gray')
    #plt.show()
    return canvas


def p_grad_img(x_dim, y_dim, path):
    assert os.path.isfile(path), "file provided in invalid"
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (x_dim, y_dim))
    img = cv2.Laplacian(img,cv2.CV_64F)
    img = cv2.GaussianBlur(img, (21, 21), 0)
    #plt.imshow(img)
    #plt.show()
    return img
