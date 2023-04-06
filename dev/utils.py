import os
import cv2
import torch
import tifffile
import numpy as np
from ast import literal_eval



def lerp(z1, z2, n):
    delta = (z2 - z1) / (n + 1)
    total_frames = n + 2
    states = []
    for i in range(total_frames):
        z = z1 + delta * float(i)
        states.append(z)
    states = torch.stack(states)
    return states


def write_image(path, img, suffix='jpg', metadata=None):
    if suffix in ['jpg', 'png']:
        path = path + f'.{suffix}'
        cv2.imwrite(path, img)
        assert os.path.isfile(path)
    elif suffix == 'tif':
        print ('writing tif')
        #assert metadata is not None, "metadata must be included to save tif"
        path = path + '.tif'
        tifffile.imwrite(path, img, metadata=metadata)
    else:
        raise NotImplementedError

def load_tif_metadata(path):
    assert os.path.isfile(path)
    try:
        with tifffile.TiffFile(path) as tif:
            img = tif.asarray()
            data = tif.shaped_metadata[0]
    except:
        return None, None

    data['seed']  = int(data['seed'])
    data['noise_scale'] = float(data['scale'])
    data['layer_width']   = int(data['net'])
    data['c_dim'] = int(data['c_dim'])
    if 'weight_init_mean' in data:
        data['weight_init_mean'] = float(data['weight_init_mean'])
    else:
        data['weight_init_mean'] = 0.0
    if 'weight_init_std' in data:
        data['weight_init_std'] = float(data['weight_init_std'])
    else:
        data['weight_init_std'] = 1.0
    try:
        data['noise_dim'] = int(data['noise'])
        data['noise_sample'] = torch.tensor(literal_eval(data['noise_sample']))
    except:
        data['noise_dim'] = int(data['z'])
        data['noise_sample'] = torch.tensor(literal_eval(data['z_sample']))
    if 'graph' not in data:
        data['graph'] = None
    elif data['graph'] == '':
        data['graph'] = None
    return img, data


def draw_graph(num_nodes, random_graph, graph):
    graph.dpi = 1000
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
    if random_graph:
        if num_nodes > 40:
            plot_size = 30
        elif num_nodes > 20:
            plot_size = 90
        elif num_nodes > 10:
            plot_size = 200
        else:
            plot_size = 250
        options['node_size'] = plot_size

    H_layout = networkx.nx_pydot.pydot_layout(graph, prog='dot')
    networkx.draw_networkx(graph, H_layout, **options)
    ax = plt.gca()
    ax.margins(0.20)
    plt.axis("off")
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
    return img_trans


def write_video(frames, save_dir, save_name):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    width = frames[0].shape[0]
    height = frames[0].shape[1]
    path = os.path.join(save_dir, save_name)
    video = cv2.VideoWriter(f'{path}.avi', fourcc, 20., (width, height))
    for frame in frames: 
        video.write(frame.astype(np.uint8))
        cv2.destroyAllWindows()
        video.release()
