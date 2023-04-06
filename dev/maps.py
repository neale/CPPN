import sys
import math
import os
import io
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
import numpy as np
import networkx
import collections

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from networkx.drawing.nx_agraph import graphviz_layout, to_agraph

from random import random

Node = collections.namedtuple('Node', ['id', 'inputs', 'type'])

def get_graph_info(graph):
    input_nodes = []
    output_nodes = []
    nodes = []
    #print ('nodes', graph.number_of_nodes())
    for node in range(graph.number_of_nodes()):
        tmp = list(graph.neighbors(node))
        tmp.sort()
        type = -1
        if node < tmp[0]:
            input_nodes.append(node)
            type = 0
        if node > tmp[-1]:
            output_nodes.append(node)
            type = 1
        nodes.append(Node(node, [n for n in tmp if n < node], type))
    return nodes, input_nodes, output_nodes


def build_random_graph(nodes, input_nodes, output_nodes, p, k):
    g = networkx.random_graphs.connected_watts_strogatz_graph(
        nodes,
        k, p,
        tries=200)
    assert nodes > input_nodes, "number of nodes must be > input ndoes"
    unodes = np.random.randint(input_nodes, nodes, size=(nodes+input_nodes,))
    lnodes = np.random.randint(0, input_nodes, size=(nodes+input_nodes,))
    # make sure input nodes don't have edges between them
    for i in range(input_nodes):
        for j in range(input_nodes):
            try:
                g.remove_edge(i, j)
            except: # no edge exists (easier to ask forgiveness)
                pass
            try:
                g.remove_edge(j, i)
            except:  # no edge exists
                pass
        g.add_edge(i, unodes[i])
    # handle unconnected nodes other than specified input nodes
    # loop through nodes, and add connections from previous nodes if none exist
    for iter, unode in enumerate(range(input_nodes, nodes)):
        if k < input_nodes + k:
            if not any([g.has_edge(lnode, unode) for lnode in range(unode)]):
                n = lnodes[iter] # get one of the input nodes
                g.add_edge(n, unode)
        else:
            if not any([g.has_edge(lnode, unode) for lnode in range(unode)]):
                n = unodes[i+1] # get one of the preceeding nodes
                g.add_edge(n, unode)
                i += 1
    if output_nodes > 1:
        # handle output layers, we want 3 nodes at the output
        # hueristic: try to just use top info, only connect top layers ot 
        for new_node in range(nodes, nodes+output_nodes):
            g.add_node(new_node)
        for node in range(input_nodes, nodes):
            if not any([g.has_edge(lnode, node) for lnode in range(node, nodes)]):
                #output node
                out_node = np.random.choice(np.arange(nodes, nodes+output_nodes))
                g.add_edge(node, out_node)
                #print ('output', node, 'edge: ', node, out_node)
        for out_node in range(nodes, nodes+output_nodes):
            if g.degree[out_node] == 0:
                g.add_edge(np.random.choice(np.arange(nodes//2, nodes)), out_node)
    return g


def plot_graph(g, path=None, plot=False):
    dot = networkx.nx_pydot.to_pydot(g)
    png_str = dot.create_png(prog='dot')
    # treat the DOT output as an image file
    sio = io.BytesIO()
    sio.write(png_str)
    sio.seek(0)
    img = mpimg.imread(sio)
    imgplot = plt.imshow(img, aspect='equal')
    if path:
        plt.savefig(path)
    if plot:
        plt.show() 
    plt.close('all')


def randact(activation_set='large'):
    if activation_set == 'large':
        acts = [nn.ELU, nn.Hardtanh, nn.LeakyReLU, nn.LogSigmoid,
                nn.SELU, nn.GELU, nn.CELU, nn.Softshrink, nn.Sigmoid,
                SinLayer, CosLayer, Gaussian, nn.Softplus, nn.Mish,
                nn.Tanh, nn.ReLU]
    else:
        acts = [nn.Sigmoid, SinLayer, CosLayer, Gaussian, nn.Softplus, nn.Mish,
                 nn.Tanh, nn.ReLU]

    x = torch.randint(0, len(acts), (1,))
    return acts[x]()


class Gaussian(nn.Module):
    def __init__(self):
        super(Gaussian, self).__init__()

    def forward(self, x, a=1.0):
        return a * torch.exp((-x ** 2) / (2 * a ** 2))  # big?


class SinLayer(nn.Module):
    def __init__(self):
        super(SinLayer, self).__init__()

    def forward(self, x):
        return torch.sin(x)


class CosLayer(nn.Module):
    def __init__(self):
        super(CosLayer, self).__init__()

    def forward(self, x):
        return torch.cos(x)


class ScaleOp(nn.Module):
    def __init__(self):
        super(ScaleOp, self).__init__()
        self.r = torch.ones(1,).uniform_(-1, 1)
    
    def forward(self, x):
        return x * self.r


class AddOp(nn.Module):
    def __init__(self):
        super(AddOp, self).__init__()
        self.r = torch.ones(1,).uniform_(-.5, .5)

    def forward(self, x):
        return x + self.r


class LinearActOp(nn.Module):
    def __init__(self, in_d, out_d, actset):
        super(LinearActOp, self).__init__()
        self.linear = nn.Linear(in_d, out_d)
        self.act = randact(actset)

    def forward(self, x):
        return self.act(self.linear(x))


class ConvActOp(nn.Module):
    def __init__(self, in_d, out_d, actset):
        super(ConvActOp, self).__init__()
        self.conv = nn.Conv2d(in_d, out_d, kernel_size=1, stride=1)
        self.act = randact(actset)

    def forward(self, x):
        x = x.reshape(x.size(0), x.size(1), 1, 1)
        out = self.act(self.conv(x))
        out = out.reshape(out.size(0), out.size(1))
        return out


class RandOp(nn.Module):
    def __init__(self, in_dim, out_dim, actset):
        super(RandOp, self).__init__()
        r_id = torch.randint(0, 3, size=(1,))
        if r_id == 0:
            self.op = ScaleOp()
        elif r_id == 1:
            self.op = AddOp()
        elif r_id == 2:
            self.op = LinearActOp(in_dim, out_dim, actset)
        #elif r_id == 1:
        #    self.op = ConvActOp(in_dim, out_dim, actset)
        else:
            raise ValueError

    def forward(self, x):
        return self.op(x)

            
class RandNodeOP(nn.Module):
    def __init__(self, node, in_dim, out_dim, actset):
        super(RandNodeOP, self).__init__()
        self.is_input_node = Node.type == 0
        self.input_nums = len(node.inputs)
        if self.input_nums > 1:
            self.mean_weight = nn.Parameter(torch.ones(self.input_nums))
            self.sigmoid = nn.Sigmoid()
        self.op = RandOp(in_dim, out_dim, actset)

    def forward(self, *input):
        if self.input_nums > 1:
            #out = self.sigmoid(self.mean_weight[0]) * input[0]
            out = input[0]
        for i in range(1, self.input_nums):
            #out = out + self.sigmoid(self.mean_weight[i]) * input[i]
            out = out + input[i]
        else:
            out = input[0]
        out = self.op(out)
        return out


class TorchGraph(nn.Module):
    def __init__(self, graph, in_dim, hidden_dim, out_dim, combine, actset):
        super(TorchGraph, self).__init__()
        self.nodes, self.input_nodes, self.output_nodes = get_graph_info(graph)
        self.combine = combine
        self.node_ops = nn.ModuleList()
        for node in self.nodes:
            self.node_ops.append(RandNodeOP(node, in_dim, hidden_dim, actset))
        if combine:
            self.linear_out = nn.Linear(hidden_dim, out_dim)
            self.act_out = randact(actset)
        else:
            self.linear_out = [nn.Linear(hidden_dim, 1) for _ in range(len(
                self.output_nodes))]
            self.act_out = [randact(actset) for _ in range(len(self.output_nodes))]
    
    def forward(self, x):
        out = {}
        for id in self.input_nodes:
            out[id] = self.node_ops[id](x)
        for id, node in enumerate(self.nodes):
            if id not in self.input_nodes:
                out[id] = self.node_ops[id](*[out[_id] for _id in node.inputs])
        if self.combine:
            result = out[self.output_nodes[0]]
            for idx, id in enumerate(self.output_nodes):
                if idx > 0:
                    result = result + out[id]
            result = self.act_out(self.linear_out(result))
            return result
        else:
            outputs = [out[id] for id in self.output_nodes]
            outputs = [self.linear_out[i](out[i]) for i in range
                (len(self.output_nodes))]
            result = torch.cat(outputs, dim=-1)
            result = self.act_out[0](result)
        return result


class MapRandomGraph(nn.Module):
    def __init__(self, 
                 z_dim,
                 c_dim,
                 layer_width,
                 scale_z,
                 nodes,
                 graph=None,
                 activation_set='large',
                 activations='permute'):
        super(MapRandomGraph, self).__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.layer_width = layer_width
        self.input_nodes = 1
        self.name = 'MapRandomGraph'
        self.nodes = nodes
        self.activations = activations

        self.linear_z = nn.Linear(self.z_dim, self.layer_width)
        self.linear_x = nn.Linear(1, self.layer_width, bias=False)
        self.linear_y = nn.Linear(1, self.layer_width, bias=False)
        self.linear_r = nn.Linear(1, self.layer_width, bias=False)
        self.linear1 = nn.Linear(self.layer_width, self.layer_width)
        self.linear2 = nn.Linear(self.layer_width, self.layer_width)
        if self.activations == 'permute':
            self.act1 = randact(activation_set)
            self.act2 = randact(activation_set)
            self.act3 = randact(activation_set)
            self.act4 = randact(activation_set)
            self.act5 = randact(activation_set)
            self.act6 = randact(activation_set)
        elif self.activations == 'fixed':
            self.act1 = nn.Tanh()
            self.act2 = nn.ELU()
            self.act3 = nn.Softplus()
            self.act4 = nn.Tanh()
            self.act5 = Gaussian()
            self.act6 = SinLayer()
        else:
            raise ValueError('activations must be fixed or permute')

        k = 4
        p = .75
        out_nodes = 1
        combine = False
        if out_nodes == 1:
            combine=True
        if graph is None:
            self.graph = build_random_graph(
                self.nodes,
                self.input_nodes,
                out_nodes,
                p,
                k)
        else:
            print ('loading old graph')
            self.graph = self.load_graph_str(graph)

        self.network = TorchGraph(self.graph,
                                  self.layer_width,
                                  self.layer_width,
                                  self.c_dim,
                                  combine,
                                  activation_set)
        
    def generate_act_list(self):
        a = [self.all_acts[i] for i in self.order]
        self.acts = nn.ModuleList(a)    

    def get_graph_str(self):
        s = ''.join(networkx.generate_graphml(self.graph))
        return s

    def load_graph_str(self, s):
        return networkx.parse_graphml(s, node_type=int)

    def forward(self, x, y, r, z):
        z_ = self.act1(self.linear_z(z))
        r_ = self.act2(self.linear_r(r))
        y_ = self.act3(self.linear_y(y))
        x_ = self.act4(self.linear_x(x))
        f = self.act5(z_ + x_ + y_ + r_)
        f = self.act6(self.linear1(f))
        res = self.network(f)
        return res


def plot(x, c=1):
    from torchvision.utils import make_grid
    if c == 1:
        g = make_grid(x.reshape(-1, c, 512, 512))
        plt.imshow(g.permute(1, 2, 0).numpy())
    elif c == 3:
        plt.imshow(x.reshape(512, 512, 3))
    plt.show()


class Map(nn.Module):
    def __init__(self,
                 noise_dim,
                 c_dim,
                 layer_width,
                 noise_scale,
                 name='Map'):
        super(Map, self).__init__()
        self.noise_dim = noise_dim
        self.c_dim = c_dim
        self.layer_width = layer_width
        self.noise_scale = noise_scale
        self.name = name

        self.linear_noise = nn.Linear(self.noise_dim, self.layer_width)
        self.linear_x = nn.Linear(1, self.layer_width, bias=False)
        self.linear_y = nn.Linear(1, self.layer_width, bias=False)
        self.linear_r = nn.Linear(1, self.layer_width, bias=False)

        #self.linear_gmm = nn.Linear(1, self.layer_width, bias=False)
        #self.linear_sqr = nn.Linear(1, self.layer_width, bias=False)
        #self.linear_sql = nn.Linear(1, self.layer_width, bias=False)
        #self.linear_grad_img = nn.Linear(1, self.layer_width, bias=False)

        self.linear_h = nn.Linear(self.layer_width, self.layer_width)
        self.linear_h1 = nn.Linear(self.layer_width, self.layer_width)
        self.linear_h2 = nn.Linear(self.layer_width, self.layer_width)
        self.linear_out = nn.Linear(self.layer_width, self.c_dim)

        #self.ste = StraightThroughEstimator()
    
    def forward(self, x, y, r, noise_scaled, extra=None):
        noise_pt = self.linear_noise(noise_scaled)
        x_pt = self.linear_x(x)
        y_pt = self.linear_y(y)
        r_pt = self.linear_r(r)
        U = noise_pt + x_pt + y_pt + r_pt
        if extra is not None:
            assert type(extra) == dict
            if 'gmm' in extra:
                extra_output = self.linear_gmm(extra['gmm'])
            if 'sqr' in extra:
                extra_output += self.linear_sqr(extra['sqr'])
            if 'sql' in extra:
                extra_output += self.linear_sql(extra['sql'])
            if 'grad' in extra:
                extra_output += self.linear_grad_img(extra['grad'])
            U += extra_output
        
        H1 = torch.tanh(U)
        H2 = F.elu(self.linear_h(H1))
        H3 = F.softplus(self.linear_h1(H2))
        #H4 = torch.tanh(self.linear_h(H3))
        H4 = torch.tanh(self.linear_h2(H3))
        x =  torch.sigmoid(self.linear_out(H4))
        #x = .5 * torch.sin(self.linear_out(H4)) + .5
        #x = self.ste(H4)
        #breakpoint()
        return x


class MapConv(nn.Module):
    def __init__(self,
                 noise_dim,
                 c_dim,
                 layer_width,
                 noise_scale,
                 act_order=None,
                 clip_loss=False,
                 name='MapConv'):
        super(MapConv, self).__init__()
        self.noise_dim = noise_dim
        self.c_dim = c_dim
        self.layer_width = layer_width
        self.noise_scale = noise_scale
        self.name = name
        self.feat_dim = 100

        self.conv1 = nn.Conv2d(4, self.feat_dim, kernel_size=1, stride=1, padding='same')
        self.conv2 = nn.Conv2d(self.feat_dim, self.feat_dim, kernel_size=1, stride=1, padding='same')
        self.conv3 = nn.Conv2d(self.feat_dim, self.feat_dim, kernel_size=1, stride=1, padding='same')
        self.conv4 = nn.Conv2d(self.feat_dim, self.feat_dim, kernel_size=1, stride=1, padding='same')
        self.conv5 = nn.Conv2d(self.feat_dim, self.feat_dim, kernel_size=1, stride=1, padding='same')
        self.conv6 = nn.Conv2d(self.feat_dim, self.feat_dim, kernel_size=1, stride=1, padding='same')
        #self.conv7 = nn.Conv2d(self.feat_dim, self.feat_dim, kernel_size=1, stride=1, padding='same')
        #self.conv8 = nn.Conv2d(self.feat_dim, self.feat_dim, kernel_size=1, stride=1, padding='same')
        self.conv_rgb = nn.Conv2d(self.feat_dim, 3, kernel_size=1, stride=1, padding='same')
        #self.act = self.relu
        self.acts = [nn.GELU() for _ in range(6)]
        self.norms = nn.ModuleList([nn.LayerNorm([self.feat_dim, 256, 256]) for _ in range(6)])
        self.all_acts = [nn.ELU(), nn.Hardtanh(), nn.LeakyReLU(), nn.LogSigmoid(),
            nn.SELU(), nn.GELU(), nn.CELU(), nn.Sigmoid(), nn.Mish(),
            nn.Softplus(), nn.Softshrink(), nn.Tanh(), torch.nn.ReLU(),
            SinLayer(), CosLayer()]
        if act_order is None:
            self.order = torch.randint(0, 15, size=(9,))
        else:
            assert isinstance(act_order, torch.Tensor) and act_order.shape == (9,)
            self.order = act_order
        self.generate_act_list()
        if clip_loss:
            self.act_out = torch.tanh
        else:
            self.act_out = torch.sigmoid

    def generate_act_list(self):
        a = [self.all_acts[i] for i in self.order]
        #self.acts = nn.ModuleList(a)    

    def atan(self, x):
        a = torch.atan(x)
        return torch.cat([a/.67, (a**2 - .45)/.396], 1)
    
    def relu(self, x):
        x = F.relu(x)
        return (x-.4)/.58
    
    def forward(self, x, y, r, noise_scaled, extra=None):

        z = noise_scaled[..., 0][0].reshape(*x.shape)
        x = torch.stack([x, y, r, z], 0).unsqueeze(0)
        x = self.acts[0](self.norms[0](self.conv1(x)))
        x = self.acts[1](self.norms[1](self.conv2(x)))
        x = self.acts[2](self.norms[2](self.conv3(x)))
        x = self.acts[3](self.norms[3](self.conv4(x)))
        x = self.acts[4](self.norms[4](self.conv5(x)))
        x = self.acts[5](self.norms[5](self.conv6(x)))
        #x = self.act(self.conv7(x))
        #x = self.act(self.conv8(x))
        x = self.act_out(self.conv_rgb(x))

        return x
   

class MapRandomAct(nn.Module):
    def __init__(self,
                 z_dim,
                 c_dim,
                 layer_width,
                 scale_z,
                 act_order=None,
                 name='MapRandomAct'):
        super(MapRandomAct, self).__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.layer_width = layer_width
        self.scale_z = scale_z
        self.name = name
        self.linear_z = nn.Linear(self.z_dim, self.layer_width)
        self.linear_x = nn.Linear(1, self.layer_width, bias=False)
        self.linear_y = nn.Linear(1, self.layer_width, bias=False)
        self.linear_r = nn.Linear(1, self.layer_width, bias=False)

        self.linear_gmm = nn.Linear(1, self.layer_width, bias=False)
        self.linear_sqr = nn.Linear(1, self.layer_width, bias=False)
        self.linear_sql = nn.Linear(1, self.layer_width, bias=False)

        self.linear_h = nn.Linear(self.layer_width, self.layer_width)
        self.linear_out = nn.Linear(self.layer_width, self.c_dim)

        self.all_acts = [nn.ELU(), nn.Hardtanh(), nn.LeakyReLU(), nn.LogSigmoid(),
            nn.SELU(), nn.GELU(), nn.CELU(), nn.Sigmoid(), nn.Mish(),
            nn.Softplus(), nn.Softshrink(), nn.Tanh(), torch.nn.ReLU(),
            SinLayer(), CosLayer()]
        if act_order is None:
            self.order = torch.randint(0, 15, size=(9,))
        else:
            assert isinstance(act_order, torch.Tensor) and act_order.shape == (9,)
            self.order = act_order
        self.generate_act_list()

    def generate_act_list(self):
        a = [self.all_acts[i] for i in self.order]
        self.acts = nn.ModuleList(a)    

    def get_graph(self):
        g = networkx.Graph()
        for i in range(6):
            g.add_node(i)
        for i in range(4):
            g.add_edge(i, 4)
        g.add_edge(4, 4)
        g.add_edge(4, 4)
        g.add_edge(4, 4)
        g.add_edge(4, 5)

        for node in range(6):
            g.nodes[node]['forcelabels'] = False
            g.nodes[node]['shape'] = 'circle'
            g.nodes[node]['id'] = ''
            g.nodes[node]['label'] = ''
            g.nodes[node]['rotatation'] = 180
            g.nodes[node]['bgcolor'] = 'transparent'
        g.bgcolor = "transparent"

        return g
 
    def forward(self, x, y, r, z_scaled, extra=None):
        z_pt = self.acts[0](self.linear_z(z_scaled))
        x_pt = self.acts[1](self.linear_x(x))
        y_pt = self.acts[2](self.linear_y(y))
        r_pt = self.acts[3](self.linear_r(r))
        U = z_pt.add_(x_pt).add_(y_pt).add_(r_pt)
        H = self.acts[4](U)
        H = self.acts[5](self.linear_h(H))
        H = self.acts[6](self.linear_h(H))
        H = self.acts[7](self.linear_h(H))
        x = .5 * self.acts[8](self.linear_out(H)) + .5
        x = torch.sigmoid(x)
        return x


class STEFunction(autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)  # clamp grads


class StraightThroughEstimator(nn.Module):
    def __init__(self):
        super(StraightThroughEstimator, self).__init__()

    def forward(self, x):
        x = STEFunction.apply(x)
        return x


class LinBnReLU(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinBnReLU, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        #self.bn = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        x = self.linear(x)
        #x = self.bn(x)
        #x = F.silu(x)
        x = torch.tanh(x)
        return x

class LinDecoder(nn.Module):
    def __init__(self, z_dim):
        super(LinDecoder, self).__init__()
        self.block1 = LinBnReLU(128, 512)
        self.block2 = LinBnReLU(512, 2048)
        self.block3 = LinBnReLU(2048, 256*256*3)
        self.fc_input = nn.Linear(z_dim, 128)

    def forward(self, z):
        z = F.silu(self.fc_input(z))
        out = self.block1(z)
        out = self.block2(out)
        out = self.block3(out)
        out = torch.tanh(out)
        return out

class ConvTBnReLU(nn.Module):
    def __init__(self, filters_in, filters_out, padding=0, output_padding=0, upsample=1):
        super(ConvTBnReLU, self).__init__()
        self.conv = nn.ConvTranspose2d(filters_in, filters_out, kernel_size=3,
            stride=2, padding=padding, output_padding=output_padding)
        self.bn = nn.BatchNorm2d(filters_out)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x


class ConvDecoder(nn.Module):  # only works for 256 sized square images
    def __init__(self, z_dim):
        super(ConvDecoder, self).__init__()
        nf = 2
        self.block1 = ConvTBnReLU(256, nf*8, padding=1, output_padding=1, upsample=2)
        self.block2 = ConvTBnReLU(nf*8, nf*8, padding=1, output_padding=1, upsample=2)
        self.block3 = ConvTBnReLU(nf*8, nf*4, padding=1, output_padding=1, upsample=2)
        self.block4 = ConvTBnReLU(nf*4, nf*4, padding=1, output_padding=1, upsample=2)
        self.block5 = ConvTBnReLU(nf*4, nf*2, padding=1, output_padding=1, upsample=2)
        self.block6 = ConvTBnReLU(nf*2, nf, padding=1, output_padding=0, upsample=2)
        self.block7 = ConvTBnReLU(nf, 3, padding=0, output_padding=1, upsample=1)
        self.fc_input = nn.Linear(z_dim, 256*2*2)

    def forward(self, z):
        z = F.silu(self.fc_input(z))
        out = z.view(-1, 256, 2, 2)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.block7(out)
        out = torch.tanh(out)
        return out


if __name__ == '__main__':
    import time
    times = []
    for _ in range(20):
        model = MapRandomGraph()
    #print(model)

