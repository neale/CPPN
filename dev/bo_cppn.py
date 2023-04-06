import time
from skopt import Optimizer
from skopt.space import Real, Space, Categorical
from functools import partial
from pytorch_msssim import ssim
import numpy as np
import torch
from torch import nn
from PIL import Image
import argparse
from run_cppn import CPPN
from utils import write_image
import lpips


class BOPPN(nn.Module):
    def __init__(self, 
                 noise_dim, 
                 target_image, 
                 map_net, 
                 coord_fn,
                 x_dim, y_dim,
                 output_dir):
        super().__init__()
        self.noise_dim = noise_dim
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.target_image = target_image
        self.map_net = map_net
        self.batch_size = 32
        self.act_dim = 9
        self.coord_fn = coord_fn
        self.x_vec, self.y_vec, self.r_vec, self.n_pts, _ = coord_fn(
            x_dim, 
            y_dim, 
            batch_size=1, 
            zoom=(.5,.5), 
            pan=(2,2))
        self.one_vec = torch.ones(self.n_pts, 1, dtype=torch.float)
        self.param_shapes = [p.shape for p in self.map_net.parameters()]
        self.param_sizes = [p.numel() for p in self.map_net.parameters()]
        self.total_param_size = sum(self.param_sizes)
        self.perceptual_loss = lpips.LPIPS(net='vgg').eval() # to(cppn.device)

        self.output_dir = output_dir
        target_image_th = torch.from_numpy(self.target_image).unsqueeze(0)
        self.target_image_th = target_image_th.permute(0, -1, 1, 2).repeat(self.batch_size, 1, 1, 1)

    def forward_map(self, weights, input_sample, act_order):
        for p, w in zip(self.map_net.parameters(), weights):
            p.data = w

        self.map_net.order = act_order
        self.map_net.generate_act_list()
        input_sample = torch.tensor(input_sample, dtype=torch.float32)
        input_sample = input_sample.view(1, 1, self.noise_dim) * self.one_vec * 10.0
        input_sample = input_sample.view(1*self.n_pts, self.noise_dim)
        output = self.map_net(self.x_vec, self.y_vec, self.r_vec, input_sample)
        output = output.reshape(*self.target_image_th.shape[1:])
        return output


    def compute_loss(self, output):

        # l1_loss = torch.nn.functional.l1_loss(output, target_image)
        ssim_loss = 1 - ssim(output, self.target_image_th, size_average=False)
        perceptual_loss = self.perceptual_loss(output, self.target_image_th)[:, 0, 0, 0]
        total_loss = perceptual_loss.detach().cpu().numpy() + ssim_loss.detach().cpu().numpy()
        # print (perceptual_loss.shape, ssim_loss.shape, total_loss.shape)
        return total_loss.tolist()


    def objective_function(self, param_set):
        outputs = []
        for params in param_set:
            weights = params[:-(self.noise_dim+self.act_dim)]
            input_sample = params[-(self.noise_dim+self.act_dim):-self.act_dim]
            act_order = params[-self.act_dim:]

            weights = torch.tensor(weights, dtype=torch.float32)
            param_splits = torch.split(weights, self.param_sizes, dim=-1)
            weights = [param_split.reshape(param_shape)
                    for param_split, param_shape in zip(param_splits, self.param_shapes)]
            
            output = self.forward_map(weights, input_sample, act_order)
            outputs.append(output)
        outputs = torch.stack(outputs)
        losses = self.compute_loss(outputs)
        return losses, outputs


    def optimize_map(self, n_calls=100, n_initial_points=10):
        # Create the search space
        space = []
        for p in self.map_net.parameters():
            space.extend([Real(-5, 5, dtype=float) for _ in range(np.prod(p.shape))])
        space.extend([Real(-2, 2) for _ in range(self.noise_dim)])  # Input sample
        space.extend([Categorical(np.arange(len(self.map_net.all_acts))) for _ in range(self.act_dim)])  # act_order

        # Initialize the Bayesian optimizer
        _optimizer = Optimizer(space, base_estimator="gp", n_initial_points=n_initial_points)

        # Optimize the objective function
        for call_iter in range(n_calls):
            x = _optimizer.ask(n_points=self.batch_size)
            loss, outputs = self.objective_function(x)
            _optimizer.tell(x, loss)
            print (f'[Iter] {call_iter} [Loss min/max] {min(loss), max(loss)}')
            if call_iter % 5 == 0:
                for i, output in enumerate(outputs):
                    output = output.reshape(self.x_dim, self.y_dim, 3)
                    output = output.detach().cpu().numpy() * 255
                    write_image(f'{self.output_dir}/output_{call_iter}_{i}', output, suffix='jpg')
        self.optimizer = _optimizer


def load_args():
    parser = argparse.ArgumentParser(description='cppn-pytorch')
    parser.add_argument('--noise_dim', default=8, type=int, help='latent space width')
    parser.add_argument('--n_samples', default=1, type=int, help='images to generate')
    parser.add_argument('--x_dim', default=256, type=int, help='out image width')
    parser.add_argument('--y_dim', default=256, type=int, help='out image height')
    parser.add_argument('--noise_scale', default=10, type=float, help='mutiplier on z')
    parser.add_argument('--c_dim', default=3, type=int, help='channels')
    parser.add_argument('--layer_width', default=16, type=int, help='net width')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--graph_nodes', default=10, type=int, help='number of graph_nodes in graph')
    parser.add_argument('--output_dir', default='trial', type=str, help='output fn')
    parser.add_argument('--no_tiff', action='store_true', help='save tiff metadata')
    parser.add_argument('--sweep_settings', action='store_true', help='sweep hps')
    parser.add_argument('--activations', default='permute', type=str, help='')
    parser.add_argument('--target_img_path', default='trial', type=str, help='image to match')
    parser.add_argument('--graph_topology', default='fixed', type=str, help='')
    args = parser.parse_args()
    return args

args = load_args()
cppn = CPPN(noise_dim=args.noise_dim,
            n_samples=args.n_samples,
            x_dim=args.x_dim,
            y_dim=args.y_dim,
            c_dim=args.c_dim,
            noise_scale=args.noise_scale,
            layer_width=args.layer_width,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            graph_nodes=args.graph_nodes)
cppn.init_map_fn(activations=args.activations,
                 graph_topology=args.graph_topology)


# Load the target image and preprocess it
target_image = Image.open(args.target_img_path).convert("RGB")
target_image = np.array(target_image).astype(np.float32) / 255.0

bo = BOPPN(args.noise_dim, 
           target_image, 
           cppn.map_fn, 
           coord_fn=cppn._coordinates,
           x_dim=cppn.x_dim,
           y_dim=cppn.y_dim,
           output_dir=args.output_dir,)

# Run the Bayesian optimization
bo.optimize_map(n_calls=1000, n_initial_points=128) 
# Get the best parameters and input found by the Bayesian optimization
best_params = bo.optimizer.Xi[np.argmin(bo.optimizer.yi)]

# Set the parameters of the Map network to the best weights found
best_weights = torch.tensor(best_params[:-(bo.noise_dim+bo.act_dim)])
best_input = torch.tensor(best_params[-(bo.noise_dim + bo.act_dim):-bo.act_dim])
best_acts = best_params[-bo.act_dim:]

param_splits = torch.split(best_weights, bo.param_sizes, dim=-1)
best_weights = [param_split.reshape(param_shape)
            for param_split, param_shape in zip(param_splits, bo.param_shapes)]

for p, w in zip(cppn.map_fn.parameters(), best_weights):
    p.data = w

cppn.map_fn.order = best_acts
cppn.map_fn.generate_act_list()
input_sample = best_input.view(bo.batch_size, 1, bo.noise_dim) * bo.one_vec * args.noise_scale
input_sample = input_sample.view(bo.batch_size*bo.n_pts, bo.noise_dim)
# Generate the final output image using the best input found
x = bo.x_vec
y = bo.y_vec
r = bo.r_vec
final_output = cppn.map_fn(x, y, r, input_sample)
final_output = final_output.reshape(*target_image.shape)

# Convert the final output to a NumPy array
final_output_np = final_output.detach().cpu().numpy() * 255.0
write_image(f'{args.output_dir}/final_output.png', final_output_np, suffix='jpg')