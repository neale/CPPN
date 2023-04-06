import os, gc
import time
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
import torch.nn.functional as F

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import qExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.generation.sampling import MaxPosteriorSampling
from botorch.optim.initializers import initialize_q_batch, gen_batch_initial_conditions
from botorch.utils.transforms import standardize, normalize, unnormalize

from maps import ConvDecoder


class BOPPN(nn.Module):
    def __init__(self, 
                 noise_dim, 
                 target_image, 
                 map_net, 
                 coord_fn,
                 x_dim, y_dim,
                 output_dir,
                 device):
        super().__init__()
        self.noise_dim = noise_dim
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.target_image = target_image
        self.map_net = map_net
        self.batch_size = 128
        self.act_dim = 9
        self.coord_fn = coord_fn
        self.device = device

        if coord_fn is not None:
            self.x_vec, self.y_vec, self.r_vec, self.n_pts = coord_fn(
                x_dim, 
                y_dim, 
                batch_size=1, 
                zoom=(.5,.5), 
                pan=(2,2),
                noise_scale=10,
                device=device)
            self.one_vec = torch.ones(self.n_pts, 1, dtype=torch.float).to(device)
            self.conv_decoder = False
        else:
            self.x_vec = None
            self.y_vec = None 
            self.r_vec = None 
            self.n_pts = None
            self.one_vec = None
            self.conv_decoder = True

        self.param_shapes = [p.shape for p in self.map_net.parameters()]
        self.param_sizes = [p.numel() for p in self.map_net.parameters()]
        self.total_param_size = sum(self.param_sizes)
        self.perceptual_loss = lpips.LPIPS(net='vgg').eval().to(self.device)

        self.output_dir = output_dir
        target_image_th = torch.from_numpy(self.target_image).unsqueeze(0)
        self.target_image_th = target_image_th.permute(0, -1, 1, 2)

    @torch.no_grad()
    def forward_map(self, weights, input_sample, act_order=None):
        for p, w in zip(self.map_net.parameters(), weights):
            p.data = w
        if self.conv_decoder == False:
            x_vec = self.x_vec.detach().clone()
            y_vec = self.y_vec.detach().clone()
            r_vec = self.r_vec.detach().clone()
            one_vec = self.one_vec.detach().clone()
            self.map_net.order = act_order
            self.map_net.generate_act_list()
            input_sample = input_sample.view(1, 1, self.noise_dim) * one_vec * 10.0
            input_sample = input_sample.view(1*self.n_pts, self.noise_dim).to(self.device)
            output = self.map_net(x_vec, y_vec, r_vec, input_sample)
        else:
            input_sample = input_sample.view(1, 1, self.noise_dim)
            output = self.map_net(input_sample)
        output = output.reshape(*self.target_image_th.shape[1:])
        return output

    @torch.no_grad()
    def compute_loss(self, output):
        #return output.mean(dim=(1,2,3))
        # l1_loss = torch.nn.functional.l1_loss(output, target_image)
        target = self.target_image_th.repeat(len(output), 1, 1, 1).float().to(self.device)
        #ssim_loss = 1 - ssim(output, target, size_average=False)
        #perceptual_loss = self.perceptual_loss(output, target)[:, 0, 0, 0]
        #total_loss = perceptual_loss + ssim_loss
        l2_loss = torch.tensor([F.mse_loss(o, t) for o, t in zip(output, target)])
        total_loss = l2_loss
        # print (perceptual_loss.shape, ssim_loss.shape, total_loss.shape)
        return total_loss

    def objective_function(self, param_set):
        outputs = []
        input_shape = param_set.shape
        if param_set.ndim == 3:
            param_set = param_set.reshape(param_set.shape[0]*param_set.shape[1], -1)
        for params in param_set:
            if self.conv_decoder:
                weights = params[:-self.noise_dim]
                input_sample = params[-self.noise_dim:]
                act_order=None
            else:
                weights = params[:-(self.noise_dim+self.act_dim)]
                input_sample = params[-(self.noise_dim+self.act_dim):-self.act_dim]
                act_order = params[-self.act_dim:].long()

            param_splits = torch.split(weights, self.param_sizes, dim=-1)
            weights = [param_split.reshape(param_shape)
                    for param_split, param_shape in zip(param_splits, self.param_shapes)]
            output = self.forward_map(weights, input_sample, act_order)
            outputs.append(output)
        outputs = torch.stack(outputs)
        losses = self.compute_loss(outputs)
        losses = losses.reshape(*input_shape[:-1])
        return losses

    def gen_initial_conditions(self, bounds, n_initial_points=4):
        x = torch.rand(n_initial_points, bounds.shape[-1]).to(self.device)
        x = unnormalize(x, bounds=bounds)
        return x
    
    def optimize_map(self, n_calls=100, n_initial_points=4):
        # Create the search space
        if self.conv_decoder:
            bounds = torch.tensor([
                            [-5]*self.total_param_size + [-2]*self.noise_dim,
                            [5] *self.total_param_size + [2] *self.noise_dim])
        else:
            bounds = torch.tensor([
                            [-5]*self.total_param_size + [-2]*self.noise_dim + [0]*self.act_dim,
                            [5] *self.total_param_size + [2] *self.noise_dim + [len(self.map_net.all_acts)-1]*self.act_dim])
        bounds = bounds.float().to(self.device)
        # Initialize the Bayesian optimizer

        #train_X = gen_batch_initial_conditions(self.objective_function, bounds, q=4, num_restarts=4, raw_samples=4) 
        #train_X = gen_batch_initial_conditions(self.simple_af, bounds, q=4, num_restarts=4, raw_samples=4) 
        #train_X = train_X.view(-1, bounds.shape[-1])

        train_X = self.gen_initial_conditions(bounds, n_initial_points)
        train_Y = self.objective_function(train_X)
        train_Y = train_Y.view(-1, 1)
        print (train_X.shape, train_Y.shape)

        model = SingleTaskGP(normalize(train_X, bounds=bounds),
                             standardize(train_Y))
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        mll.to(train_X)
        fit_gpytorch_model(mll)

        # Optimize the objective function
        for call_iter in range(n_calls):
            candidate, acq_value = optimize_acqf(
                qExpectedImprovement(model, best_f=train_Y.min()),
                bounds=bounds,
                q=self.batch_size,
                num_restarts=1,
                raw_samples=4, 
            )
            
            loss = self.objective_function(candidate)
            candidate_Y = loss.view(-1, 1)

            if len(train_X) > 2048:
                indices = torch.topk(train_Y.view(-1), self.batch_size, largest=True).indices
                train_X = torch.stack([row for i, row in enumerate(train_X) if i not in indices])
                train_Y = torch.stack([row for i, row in enumerate(train_Y) if i not in indices])
            # Update the model
            train_X = torch.cat([train_X, candidate], dim=0)
            train_Y = torch.cat([train_Y, candidate_Y], dim=0)

            print (f'set shape: train_X {train_X.shape}, train_Y {train_Y.shape}')
            model = SingleTaskGP(normalize(train_X, bounds=bounds),
                                 standardize(train_Y))
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_model(mll)

            print(f'[Iter] {call_iter} [Loss min/max] {min(loss), max(loss)}')
            if call_iter % 25 == 0:
                for i, candidate_i in enumerate(candidate):
                    if self.conv_decoder:
                        output_w = candidate_i[:-self.noise_dim]
                        output_z = candidate_i[-self.noise_dim:]
                        output_a = None
                    else:
                        output_w = candidate_i[:-(self.noise_dim+self.act_dim)]
                        output_z = candidate_i[-(self.noise_dim+self.act_dim):-self.act_dim]
                        output_a = candidate_i[-self.act_dim:].long()
                    splits_w = torch.split(output_w, self.param_sizes, dim=-1)
                    w = [param_split.reshape(param_shape)
                        for param_split, param_shape in zip(splits_w, self.param_shapes)]
                    output = self.forward_map(w, output_z, output_a)
                    output = output.reshape(self.x_dim, self.y_dim, 3).detach().cpu().numpy() * 255
                    write_image(f'{self.output_dir}/output_{call_iter}_{i}', output, suffix='jpg')
                    gc.collect()
        self.optimizer = model


def load_args():
    parser = argparse.ArgumentParser(description='cppn-pytorch')
    parser.add_argument('--noise_dim', default=8, type=int, help='latent space width')
    parser.add_argument('--n_samples', default=1, type=int, help='images to generate')
    parser.add_argument('--x_dim', default=256, type=int, help='out image width')
    parser.add_argument('--y_dim', default=256, type=int, help='out image height')
    parser.add_argument('--noise_scale', default=10, type=float, help='mutiplier on z')
    parser.add_argument('--c_dim', default=3, type=int, help='channels')
    parser.add_argument('--layer_width', default=32, type=int, help='net width')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--graph_nodes', default=10, type=int, help='number of graph_nodes in graph')
    parser.add_argument('--output_dir', default='trial', type=str, help='output fn')
    parser.add_argument('--no_tiff', action='store_true', help='save tiff metadata')
    parser.add_argument('--sweep_settings', action='store_true', help='sweep hps')
    parser.add_argument('--activations', default='permute', type=str, help='')
    parser.add_argument('--target_img_path', default='trial', type=str, help='image to match')
    parser.add_argument('--graph_topology', default='fixed', type=str, help='')
    parser.add_argument('--device', default='cuda', type=str, help='')
    parser.add_argument('--use_conv_decoder', action='store_true', help='')
    args = parser.parse_args()
    return args


args = load_args()
# Load the target image and preprocess it
target_image = Image.open(args.target_img_path).convert("RGB")
target_image = np.array(target_image).astype(np.float32) / 255.0

if args.use_conv_decoder:
    map_fn = ConvDecoder(args.noise_dim)
    x_dim = target_image.shape[0]
    y_dim = target_image.shape[1]
    coord_fn = None
    os.makedirs(args.output_dir, exist_ok=True)
else:
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
    x_dim = cppn.x_dim
    y_dim = cppn.y_dim
    coord_fn = cppn._coordinates_torch
    map_fn = cppn.map_fn

bo = BOPPN(args.noise_dim, 
           target_image, 
           map_fn.to(args.device), 
           coord_fn=coord_fn,
           x_dim=x_dim,
           y_dim=y_dim,
           output_dir=args.output_dir,
           device=args.device)

# Run the Bayesian optimization
bo.optimize_map(n_calls=20000, n_initial_points=4) 
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