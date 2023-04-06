import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from run_cppn import CPPN
from utils import write_image


class HyperNetwork(nn.Module):
    def __init__(self, target_net, input_size, hidden_size=256):
        super(HyperNetwork, self).__init__()
        self.input_size = input_size
        self.target_net = target_net
        self.param_shapes = [p.shape for p in target_net.parameters()]
        self.param_sizes = [p.numel() for p in target_net.parameters()]
        self.total_param_size = sum(self.param_sizes)

        self.hyper_net = nn.Sequential(
            nn.Linear(self.input_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, self.total_param_size)
        )

    def forward(self, z):
        param_pred = self.hyper_net(z)
        # param_pred = torch.tanh(param_pred)
        param_splits = torch.split(param_pred, self.param_sizes, dim=-1)

        params = [param_split.reshape(param_shape)
                  for param_split, param_shape in zip(param_splits, self.param_shapes)]

        return params

    def predict(self, z, x):
        with torch.no_grad():
            params = self(z)

            for param, target_param in zip(params, self.target_net.parameters()):
                target_param.data.copy_(param)

            return self.target_net(x)


def train_hypernetwork(args, hypernet, cppn, target_image, iterations, device):
    hypernet.to(device)
    cppn.map_fn.to(device)
    target_image = torch.from_numpy(target_image).float().to(device)
    optimizer = optim.Adam(hypernet.parameters(), lr=1e-2)# , weight_decay=1e-5)
    criterion = nn.MSELoss()
    target_image_norm = target_image.permute(1, 2, 0) / 255.
    
    z_cppn = cppn.init_inputs().to(device)
    x_vec, y_vec, r_vec, n_pts, _ = cppn._coordinates(
           args.x_dim, args.y_dim, args.batch_size, zoom=(.5,.5), pan=(2,2))
    one_vec = torch.ones(n_pts, 1, dtype=torch.float).to(device)
    z_cppn = z_cppn.view(cppn.batch_size, 1, cppn.noise_dim) * one_vec * cppn.noise_scale
    z_cppn = z_cppn.view(cppn.batch_size*n_pts, cppn.noise_dim)
    sum_p = 0
    for p in cppn.map_fn.parameters():
        sum_p += p.norm()
    for iteration in range(iterations):
        z_hypernet = torch.ones(args.hypernet_batch_size, args.hypernet_z_dim, device=device)
        z_hypernet = z_hypernet.normal_(0, 1)
        z_cppn = z_cppn.clone().detach().requires_grad_(True)
        x_vec = x_vec.clone().detach().requires_grad_(True)
        y_vec = y_vec.clone().detach().requires_grad_(True)
        r_vec = r_vec.clone().detach().requires_grad_(True)

        optimizer.zero_grad()
        target_net_params = hypernet(z_hypernet)

        out_norms = 0
        for param, target_param in zip(target_net_params, cppn.map_fn.parameters()):
            target_param.data.copy_(param)
            out_norms += param.norm()

        outputs = cppn.map_fn(x_vec.to(device),
                              y_vec.to(device), 
                              r_vec.to(device), 
                              z_cppn)
        
        outputs = outputs.reshape(*target_image_norm.shape)
        loss = -out_norms + criterion(outputs, target_image_norm) 
        loss.backward()
        optimizer.step()

        if iteration % 100 == 0:
            print(f'Iteration {iteration+1}, Loss: {loss}, Norm: {out_norms}, 0norm: {sum_p}')
            img_npy = outputs.reshape(*target_image.shape).detach().cpu().numpy() * 255.
            write_image(f'{args.output_dir}/iteration_{iteration}', img_npy, 'jpg')

    return hypernet


def load_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--x_dim', type=int, default=256)
    parser.add_argument('--y_dim', type=int, default=256)
    parser.add_argument('--c_dim', type=int, default=3)
    parser.add_argument('--noise_dim', type=int, default=8)
    parser.add_argument('--noise_scale', type=float, default=10.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--graph', type=str, default=None)
    parser.add_argument('--weight_init_mean', type=float, default=0.0)
    parser.add_argument('--weight_init_std', type=float, default=0.1)
    parser.add_argument('--layer_width', type=int, default=32)
    parser.add_argument('--activations', type=str, default='fixed')
    parser.add_argument('--graph_topology', type=str, default='fixed')
    parser.add_argument('--output_dir', default='trial', type=str, help='output fn')
    parser.add_argument('--target_image', type=str, default='temp')
    parser.add_argument('--hypernet_batch_size', type=int, default=1)
    parser.add_argument('--hypernet_z_dim', type=int, default=128)
    parser.add_argument('--iterations', type=int, default=10000)

    return parser.parse_args()


if __name__ == '__main__':
    args = load_args()
    cppn = CPPN(x_dim=args.x_dim, 
                y_dim=args.y_dim, 
                c_dim=args.c_dim,
                noise_dim=args.noise_dim, 
                noise_scale=args.noise_scale, 
                layer_width=args.layer_width,
                output_dir=args.output_dir)
    cppn.init_map_fn(activations=args.activations, 
                     graph_topology=args.graph_topology)
    hypernetwork = HyperNetwork(cppn.map_fn, input_size=args.hypernet_z_dim, hidden_size=256)
    target_image = cv2.imread(args.target_image)
    target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print ('hypernetowrk', hypernetwork)
    print ('cppn', cppn.map_fn)
    _ = train_hypernetwork(args, hypernetwork, cppn, target_image, args.iterations, device)

    
