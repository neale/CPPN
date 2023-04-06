import torch
from torch import nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce
import itertools
from maps import Gaussian, CosLayer, SinLayer

pair = lambda x: x if isinstance(x, tuple) else (x, x)


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        out = self.fn(x)
        #out = self.norm(out)
        #out = out + x
        return out

class LinBlock(nn.Module):
    def __init__(self, dim, expansion_factor=4, dropout=0.0, fc=nn.Linear, act=nn.GELU):
        super().__init__()
        self.inner_dim = int(dim * expansion_factor)
        self.linear1 = fc(dim, self.inner_dim)
        self.linear2 = fc(self.inner_dim, dim)
        self.gelu = act()
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(self.gelu(self.linear1(x)))
        x = self.dropout2(self.linear2(x))
        return x


class MLPMixer(nn.Module):
    def __init__(self,
                 image_size,
                 channels,
                 patch_size, 
                 dim, 
                 depth,
                 num_classes,
                 expansion_factor=2,
                 expansion_factor_token=2,#0.5,
                 randomize_act=False,
                 dropout=0.0):
        super().__init__()
        self.name = 'mlp_mixer'
        image_h, image_w = pair(image_size)
        assert (image_h % patch_size) == 0 and (image_w % patch_size) == 0, ''\
            'image must be divisible by patch size'
        num_patches = (image_h // patch_size) * (image_w // patch_size)
        chan_first = partial(nn.Conv1d, kernel_size = 1)
        chan_last = nn.Linear

        self.permute = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
            p1=patch_size, p2=patch_size)
        self.reduce_mean = Reduce('b n c -> b c', 'mean')
        self.linear1 = nn.Linear((patch_size ** 2) * channels, dim)
        self.linearx = nn.Linear((patch_size ** 2) * 1, dim)
        self.lineary = nn.Linear((patch_size ** 2) * 1, dim)
        self.linearr = nn.Linear((patch_size ** 2) * 1, dim)
        self.linear_cls = nn.Linear(dim, dim)#num_classes)
        self.linear_out = nn.Linear(dim, num_classes)
        self.ln_out = nn.LayerNorm(dim)
    
        if randomize_act:
            acts = [nn.ELU, nn.Hardtanh, nn.LeakyReLU, nn.LogSigmoid,
                    nn.SELU, nn.GELU, nn.CELU, nn.Softshrink, nn.Sigmoid,
                    SinLayer, CosLayer, Gaussian, nn.Softplus, nn.Mish,
                    nn.Tanh, nn.ReLU]
            self.order = torch.randint(0, len(acts), size=(depth*2,))
            a = [acts[i] for i in self.order]
            self.forward_fn = self.forward_base
        else:
            a = [nn.GELU for _ in range(depth*2)]
            self.forward_fn = self.forward_base
        print ('using acts: ', a)
        
        self.mixer_sequence = nn.ModuleList(
            itertools.chain.from_iterable(list(
                [PreNormResidual(num_patches, LinBlock(
                    num_patches,
                    expansion_factor,
                    dropout,
                    fc=chan_first,
                    act=a[i])),
                 PreNormResidual(dim, LinBlock(
                    dim,
                    expansion_factor_token,
                    dropout,
                    fc=chan_last,
                    act=nn.GELU))] for i in range(depth))))
                    #act=a[depth*2-(i+1)]))] for i in range(depth))))
        
        self.xblock1 = PreNormResidual(
            32, 
            LinBlock(
                1, # num_patches,
                expansion_factor=expansion_factor,
                dropout=0.0,
                fc=chan_first,
                act=nn.Identity))
        self.xblock2 = PreNormResidual(
            dim,
            LinBlock(
                dim,
                expansion_factor_token,
                dropout=0.0,
                fc=chan_last,
                act=nn.Identity))
        self.yblock1 = PreNormResidual(
            32, 
            LinBlock(
                1,#num_patches,
                expansion_factor=expansion_factor,
                dropout=0.0,
                fc=chan_first,
                act=nn.Identity))
        self.yblock2 = PreNormResidual(
            dim,
            LinBlock(
                dim,
                expansion_factor_token,
                dropout=0.0,
                fc=chan_last,
                act=nn.Identity))
        self.rblock1 = PreNormResidual(
            32, 
            LinBlock(
                1,#num_patches,
                expansion_factor=expansion_factor,
                dropout=0.0,
                fc=chan_first,
                act=nn.Identity))
        self.rblock2 = PreNormResidual(
            dim,
            LinBlock(
                dim,
                expansion_factor_token,
                dropout=0.0,
                fc=chan_last,
                act=nn.Identity))

    def forward(self, noise, x=None, y=None, r=None):
        return self.forward_fn(noise, x, y, r)

    def forward_base(self, noise, x, y, r):
        out = self.permute(noise)
        out = self.linear1(out)
        if x is not None and y is not None and r is not None:
            x = self.linearx(self.permute(x.reshape(1, 1,256,256)))
            y = self.lineary(self.permute(y.reshape(1, 1,256,256)))
            r = self.linearr(self.permute(r.reshape(1, 1,256,256)))
            #print (out.shape, x.shape, y.shape, r.shape)
            out = out.transpose(0, 1)
            x = x.transpose(0, 1)
            y = y.transpose(0, 1)
            r = r.transpose(0, 1)
            #print (out.shape, x.shape, y.shape, r.shape)

            x = self.xblock1(x)
            #print (x.shape)
            x = self.xblock2(x)
            #print (x.shape)
            y = self.yblock2(self.yblock1(y))
            r = self.rblock2(self.rblock1(r))
            #print (out.shape, x.shape, y.shape, r.shape)
            out = torch.tanh(out + x + y + r)
        for i, stack in enumerate(self.mixer_sequence):
            out = stack(out)
        out = self.ln_out(out)
        out = self.reduce_mean(out)
        #out = torch.sigmoid(self.linear_cls(out))
        out = torch.sigmoid(self.linear_out(out))#) * .5 + .5
        return out

