import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

from scipy import stats
from scipy.spatial.distance import cosine

import matplotlib.pyplot as plt
from matplotlib import colors


class Flatten(nn.Module):

    def forward(self, input):
        return input.view(input.size(0), -1)

class Reshape(nn.Module):

    def __init__(self, *target_shape):
        super().__init__()
        self.target_shape = target_shape

    def forward(self, input):
        return input.view(*self.target_shape)

def load_model(model, model_filename, device):
    model.load_state_dict(torch.load(model_filename))

    return model.to(device)

def subplots(nrow, ncol, plot_size=4, **kwargs):
    if 'figsize' not in kwargs:
        kwargs['figsize'] = (ncol * plot_size, nrow * plot_size)

    fig, axes = plt.subplots(nrow, ncol, **kwargs)

    return axes

def imshow(im, ax=plt, cmap='gray', title=None):
    if isinstance(im, torch.Tensor):
        im = im.cpu().numpy()

    if len(im.shape) == 3:
        if im.shape[0] == 1: im = im[0]
        # Transpose so it's channel_last
        else:
            im = np.moveaxis(im, 0, -1)

    ax.imshow(im, cmap=cmap)
    ax.axis('off')
    if title: ax.title(title) if ax is plt else ax.set_title(title)


def set_ticks(dataset, ax, rotation='vertical'):
    ax.set_xticks(np.arange(dataset.num_classes))
    ax.set_xticklabels(dataset.classes, rotation=rotation);


class MidpointNormalize(colors.Normalize):
    
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
    
    def invert(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        y, x = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
    

def plot_attr(
    attr, ax=None, mpn=None, cmap='RdBu_r', 
    alpha=True, colorbar=True, nticks=5, title=None
):
    if isinstance(attr, torch.Tensor):
        attr = attr.cpu().detach().numpy()
        
    if mpn is None:
        m = max(abs(np.min(attr)), abs(np.max(attr)))
        mpn = MidpointNormalize(-m, m, 0)
        
    attr = mpn(attr)
        
    if alpha:
        # Choose colormap
        cmap = plt.cm.get_cmap(cmap)
        # Get the colormap colors
        N = cmap.N
        cmap = cmap(np.arange(N))
        # Set alpha
        alphas = (np.arange(N) - (N - 1) // 2)
        alphas = (alphas / alphas.max()) ** 2
        cmap[:, -1] = alphas
        cmap = colors.ListedColormap(cmap)
        
    if ax is None:
        im = plt.imshow(attr, cmap=cmap)
        plt.axis('off')
        if title is not None: plt.title(title)
    else:
        im = ax.imshow(attr, cmap=cmap)
        ax.axis('off')
        if title is not None: ax.set_title(title)
    
    if colorbar:
        cb = im.figure.colorbar(im, ax=ax)
            
        ticks = np.linspace(0, 1, nticks)
        cb.set_ticks(ticks)
        cb.set_ticklabels(['%.2e' % tick for tick in mpn.invert(ticks)])
    
    return im


def gaussian_kernel(size, sigma=1):
    assert size % 2
    
    center = size // 2
    
    xs = np.repeat(np.arange(size).reshape((1, size)), size, axis=0)
    ys = np.repeat(np.arange(size).reshape((size, 1)), size, axis=1)
    
    k = np.sqrt((xs - center) ** 2 + (ys - center) ** 2)
    distr = stats.norm(0, sigma)
    
    k = distr.pdf(k.flatten())
    k /= k.sum()
    
    return k.reshape((size, size))
