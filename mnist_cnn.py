import os
import os.path

from collections import OrderedDict

import torch
from torch import nn, optim

import numpy as np

from torch.utils.data import Dataset, DataLoader, sampler
from torchvision import datasets

import pickle

from utils import *


class MNIST(Dataset):
    
    img_shape = (1, 28, 28)
    classes = list(map(str, range(10)))
    
    def __init__(self, folder, batch_size=32, num_workers=0, val_size=.2, seed=123):
        self.folder = folder
        
        train = datasets.MNIST(folder, train=True, download=True)
        test = datasets.MNIST(folder, train=False, download=True)
        
        data = torch.cat([
            train.train_data.unsqueeze(1).float() / 255.,
            test.test_data.unsqueeze(1).float() / 255.,
        ])
        
        labels = torch.cat([
            train.train_labels, test.test_labels
        ]).numpy()
        
        train_val_size = len(train)
        test_size = len(data) - train_val_size
        
        self.seed = seed
        rs = np.random.RandomState(seed)
        train_val = np.arange(train_val_size)
        rs.shuffle(train_val)
        
        test = np.arange(train_val_size, len(data))
        
        dataset_size = len(data)
        idx = np.arange(dataset_size)

        train_val_size = len(train_val)

        train_size = int(round(train_val_size * (1 - val_size)))
        val_size = train_val_size - train_size
        train, val = train_val[:train_size], train_val[train_size:]

        train_sampler = sampler.SubsetRandomSampler(train)
        val_sampler = sampler.SubsetRandomSampler(val)
        train_val_sampler = sampler.SubsetRandomSampler(train_val)
        test_sampler = sampler.SubsetRandomSampler(test)

        train_loader = DataLoader(self, batch_size=batch_size, num_workers=num_workers, sampler=train_sampler)
        val_loader = DataLoader(self, batch_size=batch_size, num_workers=num_workers, sampler=val_sampler)
        train_val_loader = DataLoader(self, batch_size=batch_size, num_workers=num_workers, sampler=train_val_sampler)
        test_loader = DataLoader(self, batch_size=batch_size, num_workers=num_workers, sampler=test_sampler)

        num_classes = len(self.classes)
        
        for k, v in locals().items():
            setattr(self, k, v)
        
    def __len__(self):
        return self.dataset_size
    
    def __getitem__(self, idx):
        X = self.data[idx]
        y = self.labels[idx]
        cls = self.classes[y]
        
        return X, y, cls


def mnist_model():
    feature_model = nn.Sequential( # 1, 28, 28
        OrderedDict([
            ('conv1', nn.Conv2d(1, 16, kernel_size=5, stride=3, padding=2, bias=False)), # 16, 10, 10
            ('conv1_bn', nn.BatchNorm2d(16)),
            ('conv1_relu', nn.ReLU()),
            ('conv2', nn.Conv2d(16, 32, kernel_size=3, stride=1, bias=False)), # 32, 8, 8
            ('conv2_bn', nn.BatchNorm2d(32)),
            ('conv2_relu', nn.ReLU()),
            ('conv3', nn.Conv2d(32, 64, kernel_size=3, stride=1, bias=False)), # 64, 6, 6
            ('conv3_bn', nn.BatchNorm2d(64)),
            ('conv3_relu', nn.ReLU())
        ])
    )

    classifier_model = nn.Sequential(
        OrderedDict([
            ('dense1', nn.Linear(64 * 6 * 6, 128, bias=False)),
            ('dense1_bn', nn.BatchNorm1d(128)),
            ('dense1_relu', nn.ReLU()),
            ('dense1_dropout', nn.Dropout()),
            ('output', nn.Linear(128, 10)),
        ])
    )

    model = nn.Sequential(
        OrderedDict([
            ('features', feature_model),
            ('flatten', Flatten()),
            ('classifier', classifier_model)
        ])
    )
    
    return model


class ConvVAE(nn.Module):
    
    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        
        self.encoder = nn.Sequential(OrderedDict([
            ('features', nn.Sequential(
                OrderedDict([
                    ('conv1', nn.Conv2d(1, 16, kernel_size=5, stride=3, padding=2, bias=False)), # 16, 10, 10
                    ('conv1_bn', nn.BatchNorm2d(16)),
                    ('conv1_relu', nn.ReLU()),
                    ('conv2', nn.Conv2d(16, 32, kernel_size=3, stride=1, bias=False)), # 32, 8, 8
                    ('conv2_bn', nn.BatchNorm2d(32)),
                    ('conv2_relu', nn.ReLU()),
                    ('conv3', nn.Conv2d(32, 64, kernel_size=3, stride=1, bias=False)), # 64, 6, 6
                    ('conv3_bn', nn.BatchNorm2d(64)),
                    ('conv3_relu', nn.ReLU()),
                ])
            )),
            ('dense', nn.Sequential(
                OrderedDict([
                    ('flatten', Flatten()),
                    ('dense1', nn.Linear(64 * 6 * 6, 128, bias=False)),
                    ('dense1_bn', nn.BatchNorm1d(128)),
                    ('dense1_relu', nn.ReLU()),
                    ('dense1_dropout', nn.Dropout()),
                ])
            ))
        ]))
        
        self.encoder_mu = nn.Linear(128, emb_dim)
        self.encoder_logvar = nn.Linear(128, emb_dim)
        
        self.decoder = nn.Sequential(OrderedDict([
            ('dense1', nn.Linear(emb_dim, 64 * 6 * 6)),
            ('dense1_bn', nn.BatchNorm1d(64 * 6 * 6)),
            ('dense1_relu', nn.ReLU()),
            ('dense1_dropout', nn.Dropout()),
            ('reshape', Reshape(-1, 64, 6, 6)),
            ('deconv3', nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, bias=False)),
            ('deconv3_bn', nn.BatchNorm2d(32)),
            ('deconv3_relu', nn.ReLU()),
            ('deconv2', nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, bias=False)),
            ('deconv2_bn', nn.BatchNorm2d(16)),
            ('deconv2_relu', nn.ReLU()),
            ('deconv1', nn.ConvTranspose2d(16, 1, kernel_size=5, stride=3, padding=2, bias=False)),
            ('deconv1_bn', nn.BatchNorm2d(1)),
            ('deconv1_relu', nn.Sigmoid()),
        ]))
        
    def encode(self, input):
        encoder_features = self.encoder(input)
        mu, logvar = self.encoder_mu(encoder_features), self.encoder_logvar(encoder_features)
        
        return mu, logvar
    
    def decode(self, code):
        return self.decoder(code)
        
    def forward(self, input):
        mu, logvar = self.encode(input)
        
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            code = eps.mul(std).add_(mu)
        else:
            code = mu
            
        decoded = self.decode(code)
        
        return decoded, mu, logvar, code
    
    def loss(self, recon_X, X, mu, logvar):
        BCE = F.binary_cross_entropy(recon_X, X, size_average=False)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD
        
    def find_closest_embedding(self, device, x, mu, logvar, n_samples=100, norm=2):
        std = torch.exp(0.5*logvar).to(device)
        eps = torch.randn(*std.shape, n_samples).to(device)
        samples_emb = eps.mul(std.view(*std.shape, 1)).add_(mu.view(*mu.shape, 1))

        closest_emb = torch.stack([
            embs[:, 
                (xi.unsqueeze(0) - self.decode(embs.t()))\
                .view(n_samples, -1).norm(p=norm, dim=1).argmin(dim=0)
            ]
            for xi, embs in zip(x, samples_emb)
        ], dim=0)

        return closest_emb
