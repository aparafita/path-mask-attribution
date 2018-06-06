import os
import os.path

from collections import OrderedDict

import torch
from torch import nn, optim

import numpy as np

from torch.utils.data import Dataset, DataLoader, sampler
from skimage import io

import pickle

from utils import *


class CIFAR10(Dataset):
    
    img_shape = (3, 32, 32)
    
    def __init__(self, folder, batch_size=32, num_workers=0, val_size=.2, seed=123):
        self.folder = folder
        
        with open(os.path.join(folder, 'batches.meta'), 'rb') as f:
            d = pickle.load(f, encoding='bytes')
            self.classes = list(map(lambda x: x.decode(), d[b'label_names']))
            
        self.collections = [
            io.imread_collection(os.path.join(folder, cls, '*.png'))
            for cls in self.classes
        ]
        
        data = []
        labels = []
        
        for filename in ['data_batch_%d' % i for i in range(1, 6)] + ['test_batch']:
            with open(os.path.join(folder, filename), 'rb') as f:
                d = pickle.load(f, encoding='bytes')
                imgs = d[b'data'].astype(float) / 255
                data.append(imgs.reshape([len(imgs)] + list(self.img_shape)))
                labels.append(np.array(d[b'labels']))
                
        del d, f, imgs, filename
        
        train_val_size = sum(map(len, data[:-1]))
        data = np.concatenate(data)
        labels = np.concatenate(labels)
        test_size = len(data) - train_val_size
        
        # Transpose data so its n_samples, n_channels, n_rows, n_cols
        # Also, transform it to float Tensor
        data = torch.Tensor(data).float()
        
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


def cifar10_model():
    feature_model = nn.Sequential( # 3, 32, 32
        OrderedDict([
            ('conv1', nn.Conv2d(3, 64, kernel_size=3, stride=1, bias=False)), # 64, 30, 30
            ('conv1_bn', nn.BatchNorm2d(64)),
            ('conv1_relu', nn.ReLU()),
            ('conv2', nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False)), # 64, 28, 28
            ('conv2_bn', nn.BatchNorm2d(64)),
            ('conv2_relu', nn.ReLU()),
            ('maxpool_1', nn.MaxPool2d(2)), # 64, 14, 14
            ('conv3', nn.Conv2d(64, 128, kernel_size=3, stride=1, bias=False)), # 128, 12, 12
            ('conv3_bn', nn.BatchNorm2d(128)),
            ('conv3_relu', nn.ReLU()),
            ('conv4', nn.Conv2d(128, 128, kernel_size=3, stride=1, bias=False)), # 128, 10, 10
            ('conv4_bn', nn.BatchNorm2d(128)),
            ('conv4_relu', nn.ReLU()),
            ('maxpool_2', nn.MaxPool2d(2)), # 128, 5, 5
            ('conv5', nn.Conv2d(128, 256, kernel_size=3, stride=1, bias=False)), # 256, 3, 3
            ('conv5_bn', nn.BatchNorm2d(256)),
            ('conv5_relu', nn.ReLU()),
        ])
    )

    classifier_model = nn.Sequential(
        OrderedDict([
            ('dense1', nn.Linear(256 * 3 * 3, 256, bias=False)),
            ('dense1_bn', nn.BatchNorm1d(256)),
            ('dense1_relu', nn.ReLU()),
            ('dense1_dropout', nn.Dropout()),
            ('dense2', nn.Linear(256, 128, bias=False)),
            ('dense2_bn', nn.BatchNorm1d(128)),
            ('dense2_relu', nn.ReLU()),
            ('dense2_dropout', nn.Dropout()),
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
            ('features', nn.Sequential( # 3, 32, 32
                OrderedDict([
                    ('conv1', nn.Conv2d(3, 64, kernel_size=3, stride=1, bias=False)), # 64, 30, 30
                    ('conv1_bn', nn.BatchNorm2d(64)),
                    ('conv1_relu', nn.ReLU()),
                    ('conv2', nn.Conv2d(64, 128, kernel_size=3, stride=1, bias=False)), # 128, 28, 28
                    ('conv2_bn', nn.BatchNorm2d(128)),
                    ('conv2_relu', nn.ReLU()),
                    ('maxpooling_1', nn.MaxPool2d(2)), # 128, 14, 14
                    ('conv3', nn.Conv2d(128, 256, kernel_size=3, stride=1, bias=False)), # 256, 12, 12
                    ('conv3_bn', nn.BatchNorm2d(256)),
                    ('conv3_relu', nn.ReLU()),
                    ('conv4', nn.Conv2d(256, 128, kernel_size=3, stride=1, bias=False)), # 128, 10, 10
                    ('conv4_bn', nn.BatchNorm2d(128)),
                    ('conv4_relu', nn.ReLU()),
                    ('conv5', nn.Conv2d(128, 128, kernel_size=5, stride=3, padding=2, bias=False)), # 128, 4, 4
                    ('conv5_bn', nn.BatchNorm2d(128)),
                    ('conv5_relu', nn.ReLU()),
                    ('conv6', nn.Conv2d(128, 256, kernel_size=3, stride=1, bias=False)), # 256, 2, 2
                    ('conv6_bn', nn.BatchNorm2d(256)),
                    ('conv6_relu', nn.ReLU()),
                ])
            )),
            ('dense', nn.Sequential(
                OrderedDict([
                    ('flatten', Flatten()),
                    ('dense1', nn.Linear(256 * 2 * 2, 128, bias=False)),
                    ('dense1_bn', nn.BatchNorm1d(128)),
                    ('dense1_relu', nn.ReLU()),
                    ('dense1_dropout', nn.Dropout()),
                ])
            ))
        ]))
        
        self.encoder_mu = nn.Linear(128, emb_dim)
        self.encoder_logvar = nn.Linear(128, emb_dim)
        
        self.decoder = nn.Sequential(OrderedDict([
            ('dense1', nn.Linear(emb_dim, 256 * 2 * 2)),
            ('dense1_bn', nn.BatchNorm1d(256 * 2 * 2)),
            ('dense1_relu', nn.ReLU()),
            ('dense1_dropout', nn.Dropout()),
            ('reshape', Reshape(-1, 256, 2, 2)),
            ('deconv6', nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, bias=False)),
            ('deconv6_bn', nn.BatchNorm2d(128)),
            ('deconv6_relu', nn.ReLU()),
            ('deconv5', nn.ConvTranspose2d(128, 128, kernel_size=5, stride=3, padding=2, bias=False)),
            ('deconv5_bn', nn.BatchNorm2d(128)),
            ('deconv5_relu', nn.ReLU()),
            ('deconv4', nn.ConvTranspose2d(128, 256, kernel_size=3, stride=1, bias=False)),
            ('deconv4_bn', nn.BatchNorm2d(256)),
            ('deconv4_relu', nn.ReLU()),
            ('deconv3', nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, bias=False)),
            ('deconv3_bn', nn.BatchNorm2d(128)),
            ('deconv3_relu', nn.ReLU()),
            ('upsample_1', nn.Upsample(scale_factor=2)),
            ('deconv2', nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, bias=False)),
            ('deconv2_bn', nn.BatchNorm2d(64)),
            ('deconv2_relu', nn.ReLU()),
            ('deconv1', nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, bias=False)),
            ('deconv1_bn', nn.BatchNorm2d(3)),
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