import os
import os.path

from collections import OrderedDict

import torch
from torch import nn, optim

import numpy as np

from torch.utils.data import Dataset, DataLoader, sampler
from skimage import io

from utils import *

class NotMNIST(Dataset):
    
    img_shape = (1, 28, 28)
    
    def __init__(self, folder, batch_size=32, num_workers=0, test_size=.2, val_size=.2, seed=123):
        self.folder = folder
        
        self.classes = sorted(os.listdir(os.path.join(folder)))
        self.collections = [
            io.imread_collection(os.path.join(folder, cls, '*.png'))
            for cls in self.classes
        ]
        
        dataset_size = len(self)
        rs = np.random.RandomState(seed)
        idx = rs.choice(dataset_size, size=dataset_size, replace=False)

        train_val_size = int(round(len(idx) * (1 - test_size)))
        test_size = dataset_size - train_val_size
        train_val, test = idx[:train_val_size], idx[train_val_size:]

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
        return sum(map(len, self.collections))
    
    def __getitem__(self, idx):
        if idx < 0:
            raise KeyError(idx)
            
        for y, col in enumerate(self.collections):
            if idx < len(col):
                img = col[idx]
                img = torch.from_numpy(img.astype(float) / 255.).float().unsqueeze(0)
                return img, y, self.classes[y]
            else:
                idx -= len(col)
        else:
            raise KeyError(idx)


def notmnist_model():
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