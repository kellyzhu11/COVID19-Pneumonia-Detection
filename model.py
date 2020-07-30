import numpy as np
import pandas as pd
import os
from PIL import Image
import cv2
import random
import pickle

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler

import torchvision
from torchvision import models, transforms
from torch.utils.data.dataset import Dataset
import matplotlib.pyplot as plt
import time
import copy

class MyDataset(Dataset):
    def __init__(self, image, label, transforms=None):
        self.image = image
        self.label =label
        self.transforms = transforms
        
    def __getitem__(self, index):
        feature = transforms.ToPILImage()(self.image[index])
        if self.transforms is not None:
            feature = self.transforms(feature)
        
        label = self.label[index]
        return (feature, label)

    def __len__(self):
        return len(self.label) 


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
            return False
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        return True

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss

class Resnet18Dropout(nn.Module):
    def __init__(self, classes, dropout = 0.5):
        """
        Load the pretrained ResNet-18 and add dropout layber before the final fc layer
        """
        super(Classifier, self).__init__()
        resnet = models.resnet18(pretrained=True)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self.dropout = nn.Dropout2d(p=dropout,inplace=True)
        self.fc = nn.Linear(resnet.fc.in_features, classes)
        
    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.avgpool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

