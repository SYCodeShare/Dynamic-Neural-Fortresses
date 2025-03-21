#!/usr/bin/python
"""This is a short description.
Replace this with a more detailed description of what this file contains.
"""
import argparse
import os.path as osp
import os

import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torchvision.datasets.folder import ImageFolder, default_loader

import defenses.config as cfg
from torchvision.transforms import transforms



class CUBS200(ImageFolder):
    def __init__(self, train=True, transform=None, target_transform=None,**kwargs):
        root = osp.join(cfg.DATASET_ROOT, 'CUB_200_2011')
        if not osp.exists(root):
            raise ValueError('Dataset not found at {}. Please download it from {}.'.format(
                root, 'https://data.caltech.edu/records/65de6-vp158'
            ))



        transform = transforms.Compose(
        [transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )
        print('CUBS200 Transform', transform)
        
        # Initialize ImageFolder
        super().__init__(root=osp.join(root, 'images'), transform=transform,
                         target_transform=target_transform)
        self.root = root

        self.partition_to_idxs = self.get_partition_to_idxs()
        self.pruned_idxs = self.partition_to_idxs['train' if train else 'test']

        # Prune (self.imgs, self.samples to only include examples from the required train/test partition
        self.samples = [self.samples[i] for i in self.pruned_idxs]
        self.imgs = self.samples

        print('=> done loading {} ({}) with {} examples'.format(self.__class__.__name__, 'train' if train else 'test',
                                                                len(self.samples)))

    def get_partition_to_idxs(self):
        partition_to_idxs = {
            'train': [],
            'test': []
        }

        # ----------------- Create mapping: filename -> 'train' / 'test'
        # There are two files: a) images.txt containing: <imageid> <filepath>
        #            b) train_test_split.txt containing: <imageid> <0/1>

        imageid_to_filename = dict()
        with open(osp.join(self.root, 'images.txt')) as f:
            for line in f:
                imageid, filepath = line.strip().split()
                _, filename = osp.split(filepath)
                imageid_to_filename[imageid] = filename
        filename_to_imageid = {v: k for k, v in imageid_to_filename.items()}

        imageid_to_partition = dict()
        with open(osp.join(self.root, 'train_test_split.txt')) as f:
            for line in f:
                imageid, split = line.strip().split()
                imageid_to_partition[imageid] = 'train' if int(split) else 'test'

        # Loop through each sample and group based on partition
        for idx, (filepath, _) in enumerate(self.samples):
            _, filename = osp.split(filepath)
            imageid = filename_to_imageid[filename]
            partition_to_idxs[imageid_to_partition[imageid]].append(idx)

        return partition_to_idxs
