#!/usr/bin/python
"""This is a short description.
Replace this with a more detailed description of what this file contains.
"""
from torchvision.transforms import transforms

import defenses.config as cfg




class DefaultTransforms:
    normalize = transforms.Normalize(mean=cfg.IMAGENET_MEAN,
                                     std=cfg.IMAGENET_STD)


    size = 224
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        # transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        normalize,
    ])
