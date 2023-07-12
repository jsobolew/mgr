import torch
from torch.utils.data import Dataset
import torchvision
import os
import numpy as np
from typing import Optional, Callable, Any


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]

def dataset_prep(dataset_name, no_classes, im_size = 64):
    imagefolder = f"data/{dataset_name}/"
    resize_image = True

    transform_array = []
    if resize_image:
        transform_array.append(
            torchvision.transforms.Resize((96,96))
        )

    transform_array += [
        torchvision.transforms.RandomResizedCrop(im_size, scale=(0.08, 1)),
        # torchvision.transforms.RandomHorizontalFlip(),
        # torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        # torchvision.transforms.RandomGrayscale(p=0.2),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.4988, 0.4980, 0.4980),
            (0.2074, 0.2097, 0.2056),
        ),
    ]

    #     (0.44087801806139126, 0.42790631331699347, 0.3867879370752931),
    #     (0.26826768628079806, 0.2610450402318512, 0.26866836876860795),
    transform = torchvision.transforms.Compose(transform_array)

    train_path = os.path.join(imagefolder, 'train')
    print(f'Loading data from {imagefolder} as imagefolder')
    # dataset = torchvision.datasets.ImageFolder(
    #     train_path,
    #     transform=transform)
    
    dataset = RandomLabelImageFolder(
        train_path,
        transform=transform,
        no_classes=no_classes)

    return dataset

def dataset_prep_gray_scale(dataset_name):
    imagefolder = f"data/{dataset_name}/"
    resize_image = True

    transform_array = []
    if resize_image:
        transform_array.append(
            torchvision.transforms.Resize((96,96))
        )

    transform_array += [
        torchvision.transforms.RandomResizedCrop(64, scale=(0.08, 1)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.Grayscale(),
        torchvision.transforms.ToTensor(),
    ]

    transform = torchvision.transforms.Compose(transform_array)


    train_path = os.path.join(imagefolder, 'train')
    print(f'Loading data from {imagefolder} as imagefolder')
    dataset = torchvision.datasets.ImageFolder(
        train_path,
        transform=transform)
    return dataset

def dataloader_pretraining(dataset_name, no_classes, batch_size=128):
    dataset = dataset_prep(dataset_name, no_classes, im_size=32)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=2, shuffle=True, pin_memory=True)
    return train_loader

def dataloader_pretraining_gray(dataset_name, no_classes=105, batch_size=128):
    imagefolder = f"data/{dataset_name}/"
    resize_image = True

    transform_array = []
    if resize_image:
        transform_array.append(
            torchvision.transforms.Resize((96,96))
        )

    transform_array += [
        torchvision.transforms.RandomResizedCrop(28, scale=(0.08, 1)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.Grayscale(),
        torchvision.transforms.ToTensor(),
    ]

    transform = torchvision.transforms.Compose(transform_array)

    train_path = os.path.join(imagefolder, 'train')
    print(f'Loading data from {imagefolder} as imagefolder')
    dataset = RandomLabelImageFolder(
        train_path,
        transform=transform,
        no_classes=no_classes)
    
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=2, shuffle=True, pin_memory=True)
    return train_loader

class RandomLabelImageFolder(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform, no_classes):
        super().__init__(root=root, transform=transform)
        self.no_classes = no_classes

    def __getitem__(self, index):
        sample, target = super(torchvision.datasets.ImageFolder, self).__getitem__(index)
        target = torch.randint(self.no_classes, (1, ))[0]
        return sample, target

