import torch
from torch.utils.data import Subset
import torchvision
import os
import numpy as np
from sklearn.model_selection import train_test_split


def MNIST(batch_size_train = 64, batch_size_test = 1000):
    # MNIST_train = torchvision.datasets.MNIST('/files/', train=True, download=True,
    #                             transform=torchvision.transforms.Compose([
    #                                 torchvision.transforms.ToTensor(),
    #                                 torchvision.transforms.Normalize(
    #                                 (0.1307,), (0.3081,))
    #                             ]))
    # MNIST_test = torchvision.datasets.MNIST('/files/', train=False, download=True,
    #                             transform=torchvision.transforms.Compose([
    #                                 torchvision.transforms.ToTensor(),
    #                                 torchvision.transforms.Normalize(
    #                                 (0.1307,), (0.3081,))
    #                             ]))
    
    # val_split = 1/6
    # train_idx, val_idx = train_test_split(list(range(len(MNIST_train))), test_size=val_split)
    # train_dataset = Subset(MNIST_train, train_idx).dataset
    # val_dataset = Subset(MNIST_train, val_idx).dataset

    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size_train, shuffle=True)
    # test_loader = torch.utils.data.DataLoader(MNIST_test, batch_size=batch_size_test, shuffle=True)
    MNIST_train = torchvision.datasets.MNIST('/files/', train=True, download=True,
                                transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ]))

    MNIST_val = torchvision.datasets.MNIST('/files/', train=True, download=True,
                                transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ]))

    MNIST_test = torchvision.datasets.MNIST('/files/', train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ]))

    random_label_idx = np.arange(60000)#torch.tensor(np.random.randint(10, size=60000)).type(torch.LongTensor)
    np.random.shuffle(random_label_idx)
    train_samples = 50000
    random_label_idx_train = random_label_idx[:train_samples]
    random_label_idx_val = random_label_idx[train_samples:]

    MNIST_train.data = MNIST_train.data[random_label_idx_train]
    MNIST_train.targets = MNIST_train.targets[random_label_idx_train]


    MNIST_val.data = MNIST_val.data[random_label_idx_val]
    MNIST_val.targets = MNIST_val.targets[random_label_idx_val]


    train_loader = torch.utils.data.DataLoader(MNIST_train, batch_size=batch_size_train, shuffle=True)
    val_loader = torch.utils.data.DataLoader(MNIST_val, batch_size=batch_size_train, shuffle=True)
    test_loader = torch.utils.data.DataLoader(MNIST_test, batch_size=batch_size_test, shuffle=True)
    return train_loader, val_loader, test_loader

def MNIST_train_random_lebels(batch_size_train = 64):
    label_mapping = torch.tensor(np.random.randint(10, size=60000)).type(torch.LongTensor)
    MNIST_train = torchvision.datasets.MNIST('/files/', train=True, download=True,
                                transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ]),
                                target_transform=lambda y: label_mapping[y])

    train_loader = torch.utils.data.DataLoader(MNIST_train, batch_size=batch_size_train, shuffle=True)
    return train_loader

def MNIST_for_classes(class_1, class_2, batch_size_train=128):
    MNIST_train = torchvision.datasets.MNIST('/files/', train=True, download=True,
                                    transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ]),)
    MNIST_train.data = MNIST_train.data[torch.logical_or(MNIST_train.train_labels == class_1, MNIST_train.train_labels == class_2)]
    MNIST_train.targets = MNIST_train.targets[torch.logical_or(MNIST_train.train_labels == class_1, MNIST_train.train_labels == class_2)]
    train_loader = torch.utils.data.DataLoader(MNIST_train, batch_size=batch_size_train, shuffle=True)
    return train_loader


# ------------------------ MINST -------------------------------------------------------
#
# --------------------------------------------------------------------------------------

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]

def dead_leaves_squares_prep():
    imagefolder = "data/dead_leaves-squares/"
    resize_image = True

    transform_array = []
    if resize_image:
        transform_array.append(
            torchvision.transforms.Resize((96,96))
        )

    transform_array += [
        torchvision.transforms.RandomResizedCrop(64, scale=(0.08, 1)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        torchvision.transforms.RandomGrayscale(p=0.2),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.44087801806139126, 0.42790631331699347, 0.3867879370752931),
            (0.26826768628079806, 0.2610450402318512, 0.26866836876860795),
        ),
    ]

    transform = torchvision.transforms.Compose(transform_array)
    # transform = TwoCropsTransform(transform)


    train_path = os.path.join(imagefolder, 'train')
    print(f'Loading data from {imagefolder} as imagefolder')
    dataset = torchvision.datasets.ImageFolder(
        train_path,
        transform=transform)

    return dataset
def dead_leaves_squares_prep_gray_scale():
    imagefolder = "data/dead_leaves-squares/"
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

def dead_leaves_squares(batch_size = 128):
    dataset = dead_leaves_squares_prep()

    val_split = 0.3
    train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    train_dataset = Subset(dataset, train_idx).dataset
    test_dataset = Subset(dataset, test_idx).dataset

    # loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=2, shuffle=True, pin_memory=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=2, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=2, shuffle=True, pin_memory=True)
    return train_loader, test_loader

def dead_leaves_squares_pretraining(batch_size = 128):
    dataset = dead_leaves_squares_prep()
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=2, shuffle=True, pin_memory=True)
    return train_loader

def dead_leaves_squares_pretraining_gray(batch_size = 128):
    # os.chdir( "e:\\" )
    # imagefolder = "e:\\Datasets\\mgr\\dead_leaves-squares\\"
    imagefolder = "data/dead_leaves-squares/"
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
    dataset = torchvision.datasets.ImageFolder(
        train_path,
        transform=transform)
    
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=2, shuffle=True, pin_memory=True)
    return train_loader