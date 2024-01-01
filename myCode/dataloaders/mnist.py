import torch
import torchvision
import numpy as np
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = os.getenv('DATA_PATH')

def MNIST(batch_size_train = 64, batch_size_test = 1000):
    MNIST_train = torchvision.datasets.MNIST(DATA_PATH, train=True, download=True,
                                transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ]))

    MNIST_val = torchvision.datasets.MNIST(DATA_PATH, train=True, download=True,
                                transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ]))

    MNIST_test = torchvision.datasets.MNIST(DATA_PATH, train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ]))

    random_label_idx = np.arange(60000)
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

def MNIST_train_random_labels(batch_size_train = 64):
    label_mapping = torch.tensor(np.random.randint(10, size=60000)).type(torch.LongTensor)
    MNIST_train = torchvision.datasets.MNIST(DATA_PATH, train=True, download=True,
                                transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ]),
                                target_transform=lambda y: label_mapping[y])

    train_loader = torch.utils.data.DataLoader(MNIST_train, batch_size=batch_size_train, shuffle=True)
    return train_loader

def MNIST_for_classes(class_1, class_2, batch_size_train=128):
    MNIST_train = torchvision.datasets.MNIST(DATA_PATH, train=True, download=True,
                                    transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ]),)
    MNIST_train.data = MNIST_train.data[torch.logical_or(MNIST_train.train_labels == class_1, MNIST_train.train_labels == class_2)]
    MNIST_train.targets = MNIST_train.targets[torch.logical_or(MNIST_train.train_labels == class_1, MNIST_train.train_labels == class_2)]
    train_loader = torch.utils.data.DataLoader(MNIST_train, batch_size=batch_size_train, shuffle=True)
    return train_loader

def MNIST_for_classes_TaskIL(class_1, class_2, batch_size_train=128):
    MNIST_train = torchvision.datasets.MNIST(DATA_PATH, train=True, download=True,
                                    transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ]),)
    # 5421 is a minimal length of a class a dataset can have
    MNIST_train.data = torch.cat(
        (MNIST_train.data[MNIST_train.train_labels == class_1][:5421], MNIST_train.data[MNIST_train.train_labels == class_2][:5421])
    )
    MNIST_train.targets = torch.cat(
        (MNIST_train.targets[MNIST_train.train_labels == class_1][:5421], MNIST_train.targets[MNIST_train.train_labels == class_2][:5421])
    )
    MNIST_train.targets[MNIST_train.targets == class_1] = 0
    MNIST_train.targets[MNIST_train.targets == class_2] = 1
    train_loader = torch.utils.data.DataLoader(MNIST_train, batch_size=batch_size_train, shuffle=True)
    return train_loader