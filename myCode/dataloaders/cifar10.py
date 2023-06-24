import torch
import torchvision
import numpy as np

def cifar10_for_classes_TaskIL(class_1, class_2, batch_size_train=128):
    dataset = torchvision.datasets.CIFAR10('files/', train=True, download=True,
                                    transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ]),)

    dataset.data = np.concatenate(
        (dataset.data[np.array(dataset.targets) == class_1], dataset.data[np.array(dataset.targets) == class_2])
    )
    dataset.targets = np.concatenate(
        (np.array(dataset.targets)[np.array(dataset.targets) == class_1], np.array(dataset.targets)[np.array(dataset.targets) == class_2])
    )

    dataset.targets[np.array(dataset.targets) == class_1] = 0
    dataset.targets[np.array(dataset.targets) == class_2] = 1
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size_train, shuffle=True)
    return train_loader