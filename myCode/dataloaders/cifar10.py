import torch
import torchvision
import numpy as np

def cifar10_for_classes_TaskIL(class_1, class_2, batch_size_train=128):
    dataset = torchvision.datasets.CIFAR10('files/', train=True, download=True,
                                    transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    # torchvision.transforms.Normalize(
                                    # (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    torchvision.transforms.Normalize(
                                    (0.4924, 0.4739, 0.4198),
                                    (0.1891, 0.1865, 0.1885))
                                ]),)

    dataset.data = np.concatenate(
        (dataset.data[np.array(dataset.targets) == class_1], dataset.data[np.array(dataset.targets) == class_2])
    )
    dataset.targets = torch.tensor(np.concatenate(
        (np.array(dataset.targets)[np.array(dataset.targets) == class_1], np.array(dataset.targets)[np.array(dataset.targets) == class_2])
    )).to(torch.int64)

    dataset.targets[np.array(dataset.targets) == class_1] = 0
    dataset.targets[np.array(dataset.targets) == class_2] = 1
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size_train, shuffle=True)
    return train_loader