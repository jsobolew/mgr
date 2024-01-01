from dataclasses import dataclass
from dotenv import load_dotenv
import numpy as np
import torch
import torchvision

# from dataloaders.cifar10 import cifar10_for_classes_TaskIL


dataset_to_num_classes = {
    "CIFAR10": 10,
    "CIFAR100": 100,
    "MNIST": 10,
}

load_dotenv()

DATA_PATH = os.getenv('DATA_PATH')


def prepare_classes_list(num_classes, classes_per_task, dataset) -> list:
    num_total_classes = dataset_to_num_classes[dataset]
    classes_per_task = classes_per_task
    num_tasks = num_classes // classes_per_task

    classes = np.arange(num_classes)
    np.random.shuffle(classes)
    classes = list(classes)
    classes = classes[:num_tasks * classes_per_task]

    classes_list = [sorted([classes.pop() for _ in range(classes_per_task)]) for _ in range(num_tasks)]
    return classes_list


@dataclass
class TaskList:
    def __init__(self, classes_list, batch_size, dataset, train=True, setup='taskIL'):
        self.classes_list = classes_list
        self.dataset = dataset
        self.train = train

        self.tasks = [Task(classes, batch_size, dataset, train, setup) for classes in self.classes_list]


class RehearsalTask:
    def __init__(self, rehearsal_loader):
        self.tasks = [BasicTask(rehearsal_loader)]


@dataclass()
class BasicTask:
    dataloader: torch.utils.data.DataLoader


class Task:
    def __init__(self, global_classes, batch_size, dataset_ref, train, setup='taskIL'):
        self.global_classes = global_classes
        self.batch_size_train = batch_size
        self.dataset_ref = dataset_ref
        self.setup = setup
        self.dataset = self.dataset_for_classes_task(self.dataset_ref, train)
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

    def dataset_for_classes_task(self, dataset, train=True):
        dataset = dataset(DATA_PATH, train=train, download=True,
                          transform=torchvision.transforms.Compose([
                              torchvision.transforms.ToTensor(),
                              self.provide_normalization(dataset)
                          ]), )

        # select data
        dataset.data = np.concatenate([dataset.data[np.array(dataset.targets) == i] for i in self.global_classes])
        dataset.targets = torch.tensor(np.concatenate(
            [np.array(dataset.targets)[np.array(dataset.targets) == i] for i in self.global_classes]
        )).to(torch.int64)

        if self.setup == 'taskIL':
            # translate global to local labels
            for i in range(len(self.global_classes)):
                dataset.targets[np.array(dataset.targets) == self.global_classes[i]] = i

        return dataset

    @staticmethod
    def provide_normalization(dataset) -> torchvision.transforms.Normalize:
        if dataset is torchvision.datasets.CIFAR10:
            return torchvision.transforms.Normalize(
                (0.4924, 0.4739, 0.4198),
                (0.1891, 0.1865, 0.1885))
        elif dataset is torchvision.datasets.CIFAR100:
            return torchvision.transforms.Normalize(
                (0.5071, 0.4865, 0.4409),
                (0.2009, 0.1984, 0.2023))
        elif dataset is torchvision.datasets.MNIST:
            return torchvision.transforms.Normalize(
                (0.1307,), (0.3081,))
        else:
            raise ValueError("incorrect dataset")
