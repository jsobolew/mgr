import torch
from Nets import NetTaskIL
from taskIL import train_validation_all_classes
from dataloaders.mnist import MNIST_for_classes_TaskIL
from dataloaders.noise import dataloader_pretraining_gray
import numpy as np
import torch.optim as optim
import wandb
import torch.nn.functional as F


if __name__ == "__main__":
    print("hi")