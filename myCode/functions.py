import torch
import torch.nn.functional as F
import numpy as np

def permute_mnist(mnist, seed):
    """ Given the training set, permute pixels of each img the same way. """

    np.random.seed(seed)
    print("starting permutation...")
    perm_inds = list(range(28*28))
    np.random.shuffle(perm_inds)
    for i in range(len(mnist.data)):
        mnist.data[i] = mnist.data[i].flatten()[perm_inds].reshape(28,28)
    print("done.")


# def train(model, optimizer, train_loader, epoch=1, log_interval = 100):
#     train_losses = []
#     model.train()
#     for e in range(epoch):
#         for batch_idx, (data, target) in enumerate(train_loader):
#             optimizer.zero_grad()
#             output = model(data)
#             loss = F.cross_entropy(output, target)
#             loss.backward()
#             optimizer.step()
#             if batch_idx % log_interval == 0:
#                 print(f"Epoch {e+1} [{batch_idx * len(data)} / {len(train_loader.dataset)}]       loss: {loss.item()}")
#             train_losses.append(loss.item())
#     return train_losses


def test(model, test_loader, print_accuracy=True):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    if print_accuracy:
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)

def train(model, optimizer, train_loader, val_loader = [], epoch=1, log_interval = 100):
    train_losses = []
    val_losses = []
    exemplers = []
    model.train()
    for e in range(epoch):
        for batch_idx, (data, target) in enumerate(train_loader):
            exemplers.append(train_loader.batch_size * batch_idx)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print(f"Epoch {e+1} [{batch_idx * len(data)} / {len(train_loader.dataset)}]       loss: {loss.item()}")
            if val_loader and batch_idx % log_interval == 0:
                val_losses.append(test(model, val_loader, print_accuracy=False))
            train_losses.append(loss.item())
    return train_losses, val_losses, exemplers

def save_pruned_model(model, filename):
    import torch.nn as nn
    prunedmodel = nn.Sequential(*list(model.children())[:-1])
    torch.save(prunedmodel.state_dict(), filename)