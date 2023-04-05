import torch
import torch.nn.functional as F

def train_validation_all_classes(model, optimizer, tasks, rehersal_loader, wandb, epoch=1, log_interval = 1000):
    train_losses = []
    tasks_acc = [[], [], [], [], []]
    exemplers = []

    rehersal_iterator = iter(rehersal_loader)

    model.train()
    for taskNo in range(len(tasks)):
        for batch_idx, (data, target) in enumerate(tasks[taskNo]):

            # training on task
            optimizer.zero_grad()
            output = model(taskNo, data)
            loss = F.cross_entropy(output, target)

            # noise rehersal
            rehersal_data = next(rehersal_iterator)
            output = model(taskNo, rehersal_data[0])
            loss += F.cross_entropy(output, rehersal_data[1])
            loss.backward()
            optimizer.step()

            if batch_idx % log_interval == 0:
                print(f"Train [{batch_idx * len(data)} / {len(tasks[taskNo].dataset)}]       loss: {loss.item()}")
                for i in range(len(tasks)):
                    curr_task_acc = test(model, tasks[i], i, print_accuracy=False)
                    tasks_acc[i].append(curr_task_acc)
                    wandb.log({f"acc_task_{i}": curr_task_acc})

            train_losses.append(loss.item())
            wandb.log({"loss": loss.item()})
            exemplars = tasks[taskNo].batch_size * batch_idx
            exemplers.append(exemplars)
            wandb.log({"exemplers": exemplers*2})
    return train_losses, tasks_acc, exemplers

def test(model, test_loader, taskNo, print_accuracy=True):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(taskNo, data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    if print_accuracy:
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)