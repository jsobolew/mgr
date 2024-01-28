import torch
import torch.nn.functional as F
import wandb

from loss import SupervisedContrastiveLoss
from utils import SafeIterator


def train_validation_all_classes(model, optimizer, tasks, device, tasks_test=None, rehearsal_loader=None, epoch=1,
                                 log_interval=1000, setup='taskIL', loss_func=F.cross_entropy, contrastive_epoch=0, contrastive_optimizer=None):
    assert setup == 'taskIL' or setup == 'classIL', f"setup should be either taskIL or classIL but is {setup}"
    print(f"Starting training in {setup} setup")

    if rehearsal_loader:
        rehearsal_iter = SafeIterator(rehearsal_loader)

    if contrastive_epoch > 0:
        con_loss = SupervisedContrastiveLoss(temperature=0.07)

    for taskNo in range(len(tasks.tasks)):
        if contrastive_epoch > 0:
            model.unfreeze_all()

            local_class_to_global = {}
            for i, gc in enumerate(tasks.tasks[taskNo].global_classes):
                local_class_to_global[i] = gc

        for ce in range(contrastive_epoch):
            print(sum(p.numel() for p in model.parameters() if p.requires_grad))
            for batch_idx, (data, target) in enumerate(tasks.tasks[taskNo].dataloader):
                model.train()
                output = model.features(data.to(device))
                # regardless of setup always uses global classes
                for local_class, global_class in local_class_to_global.items():
                    target[target == local_class] = global_class
                loss = con_loss(output, target.to(device))

                if rehearsal_loader:
                    # noise rehearsal
                    rehearsal_data = rehearsal_iter.next()
                    output = model.features(rehearsal_data[0].to(device))
                    loss += con_loss(output, rehearsal_data[1].to(device))

                contrastive_optimizer.zero_grad()
                loss.backward()
                contrastive_optimizer.step()

                wandb.log({"con_loss": loss.item()})

                if batch_idx % log_interval == 0:
                    print(
                        f"Task: {taskNo} Contrastive train epoch: {ce} [{batch_idx} / {len(tasks.tasks[taskNo].dataloader)}]       loss: {loss.item()}")

        if contrastive_epoch > 0:
            model.freeze_features()
        for e in range(epoch):
            print(sum(p.numel() for p in model.parameters() if p.requires_grad))
            for batch_idx, (data, target) in enumerate(tasks.tasks[taskNo].dataloader):
                model.train()

                # training on task
                if setup == 'taskIL':
                    output = model(taskNo, data.to(device))
                elif setup == 'classIL':
                    output = model(data.to(device))
                loss = loss_func(output, target.to(device))

                if rehearsal_loader:
                    # noise rehearsal
                    rehearsal_data = rehearsal_iter.next()
                    if setup == 'taskIL':
                        output = model(taskNo, rehearsal_data[0].to(device))
                    elif setup == 'classIL':
                        output = model(rehearsal_data[0].to(device))
                    loss += loss_func(output, rehearsal_data[1].to(device))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # print(len(target[output.argmax(axis=1).cpu() == target]) / len(target))

                if batch_idx % log_interval == 0:
                    print(
                        f"Task: {taskNo} Train epoch: {e} [{batch_idx} / {len(tasks.tasks[taskNo].dataloader)}]       loss: {loss.item()}")
                    if tasks_test:
                        acc_tasks, acc_test_tasks = {}, {}
                        for i in range(len(tasks.tasks)):
                            curr_task_acc = test(model, tasks.tasks[i].dataloader, i, device, loss_func=loss_func, print_accuracy=False, setup=setup)
                            acc_tasks.update({f"acc_task_{i}": float(curr_task_acc)})

                            curr_test_task_acc = test(model, tasks_test.tasks[i].dataloader, i, device, loss_func=loss_func, print_accuracy=False, setup=setup)
                            acc_test_tasks.update({f"acc_test_task_{i}": float(curr_test_task_acc)})

                        wandb.log(acc_tasks)
                        wandb.log(acc_test_tasks)
                        print(acc_tasks)

                wandb.log({"loss": loss.item()})


def test(model, test_loader, task_no, device, loss_func, print_accuracy=True, setup='taskIL'):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if setup == 'taskIL':
                output = model(task_no, data.to(device))
            elif setup == 'classIL':
                output = model(data.to(device))
            test_loss += loss_func(output, target.to(device), size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1].cpu()
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    if print_accuracy:
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)
