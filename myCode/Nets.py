from collections import OrderedDict
from typing import List, Union, cast, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock


class CLNet:
    def return_pretrained_model(self, no_classes):
        last_layer_in_features = list(list(self.children())[-1].parameters())[0].shape[1]


        pretrained_model_except_last_layer = list(self.children())[:-1]

        newModel = nn.Sequential(
            *pretrained_model_except_last_layer,
            nn.Linear(last_layer_in_features, no_classes))
        return newModel

class Net(nn.Module):
    def __init__(self, out_dim):
        super(Net, self).__init__()
        self.out_dim = out_dim
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, self.out_dim)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

class NetTaskIL(nn.Module):
    def __init__(self, classes, classes_per_task):
        super(NetTaskIL, self).__init__()
        self.classes = classes
        self.noTasks = int(classes//classes_per_task)
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)

        self.fc2 = nn.ModuleDict()
        for task in range(self.noTasks):
            self.fc2[str(task)] = nn.Linear(50,2)

    def forward(self, taskNo, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2[str(taskNo)](x)
        return x


class MNIST_net(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.activation = F.relu

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=48, kernel_size=5)
        self.c_bn1 = nn.BatchNorm2d(48)
        self.conv2 = nn.Conv2d(in_channels=48, out_channels=96, kernel_size=5)
        self.c_bn2 = nn.BatchNorm2d(96)

        self.fc1 = nn.Linear(in_features=96*4*4, out_features=512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(in_features=512, out_features=out_features)

    def forward(self, x):
        # conv
        out = self.activation(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
        out = self.c_bn1(out)
        out = self.activation(F.max_pool2d(self.conv2(out), kernel_size=2, stride=2))
        out = self.c_bn2(out)

        # fc
        out = out.reshape(-1, 96*4*4)
        out = self.activation(self.fc1(out))
        out = self.bn1(out)
        out = self.fc2(out)

        return out

class NetNoise(nn.Module):
    def __init__(self, classes):
        super(NetNoise, self).__init__()
        self.classes = classes

        # 64x64 input image
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(3380, 50)
        self.fc2 = nn.Linear(50, self.classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # x = x.view(-1, 3380)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class L2Norm(nn.Module):
    def forward(self, x):
        return x / x.norm(p=2, dim=1, keepdim=True)


class SmallAlexNet(nn.Module):
    def __init__(self, in_channel=3, out_dim=128):
        super(SmallAlexNet, self).__init__()

        blocks = []

        # conv_block_1
        blocks.append(nn.Sequential(
            nn.Conv2d(in_channel, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        ))

        # conv_block_2
        blocks.append(nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        ))

        # conv_block_3
        blocks.append(nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        ))

        # conv_block_4
        blocks.append(nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        ))

        # conv_block_5
        blocks.append(nn.Sequential(
            nn.Conv2d(384, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        ))

        # fc6
        blocks.append(nn.Sequential(
            nn.Flatten(),
            nn.Linear(192 * 3 * 3, 4096, bias=False),  # 256 * 6 * 6 if 224 * 224 # org 192 * 7 * 7 // 3*3 is to small?
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
        ))

        # fc7
        blocks.append(nn.Sequential(
            nn.Linear(4096, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
        ))

        # fc8
        blocks.append(nn.Sequential(
            nn.Linear(4096, out_dim),
            L2Norm(),
        ))

        self.blocks = nn.ModuleList(blocks)
        self.init_weights_()

    def init_weights_(self):
        def init(m):
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.normal_(m.weight, 0, 0.02)
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if getattr(m, 'weight', None) is not None:
                    nn.init.ones_(m.weight)
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)

        self.apply(init)

    def features(self, x, layer_index=-1):
        if layer_index < 0:
            layer_index += len(self.blocks)
        for layer in self.blocks[:(layer_index)]:
            x = layer(x)
        return x

    def forward(self, x, *, layer_index=-1, average=True):
        x = self.features(x, layer_index)
        x = self.blocks[-1](x)

        # NEW: spatial averaging
        if average:
            if x.ndim == 4:
                x = x.mean(dim=-1)
            if x.ndim == 3:
                x = x.mean(dim=-1)

        return x


class SmallAlexNetTaslIL(SmallAlexNet):

    def __init__(self, out_dim, num_tasks, in_channel=3):
        super(SmallAlexNetTaslIL, self).__init__()

        ModuleDict = nn.ModuleDict()
        for task in range(num_tasks):
            ModuleDict[str(task)] = nn.Linear(4096, out_dim)

        self.blocks[-1] = nn.Sequential(
            ModuleDict,
            L2Norm(),
        )

    def freeze_features(self):
        for param in self.parameters():
            param.requires_grad = False

        for param in self.blocks[-1].parameters():
            param.requires_grad = True

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True

    def features(self, x, layer_index=-1):
        if layer_index < 0:
            layer_index += len(self.blocks)
        for layer in self.blocks[:(layer_index)]:
            x = layer(x)
        return x

    def forward(self, TaskNo, x, *, layer_index=-1, average=True):
        x = self.features(x, layer_index)
        x = self.blocks[-1][0][str(TaskNo)](x)

        # NEW: spatial averaging
        if average:
            if x.ndim == 4:
                x = x.mean(dim=-1)
            if x.ndim == 3:
                x = x.mean(dim=-1)

        return x


class ResNet18(ResNet):
    def __init__(self, out_dim):
        super(ResNet18, self).__init__(BasicBlock, [2, 2, 2, 2])
        self.fc = nn.Linear(512, out_dim)

    def forward(self, task_no, x):
        # default forward, with change to fc
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class ResNet18IL(ResNet):
    def __init__(self, out_dim, num_tasks):
        super(ResNet18IL, self).__init__(BasicBlock, [2, 2, 2, 2])
        module_dict = nn.ModuleDict()
        for task in range(num_tasks):
            module_dict[str(task)] = nn.Linear(512, out_dim)
        self.fc = module_dict

    def freeze_features(self):
        for param in self.parameters():
            param.requires_grad = False

        for param in self.fc.parameters():
            param.requires_grad = True

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True

    def features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, task_no, x):
        # default forward, with change to fc
        x = self.features(x)
        x = self.fc[str(task_no)](x)
        return x


class ResNet34(ResNet):
    def __init__(self, out_dim):
        super(ResNet34, self).__init__(BasicBlock, [3, 4, 6, 3])
        self.fc = nn.Linear(512, out_dim)

    def forward(self, task_no, x):
        # default forward, with change to fc
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class ResNet34IL(ResNet):
    def __init__(self, out_dim, num_tasks):
        super(ResNet34IL, self).__init__(BasicBlock, [3, 4, 6, 3])
        module_dict = nn.ModuleDict()
        for task in range(num_tasks):
            module_dict[str(task)] = nn.Linear(512, out_dim)
        self.fc = module_dict

    def forward(self, task_no, x):
        # default forward, with change to fc
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc[str(task_no)](x)

        return x


# class ResNet152(ResNet):
#     def __init__(self, out_dim):
#         super(ResNet152, self).__init__(BasicBlock, [3, 8, 36, 3])
#         self.fc = nn.Linear(512, out_dim)
#
#     def forward(self, task_no, x):
#         # default forward, with change to fc
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)
#
#         return x


class VGGIL(nn.Module):
    # based on https://pytorch.org/vision/main/_modules/torchvision/models/vgg.html#VGG16_Weights
    def __init__(
        self, out_dim, num_tasks, init_weights: bool = True, batch_norm: bool = True, dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.features = make_layers(cfgs["A"], batch_norm=batch_norm)  # cfgs["A"] is the cfg of the VGG11 network
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.fc = nn.ModuleDict()
        for task in range(num_tasks):
            self.fc[str(task)] = nn.Linear(4096, out_dim)

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
        )
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, task_no: int, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        self.fc[str(task_no)](x)
        return x


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}
