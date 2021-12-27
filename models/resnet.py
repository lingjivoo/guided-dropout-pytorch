import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .guided_dropout import GuidedDropout,GuidedDropout2D


def dropout_selection(drop_type,dim,drop_rate):
    if 'GuidedDropout' in drop_type:
        drop = GuidedDropout(dim,drop_rate)
    else:
        drop = nn.Dropout(drop_rate)
    return drop


def dropout_selection_2D(drop_type,dim,drop_rate):
    if 'GuidedDropout' in drop_type:
        drop = GuidedDropout2D(dim,drop_rate)
    else:
        drop = nn.Dropout2D(drop_rate)
    return drop


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1,drop=None,drop_rate=0):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if drop is not None:
            self.drop = dropout_selection_2D(drop,planes,drop_rate)
        else:
            self.drop = nn.Identity()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        out = self.drop(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1,drop=None,drop_rate=0):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.shortcut = nn.Sequential()
        if drop is not None:
            self.drop = dropout_selection_2D(drop, planes, drop_rate)
        else:
            self.drop = nn.Identity()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        out = self.drop(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10,RGB_Image=True,drop=None,drop_rate=0):
        super(ResNet, self).__init__()
        self.in_planes = 64
        if RGB_Image:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1,drop=drop,drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2,drop=drop,drop_rate=drop_rate)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2,drop=drop,drop_rate=drop_rate)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2,drop=drop,drop_rate=drop_rate)
        self.layer5 = BasicBlock(512, 512, stride=2,drop=drop,drop_rate=drop_rate)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        if drop is not None:
            self.drop = dropout_selection(drop,512,drop_rate)
        else:
            self.drop = nn.Identity()
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride,drop=None,drop_rate=0):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride,drop=drop,drop_rate=drop_rate))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.drop(out)
        out = self.linear(out)
        return out

    #DR scheduler
    def dropout_scheduler(self,epoch,step_epochs=[0,2,150]):
        if epoch in step_epochs:
            for m in self.modules():
                if isinstance(m, GuidedDropout):
                    if m.begin_flag:
                        m.drop_rate = m.drop_rate - 0.05
                    else:
                        m.begin_flag = True
                elif isinstance(m, GuidedDropout2D):
                    if m.begin_flag:
                        m.drop_rate = m.drop_rate - 0.05
                    else:
                        m.begin_flag = True


def ResNet18(num_classes=10,RGB_Image=True,drop=None,drop_rate=0):
    return ResNet(BasicBlock, [2, 2, 2, 1], num_classes=num_classes,RGB_Image=RGB_Image,drop=drop,drop_rate=drop_rate)


if __name__=="__main__":
    x = torch.randn(1,3,32,32).cuda()
    nn = ResNet18(10,drop='GuidedDropout',drop_rate=0.1).cuda()
    print(nn)
    out = nn(x)
    print("out:",out.shape)
    print(out.sum())