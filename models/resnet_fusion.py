'''ResNet in PyTorch.

Reference:
https://github.com/kuangliu/pytorch-cifar
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.num_classes = num_classes
        if num_classes == 1000:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        if self.num_classes == 1000:
            out = self.maxpool(out)
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = F.avg_pool2d(out4, 4)
        out5 = out5.view(out5.size(0), -1)
        out = self.linear(out5)
        return out
    
    def forward_feature(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        if self.num_classes == 1000:
            out = self.maxpool(out)
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = F.avg_pool2d(out4, 4)
        out5 = out5.view(out5.size(0), -1)
        out = self.linear(out5)
        return out, out5
    
    def layer_fusion(self, x_a, x_b, lam):
        out_a = F.relu(self.bn1(self.conv1(x_a)))
        out_b = F.relu(self.bn1(self.conv1(x_b)))
        if self.num_classes == 1000:
            out_a = self.maxpool(out_a)
            out_b = self.maxpool(out_b)
        out_a = self.layer1(out_a)
        out_b = self.layer1(out_b)
        out = lam * out_a + (1 - lam) * out_b
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
    def layer_fusion_multi(self, x_a, x_b, lam, layer=1):
        out_a = F.relu(self.bn1(self.conv1(x_a)))
        out_b = F.relu(self.bn1(self.conv1(x_b)))
        if self.num_classes == 1000:
            out_a = self.maxpool(out_a)
            out_b = self.maxpool(out_b)
        out_a = self.layer1(out_a)
        out_b = self.layer1(out_b)
        if layer == 1:
            out = lam * out_a + (1 - lam) * out_b
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
        elif layer == 2:
            out_a = self.layer2(out_a)
            out_b = self.layer2(out_b)
            out = lam * out_a + (1 - lam) * out_b
            out = self.layer3(out)
            out = self.layer4(out)
        elif layer == 3:
            out_a = self.layer2(out_a)
            out_b = self.layer2(out_b)
            out_a = self.layer3(out_a)
            out_b = self.layer3(out_b)
            out = lam * out_a + (1 - lam) * out_b
            out = self.layer4(out)
        elif layer == 4:
            out_a = self.layer2(out_a)
            out_b = self.layer2(out_b)
            out_a = self.layer3(out_a)
            out_b = self.layer3(out_b)
            out_a = self.layer4(out_a)
            out_b = self.layer4(out_b)
            out = lam * out_a + (1 - lam) * out_b

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        return out
    
    
    def forward_ll4al(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        if self.num_classes == 1000:
            out = self.maxpool(out)
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out = F.avg_pool2d(out4, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out, [out1, out2, out3, out4]
    

    def forward_feature_ll4al(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        if self.num_classes == 1000:
            out = self.maxpool(out)
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out = F.avg_pool2d(out4, 4)
        outf = out.view(out.size(0), -1)
        out = self.linear(outf)
        return out, outf, [out1, out2, out3, out4]


def ResNet18(num_classes = 10):
    return ResNet(BasicBlock, [2,2,2,2], num_classes)

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50(num_classes = 10):
    return ResNet(Bottleneck, [3,4,6,3], num_classes)

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])