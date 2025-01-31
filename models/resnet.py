"""resnet in pytorch
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import torch
import torch.nn as nn
import torchvision.models as models




class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1


    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers
    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=2):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(3 * 512, 3 * 512)
        self.fc2 = nn.Linear(3 * 512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)


    def forward_once(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        return output.view(output.size(0), -1)

    def forward(self, x):
        output1 = self.forward_once(x[:,0:3,:,:])
        output2 = self.forward_once(x[:,3:6,:,:])
        output3 = self.forward_once(x[:,6:9,:,:])

        output = torch.cat((output1, output2, output3),dim=1)
        output = self.fc1(self.relu(output))
        output = self.fc2(self.relu(output))
        return output

def resnet18():
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2])

def resnet34():
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3])

def resnet50():
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3])

def resnet101():
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3])

def resnet152():
    """ return a ResNet 152 object
    """
    return ResNet(BottleNeck, [3, 8, 36, 3])


class ResNet18_Pretrained(nn.Module):

    def __init__(self):
        super().__init__()

        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = torch.nn.Linear(512, 512)
        self.in_channels = 64

        self.relu = nn.ReLU()
        # self.fc1 = nn.Linear(3 * 512, 3 * 512)
        self.fc2 = nn.Linear(3 * 512, 2)


    def forward_once(self, x):
        output = self.resnet(x)
        return output.view(output.size(0), -1)

    def forward(self, x):
        output1 = self.forward_once(x[:,0:3,:,:])
        output2 = self.forward_once(x[:,3:6,:,:])
        output3 = self.forward_once(x[:,6:9,:,:])

        output = torch.cat((output1, output2, output3),dim=1)
        # output = self.fc1(self.relu(output))
        output = self.fc2(self.relu(output))
        return output


class ResNet18_Pretrained_Orig(nn.Module):

    def __init__(self):
        super().__init__()

        self.resnet = models.resnet18(pretrained=True)


    def forward(self, x):

        return self.resnet(x)

class ResNet50_Pretrained_Orig(nn.Module):

    def __init__(self):
        super().__init__()

        self.resnet = models.resnet50(pretrained=True)


    def forward(self, x):

        return self.resnet(x)


class ResNet50_Pretrained_Siamese(nn.Module):

    def __init__(self, embedding_dims=1000):
        super().__init__()

        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Linear(2048, embedding_dims)

    def forward(self, x, y):

        return self.resnet(x), self.resnet(y)
