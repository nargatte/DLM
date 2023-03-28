import torch
import torch.nn.functional as F

from torch import nn


class SimpleNet(nn.Module):
    def __init__(self, conv_channels, kernel_sizes, fc_sizes):
        super().__init__()

        self.convs = nn.ModuleList()
        for i in range(len(kernel_sizes)):
            self.convs.append(nn.Conv2d(conv_channels[i], conv_channels[i + 1], kernel_sizes[i]))

        self.fcs = nn.ModuleList()
        for i in range(len(fc_sizes) - 1):
            self.fcs.append(nn.Linear(fc_sizes[i], fc_sizes[i + 1]))

        
    def forward(self, x):
        for conv in self.convs:
            x = F.relu(conv(x))
            
        x = torch.flatten(x, 1)
        for fc in self.fcs[:-1]:
            x = F.relu(fc(x))

        x = self.fcs[-1](x)
        return x


class PoolingNet(nn.Module):
    def __init__(self, conv_channels, kernel_sizes, pools, fc_sizes):
        super().__init__()

        self.convs = nn.ModuleList()
        for i in range(len(kernel_sizes)):
            if pools[i]:
                self.convs.append(nn.Sequential(
                    nn.Conv2d(conv_channels[i], conv_channels[i + 1], kernel_sizes[i]),
                    nn.MaxPool2d(2, 2)
                ))
            else:
                self.convs.append(nn.Conv2d(conv_channels[i], conv_channels[i + 1], kernel_sizes[i]))

        self.fcs = nn.ModuleList()
        for i in range(len(fc_sizes) - 1):
            self.fcs.append(nn.Linear(fc_sizes[i], fc_sizes[i + 1]))

    def forward(self, x):
        for conv in self.convs:
            x = F.relu(conv(x))

        x = torch.flatten(x, 1)
        for fc in self.fcs[:-1]:
            x = F.relu(fc(x))

        x = self.fcs[-1](x)
        return x


# based on https://medium.com/analytics-vidhya/resnet-10f4ef1b9d4c
class ResidualNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ResidualNet._conv_block(3, 64, True)
        self.conv2 = ResidualNet._conv_block(64, 128, True)
        self.conv3 = ResidualNet._conv_block(128, 128)

        self.fc1 = nn.Linear(512 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(self.conv3(x)) + x

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x
    
    @staticmethod
    def _conv_block(in_channels, out_channels, pool=False):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
                nn.BatchNorm2d(out_channels), 
                nn.ReLU(inplace=True)]
        if pool:
            layers.append(nn.MaxPool2d(2))
        return nn.Sequential(*layers)


# based on https://pytorch.org/vision/main/_modules/torchvision/models/inception.html
class InceptionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = InceptionNet._conv_block(3, 32, kernel_size=3)
        self.conv2 = InceptionNet._conv_block(32, 32, kernel_size=3)
        self.conv3 = InceptionNet._conv_block(32, 64, kernel_size=3)

        self.inc1 = InceptionNet.Inception(64, 32)
        self.inc2 = InceptionNet.Inception(256, 64)
        self.inc3 = InceptionNet.Inception(288, 64)

        self.fc1 = nn.Linear(288 * 26 * 26, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.inc1(x)
        x = self.inc2(x)
        x = self.inc3(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


    class Inception(nn.Module):
        def __init__(self, in_channels, pool_features):
            super().__init__()
            self.branch_1x1 = InceptionNet._conv_block(in_channels, 64, kernel_size=1)

            self.branch_5x5 = nn.Sequential(
                InceptionNet._conv_block(in_channels, 48, kernel_size=1),
                InceptionNet._conv_block(48, 64, kernel_size=5, padding=2)
            )

            self.branch_3x3 = nn.Sequential(
                InceptionNet._conv_block(in_channels, 64, kernel_size=1),
                InceptionNet._conv_block(64, 96, kernel_size=3, padding=1),
            )

            self.branch_pool = nn.Sequential(
                nn.MaxPool2d(3, stride=1, padding=1),
                InceptionNet._conv_block(in_channels, pool_features, kernel_size=1)
            )

        def forward(self, x):
            branch_1x1 = self.branch_1x1(x)
            branch_5x5 = self.branch_5x5(x)
            branch_3x3 = self.branch_3x3(x)
            branch_pool = self.branch_pool(x)
            return torch.cat((branch_1x1, branch_5x5, branch_3x3, branch_pool), 1)

    @staticmethod
    def _conv_block(in_channels, out_channels, **kwargs):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
