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
                self.convs.append(nn.Sequential([
                    nn.Conv2d(conv_channels[i], conv_channels[i + 1], kernel_sizes[i]),
                    nn.MaxPool2d(2, 2)
                ]))
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
