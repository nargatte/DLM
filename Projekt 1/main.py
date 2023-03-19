#!/usr/bin/env python3

import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np

import torch.nn.functional as F

from torch import nn, optim
from torchvision import datasets, transforms


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc1 = nn.Linear(32 * 20 * 20, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


classes = {
    0: "plane",
    1: "car",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck"
}


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


batch_size = 5
transform = transforms.Compose([
    transforms.ToTensor()  # this includes scaling to [0, 1]
])

cifar10_data = datasets.CIFAR10("./cifar10", download=True, transform=transform)
data_loader = torch.utils.data.DataLoader(cifar10_data, shuffle=True, batch_size=batch_size)

# for images, labels in data_loader:
#     imshow(torchvision.utils.make_grid(images))
#     print(' '.join(f'{classes[int(labels[j])]:5s}' for j in range(batch_size)))
#     break

net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())

epoch_count = 5

for epoch in range(epoch_count):
    running_loss = 0.0
    for i, data in enumerate(data_loader):
        images, labels = data

        optimizer.zero_grad()

        prediction = net(images)
        loss = criterion(prediction, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
