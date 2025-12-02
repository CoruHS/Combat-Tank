# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleDQNCNN(nn.Module):
    """
    Simple Atari-style CNN:

    Input:  (B, 4, 84, 84)
    Output: (B, num_actions) Q-values
    """

    def __init__(self, in_channels: int, num_actions: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)

        # 84x84 -> conv1 (8,4) -> 20x20
        # 20x20 -> conv2 (4,2) -> 9x9
        conv_out_dim = 32 * 9 * 9

        self.fc1 = nn.Linear(conv_out_dim, 128)
        self.fc2 = nn.Linear(128, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
