import torch.nn as nn
import torch

# class Deeper3DCNN(nn.Module):
#     def __init__(self):
#         super(Deeper3DCNN, self).__init__()
#         self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm3d(16)
#         self.pool = nn.MaxPool3d(2)
#         self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm3d(32)
#         self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
#         self.bn3 = nn.BatchNorm3d(64)
#         self.fc1 = nn.Linear(64 * 16 * 16 * 16, 128)
#         self.dropout = nn.Dropout(0.3)
#         self.fc2 = nn.Linear(128, 1)
#
#     def forward(self, x):
#         x = self.pool(torch.relu(self.bn1(self.conv1(x))))
#         x = self.pool(torch.relu(self.bn2(self.conv2(x))))
#         x = self.pool(torch.relu(self.bn3(self.conv3(x))))
#         x = x.view(x.size(0), -1)
#         x = torch.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.fc2(x)
#         return x.view(-1)


class Deeper3DCNN(nn.Module):
    def __init__(self):
        super(Deeper3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        self.pool = nn.MaxPool3d(2)
        self.dropout3d = nn.Dropout3d(0.2)
        self.adaptive_pool = nn.AdaptiveAvgPool3d((4, 4, 4))  # Ensures fixed output
        self.fc1 = nn.Linear(128 * 4 * 4 * 4, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.LeakyReLU(0.01)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.dropout3d(x)
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.dropout3d(x)
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.dropout3d(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x.view(-1)
