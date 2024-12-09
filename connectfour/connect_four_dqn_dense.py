import torch.nn as nn
import torch.nn.functional as F


class Connect4DQN(nn.Module):
    def __init__(self, input_channels=3):
        super(Connect4DQN, self).__init__()

        self.flat_features = input_channels * 6 * 7

        self.fc1 = nn.Linear(self.flat_features, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 7)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        if len(x.shape) != 4:
            raise ValueError(
                f"Expected 4D input (batch_size, channels, height, width), got shape {x.shape}"
            )

        x = x.view(-1, self.flat_features)

        identity1 = self.fc1(x)
        x = F.relu(identity1)
        x = self.dropout(x)

        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = x + F.linear(identity1, self.fc2.weight[:256, :], self.fc2.bias)

        x = F.relu(self.fc3(x))
        x = self.dropout(x)

        x = self.fc4(x)

        return x
