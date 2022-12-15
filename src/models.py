"""Models for selector implementation."""

import torch.nn as nn


class SelectiveNet(nn.Module):
    """Implements a feed-forward MLP."""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers,
        dropout=0.0,
    ):
        super(SelectiveNet, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers):
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.extend([nn.Dropout(dropout), nn.Linear(hidden_dim, 1)])

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).view(-1)
