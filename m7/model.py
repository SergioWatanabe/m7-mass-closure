"""The Base MLP from Harder et al. (2022), reimplemented.

Same layer layout and attribute names as `models.Base` in
paulaharder/aerosol-microphysics-emulation, so checkpoints trained with either
load interchangeably (state_dict keys: fc_in, hidden_layers.1, fc_out).
"""
import torch.nn as nn


class Base(nn.Module):
    def __init__(self, in_features, out_features, width, depth=2):
        super().__init__()
        self.fc_in = nn.Linear(in_features, width)
        self.hidden_layers = nn.ModuleList()
        for _ in range(depth - 1):
            self.hidden_layers.append(nn.ReLU())
            self.hidden_layers.append(nn.Linear(width, width))
            self.hidden_layers.append(nn.ReLU())
        self.fc_out = nn.Linear(width, out_features)

    def forward(self, x):
        x = self.fc_in(x)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.fc_out(x)
