import torch
import torch.nn.functional as F
from torch import nn


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class MLP(nn.Module):
    def __init__(self, hidden_size, out_features=None, layer_norm=False):
        super(MLP, self).__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear = nn.Linear(hidden_size, out_features)
        self.layer_norm = layer_norm
        if layer_norm:
            self.layer_norm = nn.LayerNorm(out_features)

    def forward(self, hidden_states):
        hidden_states = self.linear(hidden_states)
        if self.layer_norm:
            hidden_states = self.layer_norm(hidden_states)
        hidden_states = F.relu(hidden_states)
        return hidden_states
