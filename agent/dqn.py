# from policy.dqn.encoder import TLC_Encoder
import copy
import numpy as np
from agent.MLP import MLP
import torch.nn as nn
import torch
from agent.cnn import CNNBase
# from torch_geometric.nn import GATv2Conv
import torch.nn.functional as F


class DQNNet(nn.Module):
    def __init__(self, hidden_size, obs_length, phase_shape, map_shape, action_space, device):
        super(DQNNet, self).__init__()
        self.device = device
        # self.fc0 = MLP(obs_length, hidden_size)
        # self.hidden = MLP(hidden_size, hidden_size)
        # self.out = nn.Linear(hidden_size + 8, action_space)
        self.fc0 = MLP(obs_length, hidden_size)
        self.fc1 = MLP(phase_shape, phase_shape)
        self.hidden = MLP(hidden_size + phase_shape, hidden_size)
        self.out = nn.Linear(hidden_size, action_space)
        self.to(device)

    def forward(self, obs, phase, obs_map, edges, edges_feature):
        """
        obs_map (_type_): [batch, n_nodes, 12+4]
        """
        obs = torch.tensor(obs, device=self.device, dtype=torch.float)
        phase = torch.tensor(phase, device=self.device, dtype=torch.float)
        obs_map = torch.tensor(obs_map, device=self.device, dtype=torch.float)

        obs_out = self.fc0(obs)  # batch, dims
        phase_out = self.fc1(phase)  # batch, dims
        # feats = obs_out
        feats = torch.cat([obs_out, phase_out], dim=1)
        # feats = torch.cat([obs, phase], dim=1)
        # feats = self.fc0(feats)
        # hidden_state = torch.cat([hidden_state, phase], dim=1)
        hidden_state = self.hidden(feats)
        out = self.out(hidden_state)
        return out, [None]


class DQNMapNet(nn.Module):
    def __init__(self, hidden_size, obs_length, phase_shape, map_shape, action_space, device):
        super(DQNMapNet, self).__init__()
        self.device = device
        self.map_input = CNNBase(hidden_size, map_shape, map_version=True)
        self.phase_input = MLP(phase_shape, phase_shape)

        self.hidden = MLP(hidden_size + phase_shape, hidden_size)
        # self.hidden = MLP(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, action_space)  # action_space = 8
        self.to(device)

    def forward(self, obs, phase, obs_map, edges, edges_feature):
        """
        obs_map (_type_): [batch, n_nodes, 12+4]
        """
        obs = torch.tensor(obs, device=self.device, dtype=torch.float)
        phase = torch.tensor(phase, device=self.device, dtype=torch.float)
        obs_map = torch.tensor(obs_map, device=self.device, dtype=torch.float)
        # edges = torch.tensor(edges, device=self.device, dtype=torch.long)
        # edges_feature = torch.tensor(edges_feature, device=self.device, dtype=torch.float)
        # batchsize, n_nodes = obs.shape[0], obs.shape[1]

        obs_map_out = self.map_input(obs_map)  # batch, dims
        phase_out = self.phase_input(phase)  # batch, dims
        # feats = obs_map_out
        feats = torch.cat([obs_map_out, phase_out], dim=1)
        hidden_state = self.hidden(feats)
        out = self.out(hidden_state)
        return out, [None]


class DQNSeqNet(nn.Module):
    def __init__(self, hidden_size, obs_length, phase_shape, map_shape, action_space, device):
        super(DQNSeqNet, self).__init__()
        self.device = device
        self.obs_input = MLP(obs_length, hidden_size)
        self.map_input = CNNBase(hidden_size, map_shape)
        self.phase_input = MLP(phase_shape, phase_shape)
        # self.phase_input = nn.Sequential()
        self.GRU = nn.GRU(hidden_size, hidden_size, batch_first=True)

        self.hidden = MLP(hidden_size * 2 + phase_shape, hidden_size)
        self.out = nn.Linear(hidden_size, action_space)  # action_space = 8
        self.to(device)
    
    def forward(self, obs, phase, obs_map, edges, edges_feature):
        """
        obs_map (_type_): [batch, n_nodes, seq_len, 12+4]
        """
        self.GRU.flatten_parameters()
        obs = torch.tensor(obs, device=self.device, dtype=torch.float)  # batch, seq_len, dims
        phase = torch.tensor(phase, device=self.device, dtype=torch.float)
        obs_map = torch.tensor(obs_map, device=self.device, dtype=torch.float)
        map_emb = self.map_input(obs_map)

        phase_emb = self.phase_input(phase)
        # obs = obs.reshape(batch_size, *obs.shape[2:])  # batch, seq_len, dims
        obs_emb = self.obs_input(obs)  # B, T, C
        _, obs_rnn = self.GRU(obs_emb)
        obs_rnn = obs_rnn.squeeze(0)
        # obs_feat = torch.cat((obs_rnn, map_emb), dim=1)
        obs_feat = torch.cat((obs_rnn, map_emb, phase_emb), dim=1)

        hidden_state = self.hidden(obs_feat)
        out = self.out(hidden_state)
        return out, None
        
        