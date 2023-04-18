import copy
import numpy as np
from agent.MLP import MLP
import torch.nn as nn
import torch
from agent.cnn import CNNBase
from torch_geometric.nn import GATv2Conv
import torch.nn.functional as F
import time


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def flat_edges(edges, batch_size, n_nodes):
    edge_num = edges.shape[-1]
    source_idx = np.arange(batch_size) * n_nodes
    offset = source_idx.reshape(-1, 1)
    edges = edges.reshape(batch_size, 2 * edge_num) + offset.repeat(2 * edge_num, axis=-1)
    edges = edges.reshape(batch_size, 2, edge_num)
    edges = np.hstack([b_edges for b_edges in edges])  # [2, (n_nodes - 1)*batch_size]
    return edges, source_idx


class TLC_DQN(nn.Module):
    def __init__(self, hidden_size, obs_length, phase_shape, map_shape, action_space, device, dueling=True):
        super(TLC_DQN, self).__init__()
        self.device = device
        self.dueling = dueling

        self.nbrs_input = MLP(obs_length + 4, hidden_size)  # obs_space = 12
        self.map_input = CNNBase(hidden_size, map_shape)
        self.phase_input = MLP(phase_shape, phase_shape)

        gat_head_num = 5
        edge_feature_dim = 4
        self.gat = GATv2Conv(in_channels=hidden_size, out_channels=hidden_size, heads=gat_head_num, concat=False)
        self.GRU = nn.GRU(hidden_size, hidden_size, batch_first=True)

        # self.hidden = MLP(hidden_size + phase_shape, hidden_size)
        self.hidden = MLP(hidden_size * 2 + phase_shape, hidden_size)
        if not self.dueling:
            self.out = nn.Linear(hidden_size, action_space)  # action_space = 8
            # self.out = init_(self.out)
        else:
            self.out = nn.Linear(hidden_size, 1)
            self.adv = nn.Linear(hidden_size, action_space)
            # self.out = init_(self.out)
            # self.adv = init_(self.adv)
        self.to(device)

    def forward(self, obs, phase, obs_map, edges, edges_feature):
        """
        obs_map (_type_): [batch, n_nodes, seq_len, 12+4]
        """
        self.GRU.flatten_parameters()
        obs = torch.tensor(obs, device=self.device, dtype=torch.float)
        batchsize, n_nodes = obs.shape[0], obs.shape[1]
        phase = torch.tensor(phase, device=self.device, dtype=torch.float)
        obs_map = torch.tensor(obs_map, device=self.device, dtype=torch.float)
        flatten_edges, source_idx = flat_edges(edges, batchsize, n_nodes)
        flatten_edges = torch.tensor(flatten_edges, device=self.device)
        source_idx = torch.tensor(source_idx, device=self.device)
        edges_feature = torch.tensor(edges_feature, device=self.device, dtype=torch.float)

        # target_emb = self.map_input(obs_map).unsqueeze(1)
        # nbrs_input = obs[:, 1:, :, :]  # batch, n_nodes-1, seq_len, dims
        # nbrs_input = nbrs_input.reshape(batchsize * (n_nodes - 1),
        #                                 *nbrs_input.shape[2:])  # batch*(n_nodes-1), seq_len, dims
        # nbrs_emb = self.nbrs_input(nbrs_input)  # B, T, C
        # _, nbrs_rnn = self.GRU(nbrs_emb)
        # nbrs_rnn = nbrs_rnn.squeeze(0).reshape(batchsize, n_nodes - 1, -1)  # batch*(n_nodes-1), dims
        # obs = torch.cat((target_emb, nbrs_rnn), dim=1)  # batch, n_nodes, dims
        # obs = obs.reshape(batchsize*n_nodes, -1)
        # obs_gat, (_, att_weights) = self.gat(x=obs, edge_index=flatten_edges, return_attention_weights=True)
        # obs_gat = obs_gat[source_idx]
        # phase_emb = self.phase_input(phase)
        # obs_gat = torch.cat((obs_gat, phase_emb), dim=1)

        nbrs_input = obs.reshape(batchsize * n_nodes, *obs.shape[2:])  # batch*n_nodes, seq_len, dims
        nbrs_emb = self.nbrs_input(nbrs_input)  # B, T, C
        _, nbrs_rnn = self.GRU(nbrs_emb)
        nbrs_rnn = nbrs_rnn.squeeze(0).reshape(batchsize, n_nodes, -1)  # batch, n_nodes, dims
        nbrs_rnn = nbrs_rnn.reshape(batchsize * n_nodes, -1)
        obs_gat, (_, att_weights) = self.gat(x=nbrs_rnn, edge_index=flatten_edges, return_attention_weights=True)
        obs_gat = obs_gat[source_idx]

        target_emb = self.map_input(obs_map)
        phase_emb = self.phase_input(phase)
        obs_gat = torch.cat((obs_gat, target_emb, phase_emb), dim=1)  # batch, n_nodes, dims

        hidden_state = self.hidden(obs_gat)
        out = self.out(hidden_state)
        if self.dueling:
            adv = self.adv(hidden_state)
            out = out + (adv - adv.mean(axis=-1, keepdim=True))
        return out, att_weights
