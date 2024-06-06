import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from typing import Union, Optional

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, num_hidden_layers, hidden_layer_dim, use_ln=False) -> None:
        super().__init__()
        self.q_model = self.build_q_network(state_dim, action_dim, num_hidden_layers, hidden_layer_dim, use_ln)

    def build_q_network(self, state_dim, action_dim, num_hidden_layers, hidden_layer_dim, use_ln):
        layers = [nn.Linear(state_dim + action_dim, hidden_layer_dim), nn.ReLU(inplace=True)]
        for _ in range(num_hidden_layers):
            layers += [nn.Linear(hidden_layer_dim, hidden_layer_dim), nn.ReLU(inplace=True)]
            if use_ln:
                layers += [nn.LayerNorm(hidden_layer_dim)]
        layers += [nn.Linear(hidden_layer_dim, 1)]
        return nn.Sequential(*layers)

    def forward(
        self,
        state: Union[np.ndarray, torch.Tensor],
        action: Optional[Union[np.ndarray, torch.Tensor]],
    ) -> torch.Tensor:
        sa = torch.cat([state, action], dim=1)
        return self.q_model(sa)
