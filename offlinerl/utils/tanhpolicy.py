import torch
from torch import nn as nn
from torchrl.modules import TanhNormal

class TanhGaussianPolicy(nn.Module):
    def __init__(self,
				 state_dim,
				 action_dim,
				 max_action,
				 num_hidden_layers,
				 hidden_layer_dim,
				 conditioned_sigma: bool = True,
				 log_sig_max = 2.0,
				 log_sig_min = -5.0):
        super().__init__()

        self.max_action = max_action
        self.conditioned_sigma = conditioned_sigma
        self.log_sig_max = log_sig_max
        self.log_sig_min = log_sig_min

        model = [nn.Linear(state_dim, hidden_layer_dim), nn.ReLU()]
        for _ in range(num_hidden_layers):
            model += [nn.Linear(hidden_layer_dim, hidden_layer_dim), nn.ReLU()]
        self.preprocess = nn.Sequential(*model)

        self.mean = nn.Linear(hidden_layer_dim, action_dim)
        if conditioned_sigma:
            self.sigma = nn.Linear(hidden_layer_dim, action_dim)
        else:
            self.sigma = nn.Parameter(torch.zeros(action_dim, 1))

    def forward(self, state: torch.Tensor):
        """
        :param obs: Observation
        """
        logits = self.preprocess(state)

        action = self.mean(logits)

        if self.conditioned_sigma:
            log_std = torch.clamp(self.sigma(logits), min=self.log_sig_min, max=self.log_sig_max)
            std = log_std.exp()
        else:
            shape = [1] * len(action.shape)
            shape[1] = -1
            log_std = (self.sigma.view(shape) + torch.zeros_like(action))
            std = log_std.exp()

        return TanhNormal(action, std)

