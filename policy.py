"""
Actor-Critic policy for PPO with TEM abstract state input.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    """Separate policy and value MLPs for PPO.

    Policy outputs tanh-squashed Gaussian actions.
    Value network predicts scalar state value.
    """

    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        # State-independent log standard deviation
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        self.value_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for net in [self.policy_net, self.value_net]:
            for module in net:
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                    module.bias.data.zero_()
        # Small init for policy output -> initial actions near zero
        nn.init.orthogonal_(self.policy_net[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.value_net[-1].weight, gain=1.0)

    def forward(self, obs):
        """Sample action and compute log_prob, value, entropy."""
        action_mean = self.policy_net(obs)
        action_std = self.log_std.exp().expand_as(action_mean)
        dist = Normal(action_mean, action_std)

        action_raw = dist.rsample()
        action = torch.tanh(action_raw)

        # Log prob with tanh squashing correction
        log_prob = dist.log_prob(action_raw)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)

        value = self.value_net(obs).squeeze(-1)
        entropy = dist.entropy().sum(dim=-1)

        return action, log_prob, value, entropy

    def get_value(self, obs):
        """Value prediction only (for GAE bootstrap)."""
        return self.value_net(obs).squeeze(-1)

    def evaluate_actions(self, obs, actions):
        """Re-evaluate log_prob and value for stored (obs, action) pairs.
        Used during PPO update phase."""
        action_mean = self.policy_net(obs)
        action_std = self.log_std.exp().expand_as(action_mean)
        dist = Normal(action_mean, action_std)

        # Inverse tanh to recover raw action
        actions_clamped = actions.clamp(-0.999, 0.999)
        action_raw = torch.atanh(actions_clamped)

        log_prob = dist.log_prob(action_raw)
        log_prob -= torch.log(1 - actions_clamped.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)

        value = self.value_net(obs).squeeze(-1)
        entropy = dist.entropy().sum(dim=-1)

        return log_prob, value, entropy
