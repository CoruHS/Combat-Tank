# ppo/model.py
import torch
import torch.nn as nn
import numpy as np


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Orthogonal initialization for better training stability."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    """
    Actor-Critic network with shared CNN backbone.

    Input:  (B, 4, 84, 84) - 4 stacked grayscale frames
    Output:
        - policy_logits: (B, num_actions) - unnormalized action probabilities
        - value: (B, 1) - estimated state value
    """

    def __init__(self, in_channels: int = 4, num_actions: int = 18):
        super().__init__()

        # Shared CNN backbone (Nature DQN style, but larger)
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate conv output size: 84 -> 20 -> 9 -> 7
        # (84 - 8) / 4 + 1 = 20
        # (20 - 4) / 2 + 1 = 9
        # (9 - 3) / 1 + 1 = 7
        conv_out_size = 64 * 7 * 7  # 3136

        # Shared fully connected layer
        self.fc = nn.Sequential(
            layer_init(nn.Linear(conv_out_size, 512)),
            nn.ReLU(),
        )

        # Policy head (actor) - outputs action logits
        self.policy_head = layer_init(nn.Linear(512, num_actions), std=0.01)

        # Value head (critic) - outputs state value
        self.value_head = layer_init(nn.Linear(512, 1), std=1.0)

    def forward(self, x: torch.Tensor):
        """
        Forward pass returning both policy logits and value.

        Args:
            x: (B, 4, 84, 84) observation tensor, values in [0, 1]

        Returns:
            policy_logits: (B, num_actions)
            value: (B,)
        """
        features = self.conv(x)
        features = self.fc(features)

        policy_logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)

        return policy_logits, value

    def get_action_and_value(self, x: torch.Tensor, action: torch.Tensor = None):
        """
        Get action, log probability, entropy, and value for PPO.

        Args:
            x: (B, 4, 84, 84) observations
            action: (B,) optional actions to evaluate (for training)

        Returns:
            action: (B,) sampled or provided actions
            log_prob: (B,) log probability of actions
            entropy: (B,) entropy of policy
            value: (B,) state values
        """
        policy_logits, value = self.forward(x)

        # Create categorical distribution from logits
        probs = torch.softmax(policy_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Get only the value estimate (for bootstrapping)."""
        features = self.conv(x)
        features = self.fc(features)
        return self.value_head(features).squeeze(-1)
