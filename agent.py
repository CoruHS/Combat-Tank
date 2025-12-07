# agent.py
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import SimpleDQNCNN


class DQNAgent:
    """
    DQN with CNN:
      - Q-network + target-network
      - MSE TD loss
      - epsilon-greedy policy with built-in epsilon decay
    """

    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        num_actions: int,
        device: torch.device = torch.device("cpu"),
        gamma: float = 0.99,
        lr: float = 1e-4,
        target_update_freq: int = 1000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay_steps: int =  1800000,
    ):
        self.device = device
        self.num_actions = num_actions
        self.gamma = gamma
        self.target_update_freq = target_update_freq

        # ---- Epsilon schedule ----
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon = epsilon_start  # current epsilon

        in_channels = obs_shape[0]  # (C, H, W)
        self.q_net = SimpleDQNCNN(in_channels, num_actions).to(device)
        self.target_net = SimpleDQNCNN(in_channels, num_actions).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.train_steps = 0
        self.max_grad_norm = 10.0

    def update_epsilon(self, global_step: int) -> None:
        """
        Linearly decay epsilon from epsilon_start to epsilon_end
        over epsilon_decay_steps steps, then keep it at epsilon_end.
        """
        frac = min(1.0, global_step / float(self.epsilon_decay_steps))
        self.epsilon = self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def select_action(self, obs: np.ndarray) -> int:
        """
        obs: (C, H, W) numpy array in [0, 1].
        Uses self.epsilon for epsilon-greedy.
        """
        if random.random() < self.epsilon:
            return random.randrange(self.num_actions)

        obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)  # (1, C, H, W)
        with torch.no_grad():
            q_values = self.q_net(obs_t)
        return int(q_values.argmax(dim=1).item())

    def train_step(self, batch, batch_size: int):
        obs, actions, rewards, next_obs, dones = batch

        obs_t = torch.from_numpy(obs).to(self.device)          # (B, C, H, W)
        next_obs_t = torch.from_numpy(next_obs).to(self.device)
        actions_t = torch.from_numpy(actions).to(self.device)  # (B,)
        rewards_t = torch.from_numpy(rewards).to(self.device)  # (B,)
        dones_t = torch.from_numpy(dones).to(self.device)      # (B,)

        # Q(s,a)
        q_values = self.q_net(obs_t)                           # (B, A)
        q_values = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_net(next_obs_t).max(dim=1)[0]
            target = rewards_t + self.gamma * (1.0 - dones_t) * next_q_values

        loss = nn.MSELoss()(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()
