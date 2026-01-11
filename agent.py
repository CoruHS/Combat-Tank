# ppo/agent.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple

from .model import ActorCritic
from .rollout_buffer import RolloutBuffer, RolloutBatch


class PPOAgent:
    """
    Proximal Policy Optimization agent.

    Uses clipped surrogate objective with value function clipping.
    """

    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        num_actions: int,
        device: torch.device,
        lr: float = 2.5e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        num_epochs: int = 4,
    ):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.num_epochs = num_epochs

        # Actor-Critic network
        in_channels = obs_shape[0]
        self.network = ActorCritic(in_channels, num_actions).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)

        # Track training statistics
        self.train_steps = 0

    def select_action(
        self, obs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Select actions for a batch of observations.

        Args:
            obs: (num_envs, C, H, W) observations

        Returns:
            actions: (num_envs,) sampled actions
            log_probs: (num_envs,) log probabilities
            values: (num_envs,) value estimates
        """
        obs_t = torch.from_numpy(obs).to(self.device)

        with torch.no_grad():
            action, log_prob, _, value = self.network.get_action_and_value(obs_t)

        return (
            action.cpu().numpy(),
            log_prob.cpu().numpy(),
            value.cpu().numpy(),
        )

    def get_value(self, obs: np.ndarray) -> np.ndarray:
        """Get value estimate for bootstrapping."""
        obs_t = torch.from_numpy(obs).to(self.device)
        with torch.no_grad():
            value = self.network.get_value(obs_t)
        return value.cpu().numpy()

    def train(
        self, buffer: RolloutBuffer, batch_size: int
    ) -> dict:
        """
        Run PPO training on collected rollouts.

        Args:
            buffer: RolloutBuffer with collected trajectories
            batch_size: minibatch size for updates

        Returns:
            dict with training statistics
        """
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        total_clip_fraction = 0.0
        num_updates = 0

        for epoch in range(self.num_epochs):
            for batch in buffer.get_batches(batch_size):
                stats = self._update_step(batch)

                total_policy_loss += stats["policy_loss"]
                total_value_loss += stats["value_loss"]
                total_entropy += stats["entropy"]
                total_approx_kl += stats["approx_kl"]
                total_clip_fraction += stats["clip_fraction"]
                num_updates += 1

        self.train_steps += num_updates

        return {
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy": total_entropy / num_updates,
            "approx_kl": total_approx_kl / num_updates,
            "clip_fraction": total_clip_fraction / num_updates,
        }

    def _update_step(self, batch: RolloutBatch) -> dict:
        """Single PPO update step on a minibatch."""
        # Get current policy outputs for the batch
        _, new_log_probs, entropy, new_values = self.network.get_action_and_value(
            batch.observations, batch.actions
        )

        # Ratio for PPO clipping: exp(log_prob_new - log_prob_old)
        log_ratio = new_log_probs - batch.old_log_probs
        ratio = torch.exp(log_ratio)

        # Approximate KL divergence for monitoring
        with torch.no_grad():
            approx_kl = ((ratio - 1) - log_ratio).mean().item()
            clip_fraction = (
                (torch.abs(ratio - 1.0) > self.clip_epsilon).float().mean().item()
            )

        # Clipped surrogate objective
        # L^CLIP = min(r * A, clip(r, 1-eps, 1+eps) * A)
        surrogate1 = ratio * batch.advantages
        surrogate2 = (
            torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
            * batch.advantages
        )
        policy_loss = -torch.min(surrogate1, surrogate2).mean()

        # Value loss with clipping (optional but helps stability)
        value_pred_clipped = batch.values + torch.clamp(
            new_values - batch.values, -self.clip_epsilon, self.clip_epsilon
        )
        value_loss_unclipped = (new_values - batch.returns) ** 2
        value_loss_clipped = (value_pred_clipped - batch.returns) ** 2
        value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

        # Entropy bonus for exploration
        entropy_loss = -entropy.mean()

        # Total loss
        loss = (
            policy_loss
            + self.value_coef * value_loss
            + self.entropy_coef * entropy_loss
        )

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": -entropy_loss.item(),
            "approx_kl": approx_kl,
            "clip_fraction": clip_fraction,
        }

    def save(self, path: str):
        """Save model weights."""
        torch.save(self.network.state_dict(), path)

    def load(self, path: str):
        """Load model weights."""
        state_dict = torch.load(path, map_location=self.device)
        self.network.load_state_dict(state_dict)
