# ppo/rollout_buffer.py
import numpy as np
import torch
from typing import Generator, NamedTuple


class RolloutBatch(NamedTuple):
    """A batch of rollout data for training."""
    observations: torch.Tensor  # (batch_size, 4, 84, 84)
    actions: torch.Tensor       # (batch_size,)
    old_log_probs: torch.Tensor # (batch_size,)
    advantages: torch.Tensor    # (batch_size,)
    returns: torch.Tensor       # (batch_size,)
    values: torch.Tensor        # (batch_size,)


class RolloutBuffer:
    """
    Buffer for storing rollout data from vectorized environments.

    Stores trajectories and computes advantages using GAE.
    """

    def __init__(
        self,
        rollout_length: int,
        num_envs: int,
        obs_shape: tuple,
        device: torch.device,
    ):
        self.rollout_length = rollout_length
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.device = device

        # Preallocate storage
        self.observations = np.zeros(
            (rollout_length, num_envs, *obs_shape), dtype=np.float32
        )
        self.actions = np.zeros((rollout_length, num_envs), dtype=np.int64)
        self.rewards = np.zeros((rollout_length, num_envs), dtype=np.float32)
        self.dones = np.zeros((rollout_length, num_envs), dtype=np.float32)
        self.values = np.zeros((rollout_length, num_envs), dtype=np.float32)
        self.log_probs = np.zeros((rollout_length, num_envs), dtype=np.float32)

        # Computed after rollout
        self.advantages = np.zeros((rollout_length, num_envs), dtype=np.float32)
        self.returns = np.zeros((rollout_length, num_envs), dtype=np.float32)

        self.ptr = 0
        self.full = False

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        value: np.ndarray,
        log_prob: np.ndarray,
    ):
        """
        Add a single step from all environments.

        Args:
            obs: (num_envs, *obs_shape)
            action: (num_envs,)
            reward: (num_envs,)
            done: (num_envs,)
            value: (num_envs,)
            log_prob: (num_envs,)
        """
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob

        self.ptr += 1
        if self.ptr >= self.rollout_length:
            self.full = True

    def compute_advantages(
        self,
        last_value: np.ndarray,
        last_done: np.ndarray,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        """
        Compute GAE advantages and returns after collecting a full rollout.

        Args:
            last_value: (num_envs,) value estimate for state after last step
            last_done: (num_envs,) done flags for last step
            gamma: discount factor
            gae_lambda: GAE lambda parameter
        """
        last_gae = 0

        for t in reversed(range(self.rollout_length)):
            if t == self.rollout_length - 1:
                next_non_terminal = 1.0 - last_done
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_value = self.values[t + 1]

            # TD error: r + gamma * V(s') - V(s)
            delta = (
                self.rewards[t]
                + gamma * next_value * next_non_terminal
                - self.values[t]
            )

            # GAE: A_t = delta_t + gamma * lambda * A_{t+1}
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        # Returns = advantages + values
        self.returns = self.advantages + self.values

    def get_batches(
        self,
        batch_size: int,
    ) -> Generator[RolloutBatch, None, None]:
        """
        Yield minibatches for training.

        Flattens (rollout_length, num_envs) into (total_samples,)
        and shuffles before yielding batches.
        """
        total_samples = self.rollout_length * self.num_envs
        indices = np.random.permutation(total_samples)

        # Flatten all arrays
        flat_obs = self.observations.reshape(-1, *self.obs_shape)
        flat_actions = self.actions.reshape(-1)
        flat_log_probs = self.log_probs.reshape(-1)
        flat_advantages = self.advantages.reshape(-1)
        flat_returns = self.returns.reshape(-1)
        flat_values = self.values.reshape(-1)

        # Normalize advantages (important for training stability)
        flat_advantages = (flat_advantages - flat_advantages.mean()) / (
            flat_advantages.std() + 1e-8
        )

        for start in range(0, total_samples, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]

            yield RolloutBatch(
                observations=torch.from_numpy(flat_obs[batch_indices]).to(self.device),
                actions=torch.from_numpy(flat_actions[batch_indices]).to(self.device),
                old_log_probs=torch.from_numpy(flat_log_probs[batch_indices]).to(self.device),
                advantages=torch.from_numpy(flat_advantages[batch_indices]).to(self.device),
                returns=torch.from_numpy(flat_returns[batch_indices]).to(self.device),
                values=torch.from_numpy(flat_values[batch_indices]).to(self.device),
            )

    def reset(self):
        """Reset buffer for next rollout collection."""
        self.ptr = 0
        self.full = False
