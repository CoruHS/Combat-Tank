# ppo/main.py
"""
PPO training for Tank Wars - v2 with anti-passive reward shaping.

Key fixes from failed v1 training:
1. Entropy 0.001 was way too low - collapsed immediately
2. Fire bonus rewarded passive "turret" play
3. No penalty for being far from enemy (only reward for approaching)
4. No penalty for spinning

This version:
1. Entropy 0.02 (20x higher) for more exploration
2. Distance PENALTY (being far is bad, not just approaching is good)
3. Conditional fire bonus (only when approaching)
4. Spinning detection and penalty
5. Frame skip reduced to 2 for more responsive control
"""
import os
import sys
import time
import random
from collections import deque

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ppo.agent import PPOAgent
from ppo.rollout_buffer import RolloutBuffer
from ppo.vec_env import VecEnv, DummyVecEnv


# ============================================================
# HYPERPARAMETERS - FIXED FOR EXPLORATION
# ============================================================

# Environment
NUM_ENVS = 8  # Reduced to lower CPU usage (use 2 if still too high)
STACK_SIZE = 4
ENV_VERSION = "v2"  # "v2" = anti-passive, "v1" = original, "old" = combat_tankyes

# Training budget
TOTAL_TIMESTEPS = 4_000_000

# PPO rollout
ROLLOUT_LENGTH = 256
BATCH_SIZE = 512
NUM_EPOCHS = 4

# PPO algorithm
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
VALUE_COEF = 0.5
ENTROPY_COEF = 0.02       # INCREASED from 0.001 - critical fix!
MAX_GRAD_NORM = 0.5

# Learning rate
LEARNING_RATE = 3e-4
LR_ANNEAL = True
LR_MIN_FRAC = 0.1         # Don't anneal below 10% of initial LR

# Logging
LOG_INTERVAL = 5
SAVE_INTERVAL = 25

# Enemy (set to trained PPO model for self-play)
ENEMY_MODEL_PATH = "ppo_tank_final_trainer.pth"


class EpisodeTracker:
    """Track episode statistics across vectorized environments."""

    def __init__(self, num_envs: int, window_size: int = 100):
        self.num_envs = num_envs
        self.window_size = window_size

        self.episode_rewards = [0.0] * num_envs
        self.episode_lengths = [0] * num_envs

        self.reward_history = deque(maxlen=window_size)
        self.length_history = deque(maxlen=window_size)
        self.score_history = deque(maxlen=window_size)
        self.win_history = deque(maxlen=window_size)

        # Spin tracking
        self.spin_penalty_history = deque(maxlen=window_size)
        self.rotation_rate_history = deque(maxlen=window_size)
        self.movement_rate_history = deque(maxlen=window_size)

        self.total_episodes = 0

    def update(self, rewards, dones, infos):
        for i in range(self.num_envs):
            self.episode_rewards[i] += rewards[i]
            self.episode_lengths[i] += 1

            if dones[i]:
                self.reward_history.append(self.episode_rewards[i])
                self.length_history.append(self.episode_lengths[i])

                info = infos[i] if infos else {}
                my_score = info.get("my_raw_score", 0)
                enemy_score = info.get("enemy_raw_score", 0)
                self.score_history.append(my_score)

                won = 1 if my_score > enemy_score else 0
                self.win_history.append(won)

                # Track spin stats
                spin_penalties = info.get("spin_penalties_triggered", 0)
                rotation_rate = info.get("rotation_rate", 0)
                movement_rate = info.get("movement_rate", 0)
                self.spin_penalty_history.append(spin_penalties)
                self.rotation_rate_history.append(rotation_rate)
                self.movement_rate_history.append(movement_rate)

                self.episode_rewards[i] = 0.0
                self.episode_lengths[i] = 0
                self.total_episodes += 1

    def get_stats(self):
        if len(self.reward_history) == 0:
            return {
                "mean_reward": 0.0,
                "mean_length": 0.0,
                "mean_score": 0.0,
                "win_rate": 0.0,
                "spin_penalties": 0.0,
                "rotation_rate": 0.0,
                "movement_rate": 0.0,
                "episodes": 0,
            }

        return {
            "mean_reward": np.mean(self.reward_history),
            "mean_length": np.mean(self.length_history),
            "mean_score": np.mean(self.score_history),
            "win_rate": np.mean(self.win_history) if self.win_history else 0.0,
            "spin_penalties": np.mean(self.spin_penalty_history) if self.spin_penalty_history else 0.0,
            "rotation_rate": np.mean(self.rotation_rate_history) if self.rotation_rate_history else 0.0,
            "movement_rate": np.mean(self.movement_rate_history) if self.movement_rate_history else 0.0,
            "episodes": self.total_episodes,
        }


def make_env_kwargs():
    """Create environment kwargs based on version."""
    if ENV_VERSION == "v2":
        return {
            "frame_skip": 2,  # Reduced for more responsive control
            "render_mode": None,
            "enemy_model_path": ENEMY_MODEL_PATH,
            "enemy_device": "cuda" if torch.cuda.is_available() else "cpu",
            "debug": False,
        }
    elif ENV_VERSION == "v1":
        return {
            "frame_skip": 4,
            "render_mode": None,
            "enemy_model_path": ENEMY_MODEL_PATH,
            "enemy_device": "cuda" if torch.cuda.is_available() else "cpu",
        }
    else:
        return {
            "gamma": GAMMA,
            "lambda_phi": 0.0,
            "render_mode": None,
            "enemy_model_path": ENEMY_MODEL_PATH,
            "enemy_device": "cuda" if torch.cuda.is_available() else "cpu",
        }


def main():
    print("=" * 60)
    print("PPO Training for Tank Wars - v2 Anti-Passive")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    env_kwargs = make_env_kwargs()
    print(f"Environment version: {ENV_VERSION}")
    print(f"Enemy: {'Random' if ENEMY_MODEL_PATH is None else ENEMY_MODEL_PATH}")
    print(f"Entropy coefficient: {ENTROPY_COEF} (critical for exploration!)")
    print(f"Creating {NUM_ENVS} parallel environments...")

    try:
        vec_env = VecEnv(NUM_ENVS, env_kwargs, stack_size=STACK_SIZE, env_version=ENV_VERSION)
        print("Using VecEnv (multiprocessing)")
    except Exception as e:
        print(f"VecEnv failed: {e}")
        print("Falling back to DummyVecEnv (sequential)")
        vec_env = DummyVecEnv(NUM_ENVS, env_kwargs, stack_size=STACK_SIZE, env_version=ENV_VERSION)

    obs_shape = vec_env.obs_shape
    num_actions = vec_env.num_actions
    print(f"Observation shape: {obs_shape}")
    print(f"Number of actions: {num_actions}")

    agent = PPOAgent(
        obs_shape=obs_shape,
        num_actions=num_actions,
        device=device,
        lr=LEARNING_RATE,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_epsilon=CLIP_EPSILON,
        value_coef=VALUE_COEF,
        entropy_coef=ENTROPY_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        num_epochs=NUM_EPOCHS,
    )

    buffer = RolloutBuffer(
        rollout_length=ROLLOUT_LENGTH,
        num_envs=NUM_ENVS,
        obs_shape=obs_shape,
        device=device,
    )

    episode_tracker = EpisodeTracker(NUM_ENVS)

    num_updates = TOTAL_TIMESTEPS // (NUM_ENVS * ROLLOUT_LENGTH)
    print(f"\nTraining configuration:")
    print(f"  Total timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"  Updates: {num_updates}")
    print(f"  Steps per update: {NUM_ENVS * ROLLOUT_LENGTH}")
    print(f"  Entropy coef: {ENTROPY_COEF}")
    print(f"  LR annealing: {LR_ANNEAL} (min {LR_MIN_FRAC*100}%)")
    print("=" * 60)

    global_step = 0
    start_time = time.time()
    best_score = float("-inf")
    best_win_rate = 0.0

    obs = vec_env.reset(seeds=[seed + i for i in range(NUM_ENVS)])

    for update in range(1, num_updates + 1):
        update_start = time.time()

        # Learning rate annealing (but don't go below minimum)
        if LR_ANNEAL:
            frac = 1.0 - (update - 1) / num_updates
            frac = max(frac, LR_MIN_FRAC)  # Don't go below minimum
            lr_now = LEARNING_RATE * frac
            for param_group in agent.optimizer.param_groups:
                param_group["lr"] = lr_now
        else:
            lr_now = LEARNING_RATE

        buffer.reset()

        for _ in range(ROLLOUT_LENGTH):
            actions, log_probs, values = agent.select_action(obs)
            next_obs, rewards, dones, infos = vec_env.step(actions)

            buffer.add(obs, actions, rewards, dones, values, log_probs)
            episode_tracker.update(rewards, dones, infos)

            obs = next_obs
            global_step += NUM_ENVS

        last_values = agent.get_value(obs)
        last_dones = np.zeros(NUM_ENVS, dtype=np.float32)
        buffer.compute_advantages(last_values, last_dones, GAMMA, GAE_LAMBDA)

        train_stats = agent.train(buffer, BATCH_SIZE)

        update_time = time.time() - update_start
        fps = (NUM_ENVS * ROLLOUT_LENGTH) / update_time

        if update % LOG_INTERVAL == 0:
            elapsed = time.time() - start_time
            ep_stats = episode_tracker.get_stats()

            # Check for entropy collapse warning
            entropy_warning = " [LOW!]" if train_stats['entropy'] < 0.5 else ""

            print(f"\n--- Update {update}/{num_updates} ---")
            print(f"Timesteps: {global_step:,} | FPS: {fps:.0f} | Time: {elapsed/60:.1f}m")
            print(f"Episodes: {ep_stats['episodes']} | "
                  f"Mean reward: {ep_stats['mean_reward']:.2f} | "
                  f"Mean score: {ep_stats['mean_score']:.2f}")
            print(f"Win rate: {ep_stats['win_rate']*100:.1f}% | "
                  f"Mean length: {ep_stats['mean_length']:.0f}")
            print(f"Movement: {ep_stats['movement_rate']*100:.1f}% | "
                  f"Rotation: {ep_stats['rotation_rate']*100:.1f}% | "
                  f"Spin penalties: {ep_stats['spin_penalties']:.1f}/ep")
            print(f"Policy loss: {train_stats['policy_loss']:.4f} | "
                  f"Value loss: {train_stats['value_loss']:.4f}")
            print(f"Entropy: {train_stats['entropy']:.4f}{entropy_warning} | "
                  f"KL: {train_stats['approx_kl']:.4f} | "
                  f"Clip frac: {train_stats['clip_fraction']:.3f}")
            print(f"Learning rate: {lr_now:.2e}")

        if update % SAVE_INTERVAL == 0:
            ep_stats = episode_tracker.get_stats()

            agent.save("ppo_tank_latest.pth")

            if ep_stats["mean_score"] > best_score and ep_stats["episodes"] > 10:
                best_score = ep_stats["mean_score"]
                agent.save("ppo_tank_best_score.pth")
                print(f"  -> New best score: {best_score:.2f}")

            if ep_stats["win_rate"] > best_win_rate and ep_stats["episodes"] > 10:
                best_win_rate = ep_stats["win_rate"]
                agent.save("ppo_tank_best_winrate.pth")
                print(f"  -> New best win rate: {best_win_rate*100:.1f}%")

    vec_env.close()

    agent.save("ppo_tank_final.pth")

    elapsed = time.time() - start_time
    ep_stats = episode_tracker.get_stats()

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Total time: {elapsed/3600:.2f} hours")
    print(f"Total timesteps: {global_step:,}")
    print(f"Total episodes: {ep_stats['episodes']}")
    print(f"Final mean score: {ep_stats['mean_score']:.2f}")
    print(f"Final win rate: {ep_stats['win_rate']*100:.1f}%")
    print(f"Best score achieved: {best_score:.2f}")
    print(f"Best win rate achieved: {best_win_rate*100:.1f}%")
    print("\nModels saved:")
    print("  - ppo_tank_latest.pth")
    print("  - ppo_tank_best_score.pth")
    print("  - ppo_tank_best_winrate.pth")
    print("  - ppo_tank_final.pth")


if __name__ == "__main__":
    main()
