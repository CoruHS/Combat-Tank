# main_single_agent.py
import random
from collections import defaultdict

import numpy as np
import torch
import cv2  # pip install opencv-python

from combat_env import SingleAgentCombatTankShaped
from replay_buffer import ReplayBuffer
from agent import DQNAgent


def preprocess_obs(frame: np.ndarray) -> np.ndarray:
    """
    Convert RGB frame (H, W, 3) from the env into
    a CNN-friendly tensor (C, 84, 84), with C=1 (grayscale), values in [0, 1].
    """
    # frame: uint8 [0,255], (H, W, 3)
    gray = frame.mean(axis=2)  # (H, W), float64
    gray = gray.astype(np.float32) / 255.0

    # Resize to 84x84 (like Atari DQN)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)  # (84, 84)

    # Add channel dim: (1, 84, 84)
    obs_chw = resized[np.newaxis, :, :]
    return obs_chw.astype(np.float32)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Training budget ----
    total_env_steps = 1_000_000  # total steps of env.step()

    # ---- Hyperparams ----
    replay_capacity = 10_000
    batch_size = 32
    start_learning_after = 1_000
    train_every = 4
    gamma = 0.99
    lr = 1e-4
    epsilon_start = 1.0
    epsilon_end = 0.1
    epsilon_decay_steps = total_env_steps

    # ---- Env setup (your single-agent shaped env) ----
    env = SingleAgentCombatTankShaped(
        gamma=gamma,
        lambda_phi=0.1,
        render_mode=None,   # no rendering during training
    )

    # Reset and get first observation
    obs_frame = env.reset(seed=0)           # (H, W, 3)
    obs = preprocess_obs(obs_frame)         # (1, 84, 84)
    obs_shape = obs.shape                   # (C, H, W)
    num_actions = env.action_space.n

    # ---- Agent + replay buffer ----
    agent = DQNAgent(
        obs_shape=obs_shape,
        num_actions=num_actions,
        device=device,
        gamma=gamma,
        lr=lr,
        target_update_freq=1000,
    )

    replay_buffer = ReplayBuffer(replay_capacity, obs_shape)

    global_step = 0
    episode_idx = 0

    while global_step < total_env_steps:
        obs_frame = env.reset()
        obs = preprocess_obs(obs_frame)

        episode_idx += 1
        episode_start_step = global_step

        episode_return = 0.0

        done = False
        last_obs = obs
        while not done:
            # Epsilon-greedy
            epsilon = max(
                epsilon_end,
                epsilon_start
                - (epsilon_start - epsilon_end) * (global_step / epsilon_decay_steps),
            )

            action = agent.select_action(last_obs, epsilon)

            # Step env
            next_frame, shaped_reward, done, info = env.step(action)
            next_obs = preprocess_obs(next_frame)

            # Store transition with shaped reward
            replay_buffer.add(
                last_obs,
                action,
                shaped_reward,
                next_obs,
                done,
            )

            episode_return += shaped_reward
            global_step += 1
            last_obs = next_obs

            # Train DQN
            if len(replay_buffer) > start_learning_after and global_step % train_every == 0:
                batch = replay_buffer.sample(batch_size)
                loss = agent.train_step(batch, batch_size)

            if global_step >= total_env_steps:
                break

        episode_steps = global_step - episode_start_step

        print(
            f"Episode {episode_idx} | steps: {episode_steps} | "
            f"shaped_return: {episode_return:.2f} | global_step: {global_step}"
        )

    # ---- Save trained model ----
    save_path = "cnn_dqn_combat_tank_single_agent_shaped.pth"
    torch.save(agent.q_net.state_dict(), save_path)
    print(f"Saved trained model to {save_path}")


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    main()
