# ppo/test.py
"""
Play against your trained PPO model.

Usage:
    python -m ppo.test
    python -m ppo.test --weights ppo_tank_best_score.pth

Controls:
    Movement: Arrow keys or WASD
    Fire: SPACE
    Quit: ESC
"""
import sys
import os
import argparse

import numpy as np
import pygame
import torch
import cv2

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pettingzoo.atari import combat_tank_v2
from ppo.model import ActorCritic

SCALE = 4        # Window scale (4x original size)
STACK_SIZE = 4   # Must match training


def preprocess_obs(frame: np.ndarray) -> np.ndarray:
    """RGB (H, W, 3) -> grayscale (1, 84, 84) in [0, 1]."""
    gray = frame.mean(axis=2).astype(np.float32) / 255.0
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return resized[np.newaxis, :, :].astype(np.float32)


def get_human_action():
    """
    Map keyboard to Combat Tank actions.

    Actions:
      0: NOOP, 1: FIRE, 2: UP, 3: RIGHT, 4: LEFT, 5: DOWN
      6: UPRIGHT, 7: UPLEFT, 8: DOWNRIGHT, 9: DOWNLEFT
      10-17: FIRE + direction
    """
    keys = pygame.key.get_pressed()

    up = keys[pygame.K_UP] or keys[pygame.K_w]
    down = keys[pygame.K_DOWN] or keys[pygame.K_s]
    left = keys[pygame.K_LEFT] or keys[pygame.K_a]
    right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
    fire = keys[pygame.K_SPACE]

    if not (up or down or left or right or fire):
        return 0  # NOOP

    up_only = up and not down
    down_only = down and not up
    left_only = left and not right
    right_only = right and not left

    if fire:
        if up_only and right_only:
            return 14  # FIRE_UPRIGHT
        if up_only and left_only:
            return 15  # FIRE_UPLEFT
        if up_only:
            return 10  # FIRE_UP
        if down_only and right_only:
            return 16  # FIRE_DOWNRIGHT
        if down_only and left_only:
            return 17  # FIRE_DOWNLEFT
        if down_only:
            return 13  # FIRE_DOWN
        if right_only:
            return 11  # FIRE_RIGHT
        if left_only:
            return 12  # FIRE_LEFT
        return 1  # FIRE

    if up_only and right_only:
        return 6  # UPRIGHT
    if up_only and left_only:
        return 7  # UPLEFT
    if up_only:
        return 2  # UP
    if down_only and right_only:
        return 8  # DOWNRIGHT
    if down_only and left_only:
        return 9  # DOWNLEFT
    if down_only:
        return 5  # DOWN
    if right_only:
        return 3  # RIGHT
    if left_only:
        return 4  # LEFT

    return 0  # Conflicting input


def get_model_action(model, stacked_obs: np.ndarray, device, deterministic: bool = True):
    """Get action from PPO model."""
    obs_t = torch.from_numpy(stacked_obs).unsqueeze(0).to(device)
    with torch.no_grad():
        policy_logits, _ = model(obs_t)
        if deterministic:
            action = policy_logits.argmax(dim=1).item()
        else:
            probs = torch.softmax(policy_logits, dim=-1)
            action = torch.multinomial(probs, 1).item()
    return int(action)


def main():
    parser = argparse.ArgumentParser(description="Play against trained PPO model")
    parser.add_argument(
        "--weights",
        type=str,
        default="ppo_tank_final.pth",
        help="Path to model weights",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy (sample actions) instead of greedy",
    )
    args = parser.parse_args()

    pygame.init()
    pygame.display.set_caption("Combat Tank - Human vs PPO")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create environment
    env = combat_tank_v2.parallel_env(
        obs_type="rgb_image",
        full_action_space=True,
        render_mode="rgb_array",
    )

    try:
        obs_dict, info = env.reset(seed=0)
    except TypeError:
        obs_dict = env.reset()

    agents = env.agents
    model_agent = "first_0"
    human_agent = "second_0"

    if model_agent not in agents or human_agent not in agents:
        model_agent, human_agent = agents[0], agents[1]

    # Initialize frame stack
    example_frame = obs_dict[model_agent]
    frame_proc = preprocess_obs(example_frame)
    model_stack = np.repeat(frame_proc, STACK_SIZE, axis=0)

    num_actions = env.action_space(model_agent).n

    # Load PPO model
    model = ActorCritic(in_channels=STACK_SIZE, num_actions=num_actions).to(device)

    weights_path = args.weights
    if not os.path.isabs(weights_path):
        # Check in current dir and parent dir
        if os.path.exists(weights_path):
            pass
        elif os.path.exists(os.path.join("..", weights_path)):
            weights_path = os.path.join("..", weights_path)
        else:
            print(f"Could not find weights file: {args.weights}")
            print("Available .pth files:")
            for f in os.listdir(".."):
                if f.endswith(".pth"):
                    print(f"  {f}")
            sys.exit(1)

    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded PPO model from {weights_path}")
    print(f"Model controls: {model_agent}, You control: {human_agent}")
    print(f"Policy: {'stochastic' if args.stochastic else 'deterministic'}")

    # Setup pygame window
    frame = env.render()
    base_h, base_w = frame.shape[0], frame.shape[1]
    win_w, win_h = base_w * SCALE, base_h * SCALE
    screen = pygame.display.set_mode((win_w, win_h))
    clock = pygame.time.Clock()

    print("\nControls: Arrow keys / WASD to move, SPACE to fire, ESC to quit")
    print("-" * 50)

    running = True
    games_played = 0
    human_wins = 0
    model_wins = 0

    while running:
        try:
            obs_dict, info = env.reset()
        except TypeError:
            obs_dict = env.reset()

        first_frame = obs_dict[model_agent]
        frame_proc = preprocess_obs(first_frame)
        model_stack = np.repeat(frame_proc, STACK_SIZE, axis=0)

        done = False
        model_score = 0.0
        human_score = 0.0

        while not done and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            if not running:
                break

            human_action = get_human_action()
            model_action = get_model_action(
                model, model_stack, device, deterministic=not args.stochastic
            )

            actions = {
                model_agent: model_action,
                human_agent: human_action,
            }

            obs_dict, rewards, terminations, truncations, infos = env.step(actions)

            model_score += float(rewards.get(model_agent, 0.0))
            human_score += float(rewards.get(human_agent, 0.0))

            new_frame = obs_dict[model_agent]
            new_proc = preprocess_obs(new_frame)
            model_stack = np.concatenate([model_stack[1:], new_proc], axis=0)

            done = bool(
                terminations.get(model_agent, False)
                or truncations.get(model_agent, False)
                or terminations.get(human_agent, False)
                or truncations.get(human_agent, False)
            )

            frame = env.render()
            if frame is not None:
                surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
                surf = pygame.transform.scale(surf, (win_w, win_h))
                screen.blit(surf, (0, 0))
                pygame.display.flip()

            clock.tick(30)

        if running:
            games_played += 1
            if human_score > model_score:
                human_wins += 1
                result = "YOU WIN!"
            elif model_score > human_score:
                model_wins += 1
                result = "Model wins"
            else:
                result = "Draw"

            print(
                f"Game {games_played}: {result} | "
                f"You: {human_score:.0f}, Model: {model_score:.0f} | "
                f"Overall: You {human_wins}-{model_wins} Model"
            )

    env.close()
    pygame.quit()

    print("\n" + "=" * 50)
    print(f"Final score: You {human_wins} - {model_wins} Model")


if __name__ == "__main__":
    main()
