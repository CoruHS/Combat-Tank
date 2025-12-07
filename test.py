import sys
import numpy as np
import pygame
import torch
import cv2  # pip install opencv-python

from pettingzoo.atari import combat_tank_v2
from model import SimpleDQNCNN

WEIGHTS_PATH = "cnn_dqn_combat_tank_single_agent_shaped.pth"
SCALE = 4        # how big the game window is (4x original size)
STACK_SIZE = 4   # must match training


def preprocess_obs(frame: np.ndarray) -> np.ndarray:
    """
    Same preprocessing as training:
    RGB (H, W, 3) -> grayscale 84x84 -> (1, 84, 84) float32 in [0, 1]
    """
    gray = frame.mean(axis=2).astype(np.float32) / 255.0
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    obs_chw = resized[np.newaxis, :, :]  # (1, 84, 84)
    return obs_chw.astype(np.float32)


def get_human_action():
    """
    Map keyboard input to Combat Tank full_action_space indices.

    Controls:
      Movement:  Arrow keys OR WASD
      Fire:      SPACE

    Action mapping (approx Atari full_action_space for Combat):
      0: NOOP
      1: FIRE
      2: UP
      3: RIGHT
      4: LEFT
      5: DOWN
      6: UPRIGHT
      7: UPLEFT
      8: DOWNRIGHT
      9: DOWNLEFT
      10: FIRE_UP
      11: FIRE_RIGHT
      12: FIRE_LEFT
      13: FIRE_DOWN
      14: FIRE_UPRIGHT
      15: FIRE_UPLEFT
      16: FIRE_DOWNRIGHT
      17: FIRE_DOWNLEFT
    """
    keys = pygame.key.get_pressed()

    up = keys[pygame.K_UP] or keys[pygame.K_w]
    down = keys[pygame.K_DOWN] or keys[pygame.K_s]
    left = keys[pygame.K_LEFT] or keys[pygame.K_a]
    right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
    fire = keys[pygame.K_SPACE]

    # No input -> NOOP
    if not (up or down or left or right or fire):
        return 0

    up_only = up and not down
    down_only = down and not up
    left_only = left and not right
    right_only = right and not left

    # Fire controls
    if fire:
        if up_only:
            if right_only:
                return 14  # FIRE_UPRIGHT
            if left_only:
                return 15  # FIRE_UPLEFT
            return 10      # FIRE_UP

        if down_only:
            if right_only:
                return 16  # FIRE_DOWNRIGHT
            if left_only:
                return 17  # FIRE_DOWNLEFT
            return 13      # FIRE_DOWN

        if right_only:
            return 11      # FIRE_RIGHT
        if left_only:
            return 12      # FIRE_LEFT

        # Fire with no direction keys -> straight FIRE
        return 1

    # Movement (no fire)
    if up_only:
        if right_only:
            return 6   # UPRIGHT
        if left_only:
            return 7   # UPLEFT
        return 2       # UP

    if down_only:
        if right_only:
            return 8   # DOWNRIGHT
        if left_only:
            return 9   # DOWNLEFT
        return 5       # DOWN

    if right_only:
        return 3       # RIGHT
    if left_only:
        return 4       # LEFT

    # Conflicting input (e.g. up+down): just NOOP
    return 0


def get_model_action(model, stacked_obs: np.ndarray, device):
    """
    Greedy action from trained CNN DQN given the stacked observation.

    stacked_obs: (4, 84, 84) numpy array, float32 in [0, 1]
    """
    obs_t = torch.from_numpy(stacked_obs).unsqueeze(0).to(device)  # (1, 4, 84, 84)
    with torch.no_grad():
        q_values = model(obs_t)
    return int(q_values.argmax(dim=1).item())


def main():
    pygame.init()
    pygame.display.set_caption("Combat Tank - Human vs DQN")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Create parallel PettingZoo env (both agents visible) ---
    env = combat_tank_v2.parallel_env(
        obs_type="rgb_image",
        full_action_space=True,
        render_mode="rgb_array",
    )

    # Reset env
    try:
        obs_dict, info = env.reset(seed=0)
    except TypeError:
        obs_dict = env.reset()

    agents = env.agents
    if len(agents) < 2:
        print("Expected at least 2 agents in combat_tank_v2.")
        env.close()
        pygame.quit()
        sys.exit(1)

    # Decide who is who
    # We'll let the model control 'first_0' and the human control 'second_0'
    model_agent = "first_0"
    human_agent = "second_0"
    if model_agent not in agents or human_agent not in agents:
        # Fallback: just use the first two agents
        model_agent, human_agent = agents[0], agents[1]

    # --- Build frame stack for model agent and load trained model ---
    example_frame = obs_dict[model_agent]          # (H, W, 3)
    frame_proc = preprocess_obs(example_frame)     # (1, 84, 84)
    model_stack = np.repeat(frame_proc, STACK_SIZE, axis=0)  # (4, 84, 84)

    num_actions = env.action_space(model_agent).n

    # Model was trained with in_channels = 4
    model = SimpleDQNCNN(in_channels=STACK_SIZE, num_actions=num_actions).to(device)
    state_dict = torch.load(WEIGHTS_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded trained model from {WEIGHTS_PATH}")
    print(f"Model controls: {model_agent}, You control: {human_agent}")

    # --- Set up pygame window based on rendered frame ---
    frame = env.render()  # (H, W, 3)
    if frame is None:
        print("env.render() returned None. Make sure render_mode='rgb_array'.")
        env.close()
        pygame.quit()
        sys.exit(1)

    base_h, base_w = frame.shape[0], frame.shape[1]
    win_w, win_h = base_w * SCALE, base_h * SCALE
    screen = pygame.display.set_mode((win_w, win_h))
    clock = pygame.time.Clock()

    running = True
    print("Controls: Arrow keys / WASD to move, SPACE to fire. ESC or close window to quit.")

    # --- Play episodes forever until user quits ---
    while running:
        # Start a new episode
        try:
            obs_dict, info = env.reset()
        except TypeError:
            obs_dict = env.reset()

        # Reset model's frame stack using first observation of this episode
        first_model_frame = obs_dict[model_agent]
        frame_proc = preprocess_obs(first_model_frame)         # (1, 84, 84)
        model_stack = np.repeat(frame_proc, STACK_SIZE, axis=0)  # (4, 84, 84)

        done = False
        episode_reward_model = 0.0
        episode_reward_human = 0.0

        while not done and running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            if not running:
                break

            # Get current obs for each agent
            model_frame = obs_dict[model_agent]
            human_frame = obs_dict[human_agent]  # not actually needed, but here if you want HUD later

            # Human action from keyboard
            human_action = get_human_action()

            # Model action from stacked CNN input
            model_action = get_model_action(model, model_stack, device)

            actions = {
                model_agent: int(model_action),
                human_agent: int(human_action),
            }

            obs_dict, rewards, terminations, truncations, infos = env.step(actions)

            # Accumulate rewards (raw env rewards, not shaped)
            episode_reward_model += float(rewards.get(model_agent, 0.0))
            episode_reward_human += float(rewards.get(human_agent, 0.0))

            # Update model frame stack with new frame
            new_model_frame = obs_dict[model_agent]
            new_proc = preprocess_obs(new_model_frame)  # (1, 84, 84)
            model_stack = np.concatenate([model_stack[1:], new_proc], axis=0)  # (4, 84, 84)

            # Check if episode ended
            done_model = bool(terminations.get(model_agent, False) or truncations.get(model_agent, False))
            done_human = bool(terminations.get(human_agent, False) or truncations.get(human_agent, False))
            done = done_model or done_human

            # Render frame
            frame = env.render()
            if frame is not None:
                surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
                surf = pygame.transform.scale(surf, (win_w, win_h))
                screen.blit(surf, (0, 0))
                pygame.display.flip()

            clock.tick(30)  # Limit FPS so humans can react

        if running:
            print(
                f"Episode finished | model({model_agent}) reward: {episode_reward_model:.2f}, "
                f"you({human_agent}) reward: {episode_reward_human:.2f}"
            )

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
