# envs/single_agent_combat_tank.py

import gym
import numpy as np
from pettingzoo.atari import combat_tank_v2


class SingleAgentCombatTankShaped(gym.Env):
    """
    Gym-style wrapper around PettingZoo combat_tank_v3 with reward shaping.

    - Controls agent 'first_0'
    - Opponent 'second_0' currently acts randomly
    - Adds potential-based shaping to the reward:
        r'_t = r_env_t + lambda_phi * (gamma * Phi(s_{t+1}) - Phi(s_t))

      where Phi(s) is a scalar potential computed from the RGB frame that
      roughly encodes:
        * bullets near enemy tank are good (offense)
        * bullets far from our tank are good (defense)
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self,
                 gamma: float = 0.99,
                 lambda_phi: float = 0.1,
                 render_mode: str | None = None):
        super().__init__()

        # Underlying multi-agent PettingZoo env
        self.base_env = combat_tank_v2.parallel_env(
            obs_type="rgb_image",
            render_mode=render_mode,
        )

        # Which agent is "us" and which is the enemy
        self.control_agent = "first_0"
        self.enemy_agent = "second_0"

        # Expose single-agent spaces for control_agent
        self.observation_space = self.base_env.observation_space(self.control_agent)
        self.action_space = self.base_env.action_space(self.control_agent)

        # Shaping parameters
        self.gamma = gamma
        self.lambda_phi = lambda_phi
        self.prev_potential = 0.0
        self.prev_frame: np.ndarray | None = None

    # -------------------------
    # Standard Gym API: reset
    # -------------------------
    def reset(self, seed: int | None = None, options: dict | None = None):
        if options is None:
            options = {}

        obs_dict, info = self.base_env.reset(seed=seed, options=options)
        frame = obs_dict[self.control_agent]

        # Initialize potential from the first frame
        self.prev_frame = frame
        self.prev_potential = self._compute_potential(frame)

        # Gym v0-style: just return obs (no info).
        return frame

    # -------------------------
    # Standard Gym API: step
    # -------------------------
    def step(self, action):
        """
        One environment step for our single agent.

        - We pick an action for 'first_0' (the DQN agent).
        - We sample a random action for 'second_0' (the opponent).
        - Then we apply potential-based reward shaping to the reward.
        """

        # Random enemy for now
        enemy_action = self.base_env.action_space(self.enemy_agent).sample()

        actions = {
            self.control_agent: int(action),
            self.enemy_agent: int(enemy_action),
        }

        obs_dict, rewards, terminations, truncations, infos = self.base_env.step(actions)

        # Extract our stuff
        frame = obs_dict[self.control_agent]
        raw_reward = float(rewards[self.control_agent])

        done = bool(terminations[self.control_agent] or truncations[self.control_agent])
        info = infos[self.control_agent]

        # Potential-based shaping:
        #   r'_t = r_env_t + lambda_phi * (gamma * Phi(s_{t+1}) - Phi(s_t))
        phi_t = self.prev_potential
        phi_tp1 = self._compute_potential(frame)

        shaped_reward = raw_reward + self.lambda_phi * (self.gamma * phi_tp1 - phi_t)

        # Update stored state
        self.prev_frame = frame
        self.prev_potential = phi_tp1

        return frame, shaped_reward, done, info

    # ------------------------------------------------
    # Potential function: Phi(s) from RGB observation
    # ------------------------------------------------
    def _compute_potential(self, frame: np.ndarray) -> float:
        """
        Compute a scalar potential Phi(s) from the RGB frame.

        Rough idea:
        - Detect (very roughly):
            * our tank position
            * enemy tank position
            * a bullet position
        - Then:
            * reward bullet near enemy (offense)
            * reward bullet far from us (defense)

        This is deliberately hacky and noisy; it's just for shaping.
        Later, you can replace this with a CNN-based estimator.
        """

        # frame shape: (H, W, 3)
        h, w, _ = frame.shape

        # Quick grayscale for bullet detection
        gray = frame.mean(axis=2)

        # ---- 1) crude bullet detection: bright pixels ----
        # You *must* tune this threshold by looking at real frames.
        bullet_mask = gray > 220
        bullet_coords = np.argwhere(bullet_mask)  # (N, 2) as (y, x)

        # ---- 2) crude tank detection by color ----
        # You MUST adjust these color rules by inspecting combat_tank frames.
        r = frame[:, :, 0].astype(np.float32)
        g = frame[:, :, 1].astype(np.float32)
        b = frame[:, :, 2].astype(np.float32)

        # Placeholder: suppose our tank is "more greenish", enemy "more reddish"
        my_tank_mask = (g > 150) & (r < 180)
        enemy_tank_mask = (r > 150) & (g < 180)

        my_coords = np.argwhere(my_tank_mask)
        enemy_coords = np.argwhere(enemy_tank_mask)

        def center(coords: np.ndarray):
            """Return (x, y) center of mask coords, or None if empty."""
            if coords.shape[0] == 0:
                return None
            yx = coords.mean(axis=0)  # (y, x)
            return float(yx[1]), float(yx[0])  # (x, y)

        my_pos = center(my_coords)
        enemy_pos = center(enemy_coords)
        bullet_pos = center(bullet_coords)

        # If we couldn't detect needed objects, no shaping
        if my_pos is None or enemy_pos is None or bullet_pos is None:
            return 0.0

        def dist(p1, p2):
            return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

        # Distances
        d_bullet_me = dist(bullet_pos, my_pos)
        d_bullet_enemy = dist(bullet_pos, enemy_pos)

        # Normalize by diagonal for scale invariance
        diag = np.sqrt(h**2 + w**2)
        d_bullet_me /= diag
        d_bullet_enemy /= diag

        # Offensive potential: bullet close to enemy is good
        # (smaller distance => larger potential)
        phi_off = -d_bullet_enemy

        # Defensive potential: bullet far from us is good
        phi_def = d_bullet_me

        # Combine with weights (can tune)
        alpha = 1.0
        beta = 1.0

        phi = alpha * phi_off + beta * phi_def
        return float(phi)
