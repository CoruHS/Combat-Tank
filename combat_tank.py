# combat_tank.py
import gym
import numpy as np
from pettingzoo.atari import combat_tank_v2


class SingleAgentCombatTankShaped(gym.Env):
    """
    Simpler shaped wrapper for combat_tank_v2.

    - Controls agent 'first_0'
    - Opponent 'second_0' acts randomly

    final_reward =
        raw_env_reward (with a bit more negative on death)
      + movement_shaping         (encourage forward translation)
      + repeat_shaping           (discourage repeating the same action forever)
      + fire_shaping             (encourage shooting, esp. when close)
      + turn_shaping             (penalize excessive turning)
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(
        self,
        gamma: float = 0.99,
        lambda_phi: float = 0.0,  # kept for API compatibility, but not used
        render_mode: str | None = None,
    ):
        super().__init__()

        # Underlying multi-agent PettingZoo env
        self.base_env = combat_tank_v2.parallel_env(
            obs_type="rgb_image",
            render_mode=render_mode,
        )

        self.control_agent = "first_0"
        self.enemy_agent = "second_0"

        self.observation_space = self.base_env.observation_space(self.control_agent)
        self.action_space = self.base_env.action_space(self.control_agent)

        self.gamma = gamma

        # ---- Movement / position tracking ----
        self.prev_my_pos = None

        # ---- Action repetition tracking ----
        self.last_action = None
        self.action_repeat_count = 0
        self.repeat_threshold = 10       # how many same actions in a row is "ok"
        self.repeat_penalty = 0.02       # penalty per step beyond threshold
        self.spin_extra_penalty = 0.02   # extra if also not moving

        # ---- Movement shaping ----
        # Movement is measured as distance moved / screen diagonal
        self.still_move_threshold = 0.005     # normalized distance; below = "still"
        self.still_penalty = 0.01             # penalty for staying in place

        # Forward preference (W / Up)
        # Minimal action meanings for combat_tank are typically:
        # 0: NOOP, 1: FIRE, 2: UP, 3: RIGHT, 4: LEFT, 5: DOWN
        self.forward_action = 2
        # Reward when pressing forward AND moving
        self.move_forward_bonus = 0.02
        # Small reward for pressing forward even if not yet moving
        self.forward_press_bonus = 0.003

        # ---- Turn shaping ----
        # Penalize turning to avoid orbit meta
        self.TURN_ACTIONS = {3, 4}   # RIGHT, LEFT
        self.turn_penalty = 0.01     # per-step penalty when turning

        # ---- Fire shaping ----
        # Fire actions in full action space; we only care about plain FIRE in minimal
        # but keeping the set for compatibility if you ever switch to full_action_space.
        self.FIRE_ACTIONS = {1, 11, 12, 13, 14, 15, 16, 17}
        self.steps_since_fire = 0
        self.fire_bonus = 0.01            # small bonus when firing
        self.close_no_fire_penalty = 0.02 # penalty if close & refusing to fire
        self.fire_range = 0.25            # "close" tank distance (normalized)
        self.fire_wait = 10               # steps allowed without firing when close

        # Clip final shaped reward to keep things sane
        self.reward_clip = 2.0

    # -------------------------
    # reset
    # -------------------------
    def reset(self, seed: int | None = None, options: dict | None = None):
        if options is None:
            options = {}

        obs_dict, info = self.base_env.reset(seed=seed, options=options)
        frame = obs_dict[self.control_agent]

        my_pos, enemy_pos = self._detect_tanks(frame)
        self.prev_my_pos = my_pos

        self.last_action = None
        self.action_repeat_count = 0
        self.steps_since_fire = 0

        return frame

    # -------------------------
    # step
    # -------------------------
    def step(self, action):
        # Random enemy policy
        enemy_action = self.base_env.action_space(self.enemy_agent).sample()

        actions = {
            self.control_agent: int(action),
            self.enemy_agent: int(enemy_action),
        }

        obs_dict, rewards, terminations, truncations, infos = self.base_env.step(actions)

        frame = obs_dict[self.control_agent]
        raw_reward = float(rewards[self.control_agent])

        # Make getting hit slightly worse
        if raw_reward < 0.0:
            raw_reward *= 2.0

        done = bool(terminations[self.control_agent] or truncations[self.control_agent])
        info = infos[self.control_agent]

        # --- positions + movement ---
        my_pos, enemy_pos = self._detect_tanks(frame)

        if my_pos is not None and self.prev_my_pos is not None:
            move_dist = self._dist(my_pos, self.prev_my_pos)
        else:
            move_dist = 0.0

        h, w, _ = frame.shape
        diag = np.sqrt(h**2 + w**2) if (h > 0 and w > 0) else 1.0
        move_norm = move_dist / diag

        a = int(action)

        # --- movement + forward shaping ---
        move_shaping = 0.0
        if move_norm < self.still_move_threshold:
            # standing still / spinning in place
            if a == self.forward_action:
                # pressing forward but not really moving:
                # still somewhat bad, but less bad than other actions
                move_shaping -= self.still_penalty * 0.3
                move_shaping += self.forward_press_bonus
            else:
                move_shaping -= self.still_penalty
        else:
            # actually translating
            if a == self.forward_action:
                # moving AND pressing forward = clearly good
                move_shaping += self.move_forward_bonus + self.forward_press_bonus
            else:
                # moving with non-forward action: no bonus
                move_shaping += 0.0

        # --- turn shaping ---
        turn_shaping = 0.0
        if a in self.TURN_ACTIONS:
            # per-step penalty for turning to discourage endless orbiting
            turn_shaping -= self.turn_penalty

        # --- repeat shaping (bounded; forward punished less) ---
        if self.last_action is None or self.last_action != a:
            self.last_action = a
            self.action_repeat_count = 1
        else:
            self.action_repeat_count += 1

        repeat_shaping = 0.0
        if self.action_repeat_count > self.repeat_threshold:
            if a == self.forward_action:
                # holding forward: light penalty only
                repeat_shaping -= 0.5 * self.repeat_penalty
            else:
                # repeating other actions: normal penalty
                repeat_shaping -= self.repeat_penalty
                # extra if you're also not moving (spinbot)
                if move_norm < self.still_move_threshold and a not in self.FIRE_ACTIONS:
                    repeat_shaping -= self.spin_extra_penalty

        # --- fire shaping ---
        fire_shaping = 0.0
        if a in self.FIRE_ACTIONS:
            # we fired this step
            fire_shaping += self.fire_bonus
            self.steps_since_fire = 0
        else:
            self.steps_since_fire += 1

        if my_pos is not None and enemy_pos is not None:
            d_tanks = self._dist(my_pos, enemy_pos) / diag
            # if we're close, haven't fired in a while, and *still* not firing -> penalty
            if (
                d_tanks < self.fire_range
                and self.steps_since_fire > self.fire_wait
                and a not in self.FIRE_ACTIONS
            ):
                fire_shaping -= self.close_no_fire_penalty

        # --- combine ---
        final_reward = (
            raw_reward
            + move_shaping
            + turn_shaping
            + repeat_shaping
            + fire_shaping
        )
        final_reward = float(np.clip(final_reward, -self.reward_clip, self.reward_clip))

        # update state
        self.prev_my_pos = my_pos

        return frame, final_reward, done, info

    # -------------------------
    # helpers
    # -------------------------
    @staticmethod
    def _dist(p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def _detect_tanks(self, frame: np.ndarray):
        """
        Very crude detection of our tank vs enemy tank based on colors.
        Just returns centers; no bullets, no fancy stuff.
        """
        r = frame[:, :, 0].astype(np.float32)
        g = frame[:, :, 1].astype(np.float32)

        # These thresholds are guesses; tweak if needed based on actual frames.
        my_tank_mask = (g > 150) & (r < 180)
        enemy_tank_mask = (r > 150) & (g < 180)

        my_coords = np.argwhere(my_tank_mask)
        enemy_coords = np.argwhere(enemy_tank_mask)

        def center(coords):
            if coords.shape[0] == 0:
                return None
            yx = coords.mean(axis=0)  # (y, x)
            return float(yx[1]), float(yx[0])  # (x, y)

        my_pos = center(my_coords)
        enemy_pos = center(enemy_coords)
        return my_pos, enemy_pos
