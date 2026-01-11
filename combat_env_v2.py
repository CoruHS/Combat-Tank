# ppo/combat_env_v2.py
"""
Combat Tank environment v2 - Aggressive reward shaping.

Key insight from failed training:
- The agent learned to spin+shoot, getting +280 shaped reward but -0.4 game score
- Fire bonus (+0.3) rewarded passive "turret" play
- Approach reward (delta-based) gave 0 for staying still - no penalty for passivity
- Entropy collapsed to 0.03 immediately, locking in the bad policy

This version fixes these issues:
1. DISTANCE PENALTY: Actively penalize being far from enemy (not just reward approaching)
2. VELOCITY REQUIREMENT: Only reward when moving toward enemy, penalize stillness
3. CONDITIONAL FIRE BONUS: Only when approaching, not when stationary
4. SPINNING DETECTION: Detect and penalize rotation without translation
5. APPROACH MOMENTUM: Track recent movement, reward sustained approach
"""

import gym
import numpy as np
import cv2
import torch
from collections import deque

from pettingzoo.atari import combat_tank_v2


class CombatTankEnvV2(gym.Env):
    """
    Combat Tank with aggressive anti-passive reward shaping.
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(
        self,
        frame_skip: int = 2,  # Reduced from 4 for more responsive control
        render_mode: str | None = None,
        enemy_model_path: str | None = None,
        enemy_device: str | torch.device = "cpu",
        debug: bool = False,
    ):
        super().__init__()

        self.frame_skip = frame_skip
        self.debug = debug

        self.base_env = combat_tank_v2.parallel_env(
            obs_type="rgb_image",
            render_mode=render_mode,
        )

        self.control_agent = "first_0"
        self.enemy_agent = "second_0"

        self.observation_space = self.base_env.observation_space(self.control_agent)
        self.action_space = self.base_env.action_space(self.control_agent)

        # Enemy policy
        self.enemy_device = torch.device(enemy_device)
        self.enemy_model = None
        self.enemy_stack = None
        self.enemy_is_ppo = False

        if enemy_model_path is not None:
            import sys
            import os
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            num_enemy_actions = self.base_env.action_space(self.enemy_agent).n

            # Detect model type by filename
            if "ppo" in enemy_model_path.lower():
                from ppo.model import ActorCritic
                self.enemy_model = ActorCritic(
                    in_channels=4,
                    num_actions=num_enemy_actions,
                ).to(self.enemy_device)
                self.enemy_is_ppo = True
                print(f"[CombatTankEnvV2] Loading PPO enemy from {enemy_model_path}")
            else:
                from model import SimpleDQNCNN
                self.enemy_model = SimpleDQNCNN(
                    in_channels=4,
                    num_actions=num_enemy_actions,
                ).to(self.enemy_device)
                print(f"[CombatTankEnvV2] Loading DQN enemy from {enemy_model_path}")

            state_dict = torch.load(enemy_model_path, map_location=self.enemy_device)
            self.enemy_model.load_state_dict(state_dict)
            self.enemy_model.eval()

        # Position tracking
        self.prev_my_pos = None
        self.prev_enemy_pos = None
        self.prev_enemy_dist = None

        # Movement history for approach momentum
        self.recent_deltas = deque(maxlen=10)  # Track last 10 distance changes

        # Spinning detection
        self.prev_positions = deque(maxlen=8)  # Track positions for spin detection

        # Score tracking
        self.my_score = 0.0
        self.enemy_score = 0.0
        self.total_steps = 0

        # ============================================================
        # REWARD COEFFICIENTS - ANTI-PASSIVE DESIGN
        # ============================================================

        # Raw game rewards
        self.kill_reward = 5.0      # Reduced from 10 (let shaping guide learning)
        self.death_penalty = 5.0    # Symmetric

        # DISTANCE PENALTY (KEY CHANGE: penalize being far, not just reward approaching)
        # This makes passive play actively BAD
        self.distance_penalty_coef = 0.05  # Penalty per step scaled by distance
        self.max_penalty_dist = 0.5        # Distance above which max penalty applies

        # APPROACH REWARD (only when actively closing distance)
        self.approach_coef = 2.0           # Strong reward for approaching
        self.min_approach_delta = 0.001    # Minimum delta to count as "approaching"

        # APPROACH MOMENTUM (reward sustained approaching)
        self.momentum_bonus = 0.1          # Bonus when recent history shows approaching
        self.momentum_threshold = 5        # Need 5+ approaching steps in last 10

        # VELOCITY PENALTY (penalize low velocity toward enemy)
        self.stationary_penalty = 0.02     # Penalty for not moving toward enemy

        # SPINNING PENALTY (action-based, not position-based)
        self.spin_penalty = 0.1            # Strong penalty for rotation-only actions
        # DOWN does nothing in tank! Only UP moves forward. DOWN variants are useless.
        self.ROTATION_ACTIONS = {3, 4, 8, 9, 11, 12, 16, 17}  # RIGHT, LEFT, DOWNRIGHT, DOWNLEFT, and fire variants
        self.FORWARD_ACTIONS = {2, 6, 7, 10, 14, 15}  # Only UP-based actions actually move
        self.rotation_count = 0
        self.rotation_threshold = 3        # Penalize after 3 consecutive rotations

        # MOVEMENT BONUS (reward for actually moving forward)
        # Only UP-based actions move the tank! DOWN does nothing.
        self.MOVEMENT_ACTIONS = {2, 6, 7, 10, 14, 15}  # UP, UPRIGHT, UPLEFT, UPFIRE, UPRIGHTFIRE, UPLEFTFIRE
        self.movement_bonus = 0.08         # Reward for moving forward
        self.move_and_fire_bonus = 0.15    # Extra reward for firing WHILE moving (strafing)
        self.MOVE_AND_FIRE_ACTIONS = {10, 14, 15}  # UPFIRE, UPRIGHTFIRE, UPLEFTFIRE (actual movement + fire)

        # FIRE BONUS (conditional on movement)
        self.FIRE_ACTIONS = {1, 10, 11, 12, 13, 14, 15, 16, 17}
        self.fire_range = 0.20             # Tighter range
        self.fire_bonus_moving = 0.3       # Fire bonus when approaching
        self.fire_bonus_stationary = 0.0   # NO bonus for stationary firing

        # ENGAGEMENT ZONE BONUS
        self.engagement_dist = 0.15        # Close combat distance
        self.engagement_bonus = 0.05       # Per-step bonus for being in engagement range

        self.reward_clip = 5.0

        # Debug counters
        self.debug_detection_failures = 0
        self.debug_total_steps = 0

        # Spin tracking counters (per episode)
        self.spin_penalties_triggered = 0  # Times penalty was applied
        self.total_rotation_actions = 0    # Total rotation-only actions taken
        self.total_movement_actions = 0    # Total movement actions taken

    @staticmethod
    def _preprocess_obs(frame: np.ndarray) -> np.ndarray:
        gray = frame.mean(axis=2).astype(np.float32) / 255.0
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized[np.newaxis, :, :].astype(np.float32)

    def reset(self, seed: int | None = None, options: dict | None = None):
        if options is None:
            options = {}

        obs_dict, info = self.base_env.reset(seed=seed, options=options)
        frame = obs_dict[self.control_agent]

        my_pos, enemy_pos = self._detect_tanks(frame)
        self.prev_my_pos = my_pos
        self.prev_enemy_pos = enemy_pos

        if my_pos is not None and enemy_pos is not None:
            h, w, _ = frame.shape
            diag = np.sqrt(h**2 + w**2)
            self.prev_enemy_dist = self._dist(my_pos, enemy_pos) / diag
        else:
            self.prev_enemy_dist = None

        self.recent_deltas.clear()
        self.prev_positions.clear()
        self.rotation_count = 0

        # Reset spin tracking
        self.spin_penalties_triggered = 0
        self.total_rotation_actions = 0
        self.total_movement_actions = 0

        self.my_score = 0.0
        self.enemy_score = 0.0
        self.total_steps = 0

        if self.enemy_model is not None:
            enemy_frame = obs_dict[self.enemy_agent]
            enemy_proc = self._preprocess_obs(enemy_frame)
            self.enemy_stack = np.repeat(enemy_proc, 4, axis=0)

        return frame

    def step(self, action):
        total_reward = 0.0
        frame = None
        done = False

        for _ in range(self.frame_skip):
            if done:
                break

            # Enemy action
            if self.enemy_model is not None and self.enemy_stack is not None:
                obs_t = torch.from_numpy(self.enemy_stack).unsqueeze(0).to(self.enemy_device)
                with torch.no_grad():
                    if self.enemy_is_ppo:
                        # PPO model returns (logits, value)
                        logits, _ = self.enemy_model(obs_t)
                        enemy_action = int(logits.argmax(dim=1).item())
                    else:
                        # DQN model returns Q-values
                        q_values = self.enemy_model(obs_t)
                        enemy_action = int(q_values.argmax(dim=1).item())
            else:
                enemy_action = self.base_env.action_space(self.enemy_agent).sample()

            actions = {
                self.control_agent: int(action),
                self.enemy_agent: enemy_action,
            }

            obs_dict, rewards, terminations, truncations, infos = self.base_env.step(actions)
            frame = obs_dict[self.control_agent]

            raw_self = float(rewards[self.control_agent])
            raw_enemy = float(rewards[self.enemy_agent])
            self.my_score += raw_self
            self.enemy_score += raw_enemy

            if raw_self > 0:
                total_reward += raw_self * self.kill_reward
            elif raw_self < 0:
                total_reward += raw_self * self.death_penalty

            done = bool(terminations[self.control_agent] or truncations[self.control_agent])

            if self.enemy_model is not None:
                enemy_frame = obs_dict[self.enemy_agent]
                enemy_proc = self._preprocess_obs(enemy_frame)
                self.enemy_stack = np.concatenate([self.enemy_stack[1:], enemy_proc], axis=0)

        self.total_steps += 1
        self.debug_total_steps += 1

        # Detect positions
        my_pos, enemy_pos = self._detect_tanks(frame)
        h, w, _ = frame.shape
        diag = np.sqrt(h**2 + w**2)

        current_dist = None
        delta_dist = 0.0
        is_approaching = False

        if my_pos is not None and enemy_pos is not None:
            current_dist = self._dist(my_pos, enemy_pos) / diag

            if self.prev_enemy_dist is not None:
                delta_dist = self.prev_enemy_dist - current_dist  # Positive = approaching
                is_approaching = delta_dist > self.min_approach_delta
                self.recent_deltas.append(delta_dist)

            # Track position for spin detection
            self.prev_positions.append(my_pos)

            self.prev_enemy_dist = current_dist
        else:
            self.debug_detection_failures += 1
            # Don't update prev_enemy_dist if detection failed

        # ============================================================
        # REWARD SHAPING
        # ============================================================

        # 1. DISTANCE PENALTY (always active - makes being far BAD)
        distance_penalty = 0.0
        if current_dist is not None:
            # Penalty scales with distance, capped at max_penalty_dist
            penalty_dist = min(current_dist, self.max_penalty_dist)
            distance_penalty = -self.distance_penalty_coef * penalty_dist
        total_reward += distance_penalty

        # 2. APPROACH REWARD (only when actively moving toward enemy)
        approach_reward = 0.0
        if is_approaching and current_dist is not None:
            approach_reward = self.approach_coef * delta_dist
        total_reward += approach_reward

        # 3. APPROACH MOMENTUM BONUS (sustained approaching)
        momentum_bonus = 0.0
        if len(self.recent_deltas) >= self.momentum_threshold:
            approaching_count = sum(1 for d in self.recent_deltas if d > self.min_approach_delta)
            if approaching_count >= self.momentum_threshold:
                momentum_bonus = self.momentum_bonus
        total_reward += momentum_bonus

        # 4. STATIONARY PENALTY (not moving toward enemy)
        stationary_penalty = 0.0
        if not is_approaching and current_dist is not None:
            stationary_penalty = -self.stationary_penalty
        total_reward += stationary_penalty

        # 5. SPINNING PENALTY (action-based)
        spin_penalty = 0.0
        a = int(action)

        # Track rotation-only actions
        if a in self.ROTATION_ACTIONS:
            self.rotation_count += 1
            self.total_rotation_actions += 1  # Track total rotations
        elif a in self.FORWARD_ACTIONS:
            self.rotation_count = 0  # Reset if moving forward

        # Penalize excessive rotation without forward movement
        if self.rotation_count > self.rotation_threshold:
            spin_penalty = -self.spin_penalty * (self.rotation_count - self.rotation_threshold)
            spin_penalty = max(spin_penalty, -0.5)  # Cap the penalty
            self.spin_penalties_triggered += 1  # Track penalty triggers
        total_reward += spin_penalty

        # 6. CONDITIONAL FIRE BONUS
        fire_bonus = 0.0
        fired = a in self.FIRE_ACTIONS

        if fired and current_dist is not None and current_dist < self.fire_range:
            if is_approaching or (len(self.recent_deltas) > 0 and sum(self.recent_deltas) > 0):
                # Only bonus if approaching or has been approaching recently
                fire_bonus = self.fire_bonus_moving
            else:
                # No bonus for stationary firing
                fire_bonus = self.fire_bonus_stationary
        total_reward += fire_bonus

        # 7. MOVEMENT BONUS (reward for moving, especially while firing)
        movement_bonus = 0.0
        if a in self.MOVE_AND_FIRE_ACTIONS:
            # Best action: moving AND firing (strafing/dodging while shooting)
            movement_bonus = self.move_and_fire_bonus
            self.total_movement_actions += 1
        elif a in self.MOVEMENT_ACTIONS:
            # Good action: just moving
            movement_bonus = self.movement_bonus
            self.total_movement_actions += 1
        total_reward += movement_bonus

        # 8. ENGAGEMENT ZONE BONUS (being close is good)
        engagement_bonus = 0.0
        if current_dist is not None and current_dist < self.engagement_dist:
            engagement_bonus = self.engagement_bonus
        total_reward += engagement_bonus

        # Clip reward
        total_reward = float(np.clip(total_reward, -self.reward_clip, self.reward_clip))

        # Update previous position
        self.prev_my_pos = my_pos
        self.prev_enemy_pos = enemy_pos

        # Debug output
        if self.debug and self.total_steps % 500 == 0:
            detection_rate = 1.0 - (self.debug_detection_failures / max(1, self.debug_total_steps))
            print(f"[DEBUG] Step {self.total_steps}: dist={current_dist:.3f if current_dist else -1}, "
                  f"approaching={is_approaching}, delta={delta_dist:.4f}, "
                  f"detection_rate={detection_rate:.2%}")

        info = {
            "my_raw_score": self.my_score,
            "enemy_raw_score": self.enemy_score,
            "distance_to_enemy": current_dist if current_dist is not None else -1,
            "is_approaching": is_approaching,
            "spin_penalty": spin_penalty,
            "spin_penalties_triggered": self.spin_penalties_triggered,
            "total_rotation_actions": self.total_rotation_actions,
            "rotation_rate": self.total_rotation_actions / max(1, self.total_steps),
            "total_movement_actions": self.total_movement_actions,
            "movement_rate": self.total_movement_actions / max(1, self.total_steps),
            "components": {
                "distance_penalty": distance_penalty,
                "approach_reward": approach_reward,
                "momentum_bonus": momentum_bonus,
                "stationary_penalty": stationary_penalty,
                "spin_penalty": spin_penalty,
                "fire_bonus": fire_bonus,
                "movement_bonus": movement_bonus,
                "engagement_bonus": engagement_bonus,
            }
        }

        return frame, total_reward, done, info

    @staticmethod
    def _dist(p1, p2):
        if p1 is None or p2 is None:
            return 0.0
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def _detect_tanks(self, frame: np.ndarray):
        """Detect tank positions based on color."""
        r = frame[:, :, 0].astype(np.float32)
        g = frame[:, :, 1].astype(np.float32)
        b = frame[:, :, 2].astype(np.float32)

        # More robust detection with multiple conditions
        # Green tank (ours) - high green, low red
        my_mask = (g > 140) & (r < 200) & (g > r)
        # Red/orange tank (enemy) - high red, lower green
        enemy_mask = (r > 140) & (g < 200) & (r > g)

        def center(mask):
            coords = np.argwhere(mask)
            if coords.shape[0] < 10:  # Require minimum pixels for valid detection
                return None
            yx = coords.mean(axis=0)
            return float(yx[1]), float(yx[0])

        return center(my_mask), center(enemy_mask)

    def close(self):
        self.base_env.close()
