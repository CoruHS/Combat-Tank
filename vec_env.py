# ppo/vec_env.py
import numpy as np
import cv2
from multiprocessing import Process, Pipe
from typing import List, Tuple, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def preprocess_obs(frame: np.ndarray) -> np.ndarray:
    """
    Convert RGB frame (H, W, 3) to grayscale (1, 84, 84) in [0, 1].
    """
    gray = frame.mean(axis=2).astype(np.float32) / 255.0
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return resized[np.newaxis, :, :].astype(np.float32)


def worker(remote, parent_remote, env_kwargs, stack_size, env_version):
    """
    Worker process that runs a single environment.
    Communicates with main process via pipe.
    """
    parent_remote.close()

    # Create environment based on version
    if env_version == "v2":
        from ppo.combat_env_v2 import CombatTankEnvV2
        env = CombatTankEnvV2(**env_kwargs)
    elif env_version == "v1":
        from ppo.combat_env import CombatTankEnv
        env = CombatTankEnv(**env_kwargs)
    else:
        from combat_tankyes import SingleAgentCombatTankShaped
        env = SingleAgentCombatTankShaped(**env_kwargs)

    # Frame stack
    frame_stack = None

    def reset_stack(frame):
        nonlocal frame_stack
        proc_frame = preprocess_obs(frame)
        frame_stack = np.repeat(proc_frame, stack_size, axis=0)
        return frame_stack.copy()

    def update_stack(frame):
        nonlocal frame_stack
        proc_frame = preprocess_obs(frame)
        frame_stack = np.concatenate([frame_stack[1:], proc_frame], axis=0)
        return frame_stack.copy()

    while True:
        try:
            cmd, data = remote.recv()

            if cmd == "step":
                frame, reward, done, info = env.step(data)
                obs = update_stack(frame)

                # Auto-reset on done
                if done:
                    # Store final info before reset
                    final_info = info.copy() if info else {}
                    frame = env.reset()
                    obs = reset_stack(frame)
                    info = final_info
                    info["episode_done"] = True

                remote.send((obs, reward, done, info))

            elif cmd == "reset":
                frame = env.reset(seed=data)
                obs = reset_stack(frame)
                remote.send(obs)

            elif cmd == "close":
                try:
                    env.base_env.close()
                except:
                    pass
                remote.close()
                break

            elif cmd == "get_spaces":
                remote.send((env.observation_space, env.action_space))

            else:
                raise NotImplementedError(f"Unknown command: {cmd}")

        except EOFError:
            break


class VecEnv:
    """
    Vectorized environment using multiprocessing.

    Each environment runs in its own process for true parallelism.
    Handles frame stacking internally.
    """

    def __init__(
        self,
        num_envs: int,
        env_kwargs: dict,
        stack_size: int = 4,
        env_version: str = "v2",
    ):
        self.num_envs = num_envs
        self.stack_size = stack_size
        self.waiting = False
        self.closed = False

        # Create pipes and workers
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(num_envs)])

        self.processes = []
        for i, (remote, work_remote) in enumerate(zip(self.remotes, self.work_remotes)):
            kwargs = env_kwargs.copy()
            process = Process(
                target=worker,
                args=(work_remote, remote, kwargs, stack_size, env_version),
                daemon=True,
            )
            process.start()
            self.processes.append(process)
            work_remote.close()

        # Get spaces from first environment
        self.remotes[0].send(("get_spaces", None))
        obs_space, act_space = self.remotes[0].recv()
        self.observation_space = obs_space
        self.action_space = act_space
        self.num_actions = act_space.n

        # Observation shape after stacking
        self.obs_shape = (stack_size, 84, 84)

    def reset(self, seeds: Optional[List[int]] = None) -> np.ndarray:
        if seeds is None:
            seeds = [None] * self.num_envs

        for remote, seed in zip(self.remotes, seeds):
            remote.send(("reset", seed))

        observations = [remote.recv() for remote in self.remotes]
        return np.stack(observations)

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", int(action)))

        results = [remote.recv() for remote in self.remotes]
        observations, rewards, dones, infos = zip(*results)

        return (
            np.stack(observations),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            list(infos),
        )

    def close(self):
        if self.closed:
            return

        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except BrokenPipeError:
                pass

        for process in self.processes:
            process.join(timeout=1)
            if process.is_alive():
                process.terminate()

        self.closed = True


class DummyVecEnv:
    """
    Non-parallel vectorized environment for debugging.
    Runs all environments sequentially in main process.
    """

    def __init__(
        self,
        num_envs: int,
        env_kwargs: dict,
        stack_size: int = 4,
        env_version: str = "v2",
    ):
        self.num_envs = num_envs
        self.stack_size = stack_size

        # Create environments based on version
        if env_version == "v2":
            from ppo.combat_env_v2 import CombatTankEnvV2
            self.envs = [CombatTankEnvV2(**env_kwargs) for _ in range(num_envs)]
        elif env_version == "v1":
            from ppo.combat_env import CombatTankEnv
            self.envs = [CombatTankEnv(**env_kwargs) for _ in range(num_envs)]
        else:
            from combat_tankyes import SingleAgentCombatTankShaped
            self.envs = [SingleAgentCombatTankShaped(**env_kwargs) for _ in range(num_envs)]

        # Frame stacks for each env
        self.frame_stacks = [None] * num_envs

        # Get spaces
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        self.num_actions = self.action_space.n
        self.obs_shape = (stack_size, 84, 84)

    def _reset_stack(self, env_idx: int, frame: np.ndarray) -> np.ndarray:
        proc_frame = preprocess_obs(frame)
        self.frame_stacks[env_idx] = np.repeat(proc_frame, self.stack_size, axis=0)
        return self.frame_stacks[env_idx].copy()

    def _update_stack(self, env_idx: int, frame: np.ndarray) -> np.ndarray:
        proc_frame = preprocess_obs(frame)
        self.frame_stacks[env_idx] = np.concatenate(
            [self.frame_stacks[env_idx][1:], proc_frame], axis=0
        )
        return self.frame_stacks[env_idx].copy()

    def reset(self, seeds: Optional[List[int]] = None) -> np.ndarray:
        if seeds is None:
            seeds = [None] * self.num_envs

        observations = []
        for i, (env, seed) in enumerate(zip(self.envs, seeds)):
            frame = env.reset(seed=seed)
            obs = self._reset_stack(i, frame)
            observations.append(obs)

        return np.stack(observations)

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        observations = []
        rewards = []
        dones = []
        infos = []

        for i, (env, action) in enumerate(zip(self.envs, actions)):
            frame, reward, done, info = env.step(int(action))
            obs = self._update_stack(i, frame)

            if done:
                final_info = info.copy() if info else {}
                frame = env.reset()
                obs = self._reset_stack(i, frame)
                info = final_info
                info["episode_done"] = True

            observations.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

        return (
            np.stack(observations),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            infos,
        )

    def close(self):
        for env in self.envs:
            try:
                env.base_env.close()
            except:
                pass
