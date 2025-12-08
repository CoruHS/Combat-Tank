# DQN Agent for PettingZoo Atari Combat Tank

This project trains a Deep Q-Network (DQN) agent with a simple CNN backbone to play the **Combat Tank** game from the PettingZoo Atari suite.  
Training is done in a **single-agent shaped environment**, and a separate script lets you play **human vs. trained DQN**.

---

## Project Structure

- `agent.py`  
  DQN agent implementation:
  - Online and target Q-networks (CNN)
  - MSE TD loss
  - Epsilon-greedy policy with linear decay
  - Gradient clipping

- `combat_tank.py`  
  Custom single-agent Gym-style wrapper around `pettingzoo.atari.combat_tank_v2`:
  - Controls agent `first_0`
  - Enemy `second_0` acts randomly
  - Adds reward shaping:
    - Movement shaping (encourages translation, discourages standing still)
    - Action repetition penalty (discourages repeating actions forever)
    - Fire shaping (encourages shooting, punishes not shooting when close)
    - Turn penalty (discourages endless spinning)

- `main.py`  
  Training loop for the single-agent shaped environment:
  - Stacks 4 preprocessed frames
  - Uses replay buffer and target network
  - Linear epsilon decay over millions of steps
  - Saves trained weights to `cnn_dqn_combat_tank_single_agent_shaped.pth`

- `model.py`  
  Simple CNN architecture used by the DQN:
  - Input: `(B, 4, 84, 84)` stacked grayscale frames
  - Convs + 2 fully connected layers → Q-values for each action

- `replay_buffer.py`  
  Fixed-size replay buffer for `(s, a, r, s', done)` transitions.

- `test.py`  
  Human vs DQN script:
  - Loads trained model
  - Model controls one tank
  - Human controls the other via keyboard (WASD/arrow keys + SPACE)
  - Renders via `pygame`

---

## Requirements

### Python

- This project is done in python 3.11.14

### Python Packages

Install via `pip` (example below):

- `numpy`
- `torch` (CPU or CUDA build)
- `gym` (classic Gym API; used for the `Env` base class)
- `pettingzoo[atari]`
- `opencv-python` (for frame preprocessing)
- `pygame` (for human vs DQN visualization in `test.py`)

Example `pip` install:

```bash
pip install \
    numpy \
    torch \
    gym \
    "pettingzoo[atari]" \
    opencv-python \
    pygame

```
# Detailed Code Explanation

This document explains how the main components of the project work together:

- `DQNAgent` in `agent.py` (the learning agent)
- `SingleAgentCombatTankShaped` in `combat_tank.py` (the shaped environment wrapper)
- `main()` in `main.py` (the training loop)

with supporting roles from:

- `SimpleDQNCNN` in `model.py`
- `ReplayBuffer` in `replay_buffer.py`

The goal is to give a conceptual understanding so you can safely change rewards, features, and learning behavior without constantly re-reading the code.

---

## 1. High-Level Overview

The project implements a Deep Q-Network (DQN) to play the PettingZoo Atari environment `combat_tank_v2`.

The overall flow is:

1. The underlying PettingZoo environment simulates the Combat Tank game with two agents.
2. `SingleAgentCombatTankShaped` wraps this multi-agent environment and exposes a single-agent Gym-style API, while adding reward shaping.
3. Each RGB frame from the environment is preprocessed into an `84 × 84` grayscale image, and 4 such frames are stacked to form the agent’s observation.
4. `DQNAgent` uses `SimpleDQNCNN` to map this stacked observation to Q-values over actions and chooses actions with an epsilon-greedy policy.
5. The `main()` training loop:
   - Interacts with the environment,
   - Stores transitions in `ReplayBuffer`,
   - Periodically calls `DQNAgent.train_step()` to update the network,
   - Saves the final model.

Formally, the agent is learning an action-value function \( Q_\theta(s, a) \) that approximates the optimal values by minimizing the DQN loss:

\[
L(\theta) = \mathbb{E}\left[\left(r + \gamma \max_{a'} Q_{\theta^-}(s', a') - Q_\theta(s, a)\right)^2\right],
\]

where:

- \( Q_\theta \) is the online network,
- \( Q_{\theta^-} \) is the target network (a delayed copy),
- \( \gamma \) is the discount factor.

---

## 2. Observations and Preprocessing

The PettingZoo environment returns raw RGB frames of shape `(H, W, 3)` for each agent.

The preprocessing function `preprocess_obs` (defined in `main.py`) converts each frame into a model-friendly format by:

1. Converting the RGB frame to grayscale (by averaging the color channels).
2. Normalizing pixel values to the range `[0, 1]`.
3. Resizing the image to `84 × 84`.
4. Adding a channel dimension so that a single frame becomes shape `(1, 84, 84)`.

The training loop then maintains a stack of 4 consecutive frames, giving the agent a short temporal window:

- At the beginning of an episode, the first processed frame is simply repeated 4 times.
- After each step:
  - The oldest frame is dropped from the stack.
  - The newest processed frame is added.

This produces a stacked observation with shape `(4, 84, 84)` that is passed into `DQNAgent` and ultimately into `SimpleDQNCNN`.

This design captures motion and direction without using recurrent networks; the model can infer velocities and changes from differences between stacked frames.

---

## 3. The Agent: `DQNAgent` (`agent.py`)

`DQNAgent` represents the DQN algorithm. It holds the neural networks, the epsilon-greedy exploration logic, and the training step.

### 3.1. Networks and Hyperparameters

When you create a `DQNAgent`, you pass:

- `obs_shape`: e.g. `(4, 84, 84)`,
- `num_actions`: the number of discrete actions from the environment’s action space,
- a device (CPU or GPU),
- hyperparameters such as:
  - `gamma` (discount factor),
  - `lr` (learning rate),
  - `target_update_freq` (how often to sync the target network),
  - `epsilon_start`, `epsilon_end`, and `epsilon_decay_steps` (exploration schedule).

Internally, `DQNAgent` constructs two networks using `SimpleDQNCNN`:

- `q_net`: the online network, used to compute Q-values for action selection and to apply gradient updates.
- `target_net`: the target network, used only to compute target Q-values. It is periodically updated to match `q_net` but otherwise kept fixed between updates.

The agent also sets up an Adam optimizer for `q_net` and initializes epsilon to `epsilon_start`.

### 3.2. Epsilon-Greedy Exploration

The exploration strategy is handled by the methods `update_epsilon(global_step)` and `select_action(obs)`.

`update_epsilon(global_step)` implements a linear decay:

- At step 0, epsilon is `epsilon_start`.
- Over `epsilon_decay_steps` steps, epsilon is linearly interpolated down to `epsilon_end`.
- After that, it remains at `epsilon_end`.

`select_action(obs)` then:

- Decides randomly between exploration and exploitation based on the current epsilon value.
- If exploring, chooses a random valid action.
- If exploiting, passes the stacked observation into `q_net` and selects the action with the highest Q-value.

This schedule encourages exploration early in training when the agent knows nothing, and gradually shifts towards exploitation of what it has learned as training progresses.

### 3.3. Learning from Replay: `train_step`

The method `DQNAgent.train_step(batch, batch_size)` performs one gradient update using a mini-batch sampled from `ReplayBuffer`. Each batch element includes:

- A stacked observation (state),
- The action taken,
- The reward (in this project, the shaped reward provided by `SingleAgentCombatTankShaped`),
- The next stacked observation (next state),
- A `done` flag indicating if the episode terminated.

Inside `train_step`, the agent:

1. Uses `q_net` to compute Q-values for all actions at the current states and extracts the Q-values corresponding to the actions actually taken. This yields \( Q_\theta(s_i, a_i) \) for each sample.
2. Uses `target_net` to compute Q-values for all actions at the next states and takes the maximum over actions for each next state. This yields an estimate of future value \( \max_{a'} Q_{\theta^-}(s'_i, a') \).
3. Constructs target values using the Bellman equation:
   - If a sample is terminal, the target is just the reward.
   - Otherwise, the target is reward plus discounted max next-state Q-value.
4. Computes the loss as the mean squared error between current Q-values and targets.
5. Runs backpropagation, clips gradients to a maximum norm (via `max_grad_norm`), and updates the parameters of `q_net` using the optimizer.
6. Increments an internal counter of training steps and, every `target_update_freq` training steps, copies `q_net`’s parameters into `target_net`, keeping the target network slightly behind to stabilize learning.

This combination of replay, a separate target network, and gradient descent is the core of the DQN algorithm.

---

## 4. The Environment Wrapper: `SingleAgentCombatTankShaped` (`combat_tank.py`)

`SingleAgentCombatTankShaped` adapts PettingZoo’s `combat_tank_v2` to a more standard single-agent Gym-style environment and defines the reward shaping.

### 4.1. Turning Multi-Agent into Single-Agent

The underlying environment `combat_tank_v2.parallel_env` has multiple agents (e.g. `"first_0"` and `"second_0"`). `SingleAgentCombatTankShaped`:

- Chooses one agent (e.g. `"first_0"`) as the controlled agent.
- Chooses another (e.g. `"second_0"`) as the enemy.
- Exposes a single-agent API:
  - `reset(seed, options)` returns only the controlled agent’s initial frame.
  - `step(action)`:
    - Samples a random action for the enemy from its action space.
    - Calls the underlying environment’s `step` with both agents’ actions.
    - Extracts the controlled agent’s observation, reward, termination/truncation flags, and info.
    - Computes a final shaped reward.
    - Returns the frame, shaped reward, `done` flag, and info.

From the training loop’s perspective, the environment behaves like any Gym `Env` with a single agent, while under the hood it is a multi-agent game with the enemy acting randomly.

### 4.2. Reward Shaping

The raw environment reward from PettingZoo is modified in two main ways:

1. **Negative rewards** (e.g. when the agent gets hit) are made slightly more severe, to emphasize the cost of being damaged or losing.
2. Several **shaping terms** are added to encourage useful behaviors and discourage degenerate ones. These are computed each step from the current frame and internal state held by the wrapper.

The shaping terms include:

- **Movement shaping**  
  Estimates how far the tank moved between frames (using a crude tank position detection based on pixel colors) and normalizes this distance by the screen diagonal. It:
  - Rewards actual translation,
  - Mildly rewards pressing forward,
  - Penalizes standing still, especially when not attempting to move.

- **Turn shaping**  
  Recognizes turning actions (such as left and right turns) and applies a small penalty each time they are used, discouraging strategies where the agent spins endlessly in place.

- **Action repetition shaping**  
  Tracks how many times the same action has been repeated consecutively. Once a threshold is exceeded, it begins penalizing repeated actions, especially if the agent is not moving or firing. Repeated forward movement is penalized less than repeated turning or no-op behavior.

- **Fire shaping**  
  Rewards firing actions with a small bonus, tracks how many steps have passed since the last fire, and penalizes the agent if it stays close to the enemy for too long without firing.

The wrapper also maintains basic internal state such as:

- The previous detected position of the controlled tank (to measure movement),
- The last action taken and its repetition count,
- The number of steps since the agent last fired.

Each step, `SingleAgentCombatTankShaped` combines:

- The adjusted raw reward,
- Movement shaping,
- Turn shaping,
- Repetition shaping,
- Fire shaping,

and then clips the final result to a configured range (for example, between −2 and +2). This final scalar is the reward that the agent actually learns from.

Because all shaping logic lives here, this is the main file to modify when you want to add new reward components (for example, distance-based bonuses, cover usage, line-of-sight advantages, etc.).

---

## 5. The Training Loop: `main()` (`main.py`)

`main()` orchestrates the entire training process: environment setup, agent creation, interaction, training, and saving.

At a high level, `main()` does the following:

1. Selects a device (CPU or GPU) and defines global hyperparameters such as:
   - Total number of environment steps (`total_env_steps`),
   - Replay buffer capacity,
   - Batch size,
   - Warm-up period before learning starts,
   - Training frequency (how often to perform a gradient update),
   - Discount factor and learning rate,
   - Epsilon schedule parameters.
2. Creates an instance of `SingleAgentCombatTankShaped`, specifying options such as the discount factor and whether to render during training (rendering is usually disabled during training).
3. Calls `reset()` on the environment to get the first frame, preprocesses it using `preprocess_obs`, and builds an initial stack of 4 frames by repeating that processed frame.
4. Extracts the shape of the stacked observation and the number of actions from the environment’s action space.
5. Creates a `DQNAgent` with this observation shape and number of actions, plus the chosen hyperparameters.
6. Creates a `ReplayBuffer` with the defined capacity to store transitions.

The main training happens in a loop over episodes, and within each episode a loop over steps:

- At the start of an episode:
  - The environment is reset.
  - The frame stack is re-initialized.
- For each step:
  1. `DQNAgent.update_epsilon(global_step)` updates epsilon according to the schedule.
  2. `DQNAgent.select_action(last_obs)` selects an action based on the current policy and epsilon.
  3. `env.step(action)` executes the chosen action (with a random enemy action), returning the next RGB frame, a shaped reward, a `done` flag, and info.
  4. The next frame is preprocessed and used to update the stacked observation.
  5. The transition `(last_obs, action, shaped_reward, next_stacked_obs, done)` is added to `ReplayBuffer`.
  6. The episode’s cumulative shaped return is updated, and `global_step` is incremented.
  7. If the replay buffer has reached a minimum size and the current step matches the training frequency, a batch is sampled from `ReplayBuffer` and passed to `DQNAgent.train_step()`.

The loop continues until either:

- The episode ends (`done` flag), or
- The global step budget is exhausted.

After each episode, `main()` prints statistics such as:

- The number of steps taken,
- The episode’s shaped return,
- The total global steps,
- The current epsilon value.

When `global_step` reaches `total_env_steps`, training stops and `main()` saves the trained model parameters (from the agent’s online network) to a file (for example, `cnn_dqn_combat_tank_single_agent_shaped.pth`).

---

## 6. Supporting Components

### 6.1. `SimpleDQNCNN` (`model.py`)

`SimpleDQNCNN` defines the convolutional neural network architecture used by both `q_net` and `target_net`. It:

- Takes inputs with shape `(batch_size, in_channels, height, width)`,
- Applies two convolutional layers to gradually reduce spatial dimensions and extract features,
- Flattens the resulting feature maps,
- Uses a fully connected layer to produce a compact hidden representation,
- Uses a final fully connected layer to map that representation to `num_actions` Q-values.

This architecture is intentionally small and can be replaced or extended (e.g. with dueling architecture, more layers, or different activation functions) without changing the rest of the training code, as long as the input and output shapes remain consistent.

### 6.2. `ReplayBuffer` (`replay_buffer.py`)

`ReplayBuffer` manages a fixed-size circular buffer of experience tuples. It stores:

- Stacked observations (`obs`),
- The action taken,
- The reward received,
- The next stacked observation (`next_obs`),
- The `done` flag.

When the capacity is reached, new entries overwrite the oldest ones. The method `sample(batch_size)` returns a random mini-batch of transitions, which `DQNAgent.train_step()` uses to compute gradients.

Experience replay helps:

- Break the correlation between consecutive samples,
- Reuse past transitions multiple times,

which improves sample efficiency and training stability.

---

## 7. Extending and Modifying the Code

As the project grows and you add more rewards, features, or algorithmic changes, the main places to work in are:

### 7.1. Reward Shaping and Behavior Constraints

Modify `SingleAgentCombatTankShaped.step()` to add or adjust shaping terms. This might involve:

- Using more sophisticated position detection,
- Incorporating bullet information,
- Adding distance-based rewards,
- Penalizing specific undesirable behaviors.

### 7.2. State Representation and Preprocessing

Change `preprocess_obs` and the stacking logic in `main()` if you want:

- Different input sizes,
- Additional channels (e.g. map overlays, occupancy grids),
- Alternate normalization schemes.

### 7.3. Network Architecture

Modify `SimpleDQNCNN` to:

- Deepen the network,
- Add dueling heads,
- Insert normalization layers,
- Experiment with different architectures while keeping the `(stacked frames → Q-values)` interface.

### 7.4. Learning Algorithm Tweaks

Experiment in `DQNAgent` with:

- Double DQN (using the online network for action selection and the target network for evaluation),
- Different loss functions,
- Alternative target computation,
- Different epsilon schedules.

The current structure clearly separates responsibilities:

- `combat_tank.py` defines how the world behaves and how rewards are shaped.
- `agent.py` defines how the agent acts and learns.
- `main.py` defines how experience is collected and training is organized.

That separation is intentional, so you can grow the project’s complexity while keeping each part conceptually manageable.
