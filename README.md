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

- This project was done in python 3.11.14

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
