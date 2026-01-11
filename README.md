# PPO Tank Wars

A Proximal Policy Optimization (PPO) agent for PettingZoo's Combat Tank environment.

## Requirements

- Python 3.11 (other versions not tested)
- CUDA-capable GPU recommended

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training
```bash
python -m ppo.main
```

Training outputs:
- `ppo_tank_latest.pth` - Periodic checkpoint
- `ppo_tank_best_score.pth` - Best game score
- `ppo_tank_best_winrate.pth` - Best win rate
- `ppo_tank_final.pth` - Final model

### Playing Against Your Model
```bash
python -m ppo.test
python -m ppo.test --weights ppo_tank_best_score.pth
```

Controls:
- Movement: Arrow keys or WASD (only W/UP moves forward)
- Fire: SPACE
- Quit: ESC

## Project Structure

```
ppo/
├── main.py           # Training loop
├── test.py           # Play against trained model
├── model.py          # ActorCritic neural network
├── agent.py          # PPO algorithm
├── rollout_buffer.py # Trajectory storage + GAE
├── vec_env.py        # Vectorized environments
├── combat_env_v2.py  # Custom reward shaping
└── requirements.txt  # Dependencies
```

## How It Works

### Neural Network (`model.py`)

ActorCritic CNN with shared backbone:

```
Input: 4 stacked grayscale frames (4, 84, 84)
    |
    v
Conv2d(4, 32, 8x8, stride=4) -> ReLU
    |
Conv2d(32, 64, 4x4, stride=2) -> ReLU
    |
Conv2d(64, 64, 3x3, stride=1) -> ReLU
    |
Flatten -> Linear(3136, 512) -> ReLU
    |
    +---> Linear(512, 18)  # Policy head (action logits)
    |
    +---> Linear(512, 1)   # Value head (state value)
```

### PPO Algorithm (`agent.py`)

PPO with clipped surrogate objective:

```
L_CLIP = E[min(r * A, clip(r, 1-eps, 1+eps) * A)]

where:
  r = pi(a|s) / pi_old(a|s)  # probability ratio
  A = advantage (from GAE)
  eps = 0.2 (clip range)
```

Key features:
- Clipped value loss for stability
- Entropy bonus for exploration
- Multiple epochs per rollout (4 epochs)

### Generalized Advantage Estimation (`rollout_buffer.py`)

Computes advantages using GAE(lambda):

```
A_t = delta_t + (gamma * lambda) * delta_{t+1} + ...

where:
  delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
  gamma = 0.99 (discount)
  lambda = 0.95 (GAE parameter)
```

### Vectorized Environments (`vec_env.py`)

Runs multiple environments in parallel using multiprocessing:
- Each worker runs its own Combat Tank instance
- Observations are batched for efficient GPU inference
- Automatic frame stacking (4 frames)

### Reward Shaping (`combat_env_v2.py`)

Custom rewards to encourage aggressive, mobile play:

| Component | Value | Description |
|-----------|-------|-------------|
| Kill | +5.0 | Hitting the enemy |
| Death | -5.0 | Getting hit |
| Distance penalty | -0.05 * dist | Penalize staying far from enemy |
| Approach reward | +2.0 * delta | Reward for closing distance |
| Movement bonus | +0.08 | Reward for pressing W (forward) |
| Move+Fire bonus | +0.15 | Reward for strafing (forward + shoot) |
| Spin penalty | -0.1 | Penalize consecutive rotations without moving |
| Engagement bonus | +0.05 | Bonus for being in close combat range |

**Action Categories:**
- Forward actions: 2, 6, 7, 10, 14, 15 (UP, UPRIGHT, UPLEFT, + fire variants)
- Rotation actions: 3, 4, 8, 9, 11, 12, 16, 17 (LEFT, RIGHT, + fire variants)
- Note: DOWN/S does nothing in Combat Tank!

### Self-Play Training

To train against a previously trained model:

1. Set `ENEMY_MODEL_PATH` in `main.py`:
   ```python
   ENEMY_MODEL_PATH = "ppo_tank_final.pth"
   ```

2. Run training - the agent plays against the loaded model

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| NUM_ENVS | 4 | Parallel environments |
| ROLLOUT_LENGTH | 512 | Steps per rollout |
| BATCH_SIZE | 256 | Minibatch size |
| NUM_EPOCHS | 4 | PPO epochs per update |
| LEARNING_RATE | 2.5e-4 | Initial learning rate |
| GAMMA | 0.99 | Discount factor |
| GAE_LAMBDA | 0.95 | GAE parameter |
| CLIP_RANGE | 0.2 | PPO clip range |
| ENTROPY_COEF | 0.02 | Entropy bonus (important for exploration) |
| VALUE_COEF | 0.5 | Value loss coefficient |

## Training Metrics

During training, you'll see:

```
--- Update 100/4882 ---
Timesteps: 204,800 | FPS: 1200 | Time: 2.8m
Episodes: 100 | Mean reward: 150.32 | Mean score: 0.50
Win rate: 55.0% | Mean length: 1800
Movement: 45.2% | Rotation: 12.3% | Spin penalties: 2.1/ep
Policy loss: 0.0123 | Value loss: 0.4567
Entropy: 1.85 | KL: 0.0045 | Clip frac: 0.082
```

**What to watch:**
- `Entropy` should stay above 0.5 (if it collapses to ~0, the agent stopped exploring)
- `Movement %` should be high (agent is moving, not stationary)
- `Rotation %` should be moderate (some rotation for aiming is good)
- `Win rate` should increase over time

## Troubleshooting

**Agent just spins in place:**
- Check entropy isn't collapsed (should be > 0.5)
- Increase `ENTROPY_COEF` if needed

**Agent doesn't move forward:**
- Movement bonus should encourage pressing W
- Check `Movement %` in training logs

**Training is slow:**
- Reduce `NUM_ENVS` to lower CPU usage
- Ensure CUDA is available for GPU acceleration

**High CPU usage:**
- Reduce `NUM_ENVS` (default: 4, can use 2)
