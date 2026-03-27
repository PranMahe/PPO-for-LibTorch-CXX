# PPO in C++ with LibTorch

A clean, from-scratch implementation of **Proximal Policy Optimization (PPO)** in C++ using LibTorch. Supports discrete and continuous action spaces, tested on CartPole, LunarLander, and MountainCarContinuous using Gymnasium. 

---

## Features

- Pure C++ PPO with LibTorch
- Discrete policy for CartPole and LunarLander
- Continuous policy for MountainCarContinuous
- Generalized Advantage Estimation (GAE)
- Orthogonal weight initialisation
- Gradient clipping
- JSON config for hyperparameter tuning

---

## Project Structure

```
PPO-for-LibTorch-CXX/
├── src/
│   ├── config/
│   │   └── Config.h              # JSON config loader
│   ├── networks/
│   │   ├── ActorCritic.h         # Network declaration
│   │   └── ActorCritic.cpp       # Forward, action selection, log probs, entropy
│   ├── environments/
│   │   ├── CartPoleEnv.h/cpp     # Native C++ CartPole
│   │   └── GymEnv.h/cpp          # Gymnasium wrapper via pybind11
│   ├── core/
│   │   ├── PPOAgent.h            # Agent declaration + RolloutBuffer
│   │   ├── PPOAgent.cpp          # GAE, update loop, save/load
│   │   └── PPOAgent.tpp          # Templated collectRollouts()
│   └── main.cpp
├── CMakeLists.txt
└── config.json
```

---

## Manual Build

### Prerequisites

| Dependency | Version | Notes |
|---|---|---|
| CMake | ≥ 3.18 | |
| LibTorch | 2.5.1 | Download from [pytorch.org](https://pytorch.org/get-started/locally/) — select C++/Java |
| Python | 3.8 – 3.12 | Required for pybind11 + Gymnasium |
| Gymnasium | latest | `pip install "gymnasium[box2d,classic-control]"` |

### 1. Download LibTorch

Go to [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/), select:
- **Build:** Stable
- **Package:** LibTorch
- **Language:** C++/Java
- **Platform:** your OS
- **Compute:** CPU (or CUDA if you have a GPU)

Unzip it somewhere, e.g. `C:/libs/libtorch` on Windows.

### 2. Build

**Windows (Visual Studio):**
```bash
cmake -B build -G "Visual Studio 17 2022" -DTorch_DIR="C:/libs/libtorch/share/cmake/Torch"
cmake --build build --config Release
```

**Linux / Mac:**
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DTorch_DIR="/path/to/libtorch/share/cmake/Torch"
cmake --build build
```

### 3. Run

`config.json` is automatically copied next to the executable by the build step.
```bash
# Windows
./build/Release/cxx_ppo_agent.exe

# Linux / Mac
./build/cxx_ppo_agent
```

---

## Configuration

All hyperparameters live in `config.json`. No recompilation needed when changing them.

```json
{
    "env": "lunarlander",

    "network": {
        "hidden_dim": 128
    },

    "ppo": {
        "gamma":        0.99,
        "gae_lambda":   0.95,
        "clip_eps":     0.2,
        "entropy_coef": 0.01,
        "value_coef":   0.5
    },

    "training": {
        "actor_lr":   0.0001,
        "critic_lr":  0.0003,
        "epochs":     10,
        "batch_size": 128,
        "num_envs":   10
    },

    "logging": {
        "test_interval": 10,
        "test_trials":   10,
        "save_interval": 200
    }
}
```

---

## Algorithm Details

### Network Architecture

Shared two-layer MLP backbone (tanh activations, orthogonal initialization), branching into separate actor and critic heads.

```
Input (state)
    → fc1 [state_dim → hidden_dim] → tanh
    → fc2 [hidden_dim → hidden_dim] → tanh
    ├── actorHead  [hidden_dim → action_dim]   (logits or mean)
    └── criticHead [hidden_dim → 1]            (state value)
```

### Discrete Policy (CartPole, LunarLander)

Actions sampled from a Categorical distribution over softmax logits.

### Continuous Policy (MountainCar)

Actions sampled from a Gaussian: `a ~ tanh(N(mean, exp(log_std)))`. The tanh squash keeps actions in `[-1, 1]`. Log probabilities are corrected for the squash using the change-of-variables formula.

## License

MIT