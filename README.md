# PPO in C++ with LibTorch

A clean, from-scratch implementation of **Proximal Policy Optimisation (PPO)** in C++ using LibTorch. Supports discrete and continuous action spaces, tested on CartPole, LunarLander, and MountainCarContinuous.

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
├── config.json
├── Dockerfile
├── docker-compose.yml
└── .dockerignore
```

---

## Quickstart with Docker (Recommended)

Docker is the easiest way to run this — no need to install LibTorch, CMake, or manage Python paths manually.

**Prerequisites:** [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.

### 1. Clone the repo

```bash
git clone https://github.com/PranMahe/PPO-for-LibTorch-CXX.git
cd PPO-for-LibTorch-CXX
```

### 2. Choose your environment

Edit `config.json` and set `"env"` to one of:

| Value | Environment | Action Space |
|---|---|---|
| `"cartpole"` | CartPole-v1 (native C++) | Discrete |
| `"lunarlander"` | LunarLander-v3 | Discrete |
| `"mountaincar"` | MountainCarContinuous-v0 | Continuous |

### 3. Build and run

```bash
# Build the image and start training
docker compose up --build

# To run again without rebuilding (config changes don't need a rebuild)
docker compose up
```

Saved models are written to `./output/` on your machine.

### 4. Change the environment without rebuilding

The `config.json` is baked into the image at build time. To change env or hyperparameters after building:

```bash
# Rebuild with updated config
docker compose up --build
```

Or override at runtime by mounting a different config:

```bash
docker run --rm \
  -v $(pwd)/config.json:/app/config.json \
  -v $(pwd)/output:/app/output \
  ppo-libtorch-cxx
```

---

## Manual Build (Windows / Linux without Docker)

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

### 2. Update CMakeLists.txt

Edit line 8 in `CMakeLists.txt` to point to your LibTorch path:

```cmake
set(Torch_DIR "C:/libs/libtorch/share/cmake/Torch")  # Windows
# set(Torch_DIR "/home/user/libtorch/share/cmake/Torch")  # Linux
```

### 3. Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

### 4. Run

Copy `config.json` next to the executable, then run:

```bash
# Windows
./build/Release/cxx_ppo_agent.exe

# Linux
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