# Drone Racing RL

Train RL policies for quadrotor drone racing.
- End to end vectorized JAX training pipeline. (Training and Simulation)
- Converges in ~500 iterations on a single 5090 GPU in ~1 min.(~230K steps/s). 

## What is here

- `drone_racing_rl/`: environments, controllers, training code, and utilities
- `config/`: training and evaluation configs
- `scripts/run_train_jax.py`: PPO training entrypoint
- `scripts/watch_policy.py`: render a saved checkpoint
- `scripts/debug_reward_attitude.py`: inspect reward terms and motion behavior
- `artifacts/`: selected saved runs and checkpoints from the project

## Setup

Requirements:

- [Pixi](https://pixi.sh)
- Linux and macOS are both supported
- if a CUDA-capable GPU is available, training will detect it and use it automatically
- macOS works well for CPU simulation and debugging

Install:

```bash
pixi install -e default
```

Recommended setup for gpu-poors:

- develop and inspect policies locally on macOS/Linux
- train on a remote Linux machine with a CUDA GPU
- use the same repo and configs on both machines, then pull checkpoints back for evaluation
- Pixi takes care of linux/macOS differences

## Train

```bash
pixi run train --config baseline.toml --checkpoint_path baseline
```

## Inspect a policy

```bash
pixi run watch --checkpoint_path baseline
pixi run debug --checkpoint_path baseline
```

This project builds on the excellent [Crazyflow](https://github.com/utiasDSL/crazyflow) simulator.
