# Drone Racing RL

JAX PPO project for autonomous drone racing in simulation, with tools for training, evaluation, controller testing, and deployment-oriented experiments.

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
- Linux for GPU training
- macOS Apple Silicon for CPU simulation and debugging

Install:

```bash
pixi install -e default
```

## Train

```bash
pixi run train --config baseline.toml --checkpoint_path baseline
```

## Inspect a policy

```bash
pixi run watch --checkpoint_path baseline
pixi run debug --checkpoint_path baseline
```

## Project focus

- vectorized drone racing environment
- PPO training in JAX/Flax
- reward shaping and reset curriculum experiments
- policy evaluation and deployment-minded controller interfaces

## Stack

- JAX
- Flax
- Optax
- Distrax
- Crazyflow
- Pixi
