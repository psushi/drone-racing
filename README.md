# Drone Racing RL
<p align="center">
  <img width="460" height="300" src="quadrotor-sketch.png" alt="Quadrotor sketch">
</p>

Physics-based autonomous drone racing project built around a JAX PPO training loop, a functional vectorized environment, and deployment-minded reward/reset design for Crazyflie-scale vehicles.

## Highlights

- JAX-native PPO trainer for quadrotor racing
- functional vectorized environment for fast rollout collection
- reward debugging and checkpoint inspection tools that share the same config path as training
- reset curriculum plus harvested reset-bank workflows
- simulation and controller interfaces that can be adapted toward real hardware

## Repo Layout

```text
.
├── config/                 # training and evaluation configs
├── ece484_fly/
│   ├── control/            # controller interfaces and examples
│   ├── envs/               # racing environments and JAX wrappers
│   ├── train/              # PPO, obs encoding, experiment I/O
│   └── utils/              # config loading and helpers
├── scripts/
│   ├── run_train_jax.py
│   ├── watch_policy.py
│   ├── debug_reward_attitude.py
│   ├── harvest_reset_bank.py
│   ├── sim.py
│   └── vec_sim.py
└── tools/
```

## Setup

Requirements:

- Linux for GPU-backed training
- macOS Apple Silicon works for CPU simulation/debugging
- [Pixi](https://pixi.sh)

Install the environment:

```bash
pixi install -e default
```

## Train

Run the baseline config:

```bash
pixi run train --config baseline.toml --checkpoint_path baseline
```

This writes:

```text
artifacts/baseline/model.msgpack
artifacts/baseline/model.final.msgpack
artifacts/baseline/config.toml
artifacts/baseline/metadata.json
```

## Inspect a Policy

Render a saved checkpoint:

```bash
pixi run watch --checkpoint_path baseline
```

Print reward terms and motion diagnostics:

```bash
pixi run debug --checkpoint_path baseline
```

Harvest a reset bank from a trained policy:

```bash
pixi run harvest --checkpoint_path baseline
```

## Configs

Useful configs in this repo:

- `config/baseline.toml`: reproduced baseline training config
- `config/level1_flat.toml`: easier flat-track setup for early training
- `config/l1.toml` to `config/l8.toml`: curriculum-style track ladder

Typical curriculum flow:

```bash
pixi run train --config l1.toml --checkpoint_path l1
pixi run train --config l2.toml --checkpoint_path l2 --init_checkpoint_path l1
pixi run train --config l3.toml --checkpoint_path l3 --init_checkpoint_path l2
```

## Notes

- Run outputs and checkpoints are intentionally not tracked in this public branch.
- The package import path remains `ece484_fly` for compatibility with the original codebase history.
- Some hardware deployment utilities are still present, but the public focus of this repo is the simulation, training, and policy-evaluation stack.

## Tech Stack

- JAX
- Flax
- Distrax
- Optax
- Crazyflow
- Pixi

## Attribution

This project builds on public simulation and control tooling, especially Crazyflow and the JAX/Flax ecosystem. The racing environment logic, JAX training path, reward shaping, reset curriculum, and experiment tooling in this repo were developed as part of this project codebase.
