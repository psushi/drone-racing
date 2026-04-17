"""Train the Gym vector env with rsl_rl PPO."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import statistics
from typing import Any

import fire
import numpy as np
import torch
from rich.live import Live
from rich.table import Table
from tensordict import TensorDict

from rsl_rl.env import VecEnv as RslVecEnv
from rsl_rl.runners import OnPolicyRunner

from ece484_fly.envs.drone_race import VecDroneRaceEnv
from ece484_fly.train.obs import flatten_obs
from ece484_fly.train.utils import select_device
from ece484_fly.utils import load_config


@dataclass
class EpisodeStats:
    reward_sum: torch.Tensor
    length: torch.Tensor
    gates_passed: torch.Tensor


class NoOpSummaryWriter:
    """Minimal writer so rsl_rl still prints console metrics without TensorBoard."""

    def add_scalar(self, *args, **kwargs) -> None:
        pass

    def close(self) -> None:
        pass

    def flush(self) -> None:
        pass


def make_metrics_table(metrics: dict[str, float | int]) -> Table:
    table = Table(title="rsl_rl Training")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for key, value in metrics.items():
        if isinstance(value, float):
            table.add_row(key, f"{value:.4f}")
        else:
            table.add_row(key, str(value))
    return table


class DroneRaceRslVecEnv(RslVecEnv):
    """rsl_rl-compatible wrapper over the existing Gym vector env."""

    def __init__(self, env: VecDroneRaceEnv, cfg: Any, torch_device: str = "cpu"):
        self.env = env
        self.cfg = cfg
        self.device = torch.device(torch_device)

        self.num_envs = env.num_envs
        self.num_actions = int(env.single_action_space.shape[0])
        self.max_episode_length = int(np.asarray(env.data.max_episode_steps)[0])
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self._action_low = torch.as_tensor(env.single_action_space.low, dtype=torch.float32, device=self.device)
        self._action_high = torch.as_tensor(env.single_action_space.high, dtype=torch.float32, device=self.device)

        obs_dict, _ = env.reset()
        self._obs = self._to_tensordict(obs_dict)
        self._stats = EpisodeStats(
            reward_sum=torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            length=torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            gates_passed=torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
        )

    def get_observations(self) -> TensorDict:
        return self._obs

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        clipped = torch.tanh(actions.to(self.device))
        scaled = self._action_low + 0.5 * (clipped + 1.0) * (self._action_high - self._action_low)

        obs_dict, reward, terminated, truncated, info = self.env.step(scaled.cpu().numpy())
        obs = self._to_tensordict(obs_dict)

        rewards = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        time_outs = torch.as_tensor(truncated, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(np.logical_or(terminated, truncated), dtype=torch.float32, device=self.device)
        passed = torch.as_tensor(info.get("passed", np.zeros(self.num_envs)), dtype=torch.float32, device=self.device)

        self.episode_length_buf += 1
        self._stats.reward_sum += rewards
        self._stats.length += 1.0
        self._stats.gates_passed += passed

        done_mask = dones > 0
        extras: dict[str, Any] = {
            "time_outs": time_outs,
            "log": {
                "/avg_gates_passed_step": passed.mean(),
                "/avg_gates_passed_episode": self._stats.gates_passed[done_mask].mean()
                if torch.any(done_mask)
                else torch.tensor(0.0, device=self.device),
            },
        }
        if torch.any(done_mask):
            extras["episode"] = {
                "reward": self._stats.reward_sum[done_mask].mean(),
                "length": self._stats.length[done_mask].mean(),
                "gates_passed": self._stats.gates_passed[done_mask].mean(),
            }
            self._stats.reward_sum[done_mask] = 0.0
            self._stats.length[done_mask] = 0.0
            self._stats.gates_passed[done_mask] = 0.0
            self.episode_length_buf[done_mask] = 0

        self._obs = obs
        return obs, rewards, dones, extras

    def close(self) -> None:
        self.env.close()

    def _to_tensordict(self, obs_dict: dict[str, np.ndarray]) -> TensorDict:
        flat = flatten_obs(obs_dict, vectorized=True)
        policy = torch.as_tensor(flat, dtype=torch.float32, device=self.device)
        return TensorDict({"policy": policy}, batch_size=[self.num_envs], device=self.device)


def build_train_cfg(cfg: Any) -> dict[str, Any]:
    return {
        "seed": int(cfg.get("seed", 0)),
        "run_name": "drone_race_rsl_rl",
        "logger": "tensorboard",
        "save_interval": 50,
        "num_steps_per_env": int(cfg.train.num_steps),
        "obs_groups": {
            "actor": ["policy"],
            "critic": ["policy"],
        },
        "algorithm": {
            "class_name": "rsl_rl.algorithms.PPO",
            "num_learning_epochs": int(cfg.train.num_epochs),
            "num_mini_batches": int(cfg.train.num_minibatches),
            "clip_param": float(cfg.train.clip_eps),
            "gamma": float(cfg.train.gamma),
            "lam": float(cfg.train.lambda_),
            "value_loss_coef": float(cfg.train.vf_coef),
            "entropy_coef": float(cfg.train.ent_coef),
            "learning_rate": 3e-4,
            "max_grad_norm": float(cfg.train.max_grad_norm),
            "optimizer": "adam",
            "use_clipped_value_loss": True,
            "schedule": "fixed",
            "desired_kl": 0.01,
            "normalize_advantage_per_mini_batch": False,
            "rnd_cfg": None,
            "symmetry_cfg": None,
        },
        "actor": {
            "class_name": "rsl_rl.models.MLPModel",
            "hidden_dims": [256, 256, 256],
            "activation": "elu",
            "obs_normalization": False,
            "distribution_cfg": {
                "class_name": "rsl_rl.modules.GaussianDistribution",
                "init_std": 0.2,
                "std_type": "log",
            },
        },
        "critic": {
            "class_name": "rsl_rl.models.MLPModel",
            "hidden_dims": [256, 256, 256],
            "activation": "elu",
            "obs_normalization": False,
        },
    }


def _mean_extra(ep_extras: list[dict[str, Any]], keys: list[str]) -> float:
    values: list[float] = []
    for ep_info in ep_extras:
        for key in keys:
            if key not in ep_info:
                continue
            value = ep_info[key]
            if isinstance(value, torch.Tensor):
                values.append(float(value.float().mean().item()))
            else:
                values.append(float(value))
            break
    return float(sum(values) / len(values)) if values else 0.0


def enable_live_logging_without_tensorboard(runner: OnPolicyRunner, log_dir: str | None) -> None:
    """Patch rsl_rl logger to show a Rich Live dashboard without TensorBoard."""

    live_metrics: dict[str, float | int] = {
        "iteration": 0,
        "mean_reward": 0.0,
        "mean_ep_length": 0.0,
        "avg_gates_passed": 0.0,
        "value_loss": 0.0,
        "surrogate_loss": 0.0,
        "entropy": 0.0,
        "action_std": 0.0,
        "learning_rate": 0.0,
    }
    live = Live(make_metrics_table(live_metrics), refresh_per_second=4)

    def _init_logging_writer() -> None:
        runner.logger.log_dir = log_dir
        runner.logger.logger_type = "live"
        runner.logger.writer = NoOpSummaryWriter()
        runner.logger._store_code_state()
        live.start()

    def _log(
        it: int,
        start_it: int,
        total_it: int,
        collect_time: float,
        learn_time: float,
        loss_dict: dict,
        learning_rate: float,
        action_std: torch.Tensor,
        rnd_weight: float | None,
        print_minimal: bool = False,
        width: int = 80,
        pad: int = 40,
    ) -> None:
        mean_reward = float(statistics.mean(runner.logger.rewbuffer)) if runner.logger.rewbuffer else 0.0
        mean_ep_length = float(statistics.mean(runner.logger.lenbuffer)) if runner.logger.lenbuffer else 0.0
        avg_gates_passed = _mean_extra(runner.logger.ep_extras, ["gates_passed", "/avg_gates_passed_episode"])

        live_metrics.update(
            {
                "iteration": int(it),
                "mean_reward": mean_reward,
                "mean_ep_length": mean_ep_length,
                "avg_gates_passed": avg_gates_passed,
                "value_loss": float(loss_dict.get("value", 0.0)),
                "surrogate_loss": float(loss_dict.get("surrogate", 0.0)),
                "entropy": float(loss_dict.get("entropy", 0.0)),
                "action_std": float(action_std.mean().item()),
                "learning_rate": float(learning_rate),
            }
        )
        live.update(make_metrics_table(live_metrics))
        runner.logger.ep_extras.clear()

    def _stop_logging_writer() -> None:
        live.stop()

    runner.logger.init_logging_writer = _init_logging_writer
    runner.logger.log = _log
    runner.logger.stop_logging_writer = _stop_logging_writer


def run_train(
    config: str = "level1.toml",
    num_envs: int | None = None,
    seed: int = 0,
    device: str = "auto",
    iterations: int | None = None,
    save_path: str = "artifacts/rsl_rl/policy_final.pt",
) -> None:
    cfg = load_config(Path(__file__).parents[1] / "config" / config)
    env_device = select_device(device)
    torch_device = "cuda" if env_device == "gpu" and torch.cuda.is_available() else "cpu"

    resolved_num_envs = int(num_envs if num_envs is not None else cfg.train.get("num_envs", 32) or 32)

    env = VecDroneRaceEnv(
        num_envs=resolved_num_envs,
        freq=cfg.env.freq,
        sim_config=cfg.sim,
        sensor_range=cfg.env.sensor_range,
        control_mode=cfg.env.control_mode,
        track=cfg.env.track,
        disturbances=cfg.env.get("disturbances"),
        randomizations=cfg.env.get("randomizations"),
        seed=seed,
        max_episode_steps=cfg.env.get("max_episode_steps", 1500),
        device=env_device,
    )

    wrapped_env = DroneRaceRslVecEnv(env=env, cfg=cfg, torch_device=torch_device)
    train_cfg = build_train_cfg(cfg)
    train_cfg["seed"] = seed

    log_dir: str | None = None
    has_tensorboard = True
    try:
        from torch.utils.tensorboard import SummaryWriter  # noqa: F401

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = str(Path("artifacts") / "rsl_rl" / f"run_{timestamp}")
    except Exception:
        has_tensorboard = False
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = str(Path("artifacts") / "rsl_rl" / f"run_{timestamp}")

    runner = OnPolicyRunner(wrapped_env, train_cfg, log_dir=log_dir, device=torch_device)
    if not has_tensorboard:
        enable_live_logging_without_tensorboard(runner, log_dir)
    num_learning_iterations = int(cfg.train.num_iterations if iterations is None else iterations)
    print(f"Using env device: {env_device}")
    print(f"Using torch device: {torch_device}")
    print(f"Using {wrapped_env.num_envs} envs for {num_learning_iterations} iterations")
    if not has_tensorboard:
        print("TensorBoard not available; using Rich Live rsl_rl metrics.")

    try:
        runner.learn(num_learning_iterations=num_learning_iterations, init_at_random_ep_len=False)
    finally:
        save_file = Path(save_path)
        save_file.parent.mkdir(parents=True, exist_ok=True)
        runner.save(str(save_file))
        wrapped_env.close()
        print(f"Saved final policy to {save_file}")


if __name__ == "__main__":
    fire.Fire(run_train)
