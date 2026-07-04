from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from trading_rl.envs.spot_trading_env import SpotTradingConfig, SpotTradingEnv
from trading_rl.utils.config import load_yaml


def build_env(env_config: dict) -> SpotTradingEnv:
    data_path = env_config.pop("data_path")
    df = pd.read_parquet(data_path)
    config = SpotTradingConfig(**env_config)
    return SpotTradingEnv(df, config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO with Ray RLlib.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_yaml(args.config)

    try:
        import ray
        from ray.rllib.algorithms.ppo import PPOConfig
        from ray.tune.registry import register_env
    except ImportError as exc:
        raise SystemExit(
            "RLlib dependencies are not installed. Run: uv sync --extra dev --extra rllib"
        ) from exc

    env_config = dict(cfg["env_config"])
    register_env("SpotTradingEnv", lambda config: build_env(dict(config)))
    ray.init(ignore_reinit_error=True)

    algo = (
        PPOConfig()
        .environment("SpotTradingEnv", env_config=env_config)
        .training(
            lr=float(cfg["training"].get("lr", 1e-4)),
            train_batch_size=int(cfg["training"].get("train_batch_size", 4096)),
        )
        .env_runners(num_env_runners=int(cfg["training"].get("num_env_runners", 1)))
        .build()
    )

    stop_iterations = int(cfg["training"].get("stop_iterations", 10))
    checkpoint_frequency = int(cfg["training"].get("checkpoint_frequency", 5))
    for iteration in range(1, stop_iterations + 1):
        result = algo.train()
        print(
            {
                "iteration": iteration,
                "episode_return_mean": result.get("episode_return_mean"),
                "timesteps_total": result.get("timesteps_total"),
            }
        )
        if iteration % checkpoint_frequency == 0:
            checkpoint_dir = Path("artifacts/checkpoints") / cfg.get("experiment_name", "ppo")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = algo.save(str(checkpoint_dir))
            print(f"Saved checkpoint: {checkpoint}")

    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()
