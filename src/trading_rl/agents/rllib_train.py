from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from trading_rl.agents.evaluate import evaluate_and_report, named_policy, rllib_policy
from trading_rl.envs.risk_overlay_env import RiskOverlayTradingEnv
from trading_rl.envs.spot_trading_env import SpotTradingConfig, SpotTradingEnv
from trading_rl.utils.config import load_yaml


def build_env(env_config: dict) -> SpotTradingEnv:
    env_type = str(env_config.pop("env_type", "spot"))
    base_policy_config = env_config.pop("base_policy_config", None)
    overlay_reward_config = env_config.pop("overlay_reward", None)
    data_path = env_config.pop("data_path")
    df = pd.read_parquet(data_path)
    config = SpotTradingConfig(**env_config)
    env = SpotTradingEnv(df, config)
    if env_type == "spot":
        return env
    if env_type == "risk_overlay":
        return RiskOverlayTradingEnv(
            env,
            base_policy_config=base_policy_config,
            overlay_reward_config=overlay_reward_config,
        )
    raise ValueError("env_type must be one of: spot, risk_overlay")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO with Ray RLlib.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_yaml(args.config)

    try:
        import mlflow
        import ray
        from ray.rllib.algorithms.ppo import PPOConfig
        from ray.tune.registry import register_env
    except ImportError as exc:
        raise SystemExit(
            "RLlib dependencies are not installed. Run: uv sync --extra dev --extra rllib"
        ) from exc

    experiment_name = cfg.get("experiment_name", "ppo")
    tracking_uri = cfg.get("tracking", {}).get("uri", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg.get("tracking", {}).get("experiment", "trading-rl"))

    env_config = dict(cfg["env_config"])
    env_config["data_path"] = str(Path(env_config["data_path"]).expanduser().resolve())
    register_env("SpotTradingEnv", lambda config: build_env(dict(config)))
    ray.init(ignore_reinit_error=True)

    algo = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
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
    with mlflow.start_run(run_name=experiment_name):
        mlflow.log_params(_flatten_params({"config": cfg}))
        mlflow.set_tag("git_commit", _git_commit())
        mlflow.set_tag("config_path", args.config)

        final_checkpoint = None
        for iteration in range(1, stop_iterations + 1):
            result = algo.train()
            metrics = _training_metrics(result)
            mlflow.log_metrics(metrics, step=iteration)
            print({"iteration": iteration, **metrics})
            if iteration % checkpoint_frequency == 0:
                checkpoint_dir = (Path("artifacts/checkpoints") / experiment_name).resolve()
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                save_result = algo.save(str(checkpoint_dir))
                checkpoint = getattr(save_result, "checkpoint", save_result)
                checkpoint_path = Path(getattr(checkpoint, "path", str(checkpoint)))
                final_checkpoint = checkpoint_path
                if checkpoint_path.exists():
                    if checkpoint_path.is_dir():
                        mlflow.log_artifacts(str(checkpoint_path), artifact_path="checkpoints")
                    else:
                        mlflow.log_artifact(str(checkpoint_path), artifact_path="checkpoints")
                print(f"Saved checkpoint: {checkpoint_path}")

        report_cfg = cfg.get("report", {})
        if report_cfg.get("enabled", True):
            eval_env_config = dict(env_config)
            eval_env_config["split"] = report_cfg.get("split", "test")
            eval_env_config["random_start"] = False
            report_path = Path(
                report_cfg.get("output", f"artifacts/reports/{experiment_name}.html")
            )
            _, eval_metrics, html_path = evaluate_and_report(
                eval_env_config,
                rllib_policy(algo),
                report_path,
                title=f"{experiment_name} evaluation",
                start_index=report_cfg.get("start_index"),
                max_steps=report_cfg.get("max_steps"),
                seed=int(report_cfg.get("seed", 7)),
            )
            mlflow.log_metrics({f"eval_{k}": v for k, v in eval_metrics.as_dict().items()})
            mlflow.log_artifact(str(html_path), artifact_path="reports")
            print({"report": str(html_path), **eval_metrics.as_dict()})

            for baseline in report_cfg.get("baselines", ["cash", "buy_and_hold"]):
                baseline_path = report_path.with_name(f"{report_path.stem}_{baseline}.html")
                _, baseline_metrics, baseline_html = evaluate_and_report(
                    eval_env_config,
                    named_policy(baseline, seed=int(report_cfg.get("seed", 7))),
                    baseline_path,
                    title=f"{baseline} baseline",
                    start_index=report_cfg.get("start_index"),
                    max_steps=report_cfg.get("max_steps"),
                    seed=int(report_cfg.get("seed", 7)),
                )
                mlflow.log_metrics(
                    {f"baseline_{baseline}_{k}": v for k, v in baseline_metrics.as_dict().items()}
                )
                mlflow.log_artifact(str(baseline_html), artifact_path="reports")

        if final_checkpoint is not None:
            mlflow.set_tag("final_checkpoint", str(final_checkpoint))

    algo.stop()
    ray.shutdown()


def _training_metrics(result: dict[str, Any]) -> dict[str, float]:
    candidates = {
        "episode_return_mean": result.get("episode_return_mean"),
        "episode_return_min": result.get("episode_return_min"),
        "episode_return_max": result.get("episode_return_max"),
        "episodes_total": result.get("episodes_total"),
        "timesteps_total": result.get("timesteps_total"),
    }
    env_runner = result.get("env_runners")
    if isinstance(env_runner, dict):
        candidates.update(
            {
                "episode_return_mean": env_runner.get("episode_return_mean"),
                "episode_len_mean": env_runner.get("episode_len_mean"),
            }
        )
    return {
        key: float(value)
        for key, value in candidates.items()
        if isinstance(value, int | float) and value is not None
    }


def _flatten_params(value: Any, prefix: str = "") -> dict[str, str | int | float | bool]:
    params: dict[str, str | int | float | bool] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            params.update(_flatten_params(child, child_prefix))
    elif isinstance(value, str | int | float | bool) or value is None:
        params[prefix] = "" if value is None else value
    return params


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"


if __name__ == "__main__":
    main()
