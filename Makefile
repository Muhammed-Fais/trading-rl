PYTHONPATH := src
MLFLOW_DB := sqlite:///$(CURDIR)/mlflow.db
MLFLOW_PORT ?= 5001
SWEEP_CONFIG ?= configs/sweeps/btc_eth_1h.yaml
TREND_GRID_FAST_CONFIG ?= configs/sweeps/trend_risk_grid_fast_btc_eth.yaml
TREND_GRID_CONFIG ?= configs/sweeps/trend_risk_grid_btc_eth.yaml
TRAIN_CONFIG ?= configs/train/ppo.yaml

.PHONY: install install-rllib test lint check mlflow download-btc download-eth sweep multi-sweep trend-grid-fast trend-grid train

install:
	uv sync --extra dev

install-rllib:
	uv sync --extra dev --extra rllib

test:
	PYTHONPATH=$(PYTHONPATH) uv run --extra dev pytest

lint:
	PYTHONPATH=$(PYTHONPATH) uv run --extra dev ruff check .

check: test lint

mlflow:
	uv run mlflow ui --backend-store-uri $(MLFLOW_DB) --host 127.0.0.1 --port $(MLFLOW_PORT)

download-btc:
	PYTHONPATH=$(PYTHONPATH) uv run python -m trading_rl.data.binance_client \
		--symbol BTCUSDT \
		--interval 1h \
		--start 2021-01-01 \
		--end 2024-12-31 \
		--output data/raw/binance/BTCUSDT_1h.parquet \
		--features-output data/features/binance/BTCUSDT_1h_features.parquet

download-eth:
	PYTHONPATH=$(PYTHONPATH) uv run python -m trading_rl.data.binance_client \
		--symbol ETHUSDT \
		--interval 1h \
		--start 2021-01-01 \
		--end 2024-12-31 \
		--output data/raw/binance/ETHUSDT_1h.parquet \
		--features-output data/features/binance/ETHUSDT_1h_features.parquet

sweep:
	PYTHONPATH=$(PYTHONPATH) uv run python -m trading_rl.backtest.strategy_sweep \
		--config $(TRAIN_CONFIG) \
		--output-dir artifacts/strategy_sweeps/btcusdt

multi-sweep:
	PYTHONPATH=$(PYTHONPATH) uv run python -m trading_rl.backtest.multi_symbol_sweep \
		--config $(SWEEP_CONFIG) \
		--output-dir artifacts/strategy_sweeps/btc_eth

trend-grid-fast:
	PYTHONPATH=$(PYTHONPATH) uv run python -m trading_rl.backtest.trend_risk_grid \
		--config $(TREND_GRID_FAST_CONFIG) \
		--output-dir artifacts/strategy_sweeps/trend_risk_grid_fast_btc_eth

trend-grid:
	PYTHONPATH=$(PYTHONPATH) uv run python -m trading_rl.backtest.trend_risk_grid \
		--config $(TREND_GRID_CONFIG) \
		--output-dir artifacts/strategy_sweeps/trend_risk_grid_btc_eth

train:
	PYTHONPATH=$(PYTHONPATH) uv run --extra rllib --extra dev python -m trading_rl.agents.rllib_train \
		--config $(TRAIN_CONFIG)
