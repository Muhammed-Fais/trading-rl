PYTHONPATH := src
MLFLOW_DB := sqlite:///$(CURDIR)/mlflow.db
MLFLOW_PORT ?= 5001
SWEEP_CONFIG ?= configs/sweeps/btc_eth_1h.yaml
CRYPTO5_SWEEP_CONFIG ?= configs/sweeps/crypto5_1h.yaml
TREND_GRID_FAST_CONFIG ?= configs/sweeps/trend_risk_grid_fast_btc_eth.yaml
TREND_GRID_FAST_CRYPTO5_CONFIG ?= configs/sweeps/trend_risk_grid_fast_crypto5.yaml
TREND_GRID_ADAPTIVE_CRYPTO5_CONFIG ?= configs/sweeps/trend_risk_grid_adaptive_crypto5.yaml
TREND_GRID_CONFIG ?= configs/sweeps/trend_risk_grid_btc_eth.yaml
PROMOTION_GATES_CONFIG ?= configs/sweeps/promotion_gates.yaml
CALENDAR_HOLDOUT_CONFIG ?= configs/sweeps/calendar_holdout_crypto5.yaml
TUNE_TEST_CONFIG ?= configs/sweeps/tune_test_crypto5_2021_2024.yaml
TUNE_TEST_PARTICIPATION_CONFIG ?= configs/sweeps/tune_test_participation_crypto5_2021_2024.yaml
TUNE_TEST_CORE_EXPOSURE_CONFIG ?= configs/sweeps/tune_test_core_exposure_crypto5_2021_2024.yaml
TUNE_TEST_CORE_EXPOSURE_CRYPTO3_CONFIG ?= configs/sweeps/tune_test_core_exposure_crypto3_2021_2024.yaml
TUNE_TEST_ACTIVITY_CRYPTO3_CONFIG ?= configs/sweeps/tune_test_activity_crypto3_2021_2024.yaml
TUNE_TEST_REGIME_CRYPTO3_CONFIG ?= configs/sweeps/tune_test_regime_crypto3_2021_2024.yaml
FAILURE_DIAGNOSTICS_CONFIG ?= configs/sweeps/failure_diagnostics_crypto5.yaml
PORTFOLIO_REPORT_CONFIG ?= configs/reports/core_exposure_crypto3_portfolio.yaml
PORTFOLIO_GATES_CONFIG ?= configs/reports/core_exposure_crypto3_portfolio_gates.yaml
ACTIVITY_PORTFOLIO_REPORT_CONFIG ?= configs/reports/activity_crypto3_portfolio.yaml
ACTIVITY_PORTFOLIO_GATES_CONFIG ?= configs/reports/activity_crypto3_portfolio_gates.yaml
REGIME_PORTFOLIO_REPORT_CONFIG ?= configs/reports/regime_crypto3_portfolio.yaml
REGIME_PORTFOLIO_GATES_CONFIG ?= configs/reports/regime_crypto3_portfolio_gates.yaml
TRAIN_CONFIG ?= configs/train/ppo.yaml
TRAIN_OVERLAY_SMOKE_CONFIG ?= configs/train/ppo_overlay_btc_smoke.yaml
TRAIN_OVERLAY_V1_CONFIG ?= configs/train/ppo_overlay_btc_v1.yaml
TRAIN_OVERLAY_V2_CONFIG ?= configs/train/ppo_overlay_btc_v2.yaml

.PHONY: install install-rllib test lint check mlflow download-btc download-eth download-bnb download-sol download-xrp download-crypto5 sweep multi-sweep crypto5-sweep trend-grid-fast trend-grid-fast-crypto5 trend-grid-adaptive-crypto5 trend-grid promotion-gates calendar-holdout tune-test tune-test-participation tune-test-core-exposure tune-test-core-exposure-crypto3 tune-test-activity-crypto3 tune-test-regime-crypto3 failure-diagnostics portfolio-report portfolio-gates activity-portfolio-report activity-portfolio-gates regime-portfolio-report regime-portfolio-gates train train-overlay-smoke train-overlay-v1 train-overlay-v2

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

download-bnb:
	PYTHONPATH=$(PYTHONPATH) uv run python -m trading_rl.data.binance_client \
		--symbol BNBUSDT \
		--interval 1h \
		--start 2021-01-01 \
		--end 2024-12-31 \
		--output data/raw/binance/BNBUSDT_1h.parquet \
		--features-output data/features/binance/BNBUSDT_1h_features.parquet

download-sol:
	PYTHONPATH=$(PYTHONPATH) uv run python -m trading_rl.data.binance_client \
		--symbol SOLUSDT \
		--interval 1h \
		--start 2021-01-01 \
		--end 2024-12-31 \
		--output data/raw/binance/SOLUSDT_1h.parquet \
		--features-output data/features/binance/SOLUSDT_1h_features.parquet

download-xrp:
	PYTHONPATH=$(PYTHONPATH) uv run python -m trading_rl.data.binance_client \
		--symbol XRPUSDT \
		--interval 1h \
		--start 2021-01-01 \
		--end 2024-12-31 \
		--output data/raw/binance/XRPUSDT_1h.parquet \
		--features-output data/features/binance/XRPUSDT_1h_features.parquet

download-crypto5: download-btc download-eth download-bnb download-sol download-xrp

sweep:
	PYTHONPATH=$(PYTHONPATH) uv run python -m trading_rl.backtest.strategy_sweep \
		--config $(TRAIN_CONFIG) \
		--output-dir artifacts/strategy_sweeps/btcusdt

multi-sweep:
	PYTHONPATH=$(PYTHONPATH) uv run python -m trading_rl.backtest.multi_symbol_sweep \
		--config $(SWEEP_CONFIG) \
		--output-dir artifacts/strategy_sweeps/btc_eth

crypto5-sweep:
	PYTHONPATH=$(PYTHONPATH) uv run python -m trading_rl.backtest.multi_symbol_sweep \
		--config $(CRYPTO5_SWEEP_CONFIG) \
		--output-dir artifacts/strategy_sweeps/crypto5

trend-grid-fast:
	PYTHONPATH=$(PYTHONPATH) uv run python -m trading_rl.backtest.trend_risk_grid \
		--config $(TREND_GRID_FAST_CONFIG) \
		--output-dir artifacts/strategy_sweeps/trend_risk_grid_fast_btc_eth

trend-grid-fast-crypto5:
	PYTHONPATH=$(PYTHONPATH) uv run python -m trading_rl.backtest.trend_risk_grid \
		--config $(TREND_GRID_FAST_CRYPTO5_CONFIG) \
		--output-dir artifacts/strategy_sweeps/trend_risk_grid_fast_crypto5

trend-grid-adaptive-crypto5:
	PYTHONPATH=$(PYTHONPATH) uv run python -m trading_rl.backtest.trend_risk_grid \
		--config $(TREND_GRID_ADAPTIVE_CRYPTO5_CONFIG) \
		--output-dir artifacts/strategy_sweeps/trend_risk_grid_adaptive_crypto5

trend-grid:
	PYTHONPATH=$(PYTHONPATH) uv run python -m trading_rl.backtest.trend_risk_grid \
		--config $(TREND_GRID_CONFIG) \
		--output-dir artifacts/strategy_sweeps/trend_risk_grid_btc_eth

promotion-gates:
	PYTHONPATH=$(PYTHONPATH) uv run python -m trading_rl.backtest.promotion_gates \
		--config $(PROMOTION_GATES_CONFIG)

calendar-holdout:
	PYTHONPATH=$(PYTHONPATH) uv run python -m trading_rl.backtest.calendar_holdout \
		--config $(CALENDAR_HOLDOUT_CONFIG) \
		--output-dir artifacts/holdout/calendar_crypto5

tune-test:
	PYTHONPATH=$(PYTHONPATH) uv run python -m trading_rl.backtest.tune_test \
		--config $(TUNE_TEST_CONFIG) \
		--output-dir artifacts/tune_test/crypto5_2021_2024

tune-test-participation:
	PYTHONPATH=$(PYTHONPATH) uv run python -m trading_rl.backtest.tune_test \
		--config $(TUNE_TEST_PARTICIPATION_CONFIG) \
		--output-dir artifacts/tune_test/participation_crypto5_2021_2024

tune-test-core-exposure:
	PYTHONPATH=$(PYTHONPATH) uv run python -m trading_rl.backtest.tune_test \
		--config $(TUNE_TEST_CORE_EXPOSURE_CONFIG) \
		--output-dir artifacts/tune_test/core_exposure_crypto5_2021_2024

tune-test-core-exposure-crypto3:
	PYTHONPATH=$(PYTHONPATH) uv run python -m trading_rl.backtest.tune_test \
		--config $(TUNE_TEST_CORE_EXPOSURE_CRYPTO3_CONFIG) \
		--output-dir artifacts/tune_test/core_exposure_crypto3_2021_2024

tune-test-activity-crypto3:
	PYTHONPATH=$(PYTHONPATH) uv run python -m trading_rl.backtest.tune_test \
		--config $(TUNE_TEST_ACTIVITY_CRYPTO3_CONFIG) \
		--output-dir artifacts/tune_test/activity_crypto3_2021_2024

tune-test-regime-crypto3:
	PYTHONPATH=$(PYTHONPATH) uv run python -m trading_rl.backtest.tune_test \
		--config $(TUNE_TEST_REGIME_CRYPTO3_CONFIG) \
		--output-dir artifacts/tune_test/regime_crypto3_2021_2024

failure-diagnostics:
	PYTHONPATH=$(PYTHONPATH) uv run python -m trading_rl.backtest.failure_diagnostics \
		--config $(FAILURE_DIAGNOSTICS_CONFIG)

portfolio-report:
	PYTHONPATH=$(PYTHONPATH) uv run python -m trading_rl.backtest.portfolio_report \
		--config $(PORTFOLIO_REPORT_CONFIG)

portfolio-gates:
	PYTHONPATH=$(PYTHONPATH) uv run python -m trading_rl.backtest.portfolio_gates \
		--config $(PORTFOLIO_GATES_CONFIG)

activity-portfolio-report:
	PYTHONPATH=$(PYTHONPATH) uv run python -m trading_rl.backtest.portfolio_report \
		--config $(ACTIVITY_PORTFOLIO_REPORT_CONFIG)

activity-portfolio-gates:
	PYTHONPATH=$(PYTHONPATH) uv run python -m trading_rl.backtest.portfolio_gates \
		--config $(ACTIVITY_PORTFOLIO_GATES_CONFIG)

regime-portfolio-report:
	PYTHONPATH=$(PYTHONPATH) uv run python -m trading_rl.backtest.portfolio_report \
		--config $(REGIME_PORTFOLIO_REPORT_CONFIG)

regime-portfolio-gates:
	PYTHONPATH=$(PYTHONPATH) uv run python -m trading_rl.backtest.portfolio_gates \
		--config $(REGIME_PORTFOLIO_GATES_CONFIG)

train:
	PYTHONPATH=$(PYTHONPATH) uv run --extra rllib --extra dev python -m trading_rl.agents.rllib_train \
		--config $(TRAIN_CONFIG)

train-overlay-smoke:
	PYTHONPATH=$(PYTHONPATH) uv run --extra rllib --extra dev python -m trading_rl.agents.rllib_train \
		--config $(TRAIN_OVERLAY_SMOKE_CONFIG)

train-overlay-v1:
	PYTHONPATH=$(PYTHONPATH) uv run --extra rllib --extra dev python -m trading_rl.agents.rllib_train \
		--config $(TRAIN_OVERLAY_V1_CONFIG)

train-overlay-v2:
	PYTHONPATH=$(PYTHONPATH) uv run --extra rllib --extra dev python -m trading_rl.agents.rllib_train \
		--config $(TRAIN_OVERLAY_V2_CONFIG)
