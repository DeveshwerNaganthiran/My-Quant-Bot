"""
Model Training Script
=====================
Fetches historical data from MT5 and trains all models.

Usage:
    python train_models.py

Output:
    - models/xgboost_model.pkl
    - models/hmm_regime.pkl
"""

import os
import sys
from pathlib import Path
from datetime import datetime, UTC
import polars as pl
import numpy as np
from loguru import logger

# Configure logging
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
    level="INFO",
)
logger.add(
    "logs/training_{time:YYYY-MM-DD}.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
    rotation="1 day",
    level="DEBUG",
)

# Create directories
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Import modules
from src.config import TradingConfig, get_config
from src.mt5_connector import MT5Connector
from src.smc_polars import SMCAnalyzer
from src.feature_eng import FeatureEngineer
from src.regime_detector import MarketRegimeDetector
from src.ml_model import TradingModel, get_default_feature_columns
from backtests.ml_v3.triple_barrier_labeling import TripleBarrierLabeling
from backtests.ml_v2.ml_v2_feature_eng import MLV2FeatureEngineer


def build_self_play_targets(
    df: pl.DataFrame,
    lookahead: int = 36,
    min_edge: float = 0.0025,
    stop_frac: float = 0.75,
) -> pl.DataFrame:
    """Create self-supervised labels by paper-trading from each bar and scoring the outcome forward."""
    df = df.clone()
    n = df.height

    labels = np.full(n, -1, dtype=np.int8)
    weights = np.zeros(n, dtype=np.float32)
    outcomes = np.zeros(n, dtype=np.float32)

    closes = df["close"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    opens = df["open"].to_numpy()

    if "atr" in df.columns:
        atrs = df["atr"].to_numpy()
    else:
        atrs = np.full(n, np.nan, dtype=np.float32)

    for i in range(n - lookahead):
        if i < 3:
            continue

        entry_price = float(closes[i])
        if not np.isfinite(entry_price) or entry_price <= 0:
            continue

        prev_close = float(closes[i - 1])
        prev_prev_close = float(closes[i - 2])
        if not np.isfinite(prev_close) or not np.isfinite(prev_prev_close):
            continue

        trend = (entry_price - prev_close) / prev_close
        momentum = (entry_price - prev_prev_close) / prev_prev_close

        if not np.isfinite(trend) or not np.isfinite(momentum):
            continue

        has_direction = (
            (trend > 0.0015 and momentum > 0.0010)
            or (trend < -0.0015 and momentum < -0.0010)
        )
        if not has_direction:
            continue

        direction = 1 if trend > 0 else -1
        atr_proxy = atrs[i]
        if not np.isfinite(atr_proxy) or atr_proxy <= 0:
            atr_proxy = max(float(highs[i]) - float(lows[i]), abs(entry_price - float(opens[i])), 1e-6)

        edge = max(abs(entry_price) * min_edge, atr_proxy * 1.2)
        target_price = entry_price + edge if direction > 0 else entry_price - edge
        stop_price = entry_price - (edge * stop_frac) if direction > 0 else entry_price + (edge * stop_frac)

        label = -1
        outcome_return = 0.0

        for j in range(1, lookahead + 1):
            if i + j >= n:
                break

            future_high = float(highs[i + j])
            future_low = float(lows[i + j])

            if direction > 0:
                if future_low <= stop_price:
                    label = 0
                    outcome_return = (stop_price - entry_price) / max(atr_proxy, 1e-6)
                    break
                if future_high >= target_price:
                    label = 1
                    outcome_return = (target_price - entry_price) / max(atr_proxy, 1e-6)
                    break
            else:
                if future_high >= stop_price:
                    label = 0
                    outcome_return = (entry_price - stop_price) / max(atr_proxy, 1e-6)
                    break
                if future_low <= target_price:
                    label = 1
                    outcome_return = (entry_price - target_price) / max(atr_proxy, 1e-6)
                    break

        if label == -1:
            final_price = float(closes[min(i + lookahead, n - 1)])
            outcome_return = (final_price - entry_price) / max(atr_proxy, 1e-6)
            label = 1 if outcome_return >= 0 else 0

        magnitude = abs(target_price - stop_price) / max(atr_proxy, 1e-6)
        weight = min(3.0, 1.0 + (magnitude / 4.0))
        if label == 0:
            weight *= 1.25
        weight = max(weight, 1.0)

        labels[i] = label
        weights[i] = float(weight)
        outcomes[i] = float(outcome_return)

    df = df.with_columns(
        [
            pl.Series("self_play_target", labels),
            pl.Series("self_play_weight", weights),
            pl.Series("self_play_return", outcomes),
        ]
    )
    return df


def fetch_training_data(
    connector: MT5Connector,
    symbol: str,
    timeframes: list[str],
    bars: int = 5000,
) -> dict[str, pl.DataFrame]:
    """Fetch historical data for multiple timeframes for training."""
    data: dict[str, pl.DataFrame] = {}
    for timeframe in timeframes:
        logger.info(f"Fetching {bars} bars of {symbol} {timeframe} data...")
        df = connector.get_market_data(symbol, timeframe, bars)

        if len(df) == 0:
            raise ValueError(f"No data received from MT5 for {timeframe}")

        logger.info(f"Received {len(df)} bars for {timeframe}")
        logger.info(f"Date range for {timeframe}: {df['time'].min()} to {df['time'].max()}")
        if len(df) < bars:
            logger.warning(
                f"MT5 returned fewer bars than requested for {timeframe}: requested {bars}, received {len(df)}"
            )
        data[timeframe] = df

    return data


def align_timeframes(data: dict[str, pl.DataFrame], base_timeframe: str = "M5") -> pl.DataFrame:
    """Align multiple timeframe series onto the base timeframe and attach prefixed MTF features."""
    if base_timeframe not in data:
        raise ValueError(f"Base timeframe {base_timeframe} not found in training data")

    base_df = data[base_timeframe].clone()
    fe = FeatureEngineer()
    smc = SMCAnalyzer(swing_length=5)

    for timeframe, df in data.items():
        tf_df = df.clone()
        tf_df = fe.calculate_all(tf_df, include_ml_features=True)
        tf_df = smc.calculate_all(tf_df)

        renamed = {}
        for column in tf_df.columns:
            if column == "time":
                continue
            renamed[column] = f"{timeframe}_{column}"
        tf_df = tf_df.rename(renamed)

        base_df = base_df.join_asof(
            tf_df,
            on="time",
            strategy="backward",
            by=None,
            suffix="",
        )

    return base_df

def prepare_features(
    df: pl.DataFrame,
    df_h1: pl.DataFrame | None = None,
    min_samples_after_filter: int = 5000,
) -> pl.DataFrame:
    """Apply feature engineering and create a cleaner, outcome-based training target."""
    logger.info("Applying feature engineering...")

    fe = FeatureEngineer()
    df = fe.calculate_all(df, include_ml_features=True)

    smc = SMCAnalyzer(swing_length=5)
    df = smc.calculate_all(df)

    pre_v2_cols = len(df.columns)
    fe_v2 = MLV2FeatureEngineer()
    # User request: disable H1 context in training. Keep V2 schema with default placeholders.
    df = fe_v2.add_all_v2_features(df, None)
    post_v2_cols = len(df.columns)
    logger.info(f"V2 feature delta: +{post_v2_cols - pre_v2_cols} columns (from {pre_v2_cols} to {post_v2_cols})")

    df = df.fill_null(strategy="forward").fill_null(strategy="zero")

    # Use a triple-barrier labeler to create targets based on realized outcomes rather than a fixed lookahead.
    labeler = TripleBarrierLabeling(
        profit_atr_mult=0.8,
        stoploss_atr_mult=0.8,
        max_holding_bars=48,
        min_return_atr=0.8,
    )
    df = labeler.label_data(df)

    df = build_self_play_targets(df)
    df = df.with_columns(
        pl.when(pl.col("self_play_target") != -1)
        .then(pl.col("self_play_target"))
        .otherwise(pl.col("target"))
        .alias("target")
    )
    df = df.with_columns(
        pl.when(pl.col("self_play_weight") > 0)
        .then(pl.col("self_play_weight"))
        .otherwise(pl.lit(1.0))
        .alias("sample_weight")
    )

    df = df.filter(pl.col("target") != -1)

    if len(df) < min_samples_after_filter:
        logger.warning("Filtered set only has too few rows; using a wider target window")
        labeler = TripleBarrierLabeling(
            profit_atr_mult=0.6,
            stoploss_atr_mult=0.6,
            max_holding_bars=72,
            min_return_atr=0.6,
        )
        df = labeler.label_data(df)
        df = build_self_play_targets(df)
        df = df.with_columns(
            pl.when(pl.col("self_play_target") != -1)
            .then(pl.col("self_play_target"))
            .otherwise(pl.col("target"))
            .alias("target")
        )
        df = df.with_columns(
            pl.when(pl.col("self_play_weight") > 0)
            .then(pl.col("self_play_weight"))
            .otherwise(pl.lit(1.0))
            .alias("sample_weight")
        )
        df = df.filter(pl.col("target") != -1)

    target_balance = df.filter(pl.col("target") == 1).height
    target_sell = df.filter(pl.col("target") == 0).height
    logger.info(
        f"Self-play target balance -> BUYs: {target_balance} | SELLs: {target_sell}"
    )
    logger.info(f"Retained {len(df)} labeled rows for training.")
    logger.info(f"Final training columns (features + labels/metadata): {len(df.columns)}")
    return df


def train_hmm_model(
    df: pl.DataFrame,
    model_path: str = "models/hmm_regime.pkl",
) -> MarketRegimeDetector:
    """Train HMM regime detection model."""
    logger.info("=" * 60)
    logger.info("Training HMM Regime Model")
    logger.info("=" * 60)
    
    detector = MarketRegimeDetector(
        n_regimes=3,
        lookback_periods=2000,
        model_path=model_path,
    )
    
    detector.fit(df)
    
    if detector.fitted:
        # Add regime predictions to df
        df_with_regime = detector.predict(df)
        
        # Show regime distribution
        regime_counts = df_with_regime.group_by("regime_name").len()
        logger.info("Regime Distribution:")
        for row in regime_counts.iter_rows(named=True):
            if row["regime_name"]:
                logger.info(f"  {row['regime_name']}: {row['len']} bars")
        
        # Show transition matrix
        logger.info("Transition Matrix:")
        transmat = detector.get_transition_matrix()
        for i, regime in detector.regime_mapping.items():
            probs = [f"{p:.2f}" for p in transmat[i]]
            logger.info(f"  {regime.value}: {probs}")
    
    return detector


def train_xgboost_model(
    df: pl.DataFrame,
    model_path: str = "models/xgboost_model.pkl",
) -> TradingModel:
    """Train XGBoost prediction model with anti-overfitting measures."""
    logger.info("=" * 60)
    logger.info("Training XGBoost Model (Anti-Overfit Config)")
    logger.info("=" * 60)

    # ========================================================
    # DIAGNOSTIC CHECK: Ensure dataset wasn't destroyed by NaNs
    # ========================================================
    try:
        if "target" in df.columns:
            buy_count = df.filter(pl.col("target") == 1).height
            sell_count = df.filter(pl.col("target") == 0).height
            logger.info(f"DEBUG: Dataset ready for XGBoost. Total rows: {len(df)}")
            logger.info(f"DEBUG: Target balance -> BUYs: {buy_count} | SELLs: {sell_count}")
            if len(df) == 0 or (buy_count == 0 and sell_count == 0):
                logger.error("⚠️ CRITICAL: Dataset is empty or targets are missing!")
    except Exception as e:
        logger.warning(f"Could not print target diagnostics: {e}")
    # ========================================================

    # Use the richer feature set from the V2-style pipeline, while keeping the core columns first.
    default_features = get_default_feature_columns()
    excluded_cols = {
        "time", "open", "high", "low", "close", "volume", "tick_volume", "spread",
        "real_volume", "target", "target_label", "barrier_hit", "bars_to_barrier", "return_pct",
        "self_play_target", "self_play_weight", "self_play_return", "sample_weight",
    }
    available_features = [
        c for c in default_features + [col for col in df.columns if col not in excluded_cols]
        if c in df.columns and c not in excluded_cols and not c.startswith("_")
    ]
    available_features = list(dict.fromkeys(available_features))

    logger.info(f"Available features: {len(available_features)}")

    # Create model with anti-overfitting parameters
    model = TradingModel(
        confidence_threshold=0.72,
        model_path=model_path,
        min_margin=0.15,
    )

    # Train with early stopping to prevent memorization
    model.fit(
        df,
        available_features,
        target_col="target",
        sample_weight_col="sample_weight",
        train_ratio=0.82,
        num_boost_round=1200,
        early_stopping_rounds=120,
    )
    
    if model.fitted:
        # Show feature importance
        logger.info("Top 10 Feature Importance:")
        for feat, imp in model.get_feature_importance(10).items():
            logger.info(f"  {feat}: {imp:.4f}")

        walk_forward_results = model.walk_forward_train(
            df,
            available_features,
            target_col="target",
            train_window=60000,
            test_window=5000,
            step=15000,
        )
        if walk_forward_results:
            avg_test_auc = np.mean([r[1] for r in walk_forward_results])
            logger.info(f"Walk-forward validation AUC (avg test): {avg_test_auc:.4f}")

            metrics_path = Path("data/model_validation_metrics.json")
            metrics_path.parent.mkdir(exist_ok=True)
            payload = {
                "timestamp": datetime.now(UTC).isoformat(),
                "train_rows": len(df),
                "avg_test_auc": float(avg_test_auc),
                "walk_forward_results": [
                    {"train_auc": float(train_auc), "test_auc": float(test_auc)}
                    for train_auc, test_auc in walk_forward_results
                ],
            }
            with open(metrics_path, "w", encoding="utf-8") as fh:
                import json
                json.dump(payload, fh, indent=2)
            logger.info(f"Saved validation metrics to {metrics_path}")
        
    return model


def save_training_data(df: pl.DataFrame, path: str = "data/training_data.parquet"):
    """Save training data for future reference."""
    df.write_parquet(path)
    logger.info(f"Training data saved to {path}")


def main():
    """Main training pipeline."""
    logger.info("=" * 60)
    logger.info("SMART TRADING BOT - MODEL TRAINING")
    logger.info("=" * 60)
    
    # Load config
    config = get_config()
    logger.info(f"Symbol: {config.symbol}")
    logger.info(f"Capital: ${config.capital:,.2f}")
    logger.info(f"Mode: {config.capital_mode.value}")
    
    # Connect to MT5
    logger.info("Connecting to MT5...")
    connector = MT5Connector(
        login=config.mt5_login,
        password=config.mt5_password,
        server=config.mt5_server,
        path=config.mt5_path,
    )
    
    try:
        connector.connect()
        logger.info("MT5 connected successfully!")
        
        # Get account info
        balance = connector.account_balance
        equity = connector.account_equity
        logger.info(f"Account Balance: ${balance:,.2f}")
        logger.info(f"Account Equity: ${equity:,.2f}")
        
    except Exception as e:
        logger.error(f"MT5 connection failed: {e}")
        logger.info("Please ensure:")
        logger.info("  1. MT5 terminal is running")
        logger.info("  2. Auto-trading is enabled")
        logger.info("  3. Login credentials are correct")
        return
    
    try:
        # Fetch data - 1 YEAR OF DATA
        # Let's train on M5 as the primary anchor to avoid M1 noise
        training_timeframes = ["M1", "M5", "M15"]
        historical_bars = 500_000  # Pull a much larger history for better generalization
        
        logger.info(f"Initiating large-scale multi-timeframe data fetch: {historical_bars} bars on {training_timeframes}")
        
        tf_data = fetch_training_data(
            connector,
            config.symbol,
            training_timeframes,
            bars=historical_bars,
        )

        df = align_timeframes(tf_data, base_timeframe="M5")
        
        # Prepare features using the aligned multi-timeframe dataset
        df = prepare_features(df, df_h1=None)
        
        # Save raw data
        save_training_data(df)
        
        # Train HMM 
        hmm_model = train_hmm_model(df)
        
        if hmm_model.fitted:
            df = hmm_model.predict(df)
        
        # Train XGBoost
        xgb_model = train_xgboost_model(df)
        
        # Summary
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"HMM Model: {'SAVED' if hmm_model.fitted else 'FAILED'}")
        logger.info(f"XGBoost Model: {'SAVED' if xgb_model.fitted else 'FAILED'}")
        logger.info(f"Models saved in: models/")
        logger.info(f"Training data saved in: data/")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        connector.disconnect()
        logger.info("MT5 disconnected")


if __name__ == "__main__":
    main()


