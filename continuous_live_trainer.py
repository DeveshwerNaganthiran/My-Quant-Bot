import os
import time
import polars as pl
import pandas as pd
from datetime import datetime, timedelta
from loguru import logger
import schedule

from src.config import get_config
from src.mt5_connector import MT5Connector
from train_models import train_hmm_model
from src.feature_eng import FeatureEngineer
from src.smc_polars import SMCAnalyzer
from backtests.ml_v2.ml_v2_model import TradingModelV2

# Setup Directories
DATA_DIR = "data/continuous_learning"
MASTER_FILE = f"{DATA_DIR}/master_live_memory.parquet"
os.makedirs(DATA_DIR, exist_ok=True)

# In-memory storage for live snapshots
live_snapshots = []

def get_multi_tf_live_state(connector, config, fe, smc):
    """Fetches the LIVE, unclosed candle state across all required timeframes."""
    timeframes = ["M1", "M5", "M15", "M30", "H1"]
    live_state = {}
    
    for tf in timeframes:
        df = connector.get_market_data(config.symbol, tf, count=200)
        if len(df) == 0:
            continue
            
        df = fe.calculate_all(df, include_ml_features=True)
        df = smc.calculate_all(df)
        
        live_row = df.tail(1).to_pandas()
        live_row.columns = [f"{tf}_{col}" if col != "time" else col for col in live_row.columns]
        live_state[tf] = live_row

    if "M1" in live_state and "M5" in live_state:
        merged = live_state["M1"]
        for tf in ["M5", "M15", "M30", "H1"]:
            if tf in live_state:
                merged = pd.concat([merged.reset_index(drop=True), live_state[tf].drop(columns=['time']).reset_index(drop=True)], axis=1)
        
        merged['exact_live_time'] = datetime.now()
        tick = connector.get_tick(config.symbol)
        merged['current_ask'] = tick.ask if tick else merged['M1_close'].values[0]
        
        return merged
    return None

def continuous_recording_loop():
    """Takes a snapshot of the market every 5 seconds continuously."""
    global live_snapshots
    config = get_config()
    connector = MT5Connector(
        login=config.mt5_login, password=config.mt5_password, 
        server=config.mt5_server, path=config.mt5_path
    )
    
    if not connector.connect():
        logger.error("Failed to connect to MT5. Retrying in 10s...")
        time.sleep(10)
        return

    fe = FeatureEngineer()
    smc = SMCAnalyzer(swing_length=5)
    
    logger.info("Taking Live Multi-Timeframe Snapshots (M1, M5, M15, M30, H1)...")
    
    try:
        # Run loop for roughly 1 hour until the scheduler triggers the training
        end_time = datetime.now() + timedelta(minutes=59)
        while datetime.now() < end_time:
            state_df = get_multi_tf_live_state(connector, config, fe, smc)
            if state_df is not None:
                live_snapshots.append(state_df)
            
            # Check the clock to see if we hit the top of the hour
            schedule.run_pending()
            time.sleep(5)
    except Exception as e:
        logger.error(f"Recording error: {e}")
    finally:
        connector.disconnect()

def hourly_retrain():
    """Runs every hour. Evaluates the snapshots, labels them, and trains."""
    global live_snapshots
    logger.info("="*60)
    logger.info("HOURLY EVALUATION & RETRAINING TRIGGERED")
    logger.info("="*60)
    
    if len(live_snapshots) < 50:
        logger.warning("Not enough snapshots to train yet. Waiting for next hour.")
        return

    recent_df = pd.concat(live_snapshots, ignore_index=True)
    
    # 2. TARGET CREATION: Explicitly separate BUYS, SELLS, and CHOP
    labels = []
    ask_prices = recent_df['current_ask'].values
    horizon = 12 * 15 # 15 minutes of 5-sec snapshots
    
    for i in range(len(ask_prices)):
        current_price = ask_prices[i]
        future_window = ask_prices[i+1 : i+1+horizon]
        
        if len(future_window) == 0:
            labels.append(-1) # Invalid (Not enough future data)
            continue
            
        max_up = future_window.max() - current_price
        max_down = current_price - future_window.min()
        
        if max_up >= 2.0 and max_down < 1.0:
            labels.append(1) # Clean BUY success
        elif max_down >= 2.0 and max_up < 1.0:
            labels.append(0) # Clean SELL success
        else:
            labels.append(-1) # Choppy market / Whipsaw
            
    recent_df = recent_df.copy()
    recent_df['target'] = labels
    
    # CRITICAL FIX: Drop the choppy/invalid market data!
    # This prevents the AI from becoming biased towards one direction
    recent_df = recent_df[recent_df['target'] != -1]
    
    if len(recent_df) == 0:
        logger.warning("No clean Buy/Sell setups found in the last hour. Waiting for next hour.")
        return

    pl_recent = pl.from_pandas(recent_df)
    
    # Update Master Dataset
    if os.path.exists(MASTER_FILE):
        master_df = pl.read_parquet(MASTER_FILE)
        combined_df = pl.concat([master_df, pl_recent]).unique(subset=["exact_live_time"]).sort("exact_live_time")
        
        # CHANGED: 500,000 holds exactly 6 weeks for a bot running 21-22 hours a day
        if len(combined_df) > 500000:
            combined_df = combined_df.tail(500000)
    else:
        combined_df = pl_recent
        
    combined_df.write_parquet(MASTER_FILE)
    logger.info(f"Master Dataset updated. Total Live Memory: {len(combined_df)} snapshots.")
    
    # CLEAR MEMORY IMMEDIATELY so we start fresh next hour
    live_snapshots.clear()

    # FIX: PREPARE COLUMNS FOR HMM REGIME DETECTOR
    combined_df = combined_df.with_columns([
        pl.col("M5_close").alias("close"),
        pl.col("M5_high").alias("high"),
        pl.col("M5_low").alias("low"),
        pl.col("M5_open").alias("open"),
        pl.col("M5_volume").alias("volume"),
    ])

    # FIX: CHECK FOR SINGLE-CLASS CRASH
    unique_targets = combined_df["target"].n_unique()
    if unique_targets < 2:
        logger.warning("⚠️ Market has been pure chop. No winning patterns found yet.")
        logger.warning("XGBoost needs both wins (1) and losses (0) to learn. Saving memory and skipping training this hour.")
        return

    # TRAIN MODELS
    try:
        logger.info("Training HMM Regime Model...")
        hmm_model = train_hmm_model(combined_df)
        if hmm_model.fitted:
            combined_df = hmm_model.predict(combined_df)
            
        exclude_cols = {
            "time", "exact_live_time", "current_ask", "target", "regime_name", 
            "close", "high", "low", "open", "volume"
        }
        feature_cols = [
            col for col in combined_df.columns 
            if col not in exclude_cols and combined_df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int8, pl.Boolean]
        ]
        
        logger.info(f"Training XGBoost on {len(feature_cols)} Multi-Timeframe features...")
        
        # === CRITICAL FIX: PREVENT 0 SAMPLES CRASH ===
        # We must fill all empty FVG/OB gaps with 0 so the AI doesn't delete the rows!
        fill_exprs = []
        for col in feature_cols:
            dtype = combined_df[col].dtype
            if dtype in [pl.Float64, pl.Float32]:
                fill_exprs.append(pl.col(col).fill_nan(0.0).fill_null(0.0))
            elif dtype in [pl.Int64, pl.Int32, pl.Int8]:
                fill_exprs.append(pl.col(col).fill_null(0))
            elif dtype == pl.Boolean:
                fill_exprs.append(pl.col(col).fill_null(False))
                
        if fill_exprs:
            combined_df = combined_df.with_columns(fill_exprs)
        # ===============================================

        model = TradingModelV2(confidence_threshold=0.60, model_path="models/xgboost_model.pkl")
        model.fit(
            combined_df,
            feature_cols,
            target_col="target",
            train_ratio=0.8,
            num_boost_round=150,
            early_stopping_rounds=15
        )
        
        if hasattr(model, '_train_metrics'):
            test_auc = model._train_metrics.get("xgb_test_score", 0)
            logger.info(f"HOURLY MODEL ACCURACY (Test AUC): {test_auc:.4f}")
            
            if test_auc < 0.55:
                logger.error("⚠️ CRITICAL WARNING: Model accuracy dropped below 55%.")
                logger.error("The market execution is currently BAD (unpredictable chop). Consider stopping the bot!")
            elif test_auc >= 0.65:
                logger.info("✅ Market execution is GOOD. The patterns are highly predictable right now.")
                
    except Exception as e:
        logger.error(f"Hourly training failed: {e}")

# ==========================================
# STRICT HOURLY SCHEDULER
# ==========================================
schedule.every().hour.at(":00").do(hourly_retrain)

if __name__ == "__main__":
    logger.info("Continuous Live Trainer Started (Strict Hourly Mode).")
    
    while True:
        continuous_recording_loop()
        schedule.run_pending()
        time.sleep(1)