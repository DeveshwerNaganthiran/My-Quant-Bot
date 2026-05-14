"""
Main Live Trading Orchestrator (WIN/LOSS ALTERNATOR)
==============================
Asynchronous event-driven trading system.

Pipeline:
1. Load trained models & Forced Direction State
2. Fetch Data -> Convert to Polars
3. Apply SMC & Feature Engineering
4. Detect Market Regime (HMM)
5. Get AI Signal & SMC Signal -> Align with Forced Direction
6. Check Risk & Position Size
7. Execute Trade
8. Trade Closes -> Win = Same Direction | Loss = Flip Direction
"""

import asyncio
import time
import os
import json
from collections import deque
from datetime import datetime, date, timedelta
from types import SimpleNamespace
from typing import Optional, Dict, Tuple
from zoneinfo import ZoneInfo
from pathlib import Path
import polars as pl
from loguru import logger
import sys

# Configure logging
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
    level="INFO",
)
logger.add(
    "logs/trading_bot_{time:YYYY-MM-DD}.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
    rotation="1 day",
    retention="30 days",
    level="DEBUG",
    encoding="utf-8",
)

# Create directories
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Import modules
from src.config import TradingConfig, get_config
from src.mt5_connector import MT5Connector, MT5SimulationConnector
from src.smc_polars import SMCAnalyzer, SMCSignal
from src.feature_eng import FeatureEngineer
from src.regime_detector import MarketRegimeDetector, FlashCrashDetector, MarketRegime, RegimeState
from src.risk_engine import RiskEngine
from backtests.ml_v2.ml_v2_model import TradingModelV2
from backtests.ml_v2.ml_v2_feature_eng import MLV2FeatureEngineer
from src.ml_model import get_default_feature_columns
from src.position_manager import SmartPositionManager
from src.session_filter import SessionFilter, create_wib_session_filter
from src.auto_trainer import AutoTrainer, create_auto_trainer
from src.telegram_notifier import TelegramNotifier, create_telegram_notifier
from src.telegram_notifications import TelegramNotifications
from src.smart_risk_manager import SmartRiskManager, create_smart_risk_manager
from src.dynamic_confidence import DynamicConfidenceManager, create_dynamic_confidence
from src.trade_logger import TradeLogger, get_trade_logger
from src.filter_config import FilterConfigManager


class TradingBot:
    """
    Main trading bot orchestrator.
    Coordinates all components in an asynchronous event loop.
    """
    
    def __init__(
        self,
        config: Optional[TradingConfig] = None,
        simulation: bool = False,
    ):
        self.config = config or get_config()
        self.simulation = simulation
        
        if simulation:
            self.mt5 = MT5SimulationConnector()
        else:
            self.mt5 = MT5Connector(
                login=self.config.mt5_login,
                password=self.config.mt5_password,
                server=self.config.mt5_server,
                path=self.config.mt5_path,
            )
        
        self.smc = SMCAnalyzer(
            swing_length=self.config.smc.swing_length,
            ob_lookback=self.config.smc.ob_lookback,
        )
        
        self.features = FeatureEngineer()
        
        self.regime_detector = MarketRegimeDetector(
            n_regimes=self.config.regime.n_regimes,
            lookback_periods=self.config.regime.lookback_periods,
            retrain_frequency=self.config.regime.retrain_frequency,
            model_path="models/hmm_regime.pkl",
        )
        
        self.flash_crash = FlashCrashDetector(
            threshold_percent=self.config.flash_crash_threshold,
        )
        
        self.risk_engine = RiskEngine(self.config)
        self.filter_config = FilterConfigManager("data/filter_config.json")

        self.ml_model = TradingModelV2(
            confidence_threshold=0.60,
            model_path="models/xgboost_model.pkl",
        )
        self.fe_v2 = MLV2FeatureEngineer()
        self._h1_df_cached = None 

        self.position_manager = SmartPositionManager(
            breakeven_pips=150.0,       
            trail_start_pips=250.0,     
            trail_step_pips=50.0,      
            atr_be_mult=2.0,           
            atr_trail_start_mult=2.5,  
            atr_trail_step_mult=1.5,   
            min_profit_to_protect=25.0,
            max_drawdown_from_peak=50.0,  
            enable_market_close_handler=False,
            min_profit_before_close=3.0,
            max_loss_to_hold=2.0,
        )

        self.session_filter = create_wib_session_filter(aggressive=True)
        self.auto_trainer = create_auto_trainer()
        self.smart_risk = create_smart_risk_manager(capital=self.config.capital)
        self.dynamic_confidence = create_dynamic_confidence()
        self.telegram = create_telegram_notifier()
        self.notifications = TelegramNotifications(self)
        self.news_agent = None
        self.trade_logger = get_trade_logger()

        self._running = False
        self._loop_count = 0
        self._h1_bias_cache = "NEUTRAL"
        self._h1_bias_loop = 0
        self._h1_bias_score = 0.0
        self._h1_bias_strength = "weak"
        self._h1_bias_signals = {}
        self._h1_bias_regime_weights = "unknown"
        self._last_signal: Optional[SMCSignal] = None
        self._last_retrain_check: Optional[datetime] = None
        self._last_trade_time: Optional[datetime] = None
        self._execution_times: list = []
        self._current_date = date.today()
        self._models_loaded = False
        self._trade_cooldown_seconds = 10
        self._start_time = datetime.now()
        self._daily_start_balance: float = 0
        self._total_session_profit: float = 0
        self._total_session_trades: int = 0
        self._total_session_wins: int = 0
        self._last_market_update_time: Optional[datetime] = None
        self._last_hourly_report_time: Optional[datetime] = None
        self._open_trade_info: Dict = {}
        self._last_news_alert_reason: Optional[str] = None
        self._current_session_multiplier: float = 1.0
        self._is_sydney_session: bool = False
        self._last_candle_time: Optional[datetime] = None
        self._pyramid_done_tickets: set = set()
        self._last_pyramid_time: Optional[datetime] = None
        self._position_check_interval: int = 5

        # Around line 120 in main_live.py, add this line:
        self._is_verifying_trade = False
        self._last_filter_results: list = []
        self._last_model_update_time: float = 0.0
        self._h1_ema20_value: float = 0.0
        self._h1_current_price: float = 0.0

        self._dash_price_history: deque = deque(maxlen=120)
        self._dash_equity_history: deque = deque(maxlen=120)
        self._dash_balance_history: deque = deque(maxlen=120)
        self._dash_logs: deque = deque(maxlen=50)
        self._dash_last_price: float = 0.0
        self._dash_status_file = Path("data/bot_status.json")

        self._restore_dashboard_state()
        # self._forced_next_direction = self._load_forced_direction()
        self._forced_next_direction = None # Force disabled
    
    def _load_forced_direction(self):
        """Loads the last forced direction from file."""
        try:
            if os.path.exists("data/forced_direction.txt"):
                with open("data/forced_direction.txt", "r") as f:
                    direction = f.read().strip().upper()
                    if direction in ["BUY", "SELL"]:
                        return direction
        except:
            pass
        return None

    def _save_forced_direction(self, direction):
        """Saves the forced direction to file so it survives crashes."""
        try:
            os.makedirs("data", exist_ok=True)
            with open("data/forced_direction.txt", "w") as f:
                f.write(direction)
        except:
            pass

    def _invert_smc_signal(self, signal: Optional[SMCSignal]) -> Optional[SMCSignal]:
        """Invert SMC signal direction for opposite trading."""
        if not signal:
            return None
            
        new_type = "SELL" if signal.signal_type == "BUY" else "BUY"
        sl_dist = abs(signal.entry_price - signal.stop_loss)
        tp_dist = abs(signal.entry_price - signal.take_profit)
        
        tick = self.mt5.get_tick(self.config.symbol)
        if tick:
            new_entry = tick.bid if new_type == "SELL" else tick.ask
        else:
            new_entry = signal.entry_price
            
        if new_type == "SELL":
            new_sl = new_entry + sl_dist
            new_tp = new_entry - tp_dist
        else:
            new_sl = new_entry - sl_dist
            new_tp = new_entry + tp_dist
            
        return SMCSignal(
            signal_type=new_type,
            entry_price=new_entry,
            stop_loss=new_sl,
            take_profit=new_tp,
            confidence=signal.confidence,
            reason=f"[FORCED FLIP -> {new_type}] Original: {signal.reason}"
        )

    def _invert_ml_prediction(self, ml_pred):
        """Invert ML prediction direction for opposite trading."""
        if not ml_pred or getattr(ml_pred, 'signal', 'HOLD') == 'HOLD':
            return ml_pred
            
        new_signal = "SELL" if ml_pred.signal == "BUY" else ("BUY" if ml_pred.signal == "SELL" else ml_pred.signal)
        fi = getattr(ml_pred, 'feature_importance', {})
        
        return SimpleNamespace(
            signal=new_signal,
            confidence=ml_pred.confidence,
            probability=1.0 - ml_pred.probability,
            feature_importance=fi
        )

    def _restore_dashboard_state(self):
        try:
            if not self._dash_status_file.exists():
                return
            import json
            with open(self._dash_status_file, "r") as f:
                prev = json.load(f)

            for val in prev.get("priceHistory", []):
                self._dash_price_history.append(val)
            for val in prev.get("equityHistory", []):
                self._dash_equity_history.append(val)
            for val in prev.get("balanceHistory", []):
                self._dash_balance_history.append(val)
            for log in prev.get("logs", []):
                self._dash_logs.append(log)

            self._dash_last_price = prev.get("price", 0.0)

            smc = prev.get("smc", {})
            if smc.get("signal"):
                self._last_raw_smc_signal = smc["signal"]
                self._last_raw_smc_confidence = smc.get("confidence", 0.0)
                self._last_raw_smc_reason = smc.get("reason", "")
                self._last_raw_smc_updated = smc.get("updatedAt", "")

            ml = prev.get("ml", {})
            if ml.get("signal"):
                self._last_ml_signal = ml["signal"]
                self._last_ml_confidence = ml.get("confidence", 0.0)
                self._last_ml_probability = ml.get("buyProb", ml.get("confidence", 0.0))
                self._last_ml_updated = ml.get("updatedAt", "")

            regime = prev.get("regime", {})
            if regime.get("name"):
                from src.regime_detector import MarketRegime
                regime_val = regime["name"].lower().replace(" ", "_")
                try:
                    self._last_regime = MarketRegime(regime_val)
                except ValueError:
                    pass
                self._last_regime_volatility = regime.get("volatility", 0.0)
                self._last_regime_confidence = regime.get("confidence", 0.0)
                self._last_regime_updated = regime.get("updatedAt", "")

            perf = prev.get("performance", {})
            self._loop_count = perf.get("loopCount", 0)
            self._total_session_trades = perf.get("totalSessionTrades", 0)
            self._total_session_wins = perf.get("totalSessionWins", 0)
            self._total_session_profit = perf.get("totalSessionProfit", 0.0)
            prev_uptime_h = perf.get("uptimeHours", 0)
            if prev_uptime_h > 0:
                self._start_time = datetime.now() - timedelta(hours=prev_uptime_h)

            self._h1_ema20_value = prev.get("h1BiasDetails", {}).get("ema20", 0.0)
            self._h1_current_price = prev.get("h1BiasDetails", {}).get("price", 0.0)
            self._h1_bias_loop = -999  

            logger.info(f"Dashboard state restored: {len(self._dash_price_history)} prices, {len(self._dash_logs)} logs, loops={self._loop_count}, uptime={prev_uptime_h}h")
        except Exception as e:
            logger.warning(f"Could not restore dashboard state: {e}")

    def _load_models(self) -> bool:
        logger.info("Loading trained models...")
        models_ok = True
        
        try:
            self.regime_detector.load()
            if self.regime_detector.fitted:
                logger.info("HMM Regime model loaded successfully")
            else:
                logger.warning("HMM model not found or not fitted")
                models_ok = False
        except Exception as e:
            logger.error(f"Failed to load HMM model: {e}")
            models_ok = False
        
        try:
            self.ml_model.load()
            from backtests.ml_v2.ml_v2_model import ModelType
            self.ml_model.model_type = ModelType.XGBOOST_BINARY
            
            if self.ml_model.fitted:
                logger.info("ML V2 Model D loaded successfully")
                logger.info(f"  Features: {len(self.ml_model.feature_names)}")
                logger.info(f"  Type: {self.ml_model.model_type.value}")
            else:
                logger.warning("ML V2 Model D not found or not fitted")
                models_ok = False
        except Exception as e:
            logger.error(f"Failed to load ML V2 Model D: {e}")
            models_ok = False
        
        self._models_loaded = models_ok

        if models_ok:
            self._write_model_metrics()
            import os
            model_path = "models/xgboost_model.pkl"
            if os.path.exists(model_path):
                self._last_model_update_time = os.path.getmtime(model_path)

        return models_ok

    def _dash_log(self, level: str, message: str):
        now = datetime.now(ZoneInfo("Asia/Kuala_Lumpur"))
        self._dash_logs.append({
            "time": now.strftime("%H:%M:%S"),
            "level": level,
            "message": message,
        })

    def _write_model_metrics(self, retrain_results: dict = None):
        try:
            import json as _json
            metrics = {
                "featureImportance": [],
                "trainAuc": 0,
                "testAuc": 0,
                "sampleCount": 0,
                "updatedAt": datetime.now(ZoneInfo("Asia/Kuala_Lumpur")).isoformat(),
            }

            booster = getattr(self.ml_model, 'xgb_model', None) or getattr(self.ml_model, 'model', None)
            if self.ml_model.fitted and booster is not None:
                try:
                    importance = booster.get_score(importance_type='gain') if hasattr(booster, 'get_score') else {}
                    if importance and self.ml_model.feature_names:
                        mapped = {}
                        for key, val in importance.items():
                            if key.startswith('f') and key[1:].isdigit():
                                idx = int(key[1:])
                                if idx < len(self.ml_model.feature_names):
                                    mapped[self.ml_model.feature_names[idx]] = val
                                else:
                                    mapped[key] = val
                            else:
                                mapped[key] = val
                        importance = mapped
                    if not importance and hasattr(booster, 'feature_importances_'):
                        names = self.ml_model.feature_names if hasattr(self.ml_model, 'feature_names') else []
                        importance = dict(zip(names, booster.feature_importances_))

                    total = sum(importance.values()) if importance else 1
                    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                    metrics["featureImportance"] = [
                        {"name": name, "importance": round(val / total, 4)}
                        for name, val in sorted_features[:20]
                    ]
                except Exception:
                    pass

            if retrain_results:
                metrics["trainAuc"] = retrain_results.get("xgb_train_auc", 0)
                metrics["testAuc"] = retrain_results.get("xgb_test_auc", 0)
                metrics["sampleCount"] = retrain_results.get("sample_count", 0)
            elif hasattr(self.ml_model, '_train_metrics') and self.ml_model._train_metrics:
                tm = self.ml_model._train_metrics
                metrics["trainAuc"] = tm.get("train_auc", 0) or tm.get("xgb_train_score", 0) or tm.get("train_accuracy", 0)
                metrics["testAuc"] = tm.get("test_auc", 0) or tm.get("xgb_test_score", 0) or tm.get("test_accuracy", 0)
                metrics["sampleCount"] = tm.get("train_samples", 0) + tm.get("test_samples", 0)
            elif hasattr(self, 'auto_trainer') and hasattr(self.auto_trainer, 'last_auc'):
                metrics["testAuc"] = self.auto_trainer.last_auc or 0

            if not metrics["featureImportance"] and hasattr(self.ml_model, '_feature_importance') and self.ml_model._feature_importance:
                fi = self.ml_model._feature_importance
                total = sum(fi.values()) if fi else 1
                sorted_features = sorted(fi.items(), key=lambda x: x[1], reverse=True)
                metrics["featureImportance"] = [
                    {"name": name, "importance": round(val / total, 4)}
                    for name, val in sorted_features[:20] if val > 0
                ]

            metrics_file = Path("data/model_metrics.json")
            metrics_file.parent.mkdir(parents=True, exist_ok=True)
            metrics_file.write_text(_json.dumps(metrics, indent=2))
        except Exception as e:
            logger.debug(f"Failed to write model metrics: {e}")

    def _write_dashboard_status(self):
        try:
            wib = ZoneInfo("Asia/Kuala_Lumpur")
            now = datetime.now(wib)

            tick = self.mt5.get_tick(self.config.symbol)
            price = 0.0
            spread = 0.0
            price_change = 0.0
            if tick:
                price = (tick.bid + tick.ask) / 2
                spread = (tick.ask - tick.bid) * 100
                price_change = price - self._dash_last_price if self._dash_last_price > 0 else 0
                self._dash_last_price = price
                self._dash_price_history.append(price)

            balance = self.mt5.account_balance or 0
            equity = self.mt5.account_equity or 0
            profit = equity - balance
            self._dash_equity_history.append(equity)
            self._dash_balance_history.append(balance)

            session_name = "Unknown"
            can_trade = False
            try:
                session_info = self.session_filter.get_status_report()
                if session_info:
                    session_name = session_info.get("current_session", "Unknown")
                can_trade, _, _ = self.session_filter.can_trade()
            except Exception:
                pass

            is_golden_time = 19 <= now.hour < 23

            daily_loss = 0.0
            daily_profit = 0.0
            consecutive_losses = 0
            risk_percent = 0.0
            risk_file = Path("data/risk_state.txt")
            if risk_file.exists():
                try:
                    content = risk_file.read_text()
                    for line in content.strip().split("\n"):
                        if ":" in line:
                            key, value = line.split(":", 1)
                            key = key.strip()
                            value = value.strip()
                            if key == "daily_loss":
                                daily_loss = float(value)
                            elif key == "daily_profit":
                                daily_profit = float(value)
                            elif key == "consecutive_losses":
                                consecutive_losses = int(value)
                except Exception:
                    pass
            max_loss = self.config.capital * (self.config.risk.max_daily_loss / 100)
            if max_loss > 0:
                risk_percent = (daily_loss / max_loss) * 100

            smc_data = {
                "signal": getattr(self, "_last_raw_smc_signal", ""),
                "confidence": getattr(self, "_last_raw_smc_confidence", 0.0),
                "reason": getattr(self, "_last_raw_smc_reason", ""),
                "updatedAt": getattr(self, "_last_raw_smc_updated", ""),
            }

            ml_signal = getattr(self, "_last_ml_signal", "")
            ml_conf = getattr(self, "_last_ml_confidence", 0.0)
            ml_prob = getattr(self, "_last_ml_probability", ml_conf)
            ml_data = {
                "signal": ml_signal,
                "confidence": ml_conf,
                "buyProb": ml_prob,           
                "sellProb": 1.0 - ml_prob,    
                "updatedAt": getattr(self, "_last_ml_updated", ""),
            }

            regime_data = {"name": "", "volatility": 0.0, "confidence": 0.0, "updatedAt": ""}
            if hasattr(self, "_last_regime") and self._last_regime:
                regime_data = {
                    "name": self._last_regime.value.replace("_", " ").title(),
                    "volatility": getattr(self, "_last_regime_volatility", 0.0),
                    "confidence": getattr(self, "_last_regime_confidence", 0.0),
                    "updatedAt": getattr(self, "_last_regime_updated", ""),
                }

            positions_list = []
            try:
                positions = self.mt5.get_open_positions(self.config.symbol)
                if positions is not None and not positions.is_empty():
                    for row in positions.iter_rows(named=True):
                        positions_list.append({
                            "ticket": row.get("ticket", 0),
                            "type": "BUY" if row.get("type", 0) == 0 else "SELL",
                            "volume": row.get("volume", 0),
                            "priceOpen": row.get("price_open", 0),
                            "profit": row.get("profit", 0),
                        })
            except Exception:
                pass

            status = {
                "timestamp": now.strftime("%H:%M:%S"),
                "connected": True,
                "price": price,
                "spread": spread,
                "priceChange": price_change,
                "priceHistory": list(self._dash_price_history),
                "balance": balance,
                "equity": equity,
                "profit": profit,
                "equityHistory": list(self._dash_equity_history),
                "balanceHistory": list(self._dash_balance_history),
                "session": session_name,
                "isGoldenTime": is_golden_time,
                "canTrade": can_trade,
                "dailyLoss": daily_loss,
                "dailyProfit": daily_profit,
                "consecutiveLosses": consecutive_losses,
                "riskPercent": risk_percent,
                "smc": smc_data,
                "ml": ml_data,
                "regime": regime_data,
                "positions": positions_list,
                "logs": list(self._dash_logs),
                "settings": {
                    "capitalMode": self.config.capital_mode.value,
                    "capital": self.smart_risk.capital,
                    "riskPerTrade": self.smart_risk.max_loss_per_trade_percent,
                    "maxDailyLoss": self.smart_risk.max_daily_loss_percent,
                    "maxPositions": self.smart_risk.max_concurrent_positions,
                    "maxLotSize": self.smart_risk.max_lot_size,
                    "leverage": self.config.risk.max_leverage,
                    "executionTF": self.config.execution_timeframe,
                    "trendTF": self.config.trend_timeframe,
                    "minRR": 1.5,
                    "mlConfidence": self.config.ml.confidence_threshold,
                    "cooldownSeconds": self.config.thresholds.trade_cooldown_seconds,
                    "symbol": self.config.symbol,
                    "forcedDirection": self._forced_next_direction
                },
                "h1Bias": getattr(self, "_h1_bias_cache", "NEUTRAL"),
                "dynamicThreshold": getattr(self, "_last_dynamic_threshold", self.config.ml.confidence_threshold),
                "marketQuality": getattr(self, "_last_market_quality", "unknown"),
                "marketScore": getattr(self, "_last_market_score", 0),
                "entryFilters": getattr(self, "_last_filter_results", []),
                "riskMode": self._get_risk_mode_status(),
                "cooldown": self._get_cooldown_status(),
                "timeFilter": self._get_time_filter_status(),
                "sessionMultiplier": getattr(self, "_current_session_multiplier", 1.0),
                "positionDetails": self._get_position_details(),
                "autoTrainer": self._get_auto_trainer_status(),
                "performance": self._get_performance_status(),
                "marketClose": self._get_market_close_status(),
                "h1BiasDetails": {
                    "bias": getattr(self, "_h1_bias_cache", "NEUTRAL"),
                    "score": getattr(self, "_h1_bias_score", 0.0),
                    "strength": getattr(self, "_h1_bias_strength", "weak"),
                    "indicators": getattr(self, "_h1_bias_signals", {}),
                    "regimeWeights": getattr(self, "_h1_bias_regime_weights", "unknown"),
                    "ema20": getattr(self, "_h1_ema20_value", 0.0),
                    "price": getattr(self, "_h1_current_price", 0.0),
                },
            }

            json_data = json.dumps(status, default=str)
            status_path = str(self._dash_status_file)
            for attempt in range(3):
                try:
                    with open(status_path, "w", encoding="utf-8") as f:
                        f.write(json_data)
                    break
                except (PermissionError, OSError) as e:
                    if attempt < 2:
                        import time as _time
                        _time.sleep(0.05)
                    else:
                        logger.debug(f"Dashboard write failed after 3 attempts: {e}")

        except Exception as e:
            logger.debug(f"Dashboard status write error: {e}")

    def _get_risk_mode_status(self) -> dict:
        try:
            rec = self.smart_risk.get_trading_recommendation()
            return {
                "mode": rec.get("mode", "normal"),
                "reason": rec.get("reason", ""),
                "recommendedLot": rec.get("recommended_lot", 0.01),
                "maxAllowedLot": rec.get("max_lot", 0.03),
                "totalLoss": rec.get("total_loss", 0.0),
                "maxTotalLoss": self.smart_risk.max_total_loss_usd,
                "remainingDailyRisk": rec.get("remaining_daily_risk", 0.0),
            }
        except Exception:
            return {"mode": "unknown", "reason": "", "recommendedLot": 0.01, "maxAllowedLot": 0.03, "totalLoss": 0.0, "maxTotalLoss": 0.0, "remainingDailyRisk": 0.0}

    def _get_cooldown_status(self) -> dict:
        try:
            if self._last_trade_time:
                elapsed = (datetime.now() - self._last_trade_time).total_seconds()
                remaining = max(0, self._trade_cooldown_seconds - elapsed)
                return {
                    "active": remaining > 0,
                    "secondsRemaining": round(remaining),
                    "totalSeconds": self._trade_cooldown_seconds,
                }
            return {"active": False, "secondsRemaining": 0, "totalSeconds": self._trade_cooldown_seconds}
        except Exception:
            return {"active": False, "secondsRemaining": 0, "totalSeconds": 150}

    def _get_time_filter_status(self) -> dict:
        try:
            wib_hour = datetime.now(ZoneInfo("Asia/Kuala_Lumpur")).hour
            blocked_hours = []  
            return {
                "wibHour": wib_hour,
                "isBlocked": wib_hour in blocked_hours,
                "blockedHours": blocked_hours,
            }
        except Exception:
            return {"wibHour": 0, "isBlocked": False, "blockedHours": [9, 21]}

    def _get_position_details(self) -> list:
        details = []
        try:
            for ticket, guard in self.smart_risk._position_guards.items():
                trade_hours = (datetime.now(ZoneInfo("Asia/Kuala_Lumpur")) - guard.entry_time).total_seconds() / 3600
                drawdown_pct = 0.0
                if guard.peak_profit > 0:
                    drawdown_pct = ((guard.peak_profit - guard.current_profit) / guard.peak_profit) * 100

                details.append({
                    "ticket": ticket,
                    "peakProfit": guard.peak_profit,
                    "drawdownFromPeak": round(drawdown_pct, 1),
                    "momentum": round(guard.momentum_score, 1),
                    "tpProbability": round(guard.get_tp_probability(), 1),
                    "reversalWarnings": guard.reversal_warnings,
                    "stalls": guard.stall_count,
                    "tradeHours": round(trade_hours, 1),
                })
        except Exception:
            pass
        return details

    def _get_auto_trainer_status(self) -> dict:
        try:
            hours_since = 0.0
            if self.auto_trainer._last_retrain_time:
                hours_since = (datetime.now(ZoneInfo("Asia/Kuala_Lumpur")) - self.auto_trainer._last_retrain_time).total_seconds() / 3600

            current_auc = self.auto_trainer._current_auc
            if current_auc is None and hasattr(self.ml_model, '_train_metrics') and self.ml_model._train_metrics:
                tm = self.ml_model._train_metrics
                current_auc = tm.get("test_auc") or tm.get("xgb_test_score") or tm.get("test_accuracy")

            if current_auc is not None:
                import math
                if math.isnan(current_auc) or math.isinf(current_auc):
                    current_auc = None

            return {
                "lastRetrain": self.auto_trainer._last_retrain_time.strftime("%Y-%m-%d %H:%M") if self.auto_trainer._last_retrain_time else None,
                "currentAuc": current_auc,
                "minAucThreshold": self.auto_trainer.min_auc_threshold,
                "hoursSinceRetrain": round(hours_since, 1),
                "nextRetrainHour": self.auto_trainer.daily_retrain_hour,
                "modelsFitted": self.ml_model.fitted and self.regime_detector.fitted,
            }
        except Exception:
            return {"lastRetrain": None, "currentAuc": None, "minAucThreshold": 0.65, "hoursSinceRetrain": 0, "nextRetrainHour": 5, "modelsFitted": False}

    def _get_performance_status(self) -> dict:
        try:
            uptime_hours = (datetime.now() - self._start_time).total_seconds() / 3600
            avg_ms = 0.0
            if self._execution_times:
                recent = self._execution_times[-20:]
                avg_ms = (sum(recent) / len(recent)) * 1000

            return {
                "loopCount": self._loop_count,
                "avgExecutionMs": round(avg_ms, 1),
                "uptimeHours": round(uptime_hours, 1),
                "totalSessionTrades": self._total_session_trades,
                "totalSessionWins": self._total_session_wins,
                "totalSessionProfit": round(self._total_session_profit, 2),
                "winRate": round(self._total_session_wins / self._total_session_trades * 100, 1) if self._total_session_trades > 0 else 0,
            }
        except Exception:
            return {"loopCount": 0, "avgExecutionMs": 0, "uptimeHours": 0, "totalSessionTrades": 0, "totalSessionWins": 0, "totalSessionProfit": 0, "winRate": 0}

    def _get_market_close_status(self) -> dict:
        try:
            now = datetime.now(ZoneInfo("Asia/Kuala_Lumpur"))
            daily_close_hour = 5
            if now.hour >= daily_close_hour:
                hours_to_daily = (24 - now.hour + daily_close_hour) + (0 - now.minute) / 60
            else:
                hours_to_daily = (daily_close_hour - now.hour) + (0 - now.minute) / 60

            weekday = now.weekday() 
            if weekday < 4:  
                days_to_fri = 4 - weekday
                hours_to_weekend = days_to_fri * 24 + (daily_close_hour - now.hour)
            elif weekday == 4:  
                hours_to_weekend = max(0, (24 + daily_close_hour - now.hour))
            else:  
                hours_to_weekend = 0

            market_open = weekday < 5 and (now.hour >= 6 or now.hour < 4)

            return {
                "hoursToDailyClose": round(max(0, hours_to_daily), 1),
                "hoursToWeekendClose": round(max(0, hours_to_weekend), 1),
                "nearWeekend": weekday == 4 and now.hour >= 20,
                "marketOpen": market_open,
            }
        except Exception:
            return {"hoursToDailyClose": 0, "hoursToWeekendClose": 0, "nearWeekend": False, "marketOpen": False}

    async def start(self):
        try:
            from src.version import get_detailed_version, __exit_strategy__
            version_str = get_detailed_version()
            exit_str = __exit_strategy__
        except ImportError:
            version_str = "v0.0.0 (Core)"
            exit_str = "Exit v5.0"

        logger.info("=" * 60)
        logger.info(f"XAUBOT AI {version_str} (ADAPTIVE WIN/LOSS ALTERNATOR)")
        logger.info(f"Strategy: {exit_str}")
        logger.info("=" * 60)
        logger.info(f"Symbol: {self.config.symbol}")
        logger.info(f"Capital: ${self.config.capital:,.2f}")
        logger.info(f"Mode: {self.config.capital_mode.value}")
        logger.info(f"Simulation: {self.simulation}")
        logger.info("=" * 60)
        
        if not self._load_models():
            logger.error("Models not loaded. Please run train_models.py first!")
            logger.info("Run: python train_models.py")
            return
        
        try:
            self.mt5.connect()
            logger.info("MT5 connected successfully!")
            
            balance = self.mt5.account_balance
            equity = self.mt5.account_equity
            logger.info(f"Account Balance: ${balance:,.2f}")
            logger.info(f"Account Equity: ${equity:,.2f}")

            try:
                risk_file = Path("data/risk_state.txt")
                if risk_file.exists():
                    content = risk_file.read_text()
                    d_loss = 0.0
                    d_profit = 0.0
                    for line in content.strip().split("\n"):
                        if ":" in line:
                            key, value = line.split(":", 1)
                            if key.strip() == "daily_loss": d_loss = float(value.strip())
                            elif key.strip() == "daily_profit": d_profit = float(value.strip())
                    
                    net_daily = d_profit - d_loss
                    logger.info("=" * 60)
                    logger.info(f"TODAY'S P/L SO FAR: ${net_daily:+,.2f} (Profit: ${d_profit:.2f} | Loss: ${d_loss:.2f})")
                    logger.info("=" * 60)
            except Exception as e:
                logger.debug(f"Could not read daily P/L on startup: {e}")

            session_status = self.session_filter.get_status_report()
            logger.info(f"Session: {session_status['current_session']} ({session_status['volatility']} vol)")
            logger.info(f"Can Trade: {session_status['can_trade']} - {session_status['reason']}")

            self._daily_start_balance = balance
            self._start_time = datetime.now()
            self.telegram.set_daily_start_balance(balance)

            await self.notifications.send_startup()

        except Exception as e:
            logger.error(f"Failed to connect to MT5: {e}")
            if not self.simulation:
                return

        self._register_telegram_commands()
        self._sync_position_guards()

        self._running = True
        self._dash_log("info", "Bot started - trading loop active")
        logger.info("Starting main trading loop...")
        await self._main_loop()
    
    async def stop(self):
        logger.info("Stopping trading bot...")
        self._running = False

        await self.notifications.send_shutdown()
        try:
            await self.telegram.close()
        except Exception as e:
            logger.error(f"Failed to close telegram session: {e}")

        self.mt5.disconnect()
        self._log_summary()
    
    def _sync_position_guards(self):
        try:
            open_positions = self.mt5.get_open_positions(
                symbol=self.config.symbol,
                magic=self.config.magic_number,
            )
            mt5_tickets = set()
            if open_positions is not None and not open_positions.is_empty():
                mt5_tickets = set(open_positions["ticket"].to_list())

            stale_guards = set(self.smart_risk._position_guards.keys()) - mt5_tickets
            for ticket in stale_guards:
                self.smart_risk.unregister_position(ticket)

            if stale_guards:
                logger.info(f"Cleaned up {len(stale_guards)} stale position guards: {stale_guards}")
            logger.info(f"Position guards synced: {len(self.smart_risk._position_guards)} active (MT5 has {len(mt5_tickets)} positions)")
        except Exception as e:
            logger.warning(f"Position guard sync failed: {e}")

    def _get_available_features(self, df: pl.DataFrame) -> list:
        # 1. ALWAYS prioritize the exact order expected by the XGBoost booster to avoid Feature Mismatch Error
        try:
            if hasattr(self.ml_model, "xgb_model") and self.ml_model.xgb_model is not None:
                booster = self.ml_model.xgb_model
                if hasattr(booster, "feature_names") and booster.feature_names:
                    return list(booster.feature_names)
        except Exception:
            pass

        # 2. Fallback to model's saved feature names
        if self.ml_model.fitted and self.ml_model.feature_names:
            return list(self.ml_model.feature_names)

        # 3. Fallback to default
        default_features = get_default_feature_columns()
        return [f for f in default_features if f in df.columns]

    _SIGNAL_PERSISTENCE_FILE = "data/signal_persistence.json"

    def _load_signal_persistence(self) -> dict:
        import json, os
        try:
            if os.path.exists(self._SIGNAL_PERSISTENCE_FILE):
                with open(self._SIGNAL_PERSISTENCE_FILE, "r") as f:
                    raw = json.load(f)
                result = {k: (v[0], v[1]) for k, v in raw.items()}
                logger.info(f"Loaded signal persistence: {result}")
                return result
        except Exception as e:
            logger.debug(f"Could not load signal persistence: {e}")
        return {}

    def _save_signal_persistence(self):
        import json, os
        try:
            os.makedirs(os.path.dirname(self._SIGNAL_PERSISTENCE_FILE), exist_ok=True)
            with open(self._SIGNAL_PERSISTENCE_FILE, "w") as f:
                json.dump(self._signal_persistence, f)
        except Exception as e:
            logger.debug(f"Could not save signal persistence: {e}")

    def _get_h1_bias(self) -> str:
        try:
            if hasattr(self, '_h1_bias_cache') and hasattr(self, '_h1_bias_loop'):
                if self._loop_count - self._h1_bias_loop < 4:
                    return self._h1_bias_cache

            df_h1 = self.mt5.get_market_data(
                symbol=self.config.symbol,
                timeframe="H1",
                count=100,
            )

            if len(df_h1) < 30:
                return "NEUTRAL"

            df_h1 = self.features.calculate_all(df_h1, include_ml_features=False)
            df_h1 = self.smc.calculate_all(df_h1)
            self._h1_df_cached = df_h1  

            last = df_h1.row(-1, named=True)
            price = last["close"]
            ema_9 = last["ema_9"]
            ema_21 = last["ema_21"]
            rsi = last["rsi"]
            macd_hist = last["macd_histogram"]

            signals = {
                "ema_trend": 1 if price > ema_21 else (-1 if price < ema_21 else 0),
                "ema_cross": 1 if ema_9 > ema_21 else (-1 if ema_9 < ema_21 else 0),
                "rsi": 1 if rsi > 55 else (-1 if rsi < 45 else 0),
                "macd": 1 if macd_hist > 0 else (-1 if macd_hist < 0 else 0),
                "candles": self._count_candle_bias(df_h1),
            }

            weights = self._get_regime_weights()
            score = sum(signals[k] * weights[k] for k in signals)

            if score >= 0.3:
                bias = "BULLISH"
            elif score <= -0.3:
                bias = "BEARISH"
            else:
                bias = "NEUTRAL"

            abs_score = abs(score)
            if abs_score >= 0.7:
                strength = "strong"
            elif abs_score >= 0.5:
                strength = "moderate"
            else:
                strength = "weak"

            self._h1_bias_cache = bias
            self._h1_bias_loop = self._loop_count
            self._h1_bias_score = float(score)
            self._h1_bias_strength = strength
            self._h1_bias_signals = signals.copy()
            _regime_str = self._last_regime.value if hasattr(self, '_last_regime') and self._last_regime else "unknown"
            self._h1_bias_regime_weights = _regime_str
            self._h1_current_price = float(price)
            self._h1_ema20_value = float(ema_21)

            if self._loop_count % 4 == 0:
                logger.info(
                    f"H1 Bias: {bias} ({strength}, score={score:.2f}) | "
                    f"Signals: EMA_trend={signals['ema_trend']:+d}, EMA_cross={signals['ema_cross']:+d}, "
                    f"RSI={signals['rsi']:+d}, MACD={signals['macd']:+d}, Candles={signals['candles']:+d} | "
                    f"Regime: {_regime_str}"
                )

            return bias

        except Exception as e:
            logger.debug(f"H1 dynamic bias error: {e}")
            return "NEUTRAL"

    def _count_candle_bias(self, df_h1) -> int:
        try:
            last_5 = df_h1.tail(5)
            bullish = sum(1 for row in last_5.iter_rows(named=True) if row["close"] > row["open"])
            bearish = 5 - bullish

            if bullish >= 3:
                return 1
            elif bearish >= 3:
                return -1
            else:
                return 0
        except Exception:
            return 0

    def _get_regime_weights(self) -> dict:
        regime = (self._last_regime.value if hasattr(self, '_last_regime') and self._last_regime else "medium_volatility").lower()

        if "low" in regime or "ranging" in regime:
            return {
                "ema_trend": 0.15,
                "ema_cross": 0.15,
                "rsi": 0.30,
                "macd": 0.25,
                "candles": 0.15,
            }
        elif "high" in regime or "trending" in regime:
            return {
                "ema_trend": 0.30,
                "ema_cross": 0.25,
                "rsi": 0.10,
                "macd": 0.25,
                "candles": 0.10,
            }
        else:
            return {
                "ema_trend": 0.25,
                "ema_cross": 0.20,
                "rsi": 0.20,
                "macd": 0.20,
                "candles": 0.15,
            }

    def _is_filter_enabled(self, filter_key: str) -> bool:
        return self.filter_config.is_enabled(filter_key)

    def _register_telegram_commands(self):
        from src.telegram_commands import register_commands
        register_commands(self)

    async def _main_loop(self):
        last_position_check = time.time()

        while self._running:
            loop_start = time.perf_counter()

            try:
                model_path = "models/xgboost_model.pkl"
                if os.path.exists(model_path):
                    current_mtime = os.path.getmtime(model_path)
                    if current_mtime > self._last_model_update_time:
                        logger.info("=" * 60)
                        logger.info("🔥 HOT RELOAD: New AI brain detected from Continuous Trainer!")
                        logger.info("=" * 60)
                        
                        self.regime_detector.load()
                        self.ml_model.load()
                        
                        self._last_model_update_time = current_mtime
                        logger.info("✅ New AI models successfully loaded into live trading memory.")

                if date.today() != self._current_date:
                    self._on_new_day()

                if not self.mt5.ensure_connected():
                    logger.warning("MT5 disconnected, attempting reconnection...")
                    await asyncio.sleep(10)  
                    continue

                df_check = self.mt5.get_market_data(
                    symbol=self.config.symbol,
                    timeframe=self.config.execution_timeframe,
                    count=2,
                )

                if len(df_check) == 0:
                    logger.warning("No data received from MT5")
                    await asyncio.sleep(5)
                    continue

                current_candle_time = df_check["time"].tail(1).item()

                is_new_candle = True

                if is_new_candle:
                    self._last_candle_time = current_candle_time
                    await self._trading_iteration()
                    self._loop_count += 1

                    if self._loop_count % 4 == 0:  
                        avg_time = sum(self._execution_times[-4:]) / min(4, len(self._execution_times)) if self._execution_times else 0
                        logger.info(f"Candle #{self._loop_count} | Avg execution: {avg_time*1000:.1f}ms")

                    if self._loop_count % 20 == 0:
                        await self._check_auto_retrain()
                else:
                    if time.time() - last_position_check >= self._position_check_interval:
                        await self._position_check_only()
                        last_position_check = time.time()

            except Exception as e:
                logger.error(f"Loop error: {e}")
                import traceback
                logger.debug(traceback.format_exc())

            execution_time = time.perf_counter() - loop_start
            self._execution_times.append(execution_time)

            self._write_dashboard_status()

            try:
                await self.telegram.poll_commands()
            except Exception:
                pass

            await asyncio.sleep(2)

    async def _position_check_only(self):
        try:
            tick = self.mt5.get_tick(self.config.symbol)
            if not tick:
                return
            current_price = tick.bid

            df_mini = self.mt5.get_market_data(
                symbol=self.config.symbol,
                timeframe=self.config.execution_timeframe,
                count=5,
            )
            if len(df_mini) > 0:
                is_flash, move_pct = self.flash_crash.detect(df_mini)
                if is_flash:
                    logger.warning(f"FLASH CRASH detected between candles: {move_pct:.2f}% move!")
                    try:
                        await self._emergency_close_all()
                    except Exception as e:
                        logger.critical(f"CRITICAL: Emergency close failed: {e}")
                        await self.notifications.send_flash_crash_critical(move_pct, e)
                    return

            open_positions = self.mt5.get_open_positions(
                symbol=self.config.symbol,
                magic=self.config.magic_number,
            )

            # === BROKER CLOSE CATCHER (The Win/Loss Flipper) ===
            fresh_mt5 = open_positions
            mt5_tickets = set()
            if fresh_mt5 is not None and not fresh_mt5.is_empty():
                mt5_tickets = set(fresh_mt5["ticket"].to_list())
                
            stale = set(self.smart_risk._position_guards.keys()) - mt5_tickets
            for ticket in stale:
                guard = self.smart_risk._position_guards[ticket]
                last_profit = guard.current_profit
                trade_dir = guard.direction
                
                logger.info(f"Position #{ticket} closed by Broker S/L or T/P. Profit: ${last_profit:.2f}")
                self.smart_risk.record_trade_result(last_profit)
                
                # ---> WIN/LOSS ALTERNATOR LOGIC (Broker Close) <---
                if last_profit > 1.00: 
                    self._forced_next_direction = trade_dir 
                    logger.warning(f"🟢 Trade WON (+${last_profit:.2f}). Next trade stays {self._forced_next_direction}!")
                elif last_profit > -1.50:
                    self._forced_next_direction = trade_dir
                    logger.warning(f"🟡 Trade SCRATCHED/BREAKEVEN (${last_profit:.2f}). Ignoring noise, next trade stays {self._forced_next_direction}!")
                else:
                    self._forced_next_direction = "BUY" if trade_dir == "SELL" else "SELL"
                    logger.warning(f"🔴 Trade LOST (${last_profit:.2f}). Next trade FLIPS to {self._forced_next_direction}!")
                    
                self._save_forced_direction(self._forced_next_direction) 
                # --------------------------------------------------
                
                self.smart_risk.unregister_position(ticket)
                self.position_manager._peak_profits.pop(ticket, None)
                self._pyramid_done_tickets.discard(ticket)

            if len(open_positions) > 0 and not self.simulation:
                cached_ml = getattr(self, '_cached_ml_prediction', None)
                cached_df = getattr(self, '_cached_df', None)
                cached_regime = None
                if hasattr(self, '_last_regime') and self._last_regime:
                    cached_regime = RegimeState(
                        regime=self._last_regime,
                        volatility=getattr(self, '_last_regime_volatility', 0.0),
                        confidence=getattr(self, '_last_regime_confidence', 0.0),
                        probabilities={},
                        recommendation="TRADE",
                    )

                if cached_ml and cached_df is not None and len(cached_df) > 0:
                    await self._smart_position_management(
                        open_positions=open_positions,
                        df=cached_df,
                        regime_state=cached_regime,
                        ml_prediction=cached_ml,
                        current_price=current_price,
                    )
                else:
                    mtf_df = self._build_wide_mtf_features()
                    
                    is_mtf_model = False
                    if self.ml_model.feature_names:
                        is_mtf_model = any(f.startswith("M5_") or f.startswith("M1_") for f in self.ml_model.feature_names)
                    
                    if is_mtf_model and mtf_df is not None:
                        if "regime" not in mtf_df.columns:
                            mtf_df = mtf_df.with_columns(pl.lit(1).alias("regime"))
                        if "regime_confidence" not in mtf_df.columns:
                            mtf_df = mtf_df.with_columns(pl.lit(1.0).alias("regime_confidence"))
                        
                        feature_cols = self._get_available_features(mtf_df)
                        missing_cols = [c for c in feature_cols if c not in mtf_df.columns]
                        if missing_cols:
                            mtf_df = mtf_df.with_columns([pl.lit(0.0).alias(c) for c in missing_cols])
                            
                        raw_ml_prediction = self.ml_model.predict(mtf_df, feature_cols)
                        
                        if self._forced_next_direction and raw_ml_prediction.signal != self._forced_next_direction:
                            ml_prediction = self._invert_ml_prediction(raw_ml_prediction)
                        else:
                            ml_prediction = raw_ml_prediction
                            
                        eval_df = mtf_df
                    else:
                        df_fallback = self.mt5.get_market_data(self.config.symbol, self.config.execution_timeframe, count=50)
                        if len(df_fallback) == 0: return
                        df_fallback = self.features.calculate_all(df_fallback, include_ml_features=True)
                        
                        feature_cols = self._get_available_features(df_fallback)
                        missing_cols = [c for c in feature_cols if c not in df_fallback.columns]
                        if missing_cols:
                            df_fallback = df_fallback.with_columns([pl.lit(0.0).alias(c) for c in missing_cols])
                            
                        raw_ml_prediction = self.ml_model.predict(df_fallback, feature_cols)
                        
                        if self._forced_next_direction and raw_ml_prediction.signal != self._forced_next_direction:
                            ml_prediction = self._invert_ml_prediction(raw_ml_prediction)
                        else:
                            ml_prediction = raw_ml_prediction
                            
                        eval_df = df_fallback
                        
                    await self._smart_position_management(
                        open_positions=open_positions,
                        df=eval_df,
                        regime_state=cached_regime,
                        ml_prediction=ml_prediction,
                        current_price=current_price,
                    )
            if len(open_positions) > 0 and not self.simulation:
                await self._check_pyramid_opportunity(open_positions, current_price)

        except Exception as e:
            logger.debug(f"Position check error: {e}")

    async def _check_pyramid_opportunity(self, open_positions, current_price: float):
        try:
            if self._last_pyramid_time:
                seconds_since = (datetime.now() - self._last_pyramid_time).total_seconds()
                if seconds_since < 30:
                    return

            can_open, limit_reason = self.smart_risk.can_open_position()
            if not can_open:
                return

            session_info = self.session_filter.get_status_report()
            session_name = session_info.get("current_session", "Unknown")
            if session_name not in ("London", "New York", "London-NY Overlap"):
                return

            cached_smc_signal = getattr(self, '_last_raw_smc_signal', '')
            cached_smc_conf = getattr(self, '_last_raw_smc_confidence', 0.0)
            cached_ml = getattr(self, '_cached_ml_prediction', None)

            if not cached_smc_signal or not cached_ml:
                return

            _current_atr = 0.0
            _baseline_atr = 0.0
            cached_df = getattr(self, '_cached_df', None)
            if cached_df is not None and "atr" in cached_df.columns:
                atr_series = cached_df["atr"].drop_nulls()
                if len(atr_series) > 0:
                    _current_atr = atr_series.tail(1).item() or 0
                if len(atr_series) >= 96:
                    _baseline_atr = atr_series.tail(96).mean()
                elif len(atr_series) >= 20:
                    _baseline_atr = atr_series.mean()

            for row in open_positions.iter_rows(named=True):
                ticket = row["ticket"]
                profit = row.get("profit", 0)
                
                # --- FIX: Handle MT5 returning string values ---
                position_type = row.get("type", 0)  
                direction = "BUY" if position_type in [0, "BUY", "Buy", "buy"] else "SELL"
                # -----------------------------------------------
                
                lot_size = row.get("volume", 0.01)

                if ticket in self._pyramid_done_tickets:
                    continue

                atr_dollars = _current_atr * lot_size * 100 if _current_atr > 0 else 0
                sm = max(0.3, min(1.5, _current_atr / _baseline_atr)) if _baseline_atr > 0 else 1.0
                atr_unit = atr_dollars if atr_dollars > 0 else 10 * sm
                min_profit_for_pyramid = 0.5 * atr_unit  

                if profit < min_profit_for_pyramid:
                    continue

                guard = self.smart_risk._position_guards.get(ticket)
                if guard and guard.velocity <= 0:
                    continue  

                if cached_smc_signal != direction or cached_smc_conf < 0.75:
                    continue

                if cached_ml.signal != direction:
                    continue

                logger.info(f"[PYRAMID] Conditions met for #{ticket}: profit=${profit:.2f}, "
                           f"SMC={cached_smc_signal}({cached_smc_conf:.0%}), ML={cached_ml.signal}({cached_ml.confidence:.0%})")

                last_signal = getattr(self, '_last_signal', None)
                if not last_signal:
                    logger.debug("[PYRAMID] No cached signal available")
                    continue

                tick = self.mt5.get_tick(self.config.symbol)
                if not tick:
                    continue

                entry_price = tick.ask if direction == "BUY" else tick.bid

                pyramid_signal = SMCSignal(
                    signal_type=direction,
                    entry_price=entry_price,
                    stop_loss=last_signal.stop_loss,
                    take_profit=last_signal.take_profit,
                    confidence=cached_smc_conf,
                    reason=f"PYRAMID: Add to winner #{ticket} (profit=${profit:.2f})",
                )

                sl_distance = abs(entry_price - pyramid_signal.stop_loss)
                risk_amount = lot_size * sl_distance * 10
                account_balance = self.mt5.account_balance or self.config.capital
                risk_percent = (risk_amount / account_balance) * 100

                pyramid_pos = SimpleNamespace(
                    lot_size=lot_size,  
                    risk_amount=risk_amount,
                    risk_percent=risk_percent,
                )

                cached_regime = None
                if hasattr(self, '_last_regime') and self._last_regime:
                    cached_regime = RegimeState(
                        regime=self._last_regime,
                        volatility=getattr(self, '_last_regime_volatility', 0.0),
                        confidence=getattr(self, '_last_regime_confidence', 0.0),
                        probabilities={},
                        recommendation="TRADE",
                    )

                logger.info(f"[PYRAMID] Opening {direction} {lot_size} lot @ {entry_price:.2f} "
                           f"(adding to winner #{ticket})")

                trade_time_before = self._last_trade_time
                await self._execute_trade_safe(pyramid_signal, pyramid_pos, cached_regime)

                if self._last_trade_time != trade_time_before:
                    self._pyramid_done_tickets.add(ticket)
                    self._last_pyramid_time = datetime.now()
                    self._dash_log("trade", f"PYRAMID: {direction} {lot_size} lot (adding to #{ticket}, profit=${profit:.2f})")
                else:
                    logger.warning(f"[PYRAMID] Trade execution failed for #{ticket}, will retry next cycle")

                break

        except Exception as e:
            logger.debug(f"Pyramid check error: {e}")

    def _build_wide_mtf_features(self) -> Optional[pl.DataFrame]:
        import pandas as pd
        timeframes = ["M1", "M5", "M15", "M30", "H1"]
        live_state = {}
        
        for tf in timeframes:
            df_tf = self.mt5.get_market_data(
                symbol=self.config.symbol, 
                timeframe=tf, 
                count=200
            )
            if len(df_tf) == 0:
                continue
                
            df_tf = self.features.calculate_all(df_tf, include_ml_features=True)
            df_tf = self.smc.calculate_all(df_tf)
            
            live_row = df_tf.tail(1).to_pandas()
            live_row.columns = [f"{tf}_{col}" if col != "time" else col for col in live_row.columns]
            live_state[tf] = live_row
            
        if "M1" in live_state and "M5" in live_state:
            merged = live_state["M1"]
            for tf in ["M5", "M15", "M30", "H1"]:
                if tf in live_state:
                    merged = pd.concat([
                        merged.reset_index(drop=True), 
                        live_state[tf].drop(columns=['time'], errors='ignore').reset_index(drop=True)
                    ], axis=1)
            
            return pl.from_pandas(merged)
            
        return None

    async def _trading_iteration(self):
        # Add this guard to prevent multiple verifications running at the same time
        if getattr(self, '_is_verifying_trade', False):
            return

        self._last_filter_results = []
        self.filter_config.load()

        df = self.mt5.get_market_data(
            symbol=self.config.symbol,
            timeframe=self.config.execution_timeframe,
            count=200,
        )
        
        if len(df) == 0:
            logger.warning("No data received")
            return
        
        df = self.features.calculate_all(df, include_ml_features=True)
        df = self.smc.calculate_all(df)

        if self._h1_df_cached is None:
            self._get_h1_bias()

        df = self.fe_v2.add_all_v2_features(df, self._h1_df_cached)

        try:
            df = self.regime_detector.predict(df)
            regime_state = self.regime_detector.get_current_state(df)
            
            if hasattr(self, '_last_regime') and self._last_regime != regime_state.regime:
                logger.info(f"Regime changed: {self._last_regime.value} -> {regime_state.regime.value}")
            self._last_regime = regime_state.regime
            self._last_regime_volatility = regime_state.volatility
            self._last_regime_confidence = regime_state.confidence
            self._last_regime_updated = datetime.now(ZoneInfo("Asia/Kuala_Lumpur")).strftime("%H:%M:%S")
            
        except Exception as e:
            logger.warning(f"Regime detection error: {e}")
            regime_state = None

        if "regime" not in df.columns:
            df = df.with_columns(pl.lit(1).alias("regime"))
        if "regime_confidence" not in df.columns:
            df = df.with_columns(pl.lit(1.0).alias("regime_confidence"))

        is_flash, move_pct = self.flash_crash.detect(df.tail(5))
        flash_enabled = self._is_filter_enabled("flash_crash_guard")
        flash_blocked = is_flash and flash_enabled
        self._last_filter_results.append({
            "name": "Flash Crash Guard",
            "passed": not flash_blocked,
            "detail": f"{move_pct:.2f}% move" if is_flash else "OK" + (" [DISABLED]" if not flash_enabled else "")
        })
        if flash_blocked:
            logger.warning(f"Flash crash detected: {move_pct:.2f}% move")
            try:
                await self._emergency_close_all()
            except Exception as e:
                logger.critical(f"CRITICAL: Emergency close failed completely: {e}")
                await self.notifications.send_flash_crash_critical(move_pct, e)
            return

        account_balance = self.mt5.account_balance or self.config.capital
        account_equity = self.mt5.account_equity or self.config.capital
        self.smart_risk.update_capital(account_equity)

        open_positions = self.mt5.get_open_positions(
            symbol=self.config.symbol,
            magic=self.config.magic_number,
        )

        tick = self.mt5.get_tick(self.config.symbol)
        current_price = tick.bid if tick else df["close"].tail(1).item()

        mtf_df = self._build_wide_mtf_features()
        
        is_mtf_model = False
        if self.ml_model.feature_names:
            is_mtf_model = any(f.startswith("M5_") or f.startswith("M1_") for f in self.ml_model.feature_names)
            
        # 1. GENERATE RAW SIGNALS (MTF vs Single TF)
        if is_mtf_model and mtf_df is not None:
            if "regime" not in mtf_df.columns:
                reg_map = {"low_volatility": 0, "medium_volatility": 1, "high_volatility": 2, "crisis": 3}
                reg_str = regime_state.regime.value if regime_state else "medium_volatility"
                reg_val = reg_map.get(reg_str, 1)
                mtf_df = mtf_df.with_columns(pl.lit(reg_val).cast(pl.Int32).alias("regime"))
                
            if "regime_confidence" not in mtf_df.columns:
                reg_conf = regime_state.confidence if regime_state else 1.0
                mtf_df = mtf_df.with_columns(pl.lit(reg_conf).cast(pl.Float64).alias("regime_confidence"))
                
            feature_cols = self._get_available_features(mtf_df)
            missing_cols = [c for c in feature_cols if c not in mtf_df.columns]
            if missing_cols:
                mtf_df = mtf_df.with_columns([pl.lit(0.0).alias(c) for c in missing_cols])
                
            raw_ml_prediction = self.ml_model.predict(mtf_df, feature_cols)
            raw_smc_signal = self.smc.generate_signal(df)
            
            
        else:
            feature_cols = self._get_available_features(df)
            missing_cols = [c for c in feature_cols if c not in df.columns]
            if missing_cols:
                df = df.with_columns([pl.lit(0.0).alias(c) for c in missing_cols])
                
            raw_ml_prediction = self.ml_model.predict(df, feature_cols)
            raw_smc_signal = self.smc.generate_signal(df)

        # 2. APPLY FORCED DIRECTION LOGIC (Alternator)
        if self._forced_next_direction:
            mode_tag = f"[FORCED {self._forced_next_direction}]"
            
            # Invert ML Bias if it disagrees
            if raw_ml_prediction.signal != self._forced_next_direction:
                ml_prediction = self._invert_ml_prediction(raw_ml_prediction)
            else:
                ml_prediction = raw_ml_prediction
                
            # CRITICAL FIX: Discard SMC if it disagrees (Do not invert chart geometry!)
            if raw_smc_signal and raw_smc_signal.signal_type != self._forced_next_direction:
                logger.debug(f"Discarding SMC {raw_smc_signal.signal_type}. Waiting for a structural {self._forced_next_direction} pattern.")
                smc_signal = None
            else:
                smc_signal = raw_smc_signal
                
        else:
            mode_tag = "[STANDARD]"
            ml_prediction = raw_ml_prediction
            smc_signal = raw_smc_signal
            
        self._cached_ml_prediction = ml_prediction
                
        self._cached_df = df

        self._last_ml_signal = ml_prediction.signal
        self._last_ml_confidence = ml_prediction.confidence
        self._last_ml_probability = ml_prediction.probability
        self._last_ml_updated = datetime.now(ZoneInfo("Asia/Kuala_Lumpur")).strftime("%H:%M:%S")

        _wib_now = datetime.now(ZoneInfo("Asia/Kuala_Lumpur")).strftime("%H:%M:%S")
        if smc_signal:
            self._last_raw_smc_signal = smc_signal.signal_type
            self._last_raw_smc_confidence = smc_signal.confidence
            self._last_raw_smc_reason = smc_signal.reason
            self._last_raw_smc_updated = _wib_now
            self._dash_log("trade", f"{mode_tag} SMC: {smc_signal.signal_type} ({smc_signal.confidence:.0%}) - {smc_signal.reason}")
        else:
            self._last_raw_smc_signal = ""
            self._last_raw_smc_confidence = 0.0
            self._last_raw_smc_reason = ""
            self._last_raw_smc_updated = _wib_now

        h1_bias = self._get_h1_bias()

        if len(open_positions) > 0:
            if not self.simulation:
                await self._smart_position_management(
                    open_positions=open_positions,
                    df=df,
                    regime_state=regime_state,
                    ml_prediction=ml_prediction,
                    current_price=current_price,
                )

            if self._loop_count % 60 == 0:
                total_profit = 0
                for row in open_positions.iter_rows(named=True):
                    total_profit += row.get("profit", 0)
                logger.info(f"Positions: {len(open_positions)} | Total P/L: ${total_profit:.2f}")

        await self.notifications.send_hourly_analysis_if_due(
            df=df,
            regime_state=regime_state,
            ml_prediction=ml_prediction,
            open_positions=open_positions,
            current_price=current_price,
        )

        risk_metrics = self.risk_engine.check_risk(
            account_balance=account_balance,
            account_equity=account_equity,
            open_positions=open_positions,
            current_price=current_price,
        )
        
        regime_sleep = regime_state and regime_state.recommendation == "SLEEP"
        regime_enabled = self._is_filter_enabled("regime_filter")
        regime_blocked = regime_sleep and regime_enabled
        self._last_filter_results.append({
            "name": "Regime Filter",
            "passed": not regime_blocked,
            "detail": (regime_state.regime.value if regime_state else "N/A") + (" [DISABLED]" if not regime_enabled else "")
        })
        if regime_blocked:
            logger.debug(f"Regime SLEEP: {regime_state.regime.value}")
            return

        risk_enabled = self._is_filter_enabled("risk_check")
        risk_blocked = not risk_metrics.can_trade and risk_enabled
        self._last_filter_results.append({
            "name": "Risk Check",
            "passed": not risk_blocked,
            "detail": (risk_metrics.reason if not risk_metrics.can_trade else "OK") + (" [DISABLED]" if not risk_enabled else "")
        })
        if risk_blocked:
            logger.debug(f"Risk blocked: {risk_metrics.reason}")
            return

        session_ok, session_reason, session_multiplier = self.session_filter.can_trade()
        session_enabled = self._is_filter_enabled("session_filter")
        session_blocked = not session_ok and session_enabled
        self._last_filter_results.append({
            "name": "Session Filter",
            "passed": not session_blocked,
            "detail": session_reason + (" [DISABLED]" if not session_enabled else "")
        })
        if session_blocked:
            if self._loop_count % 300 == 0: 
                logger.info(f"Session filter: {session_reason}")
                next_window = self.session_filter.get_next_trading_window()
                logger.info(f"Next trading window: {next_window['session']} in {next_window['hours_until']} hours")
            #return

        self._current_session_multiplier = session_multiplier
        self._is_sydney_session = "Sydney" in session_reason or session_multiplier == 0.5

        if self._loop_count % 4 == 0:
            price = df["close"].tail(1).item()
            h1_tag = f" | H1: {h1_bias}" if h1_bias != "NEUTRAL" else ""
            direction_tag = f" | Mode: {self._forced_next_direction}" if self._forced_next_direction else ""
            logger.info(f"Price: {price:.2f} | Regime: {regime_state.regime.value if regime_state else 'N/A'} | SMC: {smc_signal.signal_type if smc_signal else 'NONE'} | ML: {ml_prediction.signal}({ml_prediction.confidence:.0%}){h1_tag}{direction_tag}")

        self._last_filter_results.append({"name": "SMC Signal", "passed": smc_signal is not None, "detail": f"{smc_signal.signal_type} ({smc_signal.confidence:.0%})" if smc_signal else "No signal"})

        final_signal = self._combine_signals(smc_signal, ml_prediction, regime_state)
        signal_enabled = self._is_filter_enabled("signal_combination")
        signal_blocked = final_signal is None and signal_enabled
        self._last_filter_results.append({
            "name": "Signal Combination",
            "passed": not signal_blocked,
            "detail": (f"{final_signal.signal_type} ({final_signal.confidence:.0%})" if final_signal else "Filtered out") + (" [DISABLED]" if not signal_enabled else "")
        })

        if signal_blocked:
            return

        h1_enabled = False 
        h1_passed = True  
        h1_detail = f"H1={h1_bias}"
        h1_penalty = 1.0

        if h1_enabled and final_signal is not None:
            h1_opposed = (
                (final_signal.signal_type == "BUY" and h1_bias == "BEARISH") or
                (final_signal.signal_type == "SELL" and h1_bias == "BULLISH")
            )
            h1_aligned = (
                (final_signal.signal_type == "BUY" and h1_bias == "BULLISH") or
                (final_signal.signal_type == "SELL" and h1_bias == "BEARISH")
            )

            if h1_aligned:
                h1_penalty = 1.05  
                h1_detail = f"Aligned {h1_bias} (+5%)"
                logger.info(f"H1 Filter: {final_signal.signal_type} aligned with H1={h1_bias} (+5% boost)")
            elif h1_opposed:
                logger.info(f"H1 Filter: {final_signal.signal_type} BLOCKED because H1={h1_bias}")
                return None
            else:
                logger.debug(f"H1 Filter: NEUTRAL — no adjustment")

            final_signal.confidence *= h1_penalty

        self._last_filter_results.append({"name": "H1 Bias (#31B)", "passed": True, "detail": h1_detail})

        wib_hour = datetime.now(ZoneInfo("Asia/Kuala_Lumpur")).hour
        time_blocked = False  
        time_enabled = self._is_filter_enabled("time_filter")
        time_filter_blocked = time_blocked and time_enabled

        is_night_hours = wib_hour >= 22 or wib_hour <= 5
        session_info = self.session_filter.get_status_report()
        current_session_name = session_info.get("current_session", "")
        is_golden = "GOLDEN" in current_session_name.upper()
        night_spread_ok = True
        night_spread_msg = ""
        if is_night_hours:
            tick = self.mt5.get_tick(self.config.symbol)
            if tick:
                current_spread_points = (tick.ask - tick.bid) / 0.01  
                night_max_spread = 80 if is_golden else 50
                if current_spread_points > night_max_spread:
                    night_spread_ok = False
                    night_spread_msg = f"spread {current_spread_points:.1f}p > {night_max_spread}p"
                else:
                    session_tag = " [GOLDEN]" if is_golden else ""
                    night_spread_msg = f"spread {current_spread_points:.1f}p OK{session_tag} (limit {night_max_spread}p)"

        self._last_filter_results.append({
            "name": "Time Filter (#34A)",
            "passed": not time_filter_blocked and night_spread_ok,
            "detail": f"WIB {wib_hour}" + (" BLOCKED" if time_blocked else "") + (" [DISABLED]" if not time_enabled else "") + (f" NIGHT: {night_spread_msg}" if is_night_hours else "")
        })
        if time_filter_blocked:
            logger.info(f"Time Filter: {final_signal.signal_type} blocked (WIB hour {wib_hour} is skip hour)")
            return
        if not night_spread_ok:
            logger.warning(f"Night Safety: {final_signal.signal_type} blocked - {night_spread_msg} (WIB {wib_hour})")
            return

        cooldown_blocked = False
        cooldown_remaining = 0
        
        if self._last_trade_time:
            time_since_last = (datetime.now() - self._last_trade_time).total_seconds()
            cooldown_remaining = self._trade_cooldown_seconds - time_since_last
            if cooldown_remaining > 0:
                cooldown_blocked = True
                
        cooldown_enabled = self._is_filter_enabled("cooldown")
        cooldown_filter_blocked = cooldown_blocked and cooldown_enabled
        
        self._last_filter_results.append({
            "name": "Trade Cooldown",
            "passed": not cooldown_filter_blocked,
            "detail": (f"{cooldown_remaining:.1f}s left" if cooldown_blocked else "OK") + (" [DISABLED]" if not cooldown_enabled else "")
        })
        
        if cooldown_filter_blocked:
            logger.info(f"Trade cooldown: {cooldown_remaining:.1f}s remaining before next trade allowed.")
            return

        can_trade_pullback, pb_reason = self._check_pullback_filter(df, final_signal.signal_type, current_price)
        
        # --- NEW: PEAK / BOTTOM PREVENTION (RSI & STOCH) ---
        rsi_val = df["rsi"].drop_nulls().tail(1).item() if "rsi" in df.columns else 50
        stoch_k = df["stoch_k"].drop_nulls().tail(1).item() if "stoch_k" in df.columns else 50
        
        if final_signal.signal_type == "BUY" and (rsi_val > 70 or stoch_k > 80):
            can_trade_pullback = False
            pb_reason = f"BUY blocked: Market is OVERBOUGHT at peak (RSI: {rsi_val:.1f})"
        elif final_signal.signal_type == "SELL" and (rsi_val < 30 or stoch_k < 20):
            can_trade_pullback = False
            pb_reason = f"SELL blocked: Market is OVERSOLD at bottom (RSI: {rsi_val:.1f})"
        # --------------------------------------------------

        self._last_filter_results.append({
            "name": "Pullback & Extreme Filter", 
            "passed": can_trade_pullback,  
            "detail": pb_reason
        })
        
        # ACTUALLY BLOCK THE TRADE IF OVEREXTENDED
        if not can_trade_pullback:
            logger.info(f"🚫 Entry Filter Blocked: {pb_reason}")
            return
        
        mtf_passed, mtf_detail = await self._check_mtf_confluence(final_signal.signal_type)
        
        if "AI OVERRIDE" in final_signal.reason:
            mtf_passed = True
            mtf_detail = "Bypassed MTF Filter (AI Genius Override in control)"

        self._last_filter_results.append({
            "name": "MTF Confluence (M5/15/30)", 
            "passed": True,  
            "detail": mtf_detail + " [IGNORED]"
        })
        
        if not mtf_passed:
            # Changed from WARNING to INFO, and removed the 'return'
            # Now the bot will take the trade even if the higher timeframes lag!
            logger.info(f"MTF warning ignored: {mtf_detail}")

        # --- 3. EXHAUSTION / BORDER FILTER (RSI) ---
        if df is not None and "rsi" in df.columns:
            current_rsi = df["rsi"].tail(1).item()
            if current_rsi is not None:
                # Do not BUY if the market is already at the ceiling (>70 RSI)
                if final_signal.signal_type == "BUY" and current_rsi > 70:
                    logger.warning(f"🚫 BUY blocked: RSI is {current_rsi:.1f} (Overbought). Market is at the ceiling!")
                    self._last_filter_results.append({"name": "Border Filter", "passed": False, "detail": f"RSI {current_rsi:.1f} > 70"})
                    return  # Stops execution
                
                # Do not SELL if the market is already at the floor (<30 RSI)
                elif final_signal.signal_type == "SELL" and current_rsi < 30:
                    logger.warning(f"🚫 SELL blocked: RSI is {current_rsi:.1f} (Oversold). Market is at the floor!")
                    self._last_filter_results.append({"name": "Border Filter", "passed": False, "detail": f"RSI {current_rsi:.1f} < 30"})
                    return  # Stops execution

        self.smart_risk.check_new_day()
        risk_rec = self.smart_risk.get_trading_recommendation()
        self._last_filter_results.append({"name": "Smart Risk Gate", "passed": risk_rec["can_trade"], "detail": risk_rec.get("reason", risk_rec["mode"])})

        if not risk_rec["can_trade"]:
            logger.warning(f"Smart Risk: Trading blocked - {risk_rec['reason']}")
            return

        regime_name = regime_state.regime.value if regime_state else "normal"
        safe_lot = self.smart_risk.calculate_lot_size(
            entry_price=final_signal.entry_price,
            confidence=final_signal.confidence,
            regime=regime_name,
            ml_confidence=ml_prediction.confidence,  
        )

        if safe_lot <= 0:
            logger.debug("Smart Risk: Lot size is 0 (Trade Rejected) - skipping trade")
            return

        # === ADAPTIVE CANDLE BEHAVIOR FILTER (FIXED) ===
        # We WANT to buy on dips (red candles) and sell on rallies (green candles)
        current_open = df["open"].tail(1).item()
        
        target_direction = self._forced_next_direction or final_signal.signal_type
        
        if target_direction == "BUY" and current_price > current_open:
            logger.info(f"🚫 Candle Behavior: BUY blocked - Candle is currently GREEN (Wait for a dip/red candle). Open: {current_open:.2f}, Price: {current_price:.2f}")
            return
        elif target_direction == "SELL" and current_price < current_open:
            logger.info(f"🚫 Candle Behavior: SELL blocked - Candle is currently RED (Wait for a rally/green candle). Open: {current_open:.2f}, Price: {current_price:.2f}")
            return

        session_mult = getattr(self, '_current_session_multiplier', 1.0)
        if session_mult < 1.0:
            original_lot = safe_lot
            safe_lot = max(0.01, round(safe_lot * session_mult, 2))  
            sydney_mode = getattr(self, '_is_sydney_session', False)
            if sydney_mode:
                logger.info(f"Sydney SAFE MODE: Lot {original_lot:.2f} -> {safe_lot:.2f} (0.5x)")

        wib_hour = datetime.now(ZoneInfo("Asia/Kuala_Lumpur")).hour
        is_night_hours = wib_hour >= 22 or wib_hour <= 5
        if is_night_hours:
            original_lot = safe_lot
            safe_lot = max(0.01, round(safe_lot * 0.5, 2))  
            logger.warning(f"NIGHT SAFETY MODE: Lot {original_lot:.2f} -> {safe_lot:.2f} (0.5x) - WIB {wib_hour}:xx")
            
        # safe_lot = 0.01

        from dataclasses import dataclass

        @dataclass
        class SafePosition:
            lot_size: float
            risk_amount: float
            risk_percent: float

        sl_distance = abs(final_signal.entry_price - final_signal.stop_loss)
        risk_amount = safe_lot * sl_distance * 10  
        risk_percent = (risk_amount / account_balance) * 100

        position_result = SafePosition(
            lot_size=safe_lot,
            risk_amount=risk_amount,
            risk_percent=risk_percent,
        )

        logger.info(f"Smart Risk: Lot={safe_lot}, Risk=${risk_amount:.2f} ({risk_percent:.2f}%), Mode={risk_rec['mode']}")

        can_open, limit_reason = self.smart_risk.can_open_position()
        self._last_filter_results.append({"name": "Position Limit", "passed": can_open, "detail": limit_reason if not can_open else "OK"})

        # --- BULLETPROOF HEDGING & STACKING PREVENTION ---
        opposite_dir_count = 0
        same_dir_count = 0
        
        for p in open_positions.iter_rows(named=True):
            p_type = p.get("type", 0)
            is_p_buy = p_type in [0, "BUY", "Buy", "buy"]
            is_signal_buy = final_signal.signal_type == "BUY"
            
            if is_p_buy == is_signal_buy:
                same_dir_count += 1
            else:
                opposite_dir_count += 1

        # 1. Prevent Hedging (Mixed Buy/Sell)
        if opposite_dir_count > 0:
            logger.warning(f"Hedging Prevention: Already holding opposite trades. Skipping {final_signal.signal_type} entry until cleared.")
            return
            
        # 2. Prevent Over-Stacking (Pyramiding Limit)
        if same_dir_count >= self.smart_risk.max_concurrent_positions:
            logger.warning(f"Stacking Prevention: Max positions ({self.smart_risk.max_concurrent_positions}) reached for {final_signal.signal_type}. Skipping new entry.")
            return
        # -------------------------------------------------

        current_type = 0 if final_signal.signal_type == "BUY" else 1
        same_dir_count = sum(1 for p in open_positions.iter_rows(named=True) if p.get("type", -1) == current_type)
        
        # Allow multiple trades up to max limit (for pyramiding)
        if same_dir_count >= self.smart_risk.max_concurrent_positions:
            logger.warning(f"Stacking Prevention: Max positions ({self.smart_risk.max_concurrent_positions}) reached for {final_signal.signal_type}. Skipping new entry.")
            return  

        # REMOVE OR COMMENT OUT THIS LINE:
        # await self._execute_trade_safe(final_signal, position_result, regime_state)
        
        # REPLACE IT WITH THIS:
        asyncio.create_task(self._verify_and_execute_delayed(final_signal, position_result, regime_state))
    
    def _combine_signals(
        self,
        smc_signal: Optional[SMCSignal],
        ml_prediction,
        regime_state,
    ) -> Optional[SMCSignal]:
        tick = self.mt5.get_tick(self.config.symbol)
        current_price = tick.bid if tick else 0

        session_status = self.session_filter.get_status_report()
        session_name = session_status.get("current_session", "Unknown")
        volatility = session_status.get("volatility", "medium")

        trend_direction = "NEUTRAL"
        if hasattr(self, '_last_regime') and regime_state:
            trend_direction = regime_state.regime.value

        market_analysis = self.dynamic_confidence.analyze_market(
            session=session_name,
            regime=regime_state.regime.value if regime_state else "unknown",
            volatility=volatility,
            trend_direction=trend_direction,
            has_smc_signal=(smc_signal is not None),
            ml_signal=ml_prediction.signal,
            ml_confidence=ml_prediction.confidence,
        )

        dynamic_threshold = market_analysis.confidence_threshold
        self._last_dynamic_threshold = dynamic_threshold
        self._last_market_quality = market_analysis.quality.value
        self._last_market_score = market_analysis.score

        if self._loop_count % 60 == 0:
            logger.info(f"Dynamic: {market_analysis.quality.value} (score={market_analysis.score}) -> threshold={dynamic_threshold:.0%}")

        from datetime import datetime
        from zoneinfo import ZoneInfo
        from src.regime_detector import MarketRegime
        
        current_hour = datetime.now(ZoneInfo("Asia/Kuala_Lumpur")).hour
        is_golden_time = 19 <= current_hour <= 23 

        if market_analysis.quality.value == "avoid":
            if self._loop_count % 120 == 0:
                logger.info(f"Skip: Market quality AVOID - BUT FORCING TRADE")
            # return None   <--- COMMENT THIS OUT

        if regime_state and regime_state.regime == MarketRegime.CRISIS:
            if self._loop_count % 120 == 0:
                logger.info(f"Skip: CRISIS regime - BUT FORCING TRADE")
            # return None   <--- COMMENT THIS OUT

        is_london = session_name == "London"
        atr_ratio = 1.0
        london_penalty = 1.0  
        cached_df = getattr(self, '_cached_df', None)
        if cached_df is not None and "atr" in cached_df.columns:
            atr_series = cached_df["atr"].drop_nulls()
            if len(atr_series) > 0:
                current_atr = atr_series.tail(1).item() or 0
                if len(atr_series) >= 96:
                    baseline_atr = atr_series.tail(96).mean()
                    atr_ratio = current_atr / baseline_atr if baseline_atr > 0 else 1.0

        if is_london and atr_ratio < 1.2:
            london_penalty = 0.90
            if self._loop_count % 120 == 0:
                logger.info(
                    f"[LONDON LOW VOL] ATR {atr_ratio:.2f}x → "
                    f"Confidence penalty 10% (whipsaw risk)"
                )

        needs_override = smc_signal is None or smc_signal.signal_type != ml_prediction.signal

        if ml_prediction.signal in ["BUY", "SELL"] and ml_prediction.confidence >= 0.85 and needs_override:
            direction = ml_prediction.signal
            
            atr = 15.0 
            if cached_df is not None and "atr" in cached_df.columns:
                val = cached_df["atr"].drop_nulls().tail(1).item()
                if val and val > 0:
                    atr = val
            
            if tick:
                entry_price = tick.ask if direction == "BUY" else tick.bid
                
                sl_dist = atr * 2.0 
                tp_dist = atr * 3.0 
                
                if direction == "BUY":
                    sl, tp = entry_price - sl_dist, entry_price + tp_dist
                else:
                    sl, tp = entry_price + sl_dist, entry_price - tp_dist
                    
                logger.info(f"🤖 AI GENIUS OVERRIDE: Forcing {direction} trade! (ML Confidence: {ml_prediction.confidence:.0%})")
                
                return SMCSignal(
                    signal_type=direction,
                    entry_price=entry_price,
                    stop_loss=sl,
                    take_profit=tp,
                    confidence=ml_prediction.confidence,
                    reason=f"AI OVERRIDE: ML highly confident ({ml_prediction.confidence:.0%})",
                )

        golden_marker = "[GOLDEN] " if is_golden_time else ""
        if smc_signal is not None:
            smc_conf = smc_signal.confidence

            if smc_conf < 0.55:
                if self._loop_count % 120 == 0:
                    logger.info(f"[SMC LOW] {smc_signal.signal_type} confidence {smc_conf:.0%} < 55% -> Skip")
                return None

            prob = ml_prediction.probability
            
            # Widen the neutral band so AI doesn't block trades unless it is highly confident
            ml_is_strongly_bullish = prob >= 0.65
            ml_is_strongly_bearish = prob <= 0.35
            ml_is_neutral = 0.35 <= prob < 0.65

            if ml_is_neutral:
                if smc_conf >= 0.60:
                    combined_confidence = smc_conf * 0.90 
                    reason_suffix = f" | AI NEUTRAL ({prob:.1%}) -> Trusting SMC"
                    logger.info(f"⚖️ AI Neutral ({prob:.1%}). Trusting SMC {smc_signal.signal_type} ({smc_conf:.0%}).")
                else:
                    logger.warning(f"🚫 BLOCKED: AI is Neutral ({prob:.1%}) and SMC is weak ({smc_conf:.0%}).")
                    return None
            else:
                ml_agrees = (
                    (smc_signal.signal_type == "BUY" and ml_is_strongly_bullish) or
                    (smc_signal.signal_type == "SELL" and ml_is_strongly_bearish)
                )

                if ml_agrees:
                    combined_confidence = smc_conf
                    reason_suffix = f" | AI STRONG AGREE ({prob:.1%})"
                    logger.info(f"✅ CONFLUENCE: SMC {smc_signal.signal_type} + AI Strongly Agrees ({prob:.1%})")
                else:
                    # THE TIE-BREAKER: If AI disagrees, but SMC is extremely confident (>= 80%)
                    if smc_conf >= 0.80:
                        combined_confidence = smc_conf * 0.80 # Penalize confidence by 20% to reduce lot size
                        reason_suffix = f" | SMC OVERRIDE (AI Disagreed at {prob:.1%})"
                        logger.warning(f"⚠️ TIE-BREAKER: AI disagrees ({prob:.1%}) but SMC is VERY STRONG ({smc_conf:.0%}). Trusting SMC!")
                    else:
                        logger.warning(f"🚫 BLOCKED: SMC says {smc_signal.signal_type} and AI disagrees ({prob:.1%}).")
                        return None

            combined_confidence *= london_penalty
            if regime_state and regime_state.regime == MarketRegime.HIGH_VOLATILITY:
                combined_confidence *= 0.9

            logger.info(
                f"{golden_marker}[SMC-ONLY] {smc_signal.signal_type} @ {smc_signal.entry_price:.2f} "
                f"(SMC={smc_conf:.0%}, ML={ml_prediction.signal} {ml_prediction.confidence:.0%}, "
                f"Final={combined_confidence:.0%})"
            )

            return SMCSignal(
                signal_type=smc_signal.signal_type,
                entry_price=smc_signal.entry_price,
                stop_loss=smc_signal.stop_loss,
                take_profit=smc_signal.take_profit,
                confidence=combined_confidence,
                reason=f"SMC-ONLY: {smc_signal.reason}{reason_suffix}",
            )

        return None

    def _check_pullback_filter(
        self,
        df: pl.DataFrame,
        signal_direction: str,
        current_price: float,
    ) -> Tuple[bool, str]:
        """
        Check if price is in a safe entry zone.
        Forces the bot to buy red candles (dips) and sell green candles (rallies).
        Prevents buying falling knives and parabolic overextensions (Rubber Band effect).
        """
        try:
            recent_5 = df.tail(5)
            recent_20 = df.tail(20) # Look back 20 minutes for parabolic moves

            if len(recent_20) < 20:
                return True, "Not enough data for pullback check"

            atr = 12.0  
            if "atr" in df.columns:
                atr_val = recent_5["atr"].to_list()[-1]
                if atr_val is not None and atr_val > 0:
                    atr = max(atr_val, 3.0)

            ema_9 = recent_5["ema_9"].to_list()[-1] if "ema_9" in df.columns else None
            
            # Look at the current active candle
            current_open = recent_5["open"].to_list()[-1]
            current_close = recent_5["close"].to_list()[-1]
            current_candle_size = current_close - current_open
            
            surge_limit = atr * 0.30
            
            # --- NEW: MACRO TREND SHIELD (50-Period Baseline) ---
            # Calculates the moving average of the last 50 candles to find the real trend
            recent_50 = df.tail(50)
            if len(recent_50) == 50:
                macro_baseline = sum(recent_50["close"].to_list()) / 50.0
            else:
                macro_baseline = current_close # Fallback if not enough data
            # ----------------------------------------------------

            # --- THE RUBBER BAND MATH ---
            recent_floor = min(recent_20["low"].to_list())
            recent_ceiling = max(recent_20["high"].to_list())

            if signal_direction == "BUY":  
                # 0. NEW: MACRO TREND BLOCK
                # If price is significantly below the 50-period average, DO NOT catch the falling knife!
                if current_price < macro_baseline - (atr * 0.5):
                    return False, f"BUY blocked: Macro trend is BEARISH (${(macro_baseline - current_price):.2f} below 50-MA). Do not buy the crash."

                # 1. Block FOMO (Buying a green candle)
                if current_candle_size > 0:
                    return False, f"BUY blocked: Candle is GREEN (+${current_candle_size:.2f}). Wait for a red dip."
                
                # 2. Block Falling Knife (Buying a violent crash)
                if current_candle_size < -surge_limit:
                    return False, f"BUY blocked: Candle is crashing hard RED (${current_candle_size:.2f}). Unsafe to buy."
                
                # 3. Block Parabolic Overextension (The Mount Everest Trap)
                # If price is stretched more than 2.5x ATR from the floor, do NOT buy the top!
                if current_price > recent_floor + (atr * 2.5):
                    return False, f"BUY blocked: Parabolic Overextension. Price is +${(current_price - recent_floor):.2f} above recent floor. Wait for crash."

                # 4. Block Structural Breaks
                if ema_9:
                    if current_price > ema_9 + (atr * 0.4):
                        return False, f"BUY blocked: Price floating too far above EMA9. Wait for pullback."
                    if current_price < ema_9 - (atr * 0.5):
                        return False, f"BUY blocked: Price crashed below EMA9! Trend broken, do not buy."

                return True, "BUY OK: Safe red dip detected."

            elif signal_direction == "SELL":  
                # 0. NEW: MACRO TREND BLOCK
                # If price is significantly above the 50-period average, DO NOT step in front of the train!
                if current_price > macro_baseline + (atr * 0.5):
                    return False, f"SELL blocked: Macro trend is BULLISH (+${(current_price - macro_baseline):.2f} above 50-MA). Do not sell the rally."

                # 1. Block FOMO (Selling a red candle)
                if current_candle_size < 0:
                    return False, f"SELL blocked: Candle is RED (${current_candle_size:.2f}). Wait for a green rally."
                
                # 2. Block Catching Rockets (Selling into a massive green surge)
                if current_candle_size > surge_limit:
                    return False, f"SELL blocked: Candle is surging hard GREEN (+${current_candle_size:.2f}). Unsafe to sell."
                    
                # 3. Block Parabolic Overextension (The Bottomless Pit Trap)
                # If price is stretched more than 2.5x ATR from the ceiling, do NOT sell the bottom!
                if current_price < recent_ceiling - (atr * 2.5):
                    return False, f"SELL blocked: Parabolic Overextension. Price is -${(recent_ceiling - current_price):.2f} below recent peak. Wait for bounce."
                
                # 4. Block Structural Breaks
                if ema_9:
                    if current_price < ema_9 - (atr * 0.4):
                        return False, f"SELL blocked: Price dumped too far below EMA9. Wait for rally."
                    if current_price > ema_9 + (atr * 0.5):
                        return False, f"SELL blocked: Price surged above EMA9! Trend broken, do not sell."

                return True, "SELL OK: Safe green rally detected."

            return True, "Pullback check passed"

        except Exception as e:
            logger.warning(f"Pullback filter error: {e}")
            return True, f"Pullback check error: {e}"

    async def _execute_trade(self, signal: SMCSignal, position):
        logger.info("=" * 50)
        logger.info(f"TRADE SIGNAL: {signal.signal_type}")
        logger.info(f"  Entry: {signal.entry_price:.2f}")
        logger.info(f"  SL: {signal.stop_loss:.2f}")
        logger.info(f"  TP: {signal.take_profit:.2f}")
        logger.info(f"  Lot: {position.lot_size}")
        logger.info(f"  Risk: ${position.risk_amount:.2f} ({position.risk_percent:.2f}%)")
        logger.info(f"  Confidence: {signal.confidence:.2%}")
        logger.info(f"  Reason: {signal.reason}")
        logger.info("=" * 50)
        
        if self.simulation:
            logger.info("[SIMULATION] Trade not executed")
            self._last_signal = signal
            self._last_trade_time = datetime.now()
            return
        
        result = self.mt5.send_order(
            symbol=self.config.symbol,
            order_type=signal.signal_type,
            volume=position.lot_size,
            sl=signal.stop_loss,
            tp=signal.take_profit,
            magic=self.config.magic_number,
            comment="AI Bot",
        )
        
        if result.success:
            logger.info(f"SAFE ORDER EXECUTED! ID: {result.order_id}")
            self._last_signal = signal
            self._last_trade_time = datetime.now()

            expected_price = signal.entry_price
            actual_price = result.price if result.price > 0 else expected_price
            slippage = abs(actual_price - expected_price)
            slippage_pips = slippage * 10  

            requested_volume = position.lot_size
            filled_volume = result.volume if result.volume > 0 else requested_volume

            entry_price_actual = actual_price if actual_price > 0 else signal.entry_price
            lot_size_actual = filled_volume

            self.smart_risk.register_position(
                ticket=result.order_id,
                entry_price=entry_price_actual,  
                lot_size=lot_size_actual,        
                direction=signal.signal_type,
            )

            regime = self._last_regime.value if hasattr(self, '_last_regime') else "unknown"
            session_status = self.session_filter.get_status_report()
            volatility = session_status.get("volatility", "unknown")

            self._open_trade_info[result.order_id] = {
                "entry_price": entry_price_actual,  
                "lot_size": lot_size_actual,        
                "direction": signal.signal_type,
                "open_time": datetime.now(),
            }

            # --- THE FIX: Define Telegram variables FIRST so they never throw UnboundLocalError ---
            smc_fvg, smc_ob, smc_bos, smc_choch = False, False, False, False
            market_quality, market_score, dynamic_threshold = "moderate", 50, 0.7

            try:
                smc_fvg = "FVG" in signal.reason.upper()
                smc_ob = "OB" in signal.reason.upper() or "ORDER BLOCK" in signal.reason.upper()
                smc_bos = "BOS" in signal.reason.upper()
                smc_choch = "CHOCH" in signal.reason.upper()

                market_quality = self.dynamic_confidence._last_quality if hasattr(self.dynamic_confidence, '_last_quality') else "moderate"
                market_score = self.dynamic_confidence._last_score if hasattr(self.dynamic_confidence, '_last_score') else 50
                dynamic_threshold = self.dynamic_confidence._last_threshold if hasattr(self.dynamic_confidence, '_last_threshold') else 0.7

                self.trade_logger.log_trade_open(
                    ticket=result.order_id,
                    symbol=self.config.symbol,
                    direction=signal.signal_type,
                    lot_size=position.lot_size,
                    entry_price=signal.entry_price,
                    stop_loss=0,
                    take_profit=signal.take_profit,
                    regime=regime,
                    volatility=volatility,
                    session=session_status.get("session", "unknown"),
                    spread=self.mt5.get_symbol_info(self.config.symbol).get("spread", 0) if hasattr(self.mt5, 'get_symbol_info') else 0,
                    atr=0,  
                    smc_signal=signal.signal_type,
                    smc_confidence=signal.confidence,
                    smc_reason=signal.reason,
                    smc_fvg=smc_fvg,
                    smc_ob=smc_ob,
                    smc_bos=smc_bos,
                    smc_choch=smc_choch,
                    ml_signal=self._last_ml_signal if hasattr(self, '_last_ml_signal') else "HOLD",
                    ml_confidence=self._last_ml_confidence if hasattr(self, '_last_ml_confidence') else 0.5,
                    market_quality=str(market_quality),
                    market_score=int(market_score) if market_score else 50,
                    dynamic_threshold=float(dynamic_threshold) if dynamic_threshold else 0.7,
                    balance=self.mt5.account_balance,
                    equity=self.mt5.account_equity,
                )
            except Exception as e:
                logger.warning(f"Failed to log trade open locally: {e}")

            # Send Telegram Notification Safely
            try:
                await self.notifications.notify_trade_open(
                    result=result,
                    signal=signal,
                    position=position,
                    regime=regime,
                    volatility=volatility,
                    session_status=session_status,
                    safe_mode=True,
                    smc_fvg=smc_fvg,
                    smc_ob=smc_ob,
                    smc_bos=smc_bos,
                    smc_choch=smc_choch,
                    dynamic_threshold=dynamic_threshold,
                    market_quality=market_quality,
                    market_score=market_score,
                )
            except Exception as e:
                logger.error(f"TELEGRAM FAILED: {e}")
        else:
            logger.error(f"Order failed: {result.comment} (code: {result.retcode})")

    async def _verify_and_execute_delayed(self, original_signal, position_result, original_regime_state):
        """Waits 5 seconds and re-evaluates the trend before executing."""
        self._is_verifying_trade = True
        logger.info(f"⏳ Trade signal found ({original_signal.signal_type}). Waiting 5 seconds to confirm trend...")
        
        try:
            await asyncio.sleep(5)
            
            # Fetch latest data to confirm trend
            df_check = self.mt5.get_market_data(
                symbol=self.config.symbol,
                timeframe=self.config.execution_timeframe,
                count=50,
            )
            
            if len(df_check) == 0:
                logger.warning("Failed to fetch data for 5s verification. Cancelling  trade.")
                return
                
            df_check = self.features.calculate_all(df_check, include_ml_features=True)
            df_check = self.smc.calculate_all(df_check)
            
            # Re-run ML Prediction exactly as done in the main loop
            mtf_df = self._build_wide_mtf_features()
            is_mtf_model = False
            if self.ml_model.feature_names:
                is_mtf_model = any(f.startswith("M5_") or f.startswith("M1_") for f in self.ml_model.feature_names)
                
            if is_mtf_model and mtf_df is not None:
                if "regime" not in mtf_df.columns:
                    reg_map = {"low_volatility": 0, "medium_volatility": 1, "high_volatility": 2, "crisis": 3}
                    reg_str = original_regime_state.regime.value if original_regime_state else "medium_volatility"
                    mtf_df = mtf_df.with_columns(pl.lit(reg_map.get(reg_str, 1)).cast(pl.Int32).alias("regime"))
                if "regime_confidence" not in mtf_df.columns:
                    reg_conf = original_regime_state.confidence if original_regime_state else 1.0
                    mtf_df = mtf_df.with_columns(pl.lit(reg_conf).cast(pl.Float64).alias("regime_confidence"))
                    
                feature_cols = self._get_available_features(mtf_df)
                missing_cols = [c for c in feature_cols if c not in mtf_df.columns]
                if missing_cols:
                    mtf_df = mtf_df.with_columns([pl.lit(0.0).alias(c) for c in missing_cols])
                    
                raw_ml = self.ml_model.predict(mtf_df, feature_cols)
            else:
                feature_cols = self._get_available_features(df_check)
                missing_cols = [c for c in feature_cols if c not in df_check.columns]
                if missing_cols:
                    df_check = df_check.with_columns([pl.lit(0.0).alias(c) for c in missing_cols])
                raw_ml = self.ml_model.predict(df_check, feature_cols)
                
            raw_smc = self.smc.generate_signal(df_check)
            
            # Apply Alternator / Forced Direction Logic
            if getattr(self, '_forced_next_direction', None):
                ml_pred = self._invert_ml_prediction(raw_ml) if raw_ml.signal != self._forced_next_direction else raw_ml
                smc_sig = self._invert_smc_signal(raw_smc) if raw_smc and raw_smc.signal_type != self._forced_next_direction else raw_smc
            else:
                ml_pred = raw_ml
                smc_sig = raw_smc
                
            # --- UPDATED VERIFICATION STEP ---
            # Pass the new signals back through your combination logic so it respects the "Neutral AI / Strong SMC" rules
            new_final_signal = self._combine_signals(smc_sig, ml_pred, original_regime_state)

            if new_final_signal is None or new_final_signal.signal_type != original_signal.signal_type:
                logger.warning(f"❌ 5s Verification Failed! Trend weakened or reversed. Trade cancelled.")
                return
                
            # ---> NEW: RE-CHECK FOMO & PULLBACK AFTER SLEEPING <---
            current_open = df_check["open"].tail(1).item()
            current_price_check = df_check["close"].tail(1).item()
            
            # 1. Re-check Candle Behavior (Don't buy if the candle became a massive green spike while sleeping)
            if new_final_signal.signal_type == "BUY" and current_price_check > current_open:
                logger.warning(f"❌ 5s Verification Failed! Candle surged GREEN during wait. Avoiding FOMO peak.")
                return
            elif new_final_signal.signal_type == "SELL" and current_price_check < current_open:
                logger.warning(f"❌ 5s Verification Failed! Candle dumped RED during wait. Avoiding FOMO bottom.")
                return
                
            # 2. Re-check Pullback Filter (Ensure it hasn't turned into a falling knife)
            can_trade_pb, pb_reason = self._check_pullback_filter(df_check, new_final_signal.signal_type, current_price_check)
            if not can_trade_pb:
                logger.warning(f"❌ 5s Verification Failed! Market overextended during wait: {pb_reason}")
                return
            # --------------------------------------------------------

            # If everything is still aligned and safe, execute!
            logger.info(f"✅ 5s Verification Passed! Trend is still {original_signal.signal_type} and entry is safe. Executing trade...")
            
            # Update entry price to the exact newest tick
            tick = self.mt5.get_tick(self.config.symbol)
            if tick:
                original_signal.entry_price = tick.ask if original_signal.signal_type == "BUY" else tick.bid

            await self._execute_trade_safe(original_signal, position_result, original_regime_state)
            
        except Exception as e:
            logger.error(f"Error during 5s verification: {e}")
        finally:
            self._is_verifying_trade = False

    async def _execute_trade_safe(self, signal: SMCSignal, position, regime_state):
        emergency_sl = self.smart_risk.calculate_emergency_sl(
            entry_price=signal.entry_price,
            direction=signal.signal_type,
            lot_size=position.lot_size,
            symbol=self.config.symbol,
        )

        logger.info("=" * 50)
        logger.info("SAFE TRADE MODE v2 - SMART S/L")
        logger.info("=" * 50)
        logger.info(f"TRADE SIGNAL: {signal.signal_type}")
        logger.info(f"  Entry: {signal.entry_price:.2f}")
        logger.info(f"  TP: {signal.take_profit:.2f}")
        logger.info(f"  Emergency SL: {emergency_sl:.2f} (broker safety net)")
        logger.info(f"  Software S/L: ${self.smart_risk.max_loss_per_trade:.2f} (smart management)")
        logger.info(f"  Lot: {position.lot_size} (Ultra Safe)")
        logger.info(f"  Confidence: {signal.confidence:.2%}")
        logger.info(f"  Reason: {signal.reason}")
        logger.info("=" * 50)

        if self.simulation:
            logger.info("[SIMULATION] Trade not executed")
            self._last_signal = signal
            self._last_trade_time = datetime.now()
            return

        # --- FIX: ATR Volatility Buffer to survive Liquidity Sweeps ---
        atr_buffer = 0.0
        cached_df = getattr(self, '_cached_df', None)
        if cached_df is not None and "atr" in cached_df.columns:
            atr_series = cached_df["atr"].drop_nulls()
            if len(atr_series) > 0:
                # Add 1.5x ATR to the SMC Stop Loss to give the trade breathing room
                atr_buffer = atr_series.tail(1).item() * 1.5  
                
        # --- LIQUIDITY SWEEP SURVIVAL FIX ---
        broker_sl = signal.stop_loss
        tick = self.mt5.get_tick(self.config.symbol)
        current_price = tick.bid if signal.signal_type == "SELL" else tick.ask

        # 1. Dynamic ATR Buffer (Give the trade room to breathe against wicks)
        atr_buffer = 0.0
        cached_df = getattr(self, '_cached_df', None)
        if cached_df is not None and "atr" in cached_df.columns:
            atr_series = cached_df["atr"].drop_nulls()
            if len(atr_series) > 0:
                atr_buffer = atr_series.tail(1).item() * 1.5  # Add 1.5x ATR padding
        
        # 2. Hard Minimum Distance (Gold fluctuates $1-$3 naturally. Force a minimum $4.00 SL)
        min_safe_distance = 4.0 
        
        if signal.signal_type == "BUY":
            broker_sl -= atr_buffer
            if current_price - broker_sl < min_safe_distance:
                broker_sl = current_price - min_safe_distance
        else:  
            broker_sl += atr_buffer
            if broker_sl - current_price < min_safe_distance:
                broker_sl = current_price + min_safe_distance
                
        logger.info(f"🛡️ Adjusted Broker SL to {broker_sl:.2f} to survive liquidity sweeps.")
        # --------------------------------------------------

        logger.info(f"  Broker SL: {broker_sl:.2f} (ATR-based protection)")

        result = self.mt5.send_order(
            symbol=self.config.symbol,
            order_type=signal.signal_type,
            volume=position.lot_size,
            sl=broker_sl,  
            tp=signal.take_profit,
            magic=self.config.magic_number,
            comment="AI Safe v3",
        )

        if not result.success and result.retcode == 10016:
            logger.warning(f"Broker SL rejected, trying without SL...")
            result = self.mt5.send_order(
                symbol=self.config.symbol,
                order_type=signal.signal_type,
                volume=position.lot_size,
                sl=0,  
                tp=signal.take_profit,
                magic=self.config.magic_number,
                comment="AI Safe v3 NoSL",
            )

        if result.success:
            logger.info(f"SAFE ORDER EXECUTED! ID: {result.order_id}")
            self._last_signal = signal
            self._last_trade_time = datetime.now()

            expected_price = signal.entry_price
            actual_price = result.price if result.price > 0 else expected_price
            slippage = abs(actual_price - expected_price)
            slippage_pips = slippage * 10  

            max_slippage = expected_price * 0.0015  

            if slippage > max_slippage:
                logger.warning(f"HIGH SLIPPAGE: Expected {expected_price:.2f}, Got {actual_price:.2f} (slip: ${slippage:.2f} / {slippage_pips:.1f} pips)")
            elif slippage > 0:
                logger.info(f"Slippage OK: ${slippage:.2f} ({slippage_pips:.1f} pips)")

            requested_volume = position.lot_size
            filled_volume = result.volume if result.volume > 0 else requested_volume

            if filled_volume < requested_volume:
                fill_ratio = filled_volume / requested_volume * 100
                logger.warning(f"PARTIAL FILL: Requested {requested_volume}, Got {filled_volume} ({fill_ratio:.1f}%)")
                position.lot_size = filled_volume
            elif filled_volume > 0:
                logger.debug(f"Full fill: {filled_volume} lots")

            entry_price_actual = actual_price if actual_price > 0 else signal.entry_price
            lot_size_actual = filled_volume

            self.smart_risk.register_position(
                ticket=result.order_id,
                entry_price=entry_price_actual,  
                lot_size=lot_size_actual,        
                direction=signal.signal_type,
            )

            regime = self._last_regime.value if hasattr(self, '_last_regime') else "unknown"
            session_status = self.session_filter.get_status_report()
            volatility = session_status.get("volatility", "unknown")

            self._open_trade_info[result.order_id] = {
                "entry_price": entry_price_actual,  
                "expected_price": signal.entry_price,
                "slippage": slippage,
                "lot_size": lot_size_actual,        
                "requested_lot_size": requested_volume,
                "open_time": datetime.now(),
                "balance_before": self.mt5.account_balance,
                "ml_confidence": signal.confidence,
                "regime": regime,
                "volatility": volatility,
                "direction": signal.signal_type,
            }

            try:
                smc_fvg = "FVG" in signal.reason.upper()
                smc_ob = "OB" in signal.reason.upper() or "ORDER BLOCK" in signal.reason.upper()
                smc_bos = "BOS" in signal.reason.upper()
                smc_choch = "CHOCH" in signal.reason.upper()

                market_quality = self.dynamic_confidence._last_quality if hasattr(self.dynamic_confidence, '_last_quality') else "moderate"
                market_score = self.dynamic_confidence._last_score if hasattr(self.dynamic_confidence, '_last_score') else 50
                dynamic_threshold = self.dynamic_confidence._last_threshold if hasattr(self.dynamic_confidence, '_last_threshold') else 0.7

                self.trade_logger.log_trade_open(
                    ticket=result.order_id,
                    symbol=self.config.symbol,
                    direction=signal.signal_type,
                    lot_size=position.lot_size,
                    entry_price=signal.entry_price,
                    stop_loss=0,
                    take_profit=signal.take_profit,
                    regime=regime,
                    volatility=volatility,
                    session=session_status.get("session", "unknown"),
                    spread=self.mt5.get_symbol_info(self.config.symbol).get("spread", 0) if hasattr(self.mt5, 'get_symbol_info') else 0,
                    atr=0,  
                    smc_signal=signal.signal_type,
                    smc_confidence=signal.confidence,
                    smc_reason=signal.reason,
                    smc_fvg=smc_fvg,
                    smc_ob=smc_ob,
                    smc_bos=smc_bos,
                    smc_choch=smc_choch,
                    ml_signal=self._last_ml_signal if hasattr(self, '_last_ml_signal') else "HOLD",
                    ml_confidence=self._last_ml_confidence if hasattr(self, '_last_ml_confidence') else 0.5,
                    market_quality=str(market_quality),
                    market_score=int(market_score) if market_score else 50,
                    dynamic_threshold=float(dynamic_threshold) if dynamic_threshold else 0.7,
                    balance=self.mt5.account_balance,
                    equity=self.mt5.account_equity,
                )
            except Exception as e:
                logger.warning(f"Failed to log trade open: {e}")

            await self.notifications.notify_trade_open(
                result=result,
                signal=signal,
                position=position,
                regime=regime,
                volatility=volatility,
                session_status=session_status,
                safe_mode=True,
                smc_fvg=smc_fvg,
                smc_ob=smc_ob,
                smc_bos=smc_bos,
                smc_choch=smc_choch,
                dynamic_threshold=dynamic_threshold,
                market_quality=market_quality,
                market_score=market_score,
            )
        else:
            logger.error(f"Order failed: {result.comment} (code: {result.retcode})")

    async def _smart_position_management(self, open_positions, df, regime_state, ml_prediction, current_price):
        try:
            fresh_mt5 = self.mt5.get_open_positions(
                symbol=self.config.symbol,
                magic=self.config.magic_number,
            )
            mt5_tickets = set()
            if fresh_mt5 is not None and not fresh_mt5.is_empty():
                mt5_tickets = set(fresh_mt5["ticket"].to_list())
            stale = set(self.smart_risk._position_guards.keys()) - mt5_tickets
            for ticket in stale:
                self.smart_risk.unregister_position(ticket)
                logger.debug(f"Cleaned stale guard #{ticket}")
        except Exception as e:
            logger.debug(f"Guard sync error: {e}")

        if df is not None and len(df) > 0:
            pm_actions = self.position_manager.analyze_positions(
                positions=open_positions,
                df_market=df,
                regime_state=regime_state,
                ml_prediction=ml_prediction,
                current_price=current_price,
            )
            for action in pm_actions:
                if action.action == "TRAIL_SL":
                    result = self.position_manager._modify_sl(action.ticket, action.new_sl)
                    if result["success"]:
                        logger.info(f"Trailing SL #{action.ticket} -> {action.new_sl:.2f}: {action.reason}")
                    else:
                        logger.debug(f"Trail SL failed #{action.ticket}: {result['message']}")
                elif action.action == "CLOSE":
                    logger.info(f"PositionManager Close #{action.ticket}: {action.reason}")
                    result = self.mt5.close_position(action.ticket)
                    if result.success:
                        profit = 0
                        direction = "BUY" # Default fallback
                        
                        # Extract BOTH profit and direction
                        for row in open_positions.iter_rows(named=True):
                            if row["ticket"] == action.ticket:
                                profit = row.get("profit", 0)
                                p_type = row.get("type", 0)
                                direction = "BUY" if p_type in [0, "BUY", "Buy", "buy"] else "SELL"
                                break
                                
                        risk_result = self.smart_risk.record_trade_result(profit)

                        # ---> WIN/LOSS ALTERNATOR LOGIC (Normal Close) <---
                        if profit > 1.00: 
                            self._forced_next_direction = direction 
                            logger.warning(f"🟢 Trade WON (+${profit:.2f}). Next trade stays {self._forced_next_direction}!")
                        elif profit > -1.50:
                            self._forced_next_direction = direction
                            logger.warning(f"🟡 Trade SCRATCHED/BREAKEVEN (${profit:.2f}). Ignoring noise, next trade stays {self._forced_next_direction}!")
                        else:
                            self._forced_next_direction = "BUY" if direction == "SELL" else "SELL"
                            logger.warning(f"🔴 Trade LOST (${profit:.2f}). Next trade FLIPS to {self._forced_next_direction}!")
                            
                        self._save_forced_direction(self._forced_next_direction) 
                        # --------------------------------------------------

                        # ---> WIN/LOSS ALTERNATOR LOGIC (Normal Close) <---
                        if profit > 1.00:
                            self._forced_next_direction = direction
                            logger.warning(f"🟢 Trade WON (+${profit:.2f}). Next trade stays {self._forced_next_direction}!")
                        elif profit > -1.50:  # The Breakeven/Slippage Buffer
                            self._forced_next_direction = direction
                            logger.warning(f"🟡 Trade SCRATCHED/BREAKEVEN (${profit:.2f}). Ignoring noise, next trade stays {self._forced_next_direction}!")
                        else:
                            self._forced_next_direction = "BUY" if direction == "SELL" else "SELL"
                            logger.warning(f"🔴 Trade LOST (${profit:.2f}). Next trade FLIPS to {self._forced_next_direction}!")
                            
                        self._save_forced_direction(self._forced_next_direction)
                        # --------------------------------------------------
                        
                        # ---> 3-LOSS ADAPTIVE REGIME FLIP <---
                        # We let the AI pick the direction. We only invert the logic if the AI gets it wrong 3 times in a row.
                        consecutive_losses = self.smart_risk.get_state().consecutive_losses
                        
                        if profit < 0 and consecutive_losses >= 3:
                            old_mode = getattr(self, "_trading_logic_mode", "STANDARD")
                            self._trading_logic_mode = "STANDARD" if old_mode == "INVERTED" else "INVERTED"
                            
                            # Reset the loss counter so it gets 3 fresh chances
                            self.smart_risk._state.consecutive_losses = 0
                            self.smart_risk._save_daily_state()
                            
                            # Delete the forced direction so the AI can think freely again
                            self._forced_next_direction = None
                            self._save_forced_direction("")
                            
                            logger.warning("=" * 50)
                            logger.warning(f"🔄 MARKET SHIFT DETECTED!")
                            logger.warning(f"Hit 3 losses. Switching bot to {self._trading_logic_mode} logic!")
                            logger.warning("=" * 50)
                        # ----------------------------------------------------
                        self.smart_risk.unregister_position(action.ticket)
                        self.position_manager._peak_profits.pop(action.ticket, None)
                        self._pyramid_done_tickets.discard(action.ticket)  
                        await self.notifications.notify_trade_close_smart(action.ticket, profit, current_price, action.reason)
                        logger.info(f"CLOSED #{action.ticket}: {action.reason}")
                        continue  

        for row in open_positions.iter_rows(named=True):
            ticket = row["ticket"]
            profit = row.get("profit", 0)
            entry_price = row.get("price_open", current_price)
            lot_size = row.get("volume", 0.01)
            
            # --- FIX: Handle MT5 returning string values ---
            position_type = row.get("type", 0)  
            direction = "BUY" if position_type in [0, "BUY", "Buy", "buy"] else "SELL"
            # -----------------------------------------------

            current_positions = self.mt5.get_open_positions(
                symbol=self.config.symbol,
                magic=self.config.magic_number,
            )
            still_open = any(
                r["ticket"] == ticket
                for r in current_positions.iter_rows(named=True)
            ) if len(current_positions) > 0 else False
            if not still_open:
                continue

            if not self.smart_risk.is_position_registered(ticket):
                self.smart_risk.auto_register_existing_position(
                    ticket=ticket,
                    entry_price=entry_price,
                    lot_size=lot_size,
                    direction=direction,
                    current_profit=profit,
                )

            _current_atr = 0.0
            _baseline_atr = 0.0
            if df is not None and "atr" in df.columns:
                atr_series = df["atr"].drop_nulls()
                if len(atr_series) > 0:
                    _current_atr = atr_series.tail(1).item() or 0
                if len(atr_series) >= 96:  
                    _baseline_atr = atr_series.tail(96).mean()
                elif len(atr_series) >= 20:
                    _baseline_atr = atr_series.mean()

            _market_ctx = None
            if df is not None:
                _market_ctx = {}
                for col in ("rsi", "stoch_k", "adx", "histogram"):
                    if col in df.columns:
                        vals = df[col].drop_nulls()
                        _market_ctx[col if col != "histogram" else "macd_hist"] = (
                            vals.tail(1).item() if len(vals) > 0 else None
                        )
                
                # ---> ADD THIS SMC AWARENESS <---
                if "market_structure" in df.columns:
                    struct_val = df["market_structure"].tail(1).item()
                    _market_ctx["smc_trend"] = "BUY" if struct_val == 1 else ("SELL" if struct_val == -1 else "NEUTRAL")
                # --------------------------------
                try:
                    _sess = self.session_filter.get_status_report()
                    _market_ctx["session_name"] = _sess.get("current_session", "")
                    _market_ctx["is_golden"] = "GOLDEN" in _sess.get("current_session", "").upper()
                    _market_ctx["session_volatility"] = _sess.get("volatility", "medium")
                except Exception:
                    _market_ctx["is_golden"] = False

            should_close, reason, message = self.smart_risk.evaluate_position(
                ticket=ticket,
                current_price=current_price,
                current_profit=profit,
                ml_signal=ml_prediction.signal,
                ml_confidence=ml_prediction.confidence,
                regime=regime_state.regime.value if regime_state else "normal",
                current_atr=_current_atr,
                baseline_atr=_baseline_atr,
                market_context=_market_ctx,
            )

            guard = self.smart_risk._position_guards.get(ticket)
            if guard and len(guard.profit_timestamps) >= 2:
                now_ts = time.time()
                if now_ts - guard.last_momentum_log_time >= 30:
                    guard.last_momentum_log_time = now_ts
                    vel_summary = guard.get_velocity_summary()
                    atr_ratio = _current_atr / _baseline_atr if _baseline_atr > 0 else 1.0
                    logger.info(
                        f"[MOMENTUM] #{ticket} profit=${profit:+.2f} | "
                        f"vel={vel_summary['velocity']:.4f}$/s | "
                        f"accel={vel_summary['acceleration']:.4f} | "
                        f"stag={vel_summary['stagnation_s']:.0f}s | "
                        f"ATR={_current_atr:.1f}({atr_ratio:.2f}x) | "
                        f"samples={vel_summary['samples']}"
                    )

            if should_close:
                logger.info(f"Smart Close #{ticket}: {reason.value if reason else 'unknown'} - {message}")

                result = self.mt5.close_position(ticket)
                if result.success:
                    logger.info(f"CLOSED #{ticket}: {message}")

                    risk_result = self.smart_risk.record_trade_result(profit)
                    
                    # ---> WIN/LOSS ALTERNATOR LOGIC <---
                    if profit > 1.00: 
                        self._forced_next_direction = direction 
                        logger.warning(f"🟢 Trade WON (+${profit:.2f}). Next trade stays {self._forced_next_direction}!")
                    elif profit > -1.50:
                        self._forced_next_direction = direction
                        logger.warning(f"🟡 Trade SCRATCHED/BREAKEVEN (${profit:.2f}). Ignoring noise, next trade stays {self._forced_next_direction}!")
                    else:
                        self._forced_next_direction = "BUY" if direction == "SELL" else "SELL"
                        logger.warning(f"🔴 Trade LOST (${profit:.2f}). Next trade FLIPS to {self._forced_next_direction}!")
                        
                    self._save_forced_direction(self._forced_next_direction) 
                    # --------------------------------------------------

                    self.smart_risk.unregister_position(ticket)
                    self._pyramid_done_tickets.discard(ticket)  

                    try:
                        trade_info = self._open_trade_info.get(ticket, {})
                        entry_price = trade_info.get("entry_price", current_price)
                        lot_size = trade_info.get("lot_size", 0.01)

                        pips = abs(current_price - entry_price) * 100
                        if profit < 0:
                            pips = -pips

                        self.trade_logger.log_trade_close(
                            ticket=ticket,
                            exit_price=current_price,
                            profit_usd=profit,
                            profit_pips=pips,
                            exit_reason=reason.value if reason else message[:30],
                            regime=regime_state.regime.value if regime_state else "normal",
                            ml_signal=ml_prediction.signal if ml_prediction else "HOLD",
                            ml_confidence=ml_prediction.confidence if ml_prediction else 0.5,
                            balance_after=self.mt5.account_balance or 0,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to log trade close: {e}")

                    await self.notifications.notify_trade_close_smart(ticket, profit, current_price, message)

                    if risk_result.get("total_limit_hit"):
                        await self.notifications.send_critical_limit_alert(
                            "TOTAL LOSS LIMIT",
                            risk_result.get("total_loss", 0),
                            self.smart_risk.max_total_loss_usd,
                            self.smart_risk.max_total_loss_percent
                        )
                    elif risk_result.get("daily_limit_hit"):
                        await self.notifications.send_critical_limit_alert(
                            "DAILY LOSS LIMIT",
                            risk_result.get("daily_loss", 0),
                            self.smart_risk.max_daily_loss_usd,
                            self.smart_risk.max_daily_loss_percent
                        )
                else:
                    logger.error(f"Failed to close #{ticket}: {result.comment}")
            else:
                if self._loop_count % 60 == 0:
                    logger.info(f"Position #{ticket}: {message}")

    async def _emergency_close_all(self, max_retries: int = 3):
        logger.warning("=" * 50)
        logger.warning("EMERGENCY: Closing all positions!")
        logger.warning("=" * 50)

        if self.simulation:
            return

        failed_tickets = []
        closed_count = 0

        for attempt in range(max_retries):
            try:
                positions = self.mt5.get_open_positions(magic=self.config.magic_number)

                if positions is None or len(positions) == 0:
                    logger.info("No positions to close")
                    break

                for row in positions.iter_rows(named=True):
                    ticket = row["ticket"]
                    try:
                        result = self.mt5.close_position(ticket)
                        if result.success:
                            logger.info(f"Closed position {ticket}")
                            closed_count += 1
                            self._pyramid_done_tickets.discard(ticket)  
                            self.smart_risk.unregister_position(ticket)  
                            if ticket in failed_tickets:
                                failed_tickets.remove(ticket)
                        else:
                            logger.error(f"Failed to close {ticket}: {result.comment}")
                            if ticket not in failed_tickets:
                                failed_tickets.append(ticket)
                    except Exception as e:
                        logger.error(f"Exception closing {ticket}: {e}")
                        if ticket not in failed_tickets:
                            failed_tickets.append(ticket)

                remaining = self.mt5.get_open_positions(magic=self.config.magic_number)
                if remaining is None or len(remaining) == 0:
                    logger.info(f"Emergency close complete: {closed_count} positions closed")
                    break

                if attempt < max_retries - 1:
                    logger.warning(f"Retry {attempt + 2}/{max_retries} - {len(remaining)} positions still open")
                    await asyncio.sleep(2)

            except Exception as e:
                logger.error(f"Emergency close attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)

        await self.notifications.send_emergency_close_result(closed_count, failed_tickets)
    
    def _on_new_day(self):
        logger.info("=" * 60)
        logger.info(f"NEW TRADING DAY: {date.today()}")
        logger.info("=" * 60)

        self._current_date = date.today()
        self.risk_engine.reset_daily_stats()

        self._daily_start_balance = self.mt5.account_balance or self.config.capital
        self.telegram.set_daily_start_balance(self._daily_start_balance)

        self._log_summary()
    
    def _log_summary(self):
        if not self._execution_times:
            return
        
        avg_time = sum(self._execution_times) / len(self._execution_times)
        max_time = max(self._execution_times)
        min_time = min(self._execution_times)
        
        logger.info("=" * 40)
        logger.info("SESSION SUMMARY")
        logger.info(f"Total loops: {self._loop_count}")
        logger.info(f"Avg execution: {avg_time*1000:.2f}ms")
        logger.info(f"Min execution: {min_time*1000:.2f}ms")
        logger.info(f"Max execution: {max_time*1000:.2f}ms")
        
        daily = self.risk_engine.get_daily_summary()
        logger.info(f"Trades today: {daily['trades']}")
        logger.info("=" * 40)

    async def _check_auto_retrain(self):
        try:
            should_train, reason = self.auto_trainer.should_retrain()

            if not should_train:
                logger.debug(f"Auto-retrain check: {reason}")
                return

            logger.info("=" * 50)
            logger.info(f"AUTO-RETRAIN TRIGGERED: {reason}")
            logger.info("=" * 50)

            session_status = self.session_filter.get_status_report()
            if session_status.get("can_trade", True):
                logger.info("Market still open - will retrain when closed")
                return

            open_positions = self.mt5.get_open_positions(
                symbol=self.config.symbol,
                magic=self.config.magic_number,
            )
            if len(open_positions) > 0:
                logger.warning(f"Skipping retrain - {len(open_positions)} open positions")
                return

            is_weekend = self.auto_trainer.should_retrain()[1] == "Weekend deep training time"

            results = self.auto_trainer.retrain(
                connector=self.mt5,
                symbol=self.config.symbol,
                timeframe=self.config.execution_timeframe,
                is_weekend=is_weekend,
            )

            if results["success"]:
                logger.info("Retraining successful! Reloading models...")

                self.regime_detector.load()
                self.ml_model.load()

                logger.info(f"  HMM: {'OK' if self.regime_detector.fitted else 'FAILED'}")
                logger.info(f"  XGBoost: {'OK' if self.ml_model.fitted else 'FAILED'}")
                logger.info(f"  Features: {len(self.ml_model.feature_names) if self.ml_model.feature_names else 0}")
                logger.info(f"  Train AUC: {results.get('xgb_train_auc', 0):.4f}")
                logger.info(f"  Test AUC: {results.get('xgb_test_auc', 0):.4f}")

                self.auto_trainer._current_auc = results.get("xgb_test_auc", 0)
                self._write_model_metrics(retrain_results=results)

                if results.get("xgb_test_auc", 0) < 0.60:
                    logger.warning("New model AUC too low - rolling back!")
                    self.auto_trainer.rollback_models()
                    self.regime_detector.load()
                    self.ml_model.load()
                    logger.info("Rollback complete")
            else:
                logger.error(f"Retraining failed: {results.get('error', 'Unknown error')}")

        except Exception as e:
            logger.error(f"Auto-retrain error: {e}")
            import traceback
            logger.debug(traceback.format_exc())


    async def _check_mtf_confluence(self, signal_direction: str) -> tuple[bool, str]:
        timeframes = ["M5", "M15", "M30"]
        aligned_tfs = []
        
        for tf in timeframes:
            try:
                df_tf = self.mt5.get_market_data(symbol=self.config.symbol, timeframe=tf, count=100)
                if len(df_tf) < 50: continue
                
                df_tf = self.smc.calculate_all(df_tf)
                struct = df_tf["market_structure"].tail(1).item()
                
                # FORCE ALIGNMENT FOR ALTERNATOR
                if self._forced_next_direction:
                    # If we are forcing a trade, bypass MTF structure blocks
                    aligned_tfs.append(tf)
                else:
                    # Standard logic
                    if signal_direction == "BUY" and struct == 1:
                        aligned_tfs.append(tf)
                    elif signal_direction == "SELL" and struct == -1:
                        aligned_tfs.append(tf)
            except Exception:
                pass
                
        is_aligned = len(aligned_tfs) >= 1
        
        reason = f"Aligned on {', '.join(aligned_tfs)}" if is_aligned else f"Not aligned on any timeframe (need at least 1/3)"
        return is_aligned, reason
    
async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Smart AI Trading Bot")
    parser.add_argument("--simulation", "-s", action="store_true", help="Run in simulation mode")
    parser.add_argument("--capital", "-c", type=float, help="Trading capital (override)")
    parser.add_argument("--symbol", type=str, help="Trading symbol (override)")
    args = parser.parse_args()
    
    config = get_config()
    
    if args.capital:
        config = TradingConfig(capital=args.capital, symbol=config.symbol)
    if args.symbol:
        config.symbol = args.symbol
    
    bot = TradingBot(config=config, simulation=args.simulation)
    
    try:
        await bot.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        await bot.stop()


def _acquire_lock():
    lockfile = Path("data/bot.lock")
    lockfile.parent.mkdir(exist_ok=True)

    if lockfile.exists():
        try:
            old_pid = int(lockfile.read_text().strip())
            import subprocess
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {old_pid}", "/NH"],
                capture_output=True, text=True, timeout=5
            )
            if f"{old_pid}" in result.stdout and "python" in result.stdout.lower():
                logger.error(f"ANOTHER BOT INSTANCE IS RUNNING (PID {old_pid})!")
                logger.error("Kill it first: taskkill /F /PID " + str(old_pid))
                sys.exit(1)
            else:
                logger.info(f"Stale lockfile found (PID {old_pid} not running), removing...")
        except (ValueError, Exception) as e:
            logger.warning(f"Could not check lockfile: {e}, removing...")

    lockfile.write_text(str(os.getpid()))
    logger.info(f"Bot lockfile acquired: PID {os.getpid()}")
    return lockfile


def _release_lock():
    lockfile = Path("data/bot.lock")
    try:
        if lockfile.exists():
            stored_pid = int(lockfile.read_text().strip())
            if stored_pid == os.getpid():
                lockfile.unlink()
                logger.info("Bot lockfile released")
    except Exception:
        pass


if __name__ == "__main__":
    lockfile = _acquire_lock()
    try:
        asyncio.run(main())
    finally:
        _release_lock()