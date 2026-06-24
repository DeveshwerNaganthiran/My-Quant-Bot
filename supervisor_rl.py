import asyncio
import random
import logging
import os
import pickle
import json
from datetime import datetime
import pandas as pd
import MetaTrader5 as mt5

# Import your Bots and Configs
from main_live import TradingBot as StandardBot, get_config as get_std_config
from main_live_inverse import TradingBot as InverseBot, get_config as get_inv_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AI_Brain")

class AIBrain:
    def __init__(self):
        # Let the bots actually send orders to MT5
        self.standard_bot = StandardBot(config=get_std_config(), simulation=False)
        self.inverse_bot = InverseBot(config=get_inv_config(), simulation=False)
        
        # 👇 ADD THESE TWO LINES HERE 👇
        self.standard_bot._load_models()
        self.inverse_bot._load_models()
        # 👆 ----------------------- 👆
        
        self.POLLING_RATE = 60      
        
        # --- THE WATCHTOWER MEMORY ---
        self.active_rl_trades = {}
        self.memory_file = "src/backtests/rl_training_memory.csv"
        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
        
        # Load the Meta-Brain model
        try:
            with open("src/backtests/ml_v3/meta_supervisor.pkl", "rb") as f:
                self.model = pickle.load(f)
            self.ai_ready = True
            self.EPSILON = 25 # Switch to 25% random if model is found
        except FileNotFoundError:
            logger.warning("No Meta-Brain found. Defaulting to 100% Random Exploration.")
            self.ai_ready = False
            self.EPSILON = 100

    async def get_current_state(self):
        """Fetches live market data using the standard bot's tools."""
        if not self.standard_bot.mt5.ensure_connected():
            return None

        # Fetch raw data
        df = self.standard_bot.mt5.get_market_data(
            symbol=self.standard_bot.config.symbol,
            timeframe=self.standard_bot.config.execution_timeframe,
            count=100
        )
        
        if df is None or len(df) == 0:
            return None
            
        # Add technical features and SMC data
        df = self.standard_bot.features.calculate_all(df, include_ml_features=True)
        df = self.standard_bot.smc.calculate_all(df)
        
        # Convert the last row (current candle) to a dictionary
        latest_row = df.to_dicts()[-1]
        
        # Check if SMC is giving a signal right now
        signal = self.standard_bot.smc.generate_signal(df)
        latest_row['setup_detected'] = signal is not None
        
        return latest_row

    def write_to_csv(self, features, action, was_random, pnl):
        """Writes the final experience directly to the hard drive."""
        try:
            win = 1 if pnl > 0 else 0
            row_data = {"timestamp": datetime.now().isoformat()}
            row_data.update(features)
            row_data.update({
                "action_taken": action,
                "was_random": int(was_random),
                "pnl": float(pnl),
                "win": win
            })
            
            df_new = pd.DataFrame([row_data])
            if not os.path.exists(self.memory_file):
                df_new.to_csv(self.memory_file, index=False)
            else:
                df_new.to_csv(self.memory_file, mode='a', header=False, index=False)
        except Exception as e:
            logger.error(f"Failed to write to AI CSV: {e}")

    def check_watchtower(self):
        """Asks MT5 directly if our tracked trades have closed!"""
        if not self.active_rl_trades:
            return
            
        mt5.initialize() # Bulletproof MT5 connection refresh
        tickets_to_remove = []
        
        for ticket, data in self.active_rl_trades.items():
            # Ask MT5 if the ticket is still open
            pos = mt5.positions_get(ticket=ticket)
            
            if pos is None or len(pos) == 0:
                # The trade is gone! It must have closed. Get the exact PnL from MT5 History.
                deals = mt5.history_deals_get(position=ticket)
                if deals and len(deals) > 0:
                    # Sum up the profit, commission, and swap fees
                    total_pnl = sum([deal.profit + deal.commission + deal.fee for deal in deals])
                    
                    # WE HAVE EVERYTHING! Save the brain file!
                    self.write_to_csv(data['features'], data['action'], data['was_random'], total_pnl)
                    logger.info(f"🧠 SUCCESS! Supervisor logged AI Memory. Ticket #{ticket} | Action: {data['action']} | PnL: ${total_pnl:.2f}")
                
                tickets_to_remove.append(ticket)
                
        # Stop tracking the closed trades
        for t in tickets_to_remove:
            del self.active_rl_trades[t]

    async def run_realtime(self):
        logger.info("⚡ Real-Time Async AI Brain Activated.")
        
        self.standard_bot.mt5.connect()
        self.inverse_bot.mt5.connect()
        mt5.initialize() # Initialize global MT5
        
        while True:
            try:
                # 0. Watchtower: Check if any trades we are tracking just finished!
                self.check_watchtower()
                
                # 1. Read Market
                current_state = await self.get_current_state()
                
                if current_state and current_state.get('setup_detected'):
                    
                    # 2. Decide (Explore vs Exploit)
                    roll = random.randint(1, 100)
                    is_random = not self.ai_ready or roll <= self.EPSILON
                    
                    if is_random:
                        action = random.choice(["STANDARD", "INVERSE"])
                        logger.info(f"🎲 EXPLORING: Rolled {roll}. Testing {action}.")
                    else:
                        state_df = pd.DataFrame([current_state]).drop(columns=['setup_detected', 'time'], errors='ignore')
                        
                        # --- CRITICAL FIX: CONVERT OBJECTS TO FLOAT FOR XGBOOST ---
                        state_df = state_df.apply(pd.to_numeric, errors='coerce').fillna(0.0)
                        # ----------------------------------------------------------
                        
                        if hasattr(self.model, 'feature_names_in_'):
                            missing_cols = [c for c in self.model.feature_names_in_ if c not in state_df.columns]
                            for col in missing_cols:
                                state_df[col] = 0.0
                            state_df = state_df[self.model.feature_names_in_]
                            
                        # Force the final dataframe to be float32 to match XGBoost expectations perfectly
                        state_df = state_df.astype(float)
                        
                        prediction = self.model.predict(state_df)[0]
                        action = "STANDARD" if prediction == 1 else "INVERSE"
                        logger.info(f"🧠 EXPLOITING: AI chose {action}.")
                    
                    # --- SNAPSHOT MT5 OPEN TICKETS BEFORE TRADE ---
                    open_positions = mt5.positions_get()
                    tickets_before = {p.ticket for p in open_positions} if open_positions else set()
                    
                    # 3. Execute the chosen bot's single cycle
                    if action == "STANDARD":
                        await self.standard_bot.execute_single_cycle()
                    else:
                        await self.inverse_bot.execute_single_cycle()
                        
                    # --- THE FIX: WAIT FOR THE BOT'S SAFETY DELAY ---
                    logger.info("⏳ Watchtower waiting 5 seconds for bot to finish verifying and executing...")
                    await asyncio.sleep(5)
                        
                    # --- SNAPSHOT MT5 OPEN TICKETS AFTER TRADE ---
                    open_positions_after = mt5.positions_get()
                    tickets_after = {p.ticket for p in open_positions_after} if open_positions_after else set()
                    
                    # Compare before and after. Did a new trade appear?
                    new_tickets = tickets_after - tickets_before
                    
                    if new_tickets:
                        ticket = list(new_tickets)[0] # Grab the new MT5 ticket ID
                        clean_features = {k: v for k, v in current_state.items() if k not in ['time', 'setup_detected']}
                        
                        # Add it to the Supervisor Watchtower!
                        self.active_rl_trades[ticket] = {
                            'features': clean_features,
                            'action': action,
                            'was_random': is_random
                        }
                        logger.info(f"👀 Supervisor Watchtower is tracking new ticket #{ticket}...")
                        
                await asyncio.sleep(self.POLLING_RATE)
                
            except Exception as e:
                logger.error(f"Brain Loop Error: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(10)

if __name__ == "__main__":
    brain = AIBrain()
    try:
        asyncio.run(brain.run_realtime())
    except KeyboardInterrupt:
        logger.info("AI Brain Shutting Down.")