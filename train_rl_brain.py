import pandas as pd
import xgboost as xgb
import os
import pickle

def train_supervisor_brain():
    print("Initiating RL Supervisor Meta-Training...")
    
    memory_file = "src/backtests/rl_training_memory.csv"
    model_save_path = "src/backtests/ml_v3/meta_supervisor.pkl"
    
    # 1. Load the RL Memory
    try:
        df = pd.read_csv(memory_file)
    except FileNotFoundError:
        print(f"❌ Error: {memory_file} not found. Run the supervisor in exploration mode first.")
        return

    # Filter out trades that resulted in zero PnL (noise)
    df = df[df['pnl'] != 0.0] 
    
    # NOTE: I lowered this from 50 to 30. If the market is slow, you might only get 
    # 35-40 trades by the end of the week. 30 is plenty for XGBoost to start learning!
    if len(df) < 30:
        print(f"⚠️ Only {len(df)} actionable trades in memory. Waiting for at least 30 to train.")
        return
    
    # 2. Define Features (X) and Target (y)
    # Exclude tracking columns that aren't market features
    exclude_cols = ['timestamp', 'action_taken', 'was_random', 'pnl', 'win', 'setup_detected']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols]
    
    # Target Mapping for the Supervisor: 
    # 1 = Standard is the right choice, 0 = Inverse is the right choice
    y = []
    for _, row in df.iterrows():
        if (row['action_taken'] == "STANDARD" and row['win'] == 1) or \
           (row['action_taken'] == "INVERSE" and row['win'] == 0):
            y.append(1)
        else:
            y.append(0)
            
    # 3. Train a dedicated XGBoost Classifier for the Meta-Brain
    print(f"Training Supervisor on {len(X)} experiences...")
    
    # We use a fresh XGBoost instance here so we don't interfere with your ml_model.py
    model = xgb.XGBClassifier(
        n_estimators=100, 
        learning_rate=0.05, 
        max_depth=4, 
        objective='binary:logistic'
    )
    
    model.fit(X, y)
    
    # 4. Save the Brain
    # --- ADDED SAFETY CHECK: CREATE FOLDER IF IT DOESN'T EXIST ---
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    with open(model_save_path, "wb") as f:
        pickle.dump(model, f)
        
    print(f"✅ AI Brain successfully trained and saved to {model_save_path}!")

if __name__ == "__main__":
    train_supervisor_brain()