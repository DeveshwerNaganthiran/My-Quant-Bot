import sys
sys.path.insert(0, 'c:\\Users\\deves\\Desktop\\Quant_Trading_Bot')
import pandas as pd
from src.data_fetcher import DataFetcher
from src.ml_model import MLPredictor
import warnings
warnings.filterwarnings('ignore')

# Fetch last 100 candles
fetcher = DataFetcher()
df = fetcher.fetch_ohlcv_data('XAUUSD', timeframe='M5', limit=100)

# Load AI model
ml = MLPredictor()

# Get predictions
predictions = []
for i in range(len(df) - 1):
    try:
        pred = ml.predict(df.iloc[:i+1])
        if pred:
            predictions.append({
                'close': df.iloc[i]['close'],
                'signal': pred.signal,
                'confidence': f"{pred.confidence:.0%}",
                'probability': pred.probability
            })
    except:
        pass

if predictions:
    last_10 = predictions[-10:]
    print("=== LAST 10 PREDICTIONS ===")
    for p in last_10:
        print(f"{p['signal']:6} | {p['confidence']:5} | Prob: {p['probability']:.3f} | Price: {p['close']:.2f}")
    
    # Count signals
    signals = pd.Series([p['signal'] for p in predictions])
    print(f"\n=== SIGNAL DISTRIBUTION (Last {len(predictions)} predictions) ===")
    print(signals.value_counts())
    print(f"\nHOLD rate: {(signals == 'HOLD').sum() / len(signals) * 100:.1f}%")
else:
    print("No predictions generated")
