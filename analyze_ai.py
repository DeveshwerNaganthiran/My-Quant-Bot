import pandas as pd
import numpy as np

# Load trade log
df = pd.read_csv('data/trade_logs/trades/trades_2026_07.csv')

# Filter to just today's trades (Jul 14)
today = df[df['open_time'].str.contains('2026-07-14', na=False)]

print('=== TODAY (JUL 14) ANALYSIS ===')
print(f'Total trades: {len(today)}')

if len(today) > 0:
    win_count = len(today[today['profit_usd'] > 0])
    loss_count = len(today[today['profit_usd'] < 0])
    print(f'Winning trades: {win_count} ({100*win_count/len(today):.1f}%)')
    print(f'Losing trades: {loss_count} ({100*loss_count/len(today):.1f}%)')
    print(f'Total P/L: ${today["profit_usd"].sum():.2f}')
    print()

    # AI confidence analysis
    print('=== ML SIGNAL CONFIDENCE DISTRIBUTION ===')
    ml_conf = today['ml_confidence'].dropna()
    print(f'AI signals generated: {len(ml_conf)}')
    if len(ml_conf) > 0:
        print(f'Min confidence: {ml_conf.min()*100:.1f}%')
        print(f'Max confidence: {ml_conf.max()*100:.1f}%')
        print(f'Mean confidence: {ml_conf.mean()*100:.1f}%')
        print(f'Median confidence: {ml_conf.median()*100:.1f}%')
        print()
        
        # Check how many AI signals were 80%+
        high_conf = len(ml_conf[ml_conf >= 0.80])
        mid_conf = len(ml_conf[(ml_conf >= 0.70) & (ml_conf < 0.80)])
        low_conf = len(ml_conf[ml_conf < 0.70])
        print(f'AI signals >= 80%: {high_conf} ({100*high_conf/len(ml_conf):.1f}%)')
        print(f'AI signals 70-80%: {mid_conf} ({100*mid_conf/len(ml_conf):.1f}%)')
        print(f'AI signals < 70%: {low_conf} ({100*low_conf/len(ml_conf):.1f}%)')
        print()

        # Performance by confidence level
        print('=== AI PERFORMANCE BY CONFIDENCE ===')
        for threshold in [0.50, 0.60, 0.70, 0.75, 0.80]:
            above_threshold = today[today['ml_confidence'] >= threshold]
            if len(above_threshold) > 0:
                win = len(above_threshold[above_threshold['profit_usd'] > 0])
                win_rate = 100*win/len(above_threshold)
                avg_pnl = above_threshold['profit_usd'].mean()
                print(f'  >{threshold*100:.0f}% confidence: {len(above_threshold)} trades, {win_rate:.1f}% win, ${avg_pnl:.2f} avg')

    # Compare to SMC
    print()
    print('=== SMC SIGNAL PERFORMANCE ===')
    smc_only = today[today['entry_reason'].str.contains('SMC', na=False)]
    print(f'SMC trades: {len(smc_only)}')
    if len(smc_only) > 0:
        smc_win = len(smc_only[smc_only['profit_usd'] > 0])
        print(f'SMC win rate: {100*smc_win/len(smc_only):.1f}%')
        print(f'SMC avg P/L: ${smc_only["profit_usd"].mean():.2f}')

else:
    print("No trades yet today")
