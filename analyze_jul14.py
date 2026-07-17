import pandas as pd
import numpy as np

df = pd.read_csv('data/trade_logs/trades/trades_2026_07.csv')
df['open_time'] = pd.to_datetime(df['open_time'])

# Today
today = df['open_time'].dt.date.max()
df_today = df[df['open_time'].dt.date == today]

print('='*70)
print(f'ANALYSIS: {today} (Today)')
print('='*70)

total_trades = len(df_today)
winning_trades = (df_today['profit_usd'] > 0).sum()
losing_trades = (df_today['profit_usd'] < 0).sum()

total_profit = df_today[df_today['profit_usd'] > 0]['profit_usd'].sum()
total_loss = df_today[df_today['profit_usd'] < 0]['profit_usd'].sum()
net_pl = total_profit + total_loss

win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
avg_win = total_profit / winning_trades if winning_trades > 0 else 0
avg_loss = total_loss / losing_trades if losing_trades > 0 else 0
profit_factor = abs(total_profit / total_loss) if total_loss != 0 else 0

print(f'\nTotal Trades: {total_trades}')
print(f'  Wins: {winning_trades} ({win_rate:.1f}%)')
print(f'  Losses: {losing_trades} ({100-win_rate:.1f}%)')

print(f'\nP/L Summary:')
print(f'  Total Profit (wins): ${total_profit:.2f}')
print(f'  Total Loss (losses): ${total_loss:.2f}')
print(f'  NET P/L: ${net_pl:.2f}')
print(f'  Profit Factor: {profit_factor:.2f}x')

print(f'\nPer Trade:')
print(f'  Avg Win: ${avg_win:.2f}')
print(f'  Avg Loss: ${avg_loss:.2f}')
print(f'  Risk/Reward: {abs(avg_loss/avg_win) if avg_win != 0 else 0:.2f}')

# Month summary
print(f'\n' + '='*70)
print('FULL MONTH (July 2026) Summary:')
print('='*70)

df_month = df[df['open_time'].dt.month == 7]
month_trades = len(df_month)
month_wins = (df_month['profit_usd'] > 0).sum()
month_pl = df_month['profit_usd'].sum()
month_wr = month_wins / month_trades * 100 if month_trades > 0 else 0

print(f'Total Trades (July): {month_trades}')
print(f'Win Rate: {month_wr:.1f}%')
print(f'Net P/L: ${month_pl:.2f}')

# Before vs After tuning (only today's trades)
before_cutoff = pd.Timestamp(2026, 7, 14, 17, 45, tz='UTC+07:00')  # Before new model deployment
df_today_full = df[df['open_time'].dt.date == today]
df_before = df_today_full[df_today_full['open_time'] < before_cutoff]
df_after = df_today_full[df_today_full['open_time'] >= before_cutoff]

print(f'\n' + '='*70)
print('BEFORE vs AFTER Model Retraining (Jul 14 17:45):')
print('='*70)

before_pl = df_before['profit_usd'].sum()
before_trades = len(df_before)
before_wr = (df_before['profit_usd'] > 0).sum() / before_trades * 100 if before_trades > 0 else 0

after_pl = df_after['profit_usd'].sum()
after_trades = len(df_after)
after_wr = (df_after['profit_usd'] > 0).sum() / after_trades * 100 if after_trades > 0 else 0

print(f'\nBEFORE (Old Model):')
print(f'  Trades: {before_trades}')
print(f'  Win Rate: {before_wr:.1f}%')
print(f'  Net P/L: ${before_pl:.2f}')

print(f'\nAFTER (Fresh Model + SMC Fallback):')
print(f'  Trades: {after_trades}')
print(f'  Win Rate: {after_wr:.1f}%')
print(f'  Net P/L: ${after_pl:.2f}')

print(f'\nIMPACT:')
print(f'  Trade Count: {after_trades - before_trades:+d}')
print(f'  Win Rate: {after_wr - before_wr:+.1f}%')
print(f'  P/L: ${after_pl - before_pl:+.2f}')

print(f'\n' + '='*70)
print('Root Cause Analysis:')
print('='*70)

# Check if issue is weak entries or bad exits
bad_entries = df_after[df_after['ml_confidence'] < 0.60]
print(f'\nTrades with ML < 60% confidence: {len(bad_entries)}')
if len(bad_entries) > 0:
    bad_entry_wr = (bad_entries['profit_usd'] > 0).sum() / len(bad_entries) * 100
    print(f'  Win Rate (weak entries): {bad_entry_wr:.1f}%')
    print(f'  Net P/L (weak entries): ${bad_entries["profit_usd"].sum():.2f}')

strong_entries = df_after[df_after['ml_confidence'] >= 0.75]
print(f'\nTrades with ML >= 75% confidence: {len(strong_entries)}')
if len(strong_entries) > 0:
    strong_wr = (strong_entries['profit_usd'] > 0).sum() / len(strong_entries) * 100
    print(f'  Win Rate (strong entries): {strong_wr:.1f}%')
    print(f'  Net P/L (strong entries): ${strong_entries["profit_usd"].sum():.2f}')
