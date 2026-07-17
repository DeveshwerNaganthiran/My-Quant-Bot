#!/usr/bin/env python3
import pandas as pd
from datetime import datetime

# Read trades
df = pd.read_csv('data/trade_logs/trades/trades_2026_07.csv')
df['open_time'] = pd.to_datetime(df['open_time'])

# Get today's date and filter from 14:00
today = df['open_time'].max().date()
cutoff = pd.Timestamp(year=today.year, month=today.month, day=today.day, hour=14)

recent = df[df['open_time'] >= cutoff].copy()

print('\n' + '='*150)
print(f'TRADES FROM 14:00 ONWARDS (2026-07-15)')
print('='*150)

if len(recent) > 0:
    # Show key columns
    for idx, row in recent.iterrows():
        reason = row['exit_reason'] if pd.notna(row['exit_reason']) else 'N/A'
        profit = row['profit_usd']
        entry = row['entry_price']
        exit_p = row['exit_price']
        direction = row['direction']
        open_t = row['open_time'].strftime('%H:%M:%S')
        
        print(f"{open_t} | {direction:5s} @ {entry:8.2f} → {exit_p:8.2f} | ${profit:7.2f} | {reason}")
    
    print('\n' + '='*150)
    print('SUMMARY:')
    wins_mask = recent['profit_usd'] > 0
    losses_mask = recent['profit_usd'] < 0
    
    wins_sum = recent[wins_mask]['profit_usd'].sum()
    losses_sum = abs(recent[losses_mask]['profit_usd'].sum())
    
    print(f"Wins:   {len(recent[wins_mask]):3d} trades × +${wins_sum:8.2f}")
    print(f"Losses: {len(recent[losses_mask]):3d} trades × -${losses_sum:8.2f}")
    print(f"Net P/L: ${wins_sum - losses_sum:8.2f}")
    print(f"Win Rate: {100*len(recent[wins_mask])/len(recent):5.1f}%")
    
    print('\n' + '='*150)
    print('EXIT REASONS BREAKDOWN:')
    for reason, count in recent['exit_reason'].value_counts().items():
        reason_trades = recent[recent['exit_reason'] == reason]
        avg_loss = reason_trades['profit_usd'].mean()
        print(f"  {reason:20s}: {count:3d} trades | Avg: ${avg_loss:8.2f}")
    
else:
    print("No trades found from 14:00 onwards")
    print("\nLast 10 trades:")
    last10 = df.tail(10)
    for idx, row in last10.iterrows():
        open_t = row['open_time']
        profit = row['profit_usd']
        reason = row['exit_reason']
        print(f"{open_t} | ${profit:7.2f} | {reason}")
