import pandas as pd
import numpy as np

def run_strategy(df):
    """
    Test Strategy using floor function without import
    This demonstrates the global fix works
    """
    
    df = df.copy()
    
    # Handle different column formats
    if 'Date' in df.columns:
        df['timestamp'] = pd.to_datetime(df['Date'], format='%d-%b-%Y')
    elif 'timestamp' not in df.columns:
        df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='D')
    
    # Find close price column
    close_col = None
    for col in ['Close', 'close', 'CLOSE']:
        if col in df.columns:
            close_col = col
            break
    
    if close_col is None:
        return {
            "error": "No close price column found",
            "summary": {},
            "trades": []
        }
    
    # Use floor function without import (global fix makes this work)
    df['floor_price'] = df[close_col].apply(floor)
    df['ceil_price'] = df[close_col].apply(ceil)
    
    # Calculate moving averages
    df['MA10'] = df[close_col].rolling(window=10).mean()
    df['MA20'] = df[close_col].rolling(window=20).mean()
    
    # Generate signals using floor function
    df['Signal'] = 0
    
    for i in range(20, len(df)):
        # Use floor to create integer-based signals
        price_level = floor(df.iloc[i][close_col] / 10) * 10
        ma_level = floor(df.iloc[i]['MA20'] / 10) * 10
        
        if price_level > ma_level:
            df.iloc[i, df.columns.get_loc('Signal')] = 1
        elif price_level < ma_level:
            df.iloc[i, df.columns.get_loc('Signal')] = -1
    
    # Generate trades
    trades = []
    position = 0
    entry_price = 0
    entry_date = None
    shares = 100
    
    for i in range(20, len(df)):
        current_signal = df.iloc[i]['Signal']
        current_price = df.iloc[i][close_col]
        current_date = df.iloc[i]['timestamp'].strftime('%Y-%m-%d')
        
        if position == 0 and current_signal == 1:
            position = 1
            entry_price = current_price
            entry_date = current_date
        elif position == 1 and current_signal == -1:
            profit = (current_price - entry_price) * shares
            
            trades.append({
                'date': entry_date,
                'type': 'BUY',
                'price': round(entry_price, 2),
                'shares': shares,
                'profit': None
            })
            trades.append({
                'date': current_date,
                'type': 'SELL',
                'price': round(current_price, 2),
                'shares': shares,
                'profit': round(profit, 2)
            })
            position = 0
    
    # Close position at end if still open
    if position == 1:
        final_price = df.iloc[-1][close_col]
        final_date = df.iloc[-1]['timestamp'].strftime('%Y-%m-%d')
        profit = (final_price - entry_price) * shares
        
        trades.append({
            'date': entry_date,
            'type': 'BUY',
            'price': round(entry_price, 2),
            'shares': shares,
            'profit': None
        })
        trades.append({
            'date': final_date,
            'type': 'SELL',
            'price': round(final_price, 2),
            'shares': shares,
            'profit': round(profit, 2)
        })
    
    # Calculate summary
    total_trades = len([t for t in trades if t['type'] == 'SELL'])
    total_profit = sum([t['profit'] for t in trades if t['profit'] is not None])
    winning_trades = len([t for t in trades if t['profit'] is not None and t['profit'] > 0])
    losing_trades = len([t for t in trades if t['profit'] is not None and t['profit'] < 0])
    
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    avg_profit = total_profit / total_trades if total_trades > 0 else 0
    
    initial_price = df.iloc[20][close_col]
    final_price = df.iloc[-1][close_col]
    total_return = ((final_price - initial_price) / initial_price) * 100
    
    return {
        'summary': {
            'total_trades': total_trades,
            'total_profit': round(total_profit, 2),
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': round(win_rate, 2),
            'avg_profit': round(avg_profit, 2),
            'total_return': round(total_return, 2),
            'initial_price': round(initial_price, 2),
            'final_price': round(final_price, 2)
        },
        'trades': trades,
        'currency': 'INR'
    } 