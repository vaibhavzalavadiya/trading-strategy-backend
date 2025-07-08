import pandas as pd
import numpy as np

def run_strategy(df):
    """
    Guaranteed Working Strategy - Always produces results
    """
    
    # Make a copy
    df = df.copy()
    
    # Handle different date formats
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
    
    # Simple strategy: Buy when price goes up 5%, sell when it goes down 3%
    df['Signal'] = 0
    df['Price_Change'] = df[close_col].pct_change() * 100
    
    for i in range(1, len(df)):
        if df.iloc[i]['Price_Change'] > 5:
            df.iloc[i, df.columns.get_loc('Signal')] = 1
        elif df.iloc[i]['Price_Change'] < -3:
            df.iloc[i, df.columns.get_loc('Signal')] = -1
    
    # Generate trades
    trades = []
    position = 0
    entry_price = 0
    entry_date = None
    shares = 100
    
    for i in range(1, len(df)):
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
    
    initial_price = df.iloc[0][close_col]
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