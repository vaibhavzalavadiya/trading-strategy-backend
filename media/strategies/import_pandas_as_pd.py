import pandas as pd
import numpy as np

def run_strategy(df):
    """
    Simple Moving Average Crossover Strategy
    This strategy generates buy/sell signals based on moving average crossovers
    """
    
    # Clean and prepare the data
    df = df.copy()
    
    # Convert date column to datetime if it exists
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%Y')
        df['timestamp'] = df['Date']
    elif 'timestamp' not in df.columns:
        # Create a timestamp column if it doesn't exist
        df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='D')
    
    # Calculate moving averages
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    
    # Generate signals
    df['Signal'] = 0
    df.loc[df['MA20'] > df['MA50'], 'Signal'] = 1  # Buy signal
    df.loc[df['MA20'] < df['MA50'], 'Signal'] = -1  # Sell signal
    
    # Generate trades
    trades = []
    position = 0
    entry_price = 0
    entry_date = None
    
    for i in range(50, len(df)):  # Start from 50 to ensure MA50 is calculated
        current_signal = df.iloc[i]['Signal']
        current_price = df.iloc[i]['Close']
        current_date = df.iloc[i]['timestamp'].strftime('%Y-%m-%d')
        
        if position == 0 and current_signal == 1:  # Buy signal
            position = 1
            entry_price = current_price
            entry_date = current_date
        elif position == 1 and current_signal == -1:  # Sell signal
            # Calculate profit/loss
            profit = current_price - entry_price
            trades.append({
                'date': entry_date,
                'type': 'BUY',
                'price': round(entry_price, 2),
                'shares': 100,
                'profit': None
            })
            trades.append({
                'date': current_date,
                'type': 'SELL',
                'price': round(current_price, 2),
                'shares': 100,
                'profit': round(profit * 100, 2)  # Profit for 100 shares
            })
            position = 0
    
    # Calculate summary metrics
    total_trades = len([t for t in trades if t['type'] == 'SELL'])
    total_profit = sum([t['profit'] for t in trades if t['profit'] is not None])
    winning_trades = len([t for t in trades if t['profit'] is not None and t['profit'] > 0])
    losing_trades = len([t for t in trades if t['profit'] is not None and t['profit'] < 0])
    
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    avg_profit = total_profit / total_trades if total_trades > 0 else 0
    
    # Calculate additional metrics
    initial_price = df.iloc[50]['Close']
    final_price = df.iloc[-1]['Close']
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