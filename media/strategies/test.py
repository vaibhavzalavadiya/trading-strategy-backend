import pandas as pd
import numpy as np

def run_strategy(df):
    """
    Simple Moving Average Crossover Strategy
    - Buy when short MA crosses above long MA
    - Sell when short MA crosses below long MA
    """
    
    # Ensure we have required columns
    required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in df.columns:
            return {
                "error": f"Missing required column: {col}",
                "summary": {},
                "trades": []
            }
    
    # Convert timestamp to datetime if it's not already
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Calculate moving averages
    df['ma_short'] = df['close'].rolling(window=5).mean()  # 5-period MA
    df['ma_long'] = df['close'].rolling(window=20).mean()  # 20-period MA
    
    # Generate signals
    df['signal'] = 0
    df.loc[df['ma_short'] > df['ma_long'], 'signal'] = 1  # Buy signal
    df.loc[df['ma_short'] < df['ma_long'], 'signal'] = -1  # Sell signal
    
    # Generate trades
    trades = []
    position = 0  # 0 = no position, 1 = long position
    entry_price = 0
    entry_date = None
    
    for i in range(1, len(df)):
        current_signal = df.iloc[i]['signal']
        prev_signal = df.iloc[i-1]['signal']
        
        # Buy signal (crossing from 0 or -1 to 1)
        if current_signal == 1 and prev_signal <= 0 and position == 0:
            position = 1
            entry_price = df.iloc[i]['close']
            entry_date = df.iloc[i]['timestamp']
            
        # Sell signal (crossing from 0 or 1 to -1) or close position
        elif (current_signal == -1 and prev_signal >= 0 and position == 1) or (i == len(df) - 1 and position == 1):
            exit_price = df.iloc[i]['close']
            exit_date = df.iloc[i]['timestamp']
            profit = (exit_price - entry_price) * 100  # Assuming 100 shares
            
            trades.append({
                'date': entry_date.strftime('%Y-%m-%d'),
                'type': 'BUY',
                'price': entry_price,
                'shares': 100,
                'profit': None
            })
            
            trades.append({
                'date': exit_date.strftime('%Y-%m-%d'),
                'type': 'SELL',
                'price': exit_price,
                'shares': 100,
                'profit': profit
            })
            
            position = 0
    
    # Calculate summary statistics
    total_trades = len([t for t in trades if t['type'] == 'SELL'])
    total_profit = sum([t['profit'] for t in trades if t['profit'] is not None])
    winning_trades = len([t for t in trades if t['profit'] is not None and t['profit'] > 0])
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    # Calculate additional metrics
    initial_price = df.iloc[0]['close']
    final_price = df.iloc[-1]['close']
    total_return = ((final_price - initial_price) / initial_price) * 100
    
    return {
        "summary": {
            "total_trades": total_trades,
            "total_profit": round(total_profit, 2),
            "winning_trades": winning_trades,
            "win_rate": round(win_rate, 2),
            "total_return": round(total_return, 2),
            "initial_price": round(initial_price, 2),
            "final_price": round(final_price, 2)
        },
        "trades": trades
    }