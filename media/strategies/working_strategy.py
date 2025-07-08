import pandas as pd
import numpy as np

def run_strategy(df):
    """
    Comprehensive Trading Strategy for NIFTY Data
    This strategy uses multiple technical indicators to generate buy/sell signals
    """
    
    # Clean and prepare the data
    df = df.copy()
    
    # Handle different column name formats
    if 'Date' in df.columns:
        df['timestamp'] = pd.to_datetime(df['Date'], format='%d-%b-%Y')
    elif 'timestamp' not in df.columns:
        df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='D')
    
    # Ensure we have the required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Shares Traded']
    if not all(col in df.columns for col in required_cols):
        return {
            "error": f"Missing required columns. Available: {list(df.columns)}",
            "summary": {},
            "trades": []
        }
    
    # Calculate technical indicators
    df['EMA20'] = df['Close'].ewm(span=20).mean()
    df['EMA50'] = df['Close'].ewm(span=50).mean()
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['RSI'] = calculate_rsi(df['Close'], 14)
    df['Volume_SMA'] = df['Shares Traded'].rolling(window=10).mean()
    
    # Generate signals
    df['Signal'] = 0
    
    for i in range(50, len(df)):
        # Buy conditions
        ema_bullish = df.iloc[i]['EMA20'] > df.iloc[i]['EMA50']
        price_above_sma = df.iloc[i]['Close'] > df.iloc[i]['SMA20']
        rsi_oversold = df.iloc[i]['RSI'] < 30
        volume_high = df.iloc[i]['Shares Traded'] > df.iloc[i]['Volume_SMA'] * 1.2
        
        # Sell conditions
        ema_bearish = df.iloc[i]['EMA20'] < df.iloc[i]['EMA50']
        price_below_sma = df.iloc[i]['Close'] < df.iloc[i]['SMA20']
        rsi_overbought = df.iloc[i]['RSI'] > 70
        
        if ema_bullish and price_above_sma and (rsi_oversold or volume_high):
            df.iloc[i, df.columns.get_loc('Signal')] = 1  # Buy signal
        elif ema_bearish and price_below_sma and rsi_overbought:
            df.iloc[i, df.columns.get_loc('Signal')] = -1  # Sell signal
    
    # Generate trades
    trades = []
    position = 0
    entry_price = 0
    entry_date = None
    shares = 100  # Fixed number of shares per trade
    
    for i in range(50, len(df)):
        current_signal = df.iloc[i]['Signal']
        current_price = df.iloc[i]['Close']
        current_date = df.iloc[i]['timestamp'].strftime('%Y-%m-%d')
        
        if position == 0 and current_signal == 1:  # Buy signal
            position = 1
            entry_price = current_price
            entry_date = current_date
        elif position == 1 and current_signal == -1:  # Sell signal
            # Calculate profit/loss
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
    
    # Close any open position at the end
    if position == 1:
        final_price = df.iloc[-1]['Close']
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
    
    # Calculate max drawdown
    max_drawdown = calculate_max_drawdown(df['Close'])
    
    return {
        'summary': {
            'total_trades': total_trades,
            'total_profit': round(total_profit, 2),
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': round(win_rate, 2),
            'avg_profit': round(avg_profit, 2),
            'total_return': round(total_return, 2),
            'max_drawdown': round(max_drawdown, 2),
            'initial_price': round(initial_price, 2),
            'final_price': round(final_price, 2),
            'profit_factor': round(abs(winning_trades / losing_trades), 2) if losing_trades > 0 else float('inf')
        },
        'trades': trades,
        'currency': 'INR'
    }

def calculate_rsi(prices, period=14):
    """Calculate RSI (Relative Strength Index)"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_max_drawdown(prices):
    """Calculate maximum drawdown"""
    peak = prices.expanding(min_periods=1).max()
    drawdown = (prices - peak) / peak
    return drawdown.min() * 100 