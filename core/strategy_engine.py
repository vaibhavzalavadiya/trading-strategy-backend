# trading_backend/core/strategy_engine.py
import pandas as pd
import numpy as np
from datetime import timedelta
import re

def parse_pine_script(script):
    """Parse Pine Script and extract strategy parameters"""
    # Extract strategy name
    strategy_name_match = re.search(r"strategy\('([^']+)'", script)
    strategy_name = strategy_name_match.group(1) if strategy_name_match else "Custom Strategy"
    
    # Extract entry conditions
    long_conditions = []
    short_conditions = []
    
    # Parse strategy.entry calls
    entry_matches = re.finditer(r"strategy\.entry\('([^']+)',\s*strategy\.(long|short),\s*when=([^)]+)\)", script)
    for match in entry_matches:
        entry_name, direction, condition = match.groups()
        if direction == 'long':
            long_conditions.append((entry_name, condition))
        else:
            short_conditions.append((entry_name, condition))
    
    return {
        'name': strategy_name,
        'long_conditions': long_conditions,
        'short_conditions': short_conditions
    }

def evaluate_condition(condition, df):
    """Evaluate a Pine Script condition on the dataframe"""
    # Replace Pine Script functions with pandas equivalents
    condition = condition.replace('ta.rsi', 'calculate_rsi')
    condition = condition.replace('ta.sma', 'calculate_sma')
    condition = condition.replace('ta.ema', 'calculate_ema')
    
    # Add helper functions
    def calculate_rsi(series, period):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_sma(series, period):
        return series.rolling(window=period).mean()
    
    def calculate_ema(series, period):
        return series.ewm(span=period, adjust=False).mean()
    
    # Replace close, open, high, low with df columns
    condition = condition.replace('close', 'df["close"]')
    condition = condition.replace('open', 'df["open"]')
    condition = condition.replace('high', 'df["high"]')
    condition = condition.replace('low', 'df["low"]')
    
    try:
        return eval(condition)
    except Exception as e:
        print(f"Error evaluating condition: {condition}")
        print(f"Error: {str(e)}")
        return pd.Series(False, index=df.index)

def resample_dataframe(df, timeframe):
    """Resample dataframe to specified timeframe"""
    timeframe_map = {
        '1m': '1T',
        '5m': '5T',
        '15m': '15T',
        '30m': '30T',
        '1h': '1H',
        '4h': '4H',
        '1d': '1D'
    }
    
    offset = timeframe_map.get(timeframe, '1H')
    
    resampled = df.resample(offset, on='timestamp').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    return resampled

def analyze_strategy(df, pine_script='', timeframe='1h'):
    """Analyze strategy using Pine Script"""
    # Parse Pine Script
    strategy_config = parse_pine_script(pine_script)
    
    # Resample data to the specified timeframe
    df = resample_dataframe(df, timeframe)
    
    # Initialize signals and trade tracking
    df['signal'] = 0
    df['position'] = 0
    df['entry_price'] = 0.0
    df['exit_price'] = 0.0
    df['trade_pnl'] = 0.0
    df['cumulative_pnl'] = 0.0
    
    # Track trades
    trades = []
    current_position = 0
    entry_price = 0
    entry_time = None
    
    # Evaluate long conditions
    for entry_name, condition in strategy_config['long_conditions']:
        long_signal = evaluate_condition(condition, df)
        df.loc[long_signal, 'signal'] = 1
    
    # Evaluate short conditions
    for entry_name, condition in strategy_config['short_conditions']:
        short_signal = evaluate_condition(condition, df)
        df.loc[short_signal, 'signal'] = -1
    
    # Process signals and track trades
    for i in range(1, len(df)):
        current_signal = df.iloc[i]['signal']
        current_price = df.iloc[i]['close']
        current_time = df.index[i]
        
        # Handle entry signals
        if current_signal != 0 and current_position == 0:
            current_position = current_signal
            entry_price = current_price
            entry_time = current_time
            df.loc[current_time, 'position'] = current_position
            df.loc[current_time, 'entry_price'] = entry_price
        
        # Handle exit signals (opposite signal or end of data)
        elif (current_signal == -current_position or i == len(df) - 1) and current_position != 0:
            exit_price = current_price
            pnl = (exit_price - entry_price) * current_position
            
            # Record trade
            trade = {
                'entry_time': entry_time,
                'exit_time': current_time,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'position': current_position,
                'pnl': pnl,
                'pnl_pct': (pnl / entry_price) * 100
            }
            trades.append(trade)
            
            # Update DataFrame
            df.loc[current_time, 'position'] = 0
            df.loc[current_time, 'exit_price'] = exit_price
            df.loc[current_time, 'trade_pnl'] = pnl
            
            # Reset position
            current_position = 0
            entry_price = 0
            entry_time = None
    
    # Calculate cumulative PnL
    df['cumulative_pnl'] = df['trade_pnl'].cumsum()
    
    # Calculate trade statistics
    winning_trades = [t for t in trades if t['pnl'] > 0]
    losing_trades = [t for t in trades if t['pnl'] <= 0]
    
    total_trades = len(trades)
    winning_trades_count = len(winning_trades)
    losing_trades_count = len(losing_trades)
    
    win_rate = winning_trades_count / total_trades if total_trades > 0 else 0
    
    # Calculate profit metrics
    total_profit = sum(t['pnl'] for t in winning_trades)
    total_loss = sum(t['pnl'] for t in losing_trades)
    net_profit = total_profit + total_loss
    
    avg_win = total_profit / winning_trades_count if winning_trades_count > 0 else 0
    avg_loss = total_loss / losing_trades_count if losing_trades_count > 0 else 0
    
    # Calculate risk metrics
    max_drawdown = 0
    peak = 0
    for pnl in df['cumulative_pnl']:
        if pnl > peak:
            peak = pnl
        drawdown = (peak - pnl) / peak if peak > 0 else 0
        max_drawdown = max(max_drawdown, drawdown)
    
    # Calculate risk-reward ratio
    risk_reward_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
    
    # Calculate Sharpe Ratio
    returns = df['trade_pnl'].pct_change().dropna()
    sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if returns.std() != 0 else 0
    
    # Prepare results
    results = {
        'strategy_name': strategy_config['name'],
        'total_trades': total_trades,
        'winning_trades': winning_trades_count,
        'losing_trades': losing_trades_count,
        'win_rate': win_rate,
        'total_profit': total_profit,
        'total_loss': total_loss,
        'net_profit': net_profit,
        'average_win': avg_win,
        'average_loss': avg_loss,
        'largest_win': max(t['pnl'] for t in trades) if trades else 0,
        'largest_loss': min(t['pnl'] for t in trades) if trades else 0,
        'risk_reward_ratio': risk_reward_ratio,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'trades': trades,
        'equity_curve': df['cumulative_pnl'].tolist(),
        'timestamps': df.index.tolist()
    }
    
    return results

# Example usage:
# df = pd.read_csv('media/data/NIFTY.csv')
# trades, final_capital = backtest_ema_strategy(df)
# print(trades, final_capital)
