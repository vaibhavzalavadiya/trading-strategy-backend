import pandas as pd
import numpy as np

def run_strategy(df, stop_loss_pct=0.033, reward_ratio=5.0):
    """
    Buy when:
    - Previous candle closes below 5 EMA and doesn't touch it (high < ema)
    - Current candle is green (close > open)
    - Current close is below EMA 21 (i.e., not in uptrend)
    
    Sell when:
    - Price hits stop-loss or take-profit target
    """

    # Ensure correct columns
    required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        return {"error": "Missing required columns", "trades": [], "summary": {}}

    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['ema5'] = df['close'].ewm(span=5).mean()
    df['ema21'] = df['close'].ewm(span=21).mean()

    trades = []
    position = None

    for i in range(2, len(df)):
        row_prev = df.iloc[i - 1]
        row = df.iloc[i]

        # Conditions
        candle_below_ema5 = row_prev['close'] < row_prev['ema5'] and row_prev['high'] < row_prev['ema5']
        current_green = row['close'] > row['open']
        not_uptrend = row['close'] < row['ema21']

        if candle_below_ema5 and current_green and not_uptrend and position is None:
            entry_price = row['close']
            stop_loss = entry_price * (1 - stop_loss_pct)
            take_profit = entry_price * (1 + stop_loss_pct * reward_ratio)
            entry_time = row['timestamp']

            # Simulate forward candles to hit SL or TP
            for j in range(i+1, len(df)):
                future_row = df.iloc[j]
                low = future_row['low']
                high = future_row['high']

                exit_price = None
                exit_type = None

                if low <= stop_loss:
                    exit_price = stop_loss
                    exit_type = 'SL'
                elif high >= take_profit:
                    exit_price = take_profit
                    exit_type = 'TP'

                if exit_price:
                    trades.append({
                        'entry_date': entry_time.strftime('%Y-%m-%d'),
                        'exit_date': future_row['timestamp'].strftime('%Y-%m-%d'),
                        'entry_price': round(entry_price, 2),
                        'exit_price': round(exit_price, 2),
                        'type': 'BUY',
                        'exit_type': exit_type,
                        'profit': round(exit_price - entry_price, 2),
                        'return_pct': round((exit_price - entry_price) / entry_price * 100, 2)
                    })
                    position = None
                    break

    # Summary
    df_initial = df.iloc[0]['close']
    df_final = df.iloc[-1]['close']
    total_profit = sum(t['profit'] for t in trades)
    total_trades = len(trades)
    win_trades = len([t for t in trades if t['exit_type'] == 'TP'])
    win_rate = (win_trades / total_trades * 100) if total_trades > 0 else 0

    return {
        "summary": {
            "total_trades": total_trades,
            "total_profit": round(total_profit, 2),
            "win_rate": round(win_rate, 2),
            "initial_price": round(df_initial, 2),
            "final_price": round(df_final, 2),
            "total_return_pct": round((df_final - df_initial) / df_initial * 100, 2)
        },
        "trades": trades
    }
