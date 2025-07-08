import pandas as pd
import numpy as np
from math import floor

def run_strategy(df, capital=100000, risk_pct=0.10, stop_loss_pct=0.02, target_pct=0.15):
    """
    Entry:
    - Previous candle closed fully below 5 EMA
    - Next candle is green
    - Price is NOT in uptrend (close < EMA 21)

    Exit:
    - Stop Loss: 2%
    - Take Profit: 15%
    - Risk: 10% of capital per trade
    """

    required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        return {"error": "Missing required columns", "trades": [], "summary": {}}

    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['ema5'] = df['close'].ewm(span=5).mean()
    df['ema21'] = df['close'].ewm(span=21).mean()

    trades = []
    capital_used = capital

    for i in range(2, len(df)):
        row_prev = df.iloc[i - 1]
        row = df.iloc[i]

        # Entry conditions
        candle_below_ema5 = row_prev['close'] < row_prev['ema5'] and row_prev['high'] < row_prev['ema5']
        current_green = row['close'] > row['open']
        not_uptrend = row['close'] < row['ema21']

        if candle_below_ema5 and current_green and not_uptrend:
            entry_price = row['close']
            stop_loss = entry_price * (1 - stop_loss_pct)
            take_profit = entry_price * (1 + target_pct)
            entry_time = row['timestamp']

            # Capital risk and quantity
            risk_amount = capital * risk_pct
            per_share_risk = entry_price - stop_loss
            if per_share_risk <= 0:
                continue
            quantity = floor(risk_amount / per_share_risk)
            if quantity <= 0:
                continue

            # Simulate future candles
            for j in range(i + 1, len(df)):
                future = df.iloc[j]
                low = future['low']
                high = future['high']
                exit_price = None
                exit_type = None

                if low <= stop_loss:
                    exit_price = stop_loss
                    exit_type = 'SL'
                elif high >= take_profit:
                    exit_price = take_profit
                    exit_type = 'TP'

                if exit_price:
                    profit = (exit_price - entry_price) * quantity
                    capital_used += profit

                    trades.append({
                        'entry_date': entry_time.strftime('%Y-%m-%d'),
                        'exit_date': future['timestamp'].strftime('%Y-%m-%d'),
                        'entry_price': round(entry_price, 2),
                        'exit_price': round(exit_price, 2),
                        'shares': quantity,
                        'type': 'BUY',
                        'exit_type': exit_type,
                        'profit': round(profit, 2),
                        'return_pct': round((exit_price - entry_price) / entry_price * 100, 2),
                        'capital_after_trade': round(capital_used, 2)
                    })
                    break

    # Summary
    total_trades = len(trades)
    total_profit = round(sum(t['profit'] for t in trades), 2)
    win_trades = len([t for t in trades if t['exit_type'] == 'TP'])
    win_rate = round((win_trades / total_trades) * 100, 2) if total_trades > 0 else 0

    df_initial = df.iloc[0]['close']
    df_final = df.iloc[-1]['close']

    return {
        "summary": {
            "initial_capital": capital,
            "final_capital": round(capital_used, 2),
            "total_profit": total_profit,
            "total_trades": total_trades,
            "winning_trades": win_trades,
            "win_rate": win_rate,
            "initial_price": round(df_initial, 2),
            "final_price": round(df_final, 2),
            "total_return_pct": round((capital_used - capital) / capital * 100, 2)
        },
        "trades": trades
    }
