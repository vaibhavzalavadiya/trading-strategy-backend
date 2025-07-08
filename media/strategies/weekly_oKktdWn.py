import pandas as pd
import numpy as np

def calculate_supertrend(df, period=10, multiplier=3):
    hl2 = (df['high'] + df['low']) / 2
    atr = df['high'].rolling(window=period).max() - df['low'].rolling(window=period).min()
    atr = atr.rolling(window=period).mean()
    upperband = hl2 + (multiplier * atr)
    lowerband = hl2 - (multiplier * atr)
    supertrend = [True] * len(df)
    for i in range(1, len(df)):
        if df['close'].iloc[i] > upperband.iloc[i - 1]:
            supertrend[i] = True
        elif df['close'].iloc[i] < lowerband.iloc[i - 1]:
            supertrend[i] = False
        else:
            supertrend[i] = supertrend[i - 1]
    return pd.Series(supertrend, index=df.index)

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def run_strategy(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    # Weekly and Monthly resampling
    weekly = df.resample('W').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    monthly = df.resample('M').agg({
        'open': 'first',
        'close': 'last'
    }).dropna()

    # Indicators
    weekly['sma_volume_10'] = weekly['volume'].rolling(10).mean()
    weekly['sma_close_18'] = weekly['close'].rolling(18).mean()
    df['sma_close_48'] = df['close'].rolling(48).mean()
    weekly['rsi_14'] = calculate_rsi(weekly['close'], 14)
    monthly['rsi_14'] = calculate_rsi(monthly['close'], 14)
    weekly['supertrend'] = calculate_supertrend(weekly)

    signals = []

    for i in range(2, len(df)):
        try:
            latest = df.iloc[i]
            prev_day = df.iloc[i - 1]
            prev_2day = df.iloc[i - 2]

            latest_date = df.index[i]
            week = weekly.loc[:latest_date].iloc[-1]
            week_ago = weekly.loc[:latest_date].iloc[-2]
            month = monthly.loc[:latest_date].iloc[-1]
            month_ago = monthly.loc[:latest_date].iloc[-2]

            condition = (
                week['close'] > week_ago['open'] and
                week['close'] > week_ago['close'] and
                month['close'] > month_ago['open'] and
                month['close'] > month_ago['close'] and
                prev_day['close'] > prev_2day['open'] and
                latest['close'] > prev_day['open'] and
                (week['high'] - week['close']) < (week['close'] - week['open']) * 0.30 and
                week['volume'] > week['sma_volume_10'] * 2 and
                monthly['rsi_14'].iloc[-1] > 60 and
                week['rsi_14'] > 60 and
                week['close'] > week['sma_close_18'] and
                latest['close'] > latest['sma_close_48'] and
                week['supertrend'] and
                latest.get('market_cap', 0) > 400
            )

            if condition:
                signals.append({
                    'date': latest_date,
                    'close': latest['close'],
                    'reason': 'Breakout condition met'
                })

        except Exception:
            continue

    return pd.DataFrame(signals)
