import pandas as pd
import numpy as np
import os

if __name__ == "__main__":
    input_path = "/opt/ml/processing/input"
    output_path = "/opt/ml/processing/output"

    df = pd.read_csv("/opt/ml/processing/input/spy/spy_daily.csv")
    qqq = pd.read_csv("/opt/ml/processing/input/qqq/qqq_daily.csv")


    df['date'] = pd.to_datetime(df['date'])
    qqq['date'] = pd.to_datetime(qqq['date'])
    qqq = qqq[['date', 'close']].rename(columns={'close': 'qqq_close'})
    df = df.merge(qqq, on='date', how='left')

    df['return'] = df['close'].pct_change()
    df['sma_5'] = df['close'].rolling(5).mean()
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['volatility_20'] = df['return'].rolling(20).std()

    def calc_rsi(series, period=14):
        delta = series.diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ma_up = up.rolling(period).mean()
        ma_down = down.rolling(period).mean()
        rsi = 100 - (100 / (1 + ma_up / ma_down))
        return rsi

    df['rsi_14'] = calc_rsi(df['close'])
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema12'] - df['ema26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    df['bb_mid'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']

    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(14).mean()

    df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    df['rel_strength_qqq'] = df['close'] / df['qqq_close']

    conditions = [
        (df['close'] > df['sma_50']) & (df['volatility_20'] < 0.015),
        (df['close'] < df['sma_50']) & (df['volatility_20'] > 0.025),
    ]
    choices = ['bull', 'bear']
    df['regime'] = np.select(conditions, choices, default='volatile')

    df['vol_spike'] = (df['volume'] > 3 * df['volume'].rolling(20).median()).astype(int)
    df['extreme_return'] = (df['return'].abs() > 3 * df['return'].rolling(20).std()).astype(int)
    df['trend_up'] = ((df['close'] > df['sma_5']) & (df['close'] > df['sma_10'])).astype(int)
    df['trend_down'] = ((df['close'] < df['sma_5']) & (df['close'] < df['sma_10'])).astype(int)

    df['avg_vol_20d'] = df['volume'].rolling(20).mean()
    df['in_play_150'] = (df['volume'] >= 1.5 * df['avg_vol_20d']).astype(int)
    df['in_play_120'] = (df['volume'] >= 1.2 * df['avg_vol_20d']).astype(int)
    df['in_play_200'] = (df['volume'] >= 2.0 * df['avg_vol_20d']).astype(int)

    final_features = [
        'date', 'open', 'high', 'low', 'close', 'volume', 'return',
        'sma_5', 'sma_10', 'sma_20', 'sma_50',
        'rsi_14', 'macd', 'macd_signal',
        'bb_mid', 'bb_upper', 'bb_lower',
        'atr_14', 'vwap', 'rel_strength_qqq', 'regime',
        'vol_spike', 'extreme_return',
        'trend_up', 'trend_down', 'in_play_150', 'in_play_120', 'in_play_200'
    ]

    df_feat = df[final_features].dropna().reset_index(drop=True)
    df_feat.to_csv(os.path.join(output_path, "spy_daily_features.csv"), index=False)