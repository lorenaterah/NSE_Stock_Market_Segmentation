# Feature Engineering for NSE Stock Clustering

import numpy as np
import pandas as pd


def calculate_returns(group):
    # Daily returns
    group = group.sort_values('Date').copy()
    group['daily_return'] = group['Day Price'].pct_change()
    return group


def calculate_volatility_features(group, windows=[5, 10, 20]): 
    # Calculate rolling volatility
    group = group.sort_values('Date').copy()
    
    for window in windows:
        group[f'volatility_{window}d'] = group['daily_return'].rolling(
            window=window, min_periods=max(3, window//2)
        ).std()
    
    return group


def calculate_risk_metrics(group):
    # Risk metrics
    group = group.sort_values('Date').copy()
    
    # Downside deviation (only negative returns)
    negative_returns = group['daily_return'].copy()
    negative_returns[negative_returns > 0] = 0
    group['downside_deviation_20d'] = negative_returns.rolling(window=20, min_periods=10).std()
    
    # Value at Risk (5th percentile)
    group['var_95'] = group['daily_return'].rolling(window=40, min_periods=20).quantile(0.05)
    
    return group


def calculate_technical_indicators(group):
    # RSI, Bollinger Bands, MACD
    group = group.sort_values('Date').copy()
    price = group['Day Price']
    
    # RSI - Relative Strength Index
    delta = price.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=10, min_periods=5).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=10, min_periods=5).mean()
    rs = gain / (loss + 1e-10)
    group['rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    ma_20 = price.rolling(window=20, min_periods=10).mean()
    std_20 = price.rolling(window=20, min_periods=10).std()
    group['bb_width'] = (std_20 / (ma_20 + 1e-10)) * 100
    
    # MACD
    ema_12 = price.ewm(span=12, adjust=False).mean()
    ema_26 = price.ewm(span=26, adjust=False).mean()
    group['macd'] = ema_12 - ema_26
    
    return group


def calculate_liquidity_features(group):
    # Liquidity metrics
    group = group.sort_values('Date').copy()
    
    group['avg_volume'] = group['Volume'].rolling(window=20, min_periods=10).mean()
    group['volume_volatility'] = group['Volume'].rolling(window=20, min_periods=10).std()
    
    # Amihud illiquidity
    group['amihud_illiquidity'] = np.abs(group['daily_return']) / (
        group['Volume'] * group['Day Price'] + 1e-10
    )
    
    return group


def calculate_momentum_features(group):
    # Momentum and trends
    group = group.sort_values('Date').copy()
    
    # Momentum
    group['momentum_5d'] = group['Day Price'].pct_change(periods=5)
    group['momentum_20d'] = group['Day Price'].pct_change(periods=20)
    group['momentum_60d'] = group['Day Price'].pct_change(periods=60)
    
    # Moving averages
    group['ma_5'] = group['Day Price'].rolling(window=5, min_periods=3).mean()
    group['ma_20'] = group['Day Price'].rolling(window=20, min_periods=10).mean()
    group['ma_60'] = group['Day Price'].rolling(window=60, min_periods=30).mean()
    
    # Price to MA ratio
    group['price_to_ma20'] = (group['Day Price'] - group['ma_20']) / (group['ma_20'] + 1e-10)
    group['price_to_ma60'] = (group['Day Price'] - group['ma_60']) / (group['ma_60'] + 1e-10)
    
    return group


def calculate_drawdown(group):
    # Drawdown analysis 
    group = group.sort_values('Date').copy()
    
    # Calculating running max and drawdown
    running_max = group['Day Price'].expanding().max()
    group['current_drawdown'] = (group['Day Price'] - running_max) / running_max
    group['max_drawdown'] = group['current_drawdown'].expanding().min()
    
    return group


def aggregate_stock_features(group):
    # Aggregating to one row per stock
    
    # Get active trading days
    active_days = group[(group['Volume'].notna()) & (group['Volume'] > 0)]
    
    if len(active_days) < 20:
        return None
    
    # Start with basic info
    features = {
        'Stock_code': group['Stock_code'].iloc[0],
        'Sector': group['Sector'].iloc[0] if 'Sector' in group.columns else None,
        'Name': group['Name'].iloc[0] if 'Name' in group.columns else None,
    }
    
    # Volatility metrics
    features['volatility_mean'] = group['volatility_20d'].mean()
    features['volatility_max'] = group['volatility_20d'].max()
    features['volatility_5d'] = group['volatility_5d'].mean() if 'volatility_5d' in group.columns else 0
    
    # Risk metrics
    if 'downside_deviation_20d' in group.columns:
        features['downside_deviation'] = group['downside_deviation_20d'].mean()
    else:
        features['downside_deviation'] = 0
    
    if 'var_95' in group.columns:
        features['var_95'] = group['var_95'].mean()
    else:
        features['var_95'] = 0
    
    features['max_drawdown'] = group['max_drawdown'].min()
    
    # Return metrics
    features['mean_return'] = active_days['daily_return'].mean()
    features['std_return'] = active_days['daily_return'].std()
    features['return_skew'] = active_days['daily_return'].skew()
    features['return_kurtosis'] = active_days['daily_return'].kurtosis()
    
    # Return consistency (coefficient of variation)
    features['return_consistency'] = features['std_return'] / (abs(features['mean_return']) + 1e-10)
    
    # Sharpe ratio
    risk_free_rate = 0.0001
    excess_return = features['mean_return'] - risk_free_rate
    sharpe = excess_return / (features['std_return'] + 1e-10)
    features['sharpe_ratio'] = np.clip(sharpe, -5, 5)  # Cap at -5 to 5
    
    # Technical indicators
    if 'rsi' in group.columns:
        features['rsi_mean'] = group['rsi'].mean()
    else:
        features['rsi_mean'] = 50
    
    if 'bb_width' in group.columns:
        features['bb_width_mean'] = group['bb_width'].mean()
    else:
        features['bb_width_mean'] = 0
    
    if 'macd' in group.columns:
        features['macd_volatility'] = group['macd'].std()
    else:
        features['macd_volatility'] = 0
    
    # Liquidity
    features['avg_volume'] = active_days['Volume'].mean()
    features['volume_volatility'] = active_days['Volume'].std()
    
    if 'amihud_illiquidity' in group.columns:
        features['amihud_illiquidity'] = group['amihud_illiquidity'].median()
    else:
        features['amihud_illiquidity'] = 0
    
    features['trading_frequency'] = len(active_days) / len(group)
    
    # Momentum
    if 'momentum_20d' in group.columns:
        features['momentum_20d'] = group['momentum_20d'].iloc[-1]
    else:
        features['momentum_20d'] = 0
    
    if 'momentum_60d' in group.columns:
        features['momentum_60d'] = group['momentum_60d'].iloc[-1]
    else:
        features['momentum_60d'] = 0
    
    if 'price_to_ma60' in group.columns:
        features['trend_strength'] = group['price_to_ma60'].iloc[-1]
    else:
        features['trend_strength'] = 0
    
    features['current_price'] = group['Day Price'].iloc[-1]
    
    return features
