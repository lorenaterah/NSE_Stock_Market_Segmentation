# Feature Engineering Functions 

# Calculate daily returns for a stock.

import numpy as np

def calculate_returns(group):
    group = group.sort_values('Date').copy()
    group['daily_return'] = group['Day Price'].pct_change()
    return group

# Calculating rolling volatility metrics.
def calculate_volatility_features(group, windows=[7, 14, 30]): 
    group = group.sort_values('Date').copy()
    
    for window in windows:        # Doing the same calculation for 7 days, then 14, then 30
        group[f'volatility_{window}d'] = group['daily_return'].rolling(
            window=window, min_periods=max(3, window//2)      # ensures we don’t calculate volatility unless we have enough data
        ).std()
    
    return group

# Calculate liquidity-related features.
def calculate_liquidity_features(group):
    
    group = group.sort_values('Date').copy()
    
    group['avg_volume'] = group['Volume'].rolling(window=30, min_periods=10).mean()
    group['volume_volatility'] = group['Volume'].rolling(window=30, min_periods=10).std()
    group['zero_volume_days'] = group['Volume'].rolling(window=30, min_periods=10).apply(
        lambda x: (x == 0).sum() if x.notna().sum() > 0 else np.nan
    )
    
    return group

# Calculating momentum and trend indicators.
def calculate_momentum_features(group):
    group = group.sort_values('Date').copy()
    
    group['momentum_30d'] = group['Day Price'].pct_change(periods=30)
    group['ma_7'] = group['Day Price'].rolling(window=7, min_periods=3).mean()
    group['ma_30'] = group['Day Price'].rolling(window=30, min_periods=10).mean()
    group['price_to_ma30'] = (group['Day Price'] - group['ma_30']) / group['ma_30']
    
    return group

# Calculate maximum drawdown.
def calculate_drawdown(group):
    group = group.sort_values('Date').copy()
    
    running_max = group['Day Price'].expanding().max()
    drawdown = (group['Day Price'] - running_max) / running_max
    
    group['current_drawdown'] = drawdown
    group['max_drawdown'] = group['current_drawdown'].expanding().min()
    
    return group

# Aggregate features at stock level.
def aggregate_stock_features(group):
    
    active_days = group[(group['Volume'].notna()) & (group['Volume'] > 0)]
    
    if len(active_days) < 20:
        return None
    
    features = {
        'Stock_code': group['Stock_code'].iloc[0],
        'Sector': group['Sector'].iloc[0] if 'Sector' in group.columns else None,
        'Name': group['Name'].iloc[0] if 'Name' in group.columns else None,
        
        'trading_days': len(active_days),
        'total_days': len(group),
        'trading_frequency': len(active_days) / len(group),
        
        'mean_return': active_days['daily_return'].mean(),
        'std_return': active_days['daily_return'].std(),
        
        'volatility_7d': group['volatility_7d'].mean() if 'volatility_7d' in group.columns else 0,
        'volatility_30d': group['volatility_30d'].mean() if 'volatility_30d' in group.columns else 0,
        
        'max_drawdown': group['max_drawdown'].min() if 'max_drawdown' in group.columns else 0,
        
        'avg_volume': active_days['Volume'].mean(),
        'zero_volume_ratio': (group['Volume'] == 0).sum() / len(group),
        
        'momentum_30d': group['momentum_30d'].iloc[-1] if 'momentum_30d' in group.columns else 0,
        'current_price': group['Day Price'].iloc[-1]
    }
    
    return features
    