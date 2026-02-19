# Clustering Functions (embedded in notebook)
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Finding optimal number of clusters.
def find_optimal_clusters(df, feature_cols, max_clusters=8):
    
    X = df[feature_cols].fillna(df[feature_cols].median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    results = {'n_clusters': [], 'inertia': [], 'silhouette': []}
    
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=50)
        labels = kmeans.fit_predict(X_scaled)
        
        results['n_clusters'].append(k)
        results['inertia'].append(kmeans.inertia_)
        results['silhouette'].append(silhouette_score(X_scaled, labels))
    
    return pd.DataFrame(results)

# Cluster stocks into risk profiles.
class StockClusterer:
    
    def __init__(self, n_clusters=4, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None
        self.feature_columns = None
    
    # Fit and predict clusters.
    def fit_predict(self, df):
        
        # Select features
        feature_cols = [
            'volatility_7d', 'volatility_30d', 'max_drawdown',
            'trading_frequency', 'zero_volume_ratio', 'avg_volume',
            'mean_return', 'std_return', 'momentum_30d'
        ]
        
        self.feature_columns = [col for col in feature_cols if col in df.columns]
        
        X = df[self.feature_columns].fillna(df[self.feature_columns].median())
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=50,
            max_iter=500
        )
        
        df['Cluster'] = self.model.fit_predict(X_scaled)
        
        sil_score = silhouette_score(X_scaled, df['Cluster'])
        print(f" Clustering complete. Silhouette: {sil_score:.3f}")
        
        return df
    
        # cluster profiles.
    def get_cluster_summary(self):
        return [
            {'cluster': 0, 'risk_profile': 'Low Risk', 'count': 0, 
             'avg_volatility': 0, 'avg_drawdown': 0, 'avg_trading_freq': 0},
            {'cluster': 1, 'risk_profile': 'Medium-Low Risk', 'count': 0,
             'avg_volatility': 0, 'avg_drawdown': 0, 'avg_trading_freq': 0},
            {'cluster': 2, 'risk_profile': 'Medium-High Risk', 'count': 0,
             'avg_volatility': 0, 'avg_drawdown': 0, 'avg_trading_freq': 0},
            {'cluster': 3, 'risk_profile': 'High Risk', 'count': 0,
             'avg_volatility': 0, 'avg_drawdown': 0, 'avg_trading_freq': 0}
        ]
