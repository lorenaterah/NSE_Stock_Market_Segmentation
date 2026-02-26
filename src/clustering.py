
# Stock Clustering Module

# grouping stocks into risk profiles using K-Means clustering.

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pickle
import os


def find_optimal_clusters(df, feature_cols, max_clusters=8):
    # Test different numbers of clusters to find the best fit.
    
    X = df[feature_cols].fillna(df[feature_cols].median())
    
    #RobustScaler works better for financial data
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    results = {'n_clusters': [], 'inertia': [], 'silhouette': []}
    
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=50, max_iter=500)
        labels = kmeans.fit_predict(X_scaled)
        
        results['n_clusters'].append(k)
        results['inertia'].append(kmeans.inertia_)
        results['silhouette'].append(silhouette_score(X_scaled, labels))
    
    return pd.DataFrame(results)


class StockClusterer:
        
    def __init__(self, n_clusters=4, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = RobustScaler()  # Back to RobustScaler - works better
        self.model = None
        self.feature_columns = None
    
    def fit_predict(self, df):
            #    Train clustering model and assign risk profiles.
        
        # BALANCED FEATURE SELECTION - best features for good separation
        feature_cols = [
            # Core volatility (most important)
            'std_return', 'volatility_mean', 'volatility_max',
            
            # Downside risk (critical)
            'max_drawdown', 'downside_deviation', 'var_95',
            
            # Risk-adjusted return
            'sharpe_ratio',
            
            # Distribution shape
            'return_skew', 'return_kurtosis',
            
            # Technical indicators
            'rsi_mean', 'bb_width_mean',
            
            # Momentum
            'momentum_20d', 'momentum_60d',
            
            # Liquidity risk
            'trading_frequency', 'amihud_illiquidity'
        ]
        
        # Only use features that exist in dataframe
        self.feature_columns = [col for col in feature_cols if col in df.columns]
        
        print(f"Using {len(self.feature_columns)} features for clustering:")
        print(", ".join(self.feature_columns))
        
        # Handle missing values with median imputation
        X = df[self.feature_columns].fillna(df[self.feature_columns].median())
        
        # Scale features (RobustScaler handles outliers better)
        X_scaled = self.scaler.fit_transform(X)
        
        # Train K-Means
        self.model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=100,  # More initializations for better results
            max_iter=500,
            algorithm='full'
        )
        
        df['Cluster'] = self.model.fit_predict(X_scaled)
        
        # Calculate cluster centers to assign risk labels
        cluster_risk = self._assign_risk_labels(df, X_scaled)
        df['Risk_Profile'] = df['Cluster'].map(cluster_risk)
        
        # Calculate silhouette score
        sil_score = silhouette_score(X_scaled, df['Cluster'])
        print(f"\n Clustering complete!")
        print(f" Silhouette Score: {sil_score:.3f}")
        
        if sil_score >= 0.5:
            print("EXCELLENT separation! (≥0.5)")
        elif sil_score >= 0.4:
            print("✓ Good separation (0.4-0.5)")
        elif sil_score >= 0.3:
            print("Moderate separation (0.3-0.4)")
        else:
            print("Weak separation (<0.3)")
        
        return df
    
    def _assign_risk_labels(self, df, X_scaled):
        """
        Assign risk labels based on cluster characteristics.
        Sorts clusters by average volatility.
        """
        cluster_stats = []
        
        for cluster_id in range(self.n_clusters):
            mask = df['Cluster'] == cluster_id
            cluster_data = X_scaled[mask]
            
            # Use first 6 features (main risk measures) for ranking
            risk_score = np.mean(np.abs(cluster_data[:, :6]), axis=0).mean()
            
            cluster_stats.append({
                'cluster': cluster_id,
                'risk_score': risk_score,
                'count': mask.sum()
            })
        
        # Sort by risk score
        cluster_stats.sort(key=lambda x: x['risk_score'])
        
        # Assign labels based on sorted risk
        risk_labels = ['Low Risk', 'Medium-Low Risk', 'Medium-High Risk', 'High Risk']
        
        risk_mapping = {}
        for idx, stats in enumerate(cluster_stats):
            label = risk_labels[min(idx, len(risk_labels)-1)]
            risk_mapping[stats['cluster']] = label
            print(f"Cluster {stats['cluster']}: {label} ({stats['count']} stocks)")
        
        return risk_mapping
    
    def get_cluster_summary(self, df):
        """Get detailed statistics for each cluster"""
        summary = df.groupby('Risk_Profile').agg({
            'Stock_code': 'count',
            'volatility_mean': 'mean',
            'sharpe_ratio': 'mean',
            'max_drawdown': 'mean',
            'trading_frequency': 'mean',
            'avg_volume': 'median'
        }).round(4)
        
        summary.columns = ['Count', 'Avg Volatility', 'Sharpe Ratio', 
                          'Avg Drawdown', 'Trading Freq', 'Median Volume']
        
        return summary
    
    def save_model(self, filepath):
        # Save trained model to disk
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'n_clusters': self.n_clusters,
            'random_state': self.random_state
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath):
        """Load trained model from disk"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create instance and restore state
        instance = cls(
            n_clusters=model_data['n_clusters'],
            random_state=model_data['random_state']
        )
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        instance.feature_columns = model_data['feature_columns']
        
        print(f"Model loaded from {filepath}")
        return instance
    
    def predict(self, df):
        # Predict clusters for new data
        if self.model is None:
            raise ValueError("Model not trained! Call fit_predict() first.")
        
        X = df[self.feature_columns].fillna(df[self.feature_columns].median())
        X_scaled = self.scaler.transform(X)
        
        return self.model.predict(X_scaled)
