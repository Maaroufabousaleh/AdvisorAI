"""
Data Normalizer Module

Handles normalization, scaling, and feature transformation for financial datasets.
Provides various scaling strategies optimized for financial time series data.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, 
    QuantileTransformer, PowerTransformer
)
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)


class FinancialNormalizer(BaseEstimator, TransformerMixin):
    """
    Specialized normalizer for financial data with multiple scaling strategies.
    Handles different types of financial features appropriately.
    """
    
    def __init__(self, 
                 price_scaling: str = 'minmax',
                 volume_scaling: str = 'log_standard',
                 indicator_scaling: str = 'standard',
                 return_scaling: str = 'standard',
                 preserve_columns: List[str] = None):
        """
        Initialize the financial normalizer.
        
        Args:
            price_scaling: Scaling method for price-related features
            volume_scaling: Scaling method for volume features  
            indicator_scaling: Scaling method for technical indicators
            return_scaling: Scaling method for return features
            preserve_columns: Columns to preserve without scaling
        """
        self.price_scaling = price_scaling
        self.volume_scaling = volume_scaling
        self.indicator_scaling = indicator_scaling
        self.return_scaling = return_scaling
        self.preserve_columns = preserve_columns or []
        
        # Will store fitted scalers
        self.scalers_ = {}
        self.feature_categories_ = {}
        self.fitted_ = False
    
    def _categorize_features(self, columns: List[str]) -> Dict[str, List[str]]:
        """
        Categorize features based on their names and characteristics.
        
        Args:
            columns: List of column names
            
        Returns:
            Dictionary mapping categories to column lists
        """
        categories = {
            'price': [],
            'volume': [],
            'returns': [],
            'indicators': [],
            'ratios': [],
            'preserve': []
        }
        
        for col in columns:
            col_lower = col.lower()
            
            # Skip non-numeric, preserve, or datetime columns
            if col in self.preserve_columns:
                categories['preserve'].append(col)
            elif any(term in col_lower for term in ['symbol', 'timestamp', 'date', 'datetime_unix', 'unix']):
                categories['preserve'].append(col)
            # Price-related features
            elif any(term in col_lower for term in ['price', 'close', 'open', 'high', 'low', 'ask', 'bid']):
                categories['price'].append(col)
            # Volume features
            elif any(term in col_lower for term in ['volume', 'vol', 'amount', 'size']):
                categories['volume'].append(col)
            # Return features
            elif any(term in col_lower for term in ['return', 'pct', 'change', 'diff']):
                categories['returns'].append(col)
            # Technical indicators
            elif any(term in col_lower for term in ['rsi', 'macd', 'ema', 'sma', 'bollinger', 'atr', 'stoch', 'williams']):
                categories['indicators'].append(col)
            # Ratio features
            elif any(term in col_lower for term in ['ratio', 'rate', 'correlation', 'beta']):
                categories['ratios'].append(col)
            else:
                # Default to indicators for unknown numeric features
                categories['indicators'].append(col)
        
        return categories
    
    def _create_scaler(self, scaling_method: str) -> BaseEstimator:
        """
        Create a scaler based on the specified method.
        
        Args:
            scaling_method: Name of scaling method
            
        Returns:
            Initialized scaler object
        """
        if scaling_method == 'standard':
            return StandardScaler()
        elif scaling_method == 'minmax':
            return MinMaxScaler(feature_range=(-1, 1))  # Scale to [-1, 1]
        elif scaling_method == 'robust':
            # Use more extreme quantiles for better outlier handling
            return RobustScaler(with_centering=True, with_scaling=True, 
                              quantile_range=(1.0, 99.0))
        elif scaling_method == 'quantile':
            return QuantileTransformer(output_distribution='normal', random_state=42)
        elif scaling_method == 'power':
            return PowerTransformer(method='yeo-johnson', standardize=True)
        elif scaling_method == 'log_standard':
            return LogStandardScaler()
        elif scaling_method == 'none':
            return IdentityScaler()
        else:
            logger.warning(f"Unknown scaling method: {scaling_method}, using standard")
            return StandardScaler()
    
    def fit(self, X: pd.DataFrame, y=None) -> 'FinancialNormalizer':
        """
        Fit the normalizer to the data.
        
        Args:
            X: Input DataFrame
            y: Ignored (for sklearn compatibility)
            
        Returns:
            Self
        """
        logger.info("Fitting financial normalizer...")
        
        # Categorize features
        numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        self.feature_categories_ = self._categorize_features(numeric_columns)
        
        # Create and fit scalers for each category
        scaling_map = {
            'price': self.price_scaling,
            'volume': self.volume_scaling,
            'returns': self.return_scaling,
            'indicators': self.indicator_scaling,
            'ratios': self.indicator_scaling,  # Use indicator scaling for ratios
        }
        
        for category, columns in self.feature_categories_.items():
            if category in scaling_map and columns:
                scaling_method = scaling_map[category]
                scaler = self._create_scaler(scaling_method)
                
                # Fit scaler on the subset of data
                try:
                    scaler.fit(X[columns])
                    self.scalers_[category] = scaler
                    logger.info(f"Fitted {scaling_method} scaler for {category} features: {len(columns)} columns")
                except Exception as e:
                    logger.error(f"Error fitting scaler for {category}: {e}")
                    # Fall back to identity scaler
                    self.scalers_[category] = IdentityScaler()
        
        self.fitted_ = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data using fitted scalers and ensure values are within [-1, 1].
        
        Args:
            X: Input DataFrame
            
        Returns:
            Transformed DataFrame with all numeric values clipped to [-1, 1]
        """
        if not self.fitted_:
            raise ValueError("Normalizer must be fitted before transform")
        
        X_transformed = X.copy()
        
        # Transform each category
        for category, scaler in self.scalers_.items():
            columns = self.feature_categories_[category]
            if columns:
                try:
                    # Transform and update the DataFrame
                    transformed_values = scaler.transform(X[columns])
                    
                    # Ensure all values are within [-1, 1] using percentile-based clipping
                    if category in ['volume', 'returns', 'indicators', 'ratios']:
                        # Calculate safe clipping thresholds based on data distribution
                        lower_percentile = np.percentile(transformed_values, 1)  # 1st percentile
                        upper_percentile = np.percentile(transformed_values, 99)  # 99th percentile
                        scale_factor = max(abs(lower_percentile), abs(upper_percentile))
                        if scale_factor > 1:
                            transformed_values = transformed_values / scale_factor
                    
                    # Final safety clip to ensure absolute bounds
                    transformed_values = np.clip(transformed_values, -1, 1)
                    X_transformed[columns] = transformed_values
                    
                except Exception as e:
                    logger.error(f"Error transforming {category} features: {e}")
                    # Keep original values if transformation fails
                    pass
        
        logger.info(f"Transformed data shape: {X_transformed.shape}")
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Fit the normalizer and transform the data in one step.
        
        Args:
            X: Input DataFrame
            y: Ignored
            
        Returns:
            Transformed DataFrame
        """
        return self.fit(X, y).transform(X)
    
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform the scaled data back to original scale.
        
        Args:
            X: Scaled DataFrame
            
        Returns:
            DataFrame in original scale
        """
        if not self.fitted_:
            raise ValueError("Normalizer must be fitted before inverse_transform")
        
        X_inverse = X.copy()
        
        # Inverse transform each category
        for category, scaler in self.scalers_.items():
            columns = self.feature_categories_[category]
            if columns and hasattr(scaler, 'inverse_transform'):
                try:
                    X_inverse[columns] = scaler.inverse_transform(X[columns])
                except Exception as e:
                    logger.error(f"Error inverse transforming {category} features: {e}")
        
        return X_inverse
    
    def save(self, filepath: str):
        """Save the fitted normalizer to disk."""
        if not self.fitted_:
            raise ValueError("Cannot save unfitted normalizer")
        
        joblib.dump(self, filepath)
        logger.info(f"Normalizer saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'FinancialNormalizer':
        """Load a fitted normalizer from disk."""
        normalizer = joblib.load(filepath)
        logger.info(f"Normalizer loaded from {filepath}")
        return normalizer
    
    def get_feature_info(self) -> Dict:
        """Get information about feature categorization and scaling."""
        if not self.fitted_:
            return {"status": "not_fitted"}
        
        info = {
            "fitted": True,
            "categories": {},
            "scaling_methods": {}
        }
        
        for category, columns in self.feature_categories_.items():
            info["categories"][category] = {
                "count": len(columns),
                "columns": columns
            }
            
            if category in self.scalers_:
                scaler = self.scalers_[category]
                info["scaling_methods"][category] = type(scaler).__name__
        
        return info


class LogStandardScaler(BaseEstimator, TransformerMixin):
    """
    Log transformation followed by standard scaling.
    Useful for volume and other highly skewed financial features.
    """
    
    def __init__(self, offset: float = 1.0):
        """
        Args:
            offset: Small value added before log to handle zeros
        """
        self.offset = offset
        self.scaler = StandardScaler()
        
    def fit(self, X, y=None):
        """Fit the log-standard scaler with safe log handling (no -inf/NaN)."""
        # Ensure strictly positive input to log by flooring at a tiny epsilon
        X_safe = np.maximum(X + self.offset, 1e-12)
        X_log = np.log(X_safe)
        self.scaler.fit(X_log)
        return self
    
    def transform(self, X):
        """Transform using log then standard scaling with safe log handling."""
        X_safe = np.maximum(X + self.offset, 1e-12)
        X_log = np.log(X_safe)
        return self.scaler.transform(X_log)
    
    def inverse_transform(self, X):
        """Inverse transform from scaled log back to original scale."""
        X_scaled_back = self.scaler.inverse_transform(X)
        return np.exp(X_scaled_back) - self.offset


class IdentityScaler(BaseEstimator, TransformerMixin):
    """
    Identity scaler that returns data unchanged.
    Used for features that should not be scaled.
    """
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X
    
    def inverse_transform(self, X):
        return X


class DataNormalizer:
    """
    Main data normalization orchestrator for the financial system.
    Handles both stocks and crypto data with appropriate scaling strategies.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the data normalizer with configuration.
        
        Args:
            config: Configuration dictionary for scaling parameters
        """
        default_config = {
            'stocks': {
                'price_scaling': 'minmax',
                'volume_scaling': 'log_standard', 
                'indicator_scaling': 'standard',
                'return_scaling': 'standard'
            },
            'crypto': {
                'price_scaling': 'robust',  # More robust for crypto volatility
                'volume_scaling': 'log_standard',
                'indicator_scaling': 'robust',
                'return_scaling': 'robust'
            }
        }
        
        self.config = config or default_config
        self.normalizers = {}
    
    def fit_stocks(self, stocks_df: pd.DataFrame) -> 'DataNormalizer':
        """
        Fit normalizer for stocks data.
        
        Args:
            stocks_df: Stocks DataFrame
            
        Returns:
            Self
        """
        logger.info("Fitting stocks normalizer...")
        
        normalizer = FinancialNormalizer(**self.config['stocks'])
        normalizer.fit(stocks_df)
        self.normalizers['stocks'] = normalizer
        
        return self
    
    def fit_crypto(self, crypto_df: pd.DataFrame) -> 'DataNormalizer':
        """
        Fit normalizer for crypto data.
        
        Args:
            crypto_df: Crypto DataFrame
            
        Returns:
            Self
        """
        logger.info("Fitting crypto normalizer...")
        
        normalizer = FinancialNormalizer(**self.config['crypto'])
        normalizer.fit(crypto_df)
        self.normalizers['crypto'] = normalizer
        
        return self
    
    def transform_stocks(self, stocks_df: pd.DataFrame) -> pd.DataFrame:
        """Transform stocks data using fitted normalizer."""
        if 'stocks' not in self.normalizers:
            raise ValueError("Stocks normalizer not fitted")
        
        return self.normalizers['stocks'].transform(stocks_df)
    
    def transform_crypto(self, crypto_df: pd.DataFrame) -> pd.DataFrame:
        """Transform crypto data using fitted normalizer."""
        if 'crypto' not in self.normalizers:
            raise ValueError("Crypto normalizer not fitted")
        
        return self.normalizers['crypto'].transform(crypto_df)
    
    def save_normalizers(self, output_dir: str):
        """Save all fitted normalizers."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for asset_type, normalizer in self.normalizers.items():
            filepath = output_path / f"{asset_type}_normalizer.joblib"
            normalizer.save(str(filepath))
    
    def load_normalizers(self, input_dir: str):
        """Load normalizers from disk."""
        input_path = Path(input_dir)
        
        for asset_type in ['stocks', 'crypto']:
            filepath = input_path / f"{asset_type}_normalizer.joblib"
            if filepath.exists():
                self.normalizers[asset_type] = FinancialNormalizer.load(str(filepath))
                logger.info(f"Loaded {asset_type} normalizer")
    
    def get_normalization_summary(self) -> Dict:
        """Get summary of all fitted normalizers."""
        summary = {}
        
        for asset_type, normalizer in self.normalizers.items():
            summary[asset_type] = normalizer.get_feature_info()
        
        return summary
