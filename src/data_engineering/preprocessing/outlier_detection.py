"""
Outlier Detection Module

Comprehensive outlier detection and handling for financial time series data.
Uses multiple methods appropriate for different types of financial features.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class OutlierDetector:
    """
    Multi-method outlier detection for financial data.
    Applies different strategies based on feature type and financial context.
    """
    
    def __init__(self, 
                 price_method: str = 'iqr',
                 volume_method: str = 'isolation_forest',
                 return_method: str = 'zscore',
                 indicator_method: str = 'iqr',
                 price_threshold: float = 3.0,
                 volume_contamination: float = 0.05,
                 return_threshold: float = 4.0,
                 preserve_extreme_moves: bool = True):
        """
        Initialize outlier detector.
        
        Args:
            price_method: Method for price outliers ('iqr', 'zscore', 'isolation_forest')
            volume_method: Method for volume outliers
            return_method: Method for return outliers
            indicator_method: Method for technical indicator outliers
            price_threshold: Threshold for price outlier detection
            volume_contamination: Expected contamination rate for volume
            return_threshold: Threshold for return outlier detection
            preserve_extreme_moves: Whether to preserve legitimate extreme market moves
        """
        self.price_method = price_method
        self.volume_method = volume_method
        self.return_method = return_method
        self.indicator_method = indicator_method
        self.price_threshold = price_threshold
        self.volume_contamination = volume_contamination
        self.return_threshold = return_threshold
        self.preserve_extreme_moves = preserve_extreme_moves
        
        self.feature_categories = {}
        self.outlier_stats = {}
    
    def detect_outliers(self, df: pd.DataFrame, asset_type: str = 'unknown') -> Dict:
        """
        Detect outliers across all feature types.
        
        Args:
            df: Input DataFrame
            asset_type: Type of asset for context
            
        Returns:
            Dictionary with outlier detection results
        """
        logger.info(f"Detecting outliers in {asset_type} data...")
        
        # Categorize features
        self.feature_categories = self._categorize_features(df)
        
        outlier_results = {
            'total_outliers': 0,
            'outliers_by_method': {},
            'outliers_by_category': {},
            'outlier_indices': set(),
            'outlier_details': {}
        }
        
        # Apply detection method for each category
        method_map = {
            'price': self.price_method,
            'volume': self.volume_method,
            'returns': self.return_method,
            'indicators': self.indicator_method,
            'other': self.indicator_method
        }
        
        for category, columns in self.feature_categories.items():
            if columns:
                method = method_map.get(category, 'iqr')
                category_outliers = self._detect_category_outliers(
                    df, columns, method, category, asset_type
                )
                
                outlier_results['outliers_by_category'][category] = category_outliers
                outlier_results['outlier_indices'].update(category_outliers['indices'])
        
        # Combine results
        outlier_results['total_outliers'] = len(outlier_results['outlier_indices'])
        outlier_results['outlier_percentage'] = (
            outlier_results['total_outliers'] / len(df) * 100 if len(df) > 0 else 0
        )
        
        logger.info(f"Detected {outlier_results['total_outliers']} outlier rows "
                   f"({outlier_results['outlier_percentage']:.2f}%)")
        
        return outlier_results
    
    def _categorize_features(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Categorize features for targeted outlier detection.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary mapping categories to column lists
        """
        categories = {
            'price': [],
            'volume': [],
            'returns': [],
            'indicators': [],
            'other': []
        }
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            col_lower = col.lower()
            
            # Price-related features
            if any(term in col_lower for term in ['price', 'close', 'open', 'high', 'low', 'ask', 'bid']):
                categories['price'].append(col)
            # Volume features
            elif any(term in col_lower for term in ['volume', 'vol', 'amount', 'size']):
                categories['volume'].append(col)
            # Return features
            elif any(term in col_lower for term in ['return', 'pct', 'change']):
                categories['returns'].append(col)
            # Technical indicators
            elif any(term in col_lower for term in ['rsi', 'macd', 'ema', 'sma', 'bollinger', 'atr', 'stoch']):
                categories['indicators'].append(col)
            else:
                categories['other'].append(col)
        
        return categories
    
    def _detect_category_outliers(self, df: pd.DataFrame, columns: List[str], 
                                 method: str, category: str, asset_type: str) -> Dict:
        """
        Detect outliers for a specific category of features.
        
        Args:
            df: Input DataFrame
            columns: Columns to analyze
            method: Detection method
            category: Feature category
            asset_type: Asset type
            
        Returns:
            Dictionary with outlier detection results for category
        """
        outlier_indices = set()
        column_stats = {}
        
        for col in columns:
            if col in df.columns:
                col_outliers = self._detect_column_outliers(
                    df[col], method, category, asset_type
                )
                outlier_indices.update(col_outliers['indices'])
                column_stats[col] = col_outliers['stats']
        
        return {
            'method': method,
            'indices': outlier_indices,
            'column_stats': column_stats,
            'total_outliers': len(outlier_indices)
        }
    
    def _detect_column_outliers(self, series: pd.Series, method: str, 
                               category: str, asset_type: str) -> Dict:
        """
        Detect outliers in a single column.
        
        Args:
            series: Pandas Series to analyze
            method: Detection method
            category: Feature category
            asset_type: Asset type
            
        Returns:
            Dictionary with outlier indices and statistics
        """
        outlier_indices = set()
        stats_info = {}
        
        # Remove NaN values for calculation
        clean_series = series.dropna()
        
        if len(clean_series) == 0:
            return {'indices': outlier_indices, 'stats': stats_info}
        
        if method == 'zscore':
            outlier_indices, stats_info = self._zscore_outliers(
                clean_series, threshold=self._get_threshold(category)
            )
        
        elif method == 'iqr':
            outlier_indices, stats_info = self._iqr_outliers(clean_series)
        
        elif method == 'isolation_forest':
            outlier_indices, stats_info = self._isolation_forest_outliers(
                clean_series, contamination=self._get_contamination(category)
            )
        
        elif method == 'modified_zscore':
            outlier_indices, stats_info = self._modified_zscore_outliers(
                clean_series, threshold=self._get_threshold(category)
            )
        
        elif method == 'percentile':
            outlier_indices, stats_info = self._percentile_outliers(clean_series)
        
        else:
            logger.warning(f"Unknown outlier detection method: {method}, using IQR")
            outlier_indices, stats_info = self._iqr_outliers(clean_series)
        
        # Filter outliers based on financial context
        if self.preserve_extreme_moves and category in ['returns', 'price']:
            outlier_indices = self._filter_legitimate_moves(
                series, outlier_indices, category, asset_type
            )
        
        return {'indices': outlier_indices, 'stats': stats_info}
    
    def _zscore_outliers(self, series: pd.Series, threshold: float = 3.0) -> Tuple[set, Dict]:
        """Detect outliers using Z-score method."""
        z_scores = np.abs(stats.zscore(series))
        outlier_mask = z_scores > threshold
        outlier_indices = set(series[outlier_mask].index)
        
        stats_info = {
            'method': 'zscore',
            'threshold': threshold,
            'mean': series.mean(),
            'std': series.std(),
            'max_zscore': z_scores.max(),
            'outlier_count': len(outlier_indices)
        }
        
        return outlier_indices, stats_info
    
    def _iqr_outliers(self, series: pd.Series, multiplier: float = 1.5) -> Tuple[set, Dict]:
        """Detect outliers using Interquartile Range method."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        outlier_mask = (series < lower_bound) | (series > upper_bound)
        outlier_indices = set(series[outlier_mask].index)
        
        stats_info = {
            'method': 'iqr',
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'outlier_count': len(outlier_indices)
        }
        
        return outlier_indices, stats_info
    
    def _isolation_forest_outliers(self, series: pd.Series, 
                                  contamination: float = 0.1) -> Tuple[set, Dict]:
        """Detect outliers using Isolation Forest."""
        # Reshape for sklearn
        X = series.values.reshape(-1, 1)
        
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outlier_labels = iso_forest.fit_predict(X)
        
        # -1 indicates outlier in sklearn
        outlier_mask = outlier_labels == -1
        outlier_indices = set(series[outlier_mask].index)
        
        stats_info = {
            'method': 'isolation_forest',
            'contamination': contamination,
            'outlier_count': len(outlier_indices),
            'decision_scores': iso_forest.decision_function(X)
        }
        
        return outlier_indices, stats_info
    
    def _modified_zscore_outliers(self, series: pd.Series, 
                                 threshold: float = 3.5) -> Tuple[set, Dict]:
        """Detect outliers using Modified Z-score (median-based)."""
        median = series.median()
        mad = np.median(np.abs(series - median))
        
        # Modified Z-score
        modified_z_scores = 0.6745 * (series - median) / mad
        outlier_mask = np.abs(modified_z_scores) > threshold
        outlier_indices = set(series[outlier_mask].index)
        
        stats_info = {
            'method': 'modified_zscore',
            'threshold': threshold,
            'median': median,
            'mad': mad,
            'max_modified_zscore': np.abs(modified_z_scores).max(),
            'outlier_count': len(outlier_indices)
        }
        
        return outlier_indices, stats_info
    
    def _percentile_outliers(self, series: pd.Series, 
                           lower_percentile: float = 0.5, 
                           upper_percentile: float = 99.5) -> Tuple[set, Dict]:
        """Detect outliers using percentile method."""
        lower_bound = series.quantile(lower_percentile / 100)
        upper_bound = series.quantile(upper_percentile / 100)
        
        outlier_mask = (series < lower_bound) | (series > upper_bound)
        outlier_indices = set(series[outlier_mask].index)
        
        stats_info = {
            'method': 'percentile',
            'lower_percentile': lower_percentile,
            'upper_percentile': upper_percentile,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'outlier_count': len(outlier_indices)
        }
        
        return outlier_indices, stats_info
    
    def _filter_legitimate_moves(self, series: pd.Series, outlier_indices: set, 
                                category: str, asset_type: str) -> set:
        """
        Filter out legitimate extreme market moves from outliers.
        
        Args:
            series: Data series
            outlier_indices: Initial outlier indices
            category: Feature category
            asset_type: Asset type
            
        Returns:
            Filtered outlier indices
        """
        filtered_indices = outlier_indices.copy()
        
        # For crypto, be more tolerant of extreme moves
        if asset_type == 'crypto' and category == 'returns':
            # Crypto can have very large moves, be more conservative
            extreme_threshold = 0.5  # 50% moves might be legitimate
            for idx in outlier_indices:
                # Safety check: ensure index exists in series
                if idx not in series.index:
                    continue
                    
                if abs(series.loc[idx]) < extreme_threshold:
                    continue  # Keep as outlier
                else:
                    # Check if this is part of a sustained move
                    if self._is_sustained_move(series, idx):
                        filtered_indices.discard(idx)
        
        # For stocks, be more strict but still consider market crashes/rallies
        elif asset_type == 'stocks' and category == 'returns':
            extreme_threshold = 0.2  # 20% moves
            for idx in outlier_indices:
                # Safety check: ensure index exists in series
                if idx not in series.index:
                    continue
                    
                if abs(series.loc[idx]) < extreme_threshold:
                    continue  # Keep as outlier
                else:
                    # Check market context
                    if self._is_market_wide_event(series, idx):
                        filtered_indices.discard(idx)
        
        removed_count = len(outlier_indices) - len(filtered_indices)
        if removed_count > 0:
            logger.info(f"Preserved {removed_count} legitimate extreme moves in {category}")
        
        return filtered_indices
    
    def _is_sustained_move(self, series: pd.Series, idx: int, window: int = 3) -> bool:
        """Check if an extreme move is part of a sustained trend."""
        try:
            # Convert to positional index if idx is a label
            if idx in series.index:
                pos_idx = series.index.get_loc(idx)
            else:
                # If idx is already positional, use as is (but check bounds)
                pos_idx = idx
                if pos_idx >= len(series):
                    return False
            
            start_idx = max(0, pos_idx - window)
            end_idx = min(len(series), pos_idx + window + 1)
            window_data = series.iloc[start_idx:end_idx]
            
            # Check if the move is consistent with surrounding data
            current_value = series.iloc[pos_idx]
            window_mean = window_data.mean()
            
            # If the extreme value is in the same direction as the window mean, 
            # it might be legitimate
            return np.sign(current_value) == np.sign(window_mean)
        except:
            return False
    
    def _is_market_wide_event(self, series: pd.Series, idx: int) -> bool:
        """
        Check if an extreme move coincides with a market-wide event.
        This is a simplified check - in practice, you'd want to check
        against market indices or news events.
        """
        # Simplified: check if multiple assets have extreme moves on the same day
        # This would require additional market data in practice
        return False
    
    def _get_threshold(self, category: str) -> float:
        """Get appropriate threshold for category."""
        thresholds = {
            'price': self.price_threshold,
            'returns': self.return_threshold,
            'indicators': 3.0,
            'other': 3.0
        }
        return thresholds.get(category, 3.0)
    
    def _get_contamination(self, category: str) -> float:
        """Get appropriate contamination rate for category."""
        contaminations = {
            'volume': self.volume_contamination,
            'price': 0.02,
            'returns': 0.03,
            'indicators': 0.05,
            'other': 0.05
        }
        return contaminations.get(category, 0.05)
    
    def handle_outliers(self, df: pd.DataFrame, outlier_results: Dict, 
                       treatment: str = 'cap') -> pd.DataFrame:
        """
        Handle detected outliers using specified treatment.
        
        Args:
            df: Input DataFrame
            outlier_results: Results from outlier detection
            treatment: Treatment method ('cap', 'remove', 'transform')
            
        Returns:
            DataFrame with outliers handled
        """
        logger.info(f"Handling outliers using {treatment} method...")
        
        df_handled = df.copy()
        
        if treatment == 'remove':
            # Remove outlier rows
            outlier_indices = list(outlier_results['outlier_indices'])
            df_handled = df_handled.drop(outlier_indices)
            logger.info(f"Removed {len(outlier_indices)} outlier rows")
        
        elif treatment == 'cap':
            # Cap outliers to reasonable bounds
            for category, category_results in outlier_results['outliers_by_category'].items():
                columns = self.feature_categories.get(category, [])
                for col in columns:
                    if col in df_handled.columns:
                        df_handled = self._cap_column_outliers(
                            df_handled, col, category_results['column_stats'].get(col, {})
                        )
        
        elif treatment == 'transform':
            # Apply transformations to reduce outlier impact
            df_handled = self._transform_outliers(df_handled, outlier_results)
        
        else:
            logger.warning(f"Unknown treatment method: {treatment}")
        
        return df_handled
    
    def _cap_column_outliers(self, df: pd.DataFrame, column: str, stats: Dict) -> pd.DataFrame:
        """Cap outliers in a specific column."""
        df_result = df.copy()
        
        if 'upper_bound' in stats and 'lower_bound' in stats:
            # Use IQR bounds
            upper_bound = stats['upper_bound']
            lower_bound = stats['lower_bound']
            
            df_result[column] = df_result[column].clip(lower=lower_bound, upper=upper_bound)
        
        elif 'method' in stats and stats['method'] == 'percentile':
            # Use percentile bounds
            upper_bound = stats['upper_bound']
            lower_bound = stats['lower_bound']
            
            df_result[column] = df_result[column].clip(lower=lower_bound, upper=upper_bound)
        
        return df_result
    
    def _transform_outliers(self, df: pd.DataFrame, outlier_results: Dict) -> pd.DataFrame:
        """Apply transformations to reduce outlier impact."""
        df_transformed = df.copy()
        
        # Apply log transformation to volume features to reduce skewness
        volume_columns = self.feature_categories.get('volume', [])
        for col in volume_columns:
            if col in df_transformed.columns:
                # Apply log(1 + x) transformation
                df_transformed[col] = np.log1p(df_transformed[col])
        
        return df_transformed
    
    def get_outlier_summary(self, outlier_results: Dict) -> Dict:
        """
        Generate comprehensive summary of outlier detection results.
        
        Args:
            outlier_results: Results from outlier detection
            
        Returns:
            Summary dictionary
        """
        summary = {
            'total_outliers': outlier_results['total_outliers'],
            'outlier_percentage': outlier_results['outlier_percentage'],
            'outliers_by_category': {},
            'methods_used': {},
            'most_problematic_features': []
        }
        
        # Summarize by category
        for category, results in outlier_results['outliers_by_category'].items():
            summary['outliers_by_category'][category] = {
                'method': results['method'],
                'outlier_count': results['total_outliers'],
                'feature_count': len(results['column_stats'])
            }
            summary['methods_used'][category] = results['method']
        
        # Find most problematic features
        feature_outlier_counts = []
        for category, results in outlier_results['outliers_by_category'].items():
            for feature, stats in results['column_stats'].items():
                feature_outlier_counts.append({
                    'feature': feature,
                    'category': category,
                    'outlier_count': stats.get('outlier_count', 0)
                })
        
        # Sort by outlier count
        feature_outlier_counts.sort(key=lambda x: x['outlier_count'], reverse=True)
        summary['most_problematic_features'] = feature_outlier_counts[:10]
        
        return summary
