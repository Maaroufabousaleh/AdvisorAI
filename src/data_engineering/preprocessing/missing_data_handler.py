"""
Missing Data Handler Module

Sophisticated handling of missing values in financial time series data.
Implements multiple strategies based on data type and financial context.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

logger = logging.getLogger(__name__)


class MissingDataHandler:
    """
    Comprehensive missing data handler for financial datasets.
    Uses different strategies based on feature type and temporal context.
    """
    
    def __init__(self, 
                 price_strategy: str = 'forward_fill',
                 volume_strategy: str = 'zero_fill',
                 indicator_strategy: str = 'interpolate',
                 news_strategy: str = 'neutral_fill',
                 max_consecutive_missing: int = 10):
        """
        Initialize missing data handler.
        
        Args:
            price_strategy: Strategy for price-related features
            volume_strategy: Strategy for volume features
            indicator_strategy: Strategy for technical indicators
            news_strategy: Strategy for news/sentiment features
            max_consecutive_missing: Maximum consecutive missing values to allow
        """
        self.price_strategy = price_strategy
        self.volume_strategy = volume_strategy
        self.indicator_strategy = indicator_strategy
        self.news_strategy = news_strategy
        self.max_consecutive_missing = max_consecutive_missing
        
        self.feature_categories = {}
        self.imputers = {}
    
    def analyze_missing_patterns(self, df: pd.DataFrame) -> Dict:
        """
        Analyze missing data patterns in the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with missing data analysis
        """
        logger.info("Analyzing missing data patterns...")
        
        analysis = {
            'total_missing': df.isnull().sum().sum(),
            'missing_percentage': (df.isnull().sum().sum() / df.size) * 100,
            'columns_with_missing': {},
            'missing_patterns': {},
            'consecutive_missing': {}
        }
        
        # Analyze missing values by column
        missing_by_column = df.isnull().sum()
        for col, count in missing_by_column.items():
            if count > 0:
                percentage = (count / len(df)) * 100
                analysis['columns_with_missing'][col] = {
                    'count': int(count),
                    'percentage': round(percentage, 2)
                }
        
        # Analyze missing patterns (which columns are missing together)
        if df.isnull().sum().sum() > 0:
            missing_patterns = df.isnull().groupby(df.columns.tolist()).size()
            analysis['missing_patterns'] = missing_patterns.to_dict()
        
        # Analyze consecutive missing values for time series
        if 'symbol' in df.columns:
            for symbol in df['symbol'].unique():
                symbol_data = df[df['symbol'] == symbol].sort_index()
                consecutive_missing = self._find_consecutive_missing(symbol_data)
                if consecutive_missing:
                    analysis['consecutive_missing'][symbol] = consecutive_missing
        else:
            consecutive_missing = self._find_consecutive_missing(df)
            if consecutive_missing:
                analysis['consecutive_missing']['all_data'] = consecutive_missing
        
        logger.info(f"Missing data analysis complete: {analysis['total_missing']} total missing values "
                   f"({analysis['missing_percentage']:.2f}%)")
        
        return analysis
    
    def _find_consecutive_missing(self, df: pd.DataFrame) -> Dict:
        """Find consecutive missing value sequences."""
        consecutive_info = {}
        
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].isnull().any():
                # Find consecutive missing sequences
                missing_mask = df[col].isnull()
                sequences = []
                current_seq_length = 0
                
                for is_missing in missing_mask:
                    if is_missing:
                        current_seq_length += 1
                    else:
                        if current_seq_length > 0:
                            sequences.append(current_seq_length)
                        current_seq_length = 0
                
                # Don't forget the last sequence if it ends with missing values
                if current_seq_length > 0:
                    sequences.append(current_seq_length)
                
                if sequences:
                    consecutive_info[col] = {
                        'max_consecutive': max(sequences),
                        'total_sequences': len(sequences),
                        'avg_sequence_length': np.mean(sequences)
                    }
        
        return consecutive_info
    
    def _categorize_features(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Categorize features based on their characteristics for targeted imputation.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary mapping categories to column lists
        """
        categories = {
            'price': [],
            'volume': [],
            'indicators': [],
            'returns': [],
            'news_sentiment': [],
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
            # Technical indicators
            elif any(term in col_lower for term in ['rsi', 'macd', 'ema', 'sma', 'bollinger', 'atr', 'stoch']):
                categories['indicators'].append(col)
            # Return features
            elif any(term in col_lower for term in ['return', 'pct', 'change']):
                categories['returns'].append(col)
            # News and sentiment features
            elif any(term in col_lower for term in ['news', 'sentiment', 'score', 'pos', 'neg', 'neu']):
                categories['news_sentiment'].append(col)
            else:
                categories['other'].append(col)
        
        self.feature_categories = categories
        return categories
    
    def handle_missing_data(self, df: pd.DataFrame, asset_type: str = 'unknown') -> pd.DataFrame:
        """
        Apply comprehensive missing data handling.
        
        Args:
            df: Input DataFrame with missing data
            asset_type: Type of asset ('stocks', 'crypto', etc.)
            
        Returns:
            DataFrame with missing data handled
        """
        logger.info(f"Handling missing data for {asset_type} data...")
        
        # Analyze missing patterns first
        missing_analysis = self.analyze_missing_patterns(df)
        
        # Categorize features
        categories = self._categorize_features(df)
        
        df_handled = df.copy()
        
        # Apply strategy for each category
        strategy_map = {
            'price': self.price_strategy,
            'volume': self.volume_strategy,
            'indicators': self.indicator_strategy,
            'returns': self.indicator_strategy,  # Treat returns like indicators
            'news_sentiment': self.news_strategy,
            'other': self.indicator_strategy
        }
        
        for category, columns in categories.items():
            if columns:
                strategy = strategy_map.get(category, 'forward_fill')
                df_handled = self._apply_strategy(df_handled, columns, strategy, asset_type)
        
        # Final validation and cleanup
        df_handled = self._validate_and_cleanup(df_handled, missing_analysis)
        
        # Log results
        final_missing = df_handled.isnull().sum().sum()
        logger.info(f"Missing data handling complete. Remaining missing values: {final_missing}")
        
        return df_handled
    
    def _apply_strategy(self, df: pd.DataFrame, columns: List[str], strategy: str, asset_type: str) -> pd.DataFrame:
        """
        Apply specific missing data strategy to columns.
        
        Args:
            df: Input DataFrame
            columns: Columns to process
            strategy: Strategy to apply
            asset_type: Asset type for context
            
        Returns:
            DataFrame with strategy applied
        """
        df_result = df.copy()
        
        if strategy == 'forward_fill':
            # Forward fill by symbol if available, otherwise globally
            if 'symbol' in df.columns:
                for col in columns:
                    df_result[col] = df_result.groupby('symbol')[col].ffill()
            else:
                df_result[columns] = df_result[columns].ffill()
        
        elif strategy == 'backward_fill':
            if 'symbol' in df.columns:
                for col in columns:
                    df_result[col] = df_result.groupby('symbol')[col].bfill()
            else:
                df_result[columns] = df_result[columns].bfill()
        
        elif strategy == 'zero_fill':
            df_result[columns] = df_result[columns].fillna(0)
        
        elif strategy == 'mean_fill':
            if 'symbol' in df.columns:
                for col in columns:
                    df_result[col] = df_result.groupby('symbol')[col].transform(
                        lambda x: x.fillna(x.mean())
                    )
            else:
                df_result[columns] = df_result[columns].fillna(df_result[columns].mean())
        
        elif strategy == 'median_fill':
            if 'symbol' in df.columns:
                for col in columns:
                    df_result[col] = df_result.groupby('symbol')[col].transform(
                        lambda x: x.fillna(x.median())
                    )
            else:
                df_result[columns] = df_result[columns].fillna(df_result[columns].median())
        
        elif strategy == 'interpolate':
            if 'symbol' in df.columns:
                for col in columns:
                    df_result[col] = df_result.groupby('symbol')[col].transform(
                        lambda x: x.interpolate(method='linear')
                    )
            else:
                df_result[columns] = df_result[columns].interpolate(method='linear')
        
        elif strategy == 'neutral_fill':
            # For sentiment features, fill with neutral values
            for col in columns:
                if 'sentiment' in col.lower() or 'score' in col.lower():
                    df_result[col] = df_result[col].fillna(0.0)  # Neutral sentiment
                elif any(term in col.lower() for term in ['pos', 'neg', 'neu']):
                    df_result[col] = df_result[col].fillna(0.33)  # Equal probability
                else:
                    df_result[col] = df_result[col].fillna(df_result[col].median())
        
        elif strategy == 'knn_impute':
            # Use KNN imputation for more sophisticated filling
            imputer = KNNImputer(n_neighbors=5)
            df_result[columns] = imputer.fit_transform(df_result[columns])
        
        elif strategy == 'iterative_impute':
            # Use iterative imputation
            imputer = IterativeImputer(random_state=42, max_iter=10)
            df_result[columns] = imputer.fit_transform(df_result[columns])
        
        else:
            logger.warning(f"Unknown strategy: {strategy}, using forward_fill")
            df_result[columns] = df_result[columns].ffill()
        
        return df_result
    
    def _validate_and_cleanup(self, df: pd.DataFrame, original_analysis: Dict) -> pd.DataFrame:
        """
        Validate results and perform final cleanup.
        
        Args:
            df: DataFrame after imputation
            original_analysis: Original missing data analysis
            
        Returns:
            Validated and cleaned DataFrame
        """
        df_clean = df.copy()
        
        # Check for remaining missing values
        remaining_missing = df_clean.isnull().sum()
        problematic_columns = remaining_missing[remaining_missing > 0]
        
        if not problematic_columns.empty:
            logger.warning(f"Remaining missing values in columns: {problematic_columns.to_dict()}")
            
            # Apply final fallback strategy
            for col in problematic_columns.index:
                if df_clean[col].dtype in [np.float64, np.int64]:
                    # Use median for numeric columns
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                    
                    # If still missing (all values were NaN), fill with 0
                    if df_clean[col].isnull().all():
                        df_clean[col] = df_clean[col].fillna(0)
        
        # Remove rows with too many consecutive missing values (data quality issues)
        if 'symbol' in df_clean.columns:
            rows_to_remove = []
            for symbol in df_clean['symbol'].unique():
                symbol_mask = df_clean['symbol'] == symbol
                symbol_data = df_clean[symbol_mask]
                
                # Check for rows with excessive missing values
                missing_ratio = symbol_data.isnull().sum(axis=1) / len(symbol_data.columns)
                problematic_rows = missing_ratio > 0.5  # More than 50% missing
                
                if problematic_rows.any():
                    rows_to_remove.extend(symbol_data[problematic_rows].index)
            
            if rows_to_remove:
                logger.info(f"Removing {len(rows_to_remove)} rows with excessive missing values")
                df_clean = df_clean.drop(rows_to_remove)
        
        return df_clean
    
    def get_imputation_summary(self, df_before: pd.DataFrame, df_after: pd.DataFrame) -> Dict:
        """
        Generate summary of imputation results.
        
        Args:
            df_before: DataFrame before imputation
            df_after: DataFrame after imputation
            
        Returns:
            Summary dictionary
        """
        before_missing = df_before.isnull().sum().sum()
        after_missing = df_after.isnull().sum().sum()
        
        summary = {
            'missing_before': int(before_missing),
            'missing_after': int(after_missing),
            'missing_reduced': int(before_missing - after_missing),
            'reduction_percentage': round(((before_missing - after_missing) / before_missing) * 100, 2) if before_missing > 0 else 0,
            'strategies_used': {
                'price': self.price_strategy,
                'volume': self.volume_strategy,
                'indicators': self.indicator_strategy,
                'news_sentiment': self.news_strategy
            },
            'feature_categories': {cat: len(cols) for cat, cols in self.feature_categories.items()},
            'shape_before': df_before.shape,
            'shape_after': df_after.shape
        }
        
        # Column-level changes
        before_by_col = df_before.isnull().sum()
        after_by_col = df_after.isnull().sum()
        
        summary['column_improvements'] = {}
        for col in before_by_col.index:
            if before_by_col[col] > after_by_col[col]:
                summary['column_improvements'][col] = {
                    'before': int(before_by_col[col]),
                    'after': int(after_by_col[col]),
                    'improved': int(before_by_col[col] - after_by_col[col])
                }
        
        return summary
