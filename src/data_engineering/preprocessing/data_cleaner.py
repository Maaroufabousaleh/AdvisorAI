"""
Data Cleaner Module

Handles data cleaning, validation, and quality assurance for financial datasets.
Processes existing stock and crypto feature files.
"""

import pandas as pd
import numpy as np
import json
import logging
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Main data cleaning class for financial feature datasets.
    Handles stocks and crypto feature files with proper validation and cleaning.
    """
    
    def __init__(self, data_path: str = "data/features"):
        """
        Initialize DataCleaner with path to feature data.
        
        Args:
            data_path: Path to directory containing feature files
        """
        self.data_path = Path(data_path)
        self.stocks_file = self.data_path / "stocks_features.parquet"
        self.crypto_file = self.data_path / "crypto_features.parquet"
        self.stocks_report = self.data_path / "stocks_report.json"
        self.crypto_report = self.data_path / "crypto_report.json"
        
        # Load metadata reports
        self.stocks_metadata = self._load_report(self.stocks_report)
        self.crypto_metadata = self._load_report(self.crypto_report)
    
    def _load_report(self, report_path: Path) -> Dict:
        """Load JSON report file."""
        try:
            with open(report_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load report {report_path}: {e}")
            return {}
    
    def load_stocks_data(self) -> pd.DataFrame:
        """
        Load stocks feature data with basic cleaning.
        
        Returns:
            Cleaned stocks DataFrame
        """
        logger.info("Loading stocks feature data...")
        df = pd.read_parquet(self.stocks_file)
        
        # Basic info logging
        logger.info(f"Loaded stocks data: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Clean the data
        df_cleaned = self._clean_dataframe(df, asset_type="stocks")
        
        logger.info(f"Cleaned stocks data: {df_cleaned.shape[0]} rows, {df_cleaned.shape[1]} columns")
        return df_cleaned
    
    def load_crypto_data(self) -> pd.DataFrame:
        """
        Load crypto feature data with basic cleaning.
        
        Returns:
            Cleaned crypto DataFrame
        """
        logger.info("Loading crypto feature data...")
        df = pd.read_parquet(self.crypto_file)
        
        # Basic info logging
        logger.info(f"Loaded crypto data: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Clean the data
        df_cleaned = self._clean_dataframe(df, asset_type="crypto")
        
        logger.info(f"Cleaned crypto data: {df_cleaned.shape[0]} rows, {df_cleaned.shape[1]} columns")
        return df_cleaned
    
    def _clean_dataframe(self, df: pd.DataFrame, asset_type: str) -> pd.DataFrame:
        """
        Apply comprehensive cleaning to DataFrame.
        
        Args:
            df: Input DataFrame
            asset_type: Type of asset ('stocks' or 'crypto')
            
        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()
        
        # 1. Handle timestamp columns
        df_clean = self._standardize_timestamps(df_clean)
        
        # 2. Fix unhashable columns before duplicate removal
        df_clean = self._fix_unhashable_columns(df_clean)
        
        # 3. Remove duplicates
        initial_shape = df_clean.shape[0]
        df_clean = df_clean.drop_duplicates()
        duplicates_removed = initial_shape - df_clean.shape[0]
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate rows")
        
        # 3. Handle missing values
        df_clean = self._handle_missing_values(df_clean, asset_type)
        
        # 4. Remove invalid values
        df_clean = self._remove_invalid_values(df_clean)
        
        # 5. Ensure data consistency
        df_clean = self._ensure_data_consistency(df_clean, asset_type)
        
        return df_clean
    
    def _standardize_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize timestamp columns and set proper index.
        More lenient approach for feature data.
        """
        df_clean = df.copy()
        
        # Look for actual timestamp columns (be more specific to avoid feature columns)
        # Only target columns that are clearly timestamps, not feature data
        timestamp_cols = []
        for col in df.columns:
            col_lower = col.lower()
            # Only include columns that are clearly timestamp-related
            if (col_lower in ['timestamp', 'time', 'date', 'datetime'] or 
                col_lower.endswith('_timestamp') or 
                col_lower.endswith('_time') or 
                col_lower.startswith('timestamp_') or
                col_lower == 'interval_timestamp'):
                timestamp_cols.append(col)
        
        for col in timestamp_cols:
            if col in df_clean.columns:
                try:
                    # Convert to datetime if not already
                    if not pd.api.types.is_datetime64_any_dtype(df_clean[col]):
                        # Handle both Unix timestamps and datetime strings
                        if pd.api.types.is_numeric_dtype(df_clean[col]):
                            # Assume Unix timestamp (milliseconds)
                            df_clean[col] = pd.to_datetime(df_clean[col], unit='ms', errors='coerce')
                        else:
                            df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
                    
                    # For feature data, only remove rows if ALL timestamp columns are invalid
                    # Count invalid timestamps but don't drop yet
                    invalid_count = df_clean[col].isna().sum()
                    if invalid_count > 0:
                        logger.info(f"Found {invalid_count} invalid timestamps in {col} (will keep rows with valid data in other columns)")
                        
                except Exception as e:
                    logger.warning(f"Error processing timestamp column {col}: {e}")
        
        # Only remove rows where ALL timestamp columns are NaT (if any exist)
        if timestamp_cols:
            timestamp_df = df_clean[timestamp_cols]
            # Remove rows only if ALL timestamp columns are invalid
            all_invalid_mask = timestamp_df.isna().all(axis=1)
            rows_to_remove = all_invalid_mask.sum()
            
            if rows_to_remove > 0:
                df_clean = df_clean[~all_invalid_mask]
                logger.info(f"Removed {rows_to_remove} rows where ALL timestamp columns were invalid")
        
        return df_clean
    
    def _handle_missing_values(self, df: pd.DataFrame, asset_type: str) -> pd.DataFrame:
        """
        Handle missing values based on column type and asset type.
        """
        df_clean = df.copy()
        
        # Get numeric columns only
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        
        # Define strategies for different column types
        price_columns = [col for col in numeric_columns if any(price_term in col.lower() 
                        for price_term in ['price', 'close', 'open', 'high', 'low'])]
        
        volume_columns = [col for col in numeric_columns if any(vol_term in col.lower() 
                         for vol_term in ['volume', 'vol', 'amount'])]
        
        indicator_columns = [col for col in numeric_columns if any(ind_term in col.lower() 
                           for ind_term in ['rsi', 'macd', 'ema', 'sma', 'bollinger', 'atr'])]
        
        # Forward fill price columns (preserve last known price)
        for col in price_columns:
            if col in df_clean.columns:
                if 'symbol' in df_clean.columns:
                    df_clean[col] = df_clean.groupby('symbol')[col].ffill()
                else:
                    df_clean[col] = df_clean[col].ffill()
        
        # Fill volume columns with 0 (no trading volume)
        for col in volume_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(0)
        
        # Interpolate technical indicators
        for col in indicator_columns:
            if col in df_clean.columns:
                if 'symbol' in df_clean.columns:
                    # Use transform instead of apply to maintain index alignment
                    df_clean[col] = df_clean.groupby('symbol')[col].transform(lambda x: x.interpolate())
                else:
                    df_clean[col] = df_clean[col].interpolate()
        
        # For remaining numeric columns, use forward fill then backward fill
        remaining_numeric = [col for col in numeric_columns 
                           if col not in price_columns + volume_columns + indicator_columns]
        
        for col in remaining_numeric:
            if col in df_clean.columns:
                if 'symbol' in df_clean.columns:
                    df_clean[col] = df_clean.groupby('symbol')[col].ffill().bfill()
                else:
                    df_clean[col] = df_clean[col].ffill().bfill()
        
        # Log missing value statistics
        missing_after = df_clean.isnull().sum().sum()
        if missing_after > 0:
            logger.info(f"Remaining missing values after cleaning: {missing_after}")
            missing_by_column = df_clean.isnull().sum()
            missing_by_column = missing_by_column[missing_by_column > 0]
            for col, count in missing_by_column.items():
                logger.info(f"  {col}: {count} missing values")
        
        return df_clean
    
    def _remove_invalid_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove obviously invalid values (infinite, extremely large, etc.).
        Be conservative - only remove clear data errors, not legitimate extreme values.
        """
        df_clean = df.copy()
        
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        
        # Only apply extreme value removal to specific problematic columns
        # where extreme values are clearly data errors
        problematic_cols = [
            'santiment_financial_price',  # Only if > $1M per coin (clearly wrong)
            'santiment_financial_marketcap'  # Only if > $100T (clearly wrong)
        ]
        
        for col in numeric_columns:
            if col in df_clean.columns:
                # Always remove infinite values
                before_count = len(df_clean)
                df_clean = df_clean[~np.isinf(df_clean[col])]
                after_count = len(df_clean)
                
                if before_count != after_count:
                    logger.info(f"Removed {before_count - after_count} rows with infinite values in {col}")
                
                # Only apply extreme value removal to clearly problematic columns
                # with very conservative, absolute thresholds
                if col in problematic_cols and len(df_clean) > 0:
                    before_count = len(df_clean)
                    
                    if col == 'santiment_financial_price':
                        # Remove prices > $1M per coin (clearly data errors)
                        # BUT preserve NaN values (they'll be handled in missing data step)
                        extreme_mask = (df_clean[col] > 1_000_000) & df_clean[col].notna()
                        df_clean = df_clean[~extreme_mask]
                    elif col == 'santiment_financial_marketcap':
                        # Remove market caps > $100 trillion (clearly data errors)
                        # BUT preserve NaN values (they'll be handled in missing data step)
                        extreme_mask = (df_clean[col] > 100_000_000_000_000) & df_clean[col].notna()
                        df_clean = df_clean[~extreme_mask]
                    
                    after_count = len(df_clean)
                    if before_count != after_count:
                        logger.info(f"Removed {before_count - after_count} rows with extreme values in {col}")
        
        return df_clean
    
    def _ensure_data_consistency(self, df: pd.DataFrame, asset_type: str) -> pd.DataFrame:
        """
        Ensure data consistency (e.g., high >= low, etc.).
        """
        df_clean = df.copy()
        
        # Check for OHLC consistency
        ohlc_columns = {
            'open': [col for col in df_clean.columns if 'open' in col.lower()],
            'high': [col for col in df_clean.columns if 'high' in col.lower()],
            'low': [col for col in df_clean.columns if 'low' in col.lower()],
            'close': [col for col in df_clean.columns if 'close' in col.lower()]
        }
        
        # For each set of OHLC columns, ensure consistency
        for i in range(max(len(cols) for cols in ohlc_columns.values())):
            try:
                open_col = ohlc_columns['open'][i] if i < len(ohlc_columns['open']) else None
                high_col = ohlc_columns['high'][i] if i < len(ohlc_columns['high']) else None
                low_col = ohlc_columns['low'][i] if i < len(ohlc_columns['low']) else None
                close_col = ohlc_columns['close'][i] if i < len(ohlc_columns['close']) else None
                
                if high_col and low_col:
                    # Ensure high >= low
                    invalid_mask = df_clean[high_col] < df_clean[low_col]
                    invalid_count = invalid_mask.sum()
                    if invalid_count > 0:
                        logger.warning(f"Found {invalid_count} rows where high < low, removing...")
                        df_clean = df_clean[~invalid_mask]
                
                # Add more consistency checks as needed
                
            except (IndexError, KeyError):
                continue
        
        return df_clean
    
    def get_data_summary(self, df: pd.DataFrame, asset_type: str) -> Dict:
        """
        Get comprehensive summary of cleaned data.
        
        Args:
            df: Cleaned DataFrame
            asset_type: Type of asset
            
        Returns:
            Summary dictionary
        """
        summary = {
            'asset_type': asset_type,
            'shape': df.shape,
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'dtypes': {str(k): v for k, v in df.dtypes.value_counts().to_dict().items()},
            'missing_values': df.isnull().sum().sum(),
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(df.select_dtypes(include=['object', 'category']).columns),
        }
        
        # Time range if timestamp column exists
        timestamp_cols = [col for col in df.columns if 'timestamp' in col.lower()]
        if timestamp_cols and len(timestamp_cols) > 0:
            timestamp_col = timestamp_cols[0]
            if pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
                summary['time_range'] = {
                    'start': df[timestamp_col].min().isoformat(),
                    'end': df[timestamp_col].max().isoformat(),
                    'duration_days': (df[timestamp_col].max() - df[timestamp_col].min()).days
                }
        
        # Asset information if symbol column exists
        if 'symbol' in df.columns:
            summary['assets'] = {
                'count': df['symbol'].nunique(),
                'symbols': sorted(df['symbol'].unique().tolist())
            }
        
        return summary
    
    def _fix_unhashable_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fix columns that contain unhashable types (numpy arrays, lists, etc.).
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with hashable columns
        """
        df_fixed = df.copy()
        
        for col in df_fixed.columns:
            try:
                # Test if column is hashable by trying to get unique values
                _ = df_fixed[col].nunique()
            except (TypeError, ValueError) as e:
                if "unhashable" in str(e).lower() or "numpy.ndarray" in str(e):
                    logger.warning(f"Fixing unhashable column: {col}")
                    
                    # Convert arrays/lists to strings or extract first element
                    def fix_unhashable_value(val):
                        if isinstance(val, (list, np.ndarray)):
                            if len(val) == 0:
                                return None
                            elif len(val) == 1:
                                return val[0]
                            else:
                                # Convert to string representation
                                return str(val)
                        return val
                    
                    df_fixed[col] = df_fixed[col].apply(fix_unhashable_value)
                    
                    # If still problematic, convert to string
                    try:
                        _ = df_fixed[col].nunique()
                    except:
                        logger.warning(f"Converting {col} to string type")
                        df_fixed[col] = df_fixed[col].astype(str)
        
        return df_fixed
