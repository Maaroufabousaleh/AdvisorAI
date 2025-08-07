"""
Comprehensive Data Processing Pipeline

Main orchestrator for data cleaning, validation, normalization, and preprocessing
of financial datasets. Handles both stocks and crypto data with appropriate
strategies for each asset type.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import json
from datetime import datetime

from .data_cleaner import DataCleaner
from .data_validator import DataValidator
from .missing_data_handler import MissingDataHandler
from .outlier_detection import OutlierDetector
from .data_normalizer import DataNormalizer

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Main data processing orchestrator that coordinates all preprocessing steps.
    Provides a unified interface for processing stocks and crypto data.
    """
    
    def __init__(self, 
                 data_path: str = "data/features",
                 output_path: str = "data/processed",
                 config: Dict = None):
        """
        Initialize the data processor.
        
        Args:
            data_path: Path to input feature data
            output_path: Path for processed output
            config: Configuration dictionary
        """
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Default configuration
        default_config = {
            'validation': {
                'strict_mode': False,
                'save_reports': True
            },
            'cleaning': {
                'remove_duplicates': True,
                'handle_missing': True,
                'detect_outliers': True
            },
            'normalization': {
                'stocks': {
                    'price_scaling': 'minmax',
                    'volume_scaling': 'log_standard',
                    'indicator_scaling': 'standard'
                },
                'crypto': {
                    'price_scaling': 'robust',
                    'volume_scaling': 'log_standard', 
                    'indicator_scaling': 'robust'
                }
            },
            'outlier_handling': {
                'method': 'cap',  # 'cap', 'remove', 'transform'
                'preserve_extreme_moves': True
            },
            'missing_data': {
                'price_strategy': 'forward_fill',
                'volume_strategy': 'zero_fill',
                'indicator_strategy': 'interpolate'
            }
        }
        
        self.config = {**default_config, **(config or {})}
        
        # Initialize components
        self.cleaner = DataCleaner(str(self.data_path))
        self.validator = DataValidator()
        self.missing_handler = MissingDataHandler(**self.config['missing_data'])
        self.outlier_detector = OutlierDetector(**self._get_outlier_config())
        self.normalizer = DataNormalizer(self.config['normalization'])
        
        # Processing history
        self.processing_history = {}
    
    def _get_outlier_config(self) -> Dict:
        """Extract outlier detection configuration."""
        return {
            'preserve_extreme_moves': self.config['outlier_handling']['preserve_extreme_moves']
        }
    
    def process_all_data(self, save_intermediate: bool = True) -> Dict:
        """
        Process both stocks and crypto data through the complete pipeline.
        
        Args:
            save_intermediate: Whether to save intermediate results
            
        Returns:
            Dictionary with processing results and final datasets
        """
        logger.info("Starting comprehensive data processing pipeline...")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'stocks': {},
            'crypto': {},
            'summary': {}
        }
        
        # Process stocks data
        logger.info("Processing stocks data...")
        stocks_results = self.process_stocks_data(save_intermediate)
        results['stocks'] = stocks_results
        
        # Process crypto data
        logger.info("Processing crypto data...")
        crypto_results = self.process_crypto_data(save_intermediate)
        results['crypto'] = crypto_results
        
        # Generate overall summary
        results['summary'] = self._generate_overall_summary(stocks_results, crypto_results)
        
        # Save complete processing report
        if save_intermediate:
            self._save_processing_report(results)
        
        logger.info("Data processing pipeline completed successfully!")
        return results
    
    def process_stocks_data(self, save_intermediate: bool = True) -> Dict:
        """
        Process stocks data through the complete pipeline.
        
        Args:
            save_intermediate: Whether to save intermediate results
            
        Returns:
            Processing results and final dataset
        """
        logger.info("=== Processing Stocks Data ===")
        
        results = {
            'asset_type': 'stocks',
            'steps': {},
            'final_data': None,
            'metrics': {}
        }
        
        try:
            # Step 1: Load raw data
            logger.info("Step 1: Loading stocks data...")
            stocks_df = self.cleaner.load_stocks_data()
            results['steps']['loading'] = {
                'status': 'success',
                'shape_after': stocks_df.shape,
                'summary': self.cleaner.get_data_summary(stocks_df, 'stocks')
            }
            
            if save_intermediate:
                self._save_dataframe(stocks_df, 'stocks_01_loaded.parquet')
            
            # Step 2: Initial validation
            logger.info("Step 2: Validating stocks data...")
            validation_results = self.validator.validate_dataset(stocks_df, 'stocks', 'stocks_raw')
            results['steps']['validation'] = validation_results
            
            if validation_results['overall_status'] == 'FAIL' and self.config['validation']['strict_mode']:
                raise ValueError("Data validation failed in strict mode")
            
            # Step 3: Handle missing data
            logger.info("Step 3: Handling missing data...")
            missing_before = stocks_df.isnull().sum().sum()
            stocks_df = self.missing_handler.handle_missing_data(stocks_df, 'stocks')
            missing_after = stocks_df.isnull().sum().sum()
            
            results['steps']['missing_data'] = {
                'status': 'success',
                'missing_before': int(missing_before),
                'missing_after': int(missing_after),
                'shape_after': stocks_df.shape
            }
            
            if save_intermediate:
                self._save_dataframe(stocks_df, 'stocks_02_missing_handled.parquet')
            
            # Step 4: Outlier detection and handling
            logger.info("Step 4: Detecting and handling outliers...")
            outlier_results = self.outlier_detector.detect_outliers(stocks_df, 'stocks')
            stocks_df = self.outlier_detector.handle_outliers(
                stocks_df, outlier_results, self.config['outlier_handling']['method']
            )
            
            results['steps']['outliers'] = {
                'status': 'success',
                'outliers_detected': outlier_results['total_outliers'],
                'treatment_method': self.config['outlier_handling']['method'],
                'shape_after': stocks_df.shape
            }
            
            if save_intermediate:
                self._save_dataframe(stocks_df, 'stocks_03_outliers_handled.parquet')
            
            # Step 5: Normalization
            logger.info("Step 5: Normalizing stocks data...")
            self.normalizer.fit_stocks(stocks_df)
            stocks_normalized = self.normalizer.transform_stocks(stocks_df)
            
            results['steps']['normalization'] = {
                'status': 'success',
                'normalization_info': self.normalizer.normalizers['stocks'].get_feature_info(),
                'shape_after': stocks_normalized.shape
            }
            
            if save_intermediate:
                self._save_dataframe(stocks_normalized, 'stocks_04_normalized_raw.parquet')
            
            # Step 6: Apply data quality fixes
            logger.info("Step 6: Applying data quality fixes...")
            stocks_normalized = self._apply_data_quality_fixes(stocks_normalized)
            
            if save_intermediate:
                self._save_dataframe(stocks_normalized, 'stocks_04_normalized.parquet')
            
            # Step 7: Final validation
            logger.info("Step 7: Final validation...")
            final_validation = self.validator.validate_dataset(stocks_normalized, 'stocks', 'stocks_final')
            results['steps']['final_validation'] = final_validation
            
            # Set final data
            results['final_data'] = stocks_normalized
            
            # Generate metrics
            results['metrics'] = self._generate_processing_metrics(
                original_shape=results['steps']['loading']['shape_after'],
                final_shape=stocks_normalized.shape,
                missing_reduced=results['steps']['missing_data']['missing_before'] - results['steps']['missing_data']['missing_after'],
                outliers_handled=results['steps']['outliers']['outliers_detected']
            )
            
            logger.info(f"Stocks processing completed. Final shape: {stocks_normalized.shape}")
            
        except Exception as e:
            logger.error(f"Error processing stocks data: {e}")
            results['steps']['error'] = {'status': 'failed', 'error': str(e)}
            raise
        
        return results
    
    def process_crypto_data(self, save_intermediate: bool = True) -> Dict:
        """
        Process crypto data through the complete pipeline.
        
        Args:
            save_intermediate: Whether to save intermediate results
            
        Returns:
            Processing results and final dataset
        """
        logger.info("=== Processing Crypto Data ===")
        
        results = {
            'asset_type': 'crypto',
            'steps': {},
            'final_data': None,
            'metrics': {}
        }
        
        try:
            # Step 1: Load raw data
            logger.info("Step 1: Loading crypto data...")
            crypto_df = self.cleaner.load_crypto_data()
            results['steps']['loading'] = {
                'status': 'success',
                'shape_after': crypto_df.shape,
                'summary': self.cleaner.get_data_summary(crypto_df, 'crypto')
            }
            
            if save_intermediate:
                self._save_dataframe(crypto_df, 'crypto_01_loaded.parquet')
            
            # Step 2: Initial validation
            logger.info("Step 2: Validating crypto data...")
            validation_results = self.validator.validate_dataset(crypto_df, 'crypto', 'crypto_raw')
            results['steps']['validation'] = validation_results
            
            if validation_results['overall_status'] == 'FAIL' and self.config['validation']['strict_mode']:
                raise ValueError("Data validation failed in strict mode")
            
            # Step 3: Handle missing data
            logger.info("Step 3: Handling missing data...")
            missing_before = crypto_df.isnull().sum().sum()
            crypto_df = self.missing_handler.handle_missing_data(crypto_df, 'crypto')
            missing_after = crypto_df.isnull().sum().sum()
            
            results['steps']['missing_data'] = {
                'status': 'success',
                'missing_before': int(missing_before),
                'missing_after': int(missing_after),
                'shape_after': crypto_df.shape
            }
            
            if save_intermediate:
                self._save_dataframe(crypto_df, 'crypto_02_missing_handled.parquet')
            
            # Step 4: Outlier detection and handling
            logger.info("Step 4: Detecting and handling outliers...")
            outlier_results = self.outlier_detector.detect_outliers(crypto_df, 'crypto')
            crypto_df = self.outlier_detector.handle_outliers(
                crypto_df, outlier_results, self.config['outlier_handling']['method']
            )
            
            results['steps']['outliers'] = {
                'status': 'success',
                'outliers_detected': outlier_results['total_outliers'],
                'treatment_method': self.config['outlier_handling']['method'],
                'shape_after': crypto_df.shape
            }
            
            if save_intermediate:
                self._save_dataframe(crypto_df, 'crypto_03_outliers_handled.parquet')
            
            # Step 5: Normalization
            logger.info("Step 5: Normalizing crypto data...")
            self.normalizer.fit_crypto(crypto_df)
            crypto_normalized = self.normalizer.transform_crypto(crypto_df)
            
            results['steps']['normalization'] = {
                'status': 'success',
                'normalization_info': self.normalizer.normalizers['crypto'].get_feature_info(),
                'shape_after': crypto_normalized.shape
            }
            
            if save_intermediate:
                self._save_dataframe(crypto_normalized, 'crypto_04_normalized_raw.parquet')
            
            # Step 6: Apply data quality fixes
            logger.info("Step 6: Applying data quality fixes...")
            crypto_normalized = self._apply_data_quality_fixes(crypto_normalized)
            
            if save_intermediate:
                self._save_dataframe(crypto_normalized, 'crypto_04_normalized.parquet')
            
            # Step 7: Final validation
            logger.info("Step 7: Final validation...")
            final_validation = self.validator.validate_dataset(crypto_normalized, 'crypto', 'crypto_final')
            results['steps']['final_validation'] = final_validation
            
            # Set final data
            results['final_data'] = crypto_normalized
            
            # Generate metrics
            results['metrics'] = self._generate_processing_metrics(
                original_shape=results['steps']['loading']['shape_after'],
                final_shape=crypto_normalized.shape,
                missing_reduced=results['steps']['missing_data']['missing_before'] - results['steps']['missing_data']['missing_after'],
                outliers_handled=results['steps']['outliers']['outliers_detected']
            )
            
            logger.info(f"Crypto processing completed. Final shape: {crypto_normalized.shape}")
            
        except Exception as e:
            logger.error(f"Error processing crypto data: {e}")
            results['steps']['error'] = {'status': 'failed', 'error': str(e)}
            raise
        
        return results
    
    def _save_dataframe(self, df: pd.DataFrame, filename: str):
        """Save DataFrame to the output path with parquet compatibility fixes."""
        # Final check for any remaining object columns with problematic types
        df_safe = df.copy()
        
        # Ensure metadata is JSON-serializable before saving
        if hasattr(df_safe, 'attrs') and df_safe.attrs:
            # Create a JSON-serializable version of attrs
            safe_attrs = {}
            for key, value in df_safe.attrs.items():
                try:
                    # Test if the value can be JSON serialized
                    json.dumps(value)
                    safe_attrs[key] = value
                except (TypeError, ValueError):
                    # Convert non-serializable values to strings
                    if isinstance(value, dict):
                        safe_value = {}
                        for k, v in value.items():
                            try:
                                json.dumps(v)
                                safe_value[k] = v
                            except (TypeError, ValueError):
                                if hasattr(v, 'isoformat'):  # DateTime objects
                                    safe_value[k] = v.isoformat()
                                else:
                                    safe_value[k] = str(v)
                        safe_attrs[key] = safe_value
                    else:
                        safe_attrs[key] = str(value)
            df_safe.attrs = safe_attrs
        
        # Convert any remaining object columns to string
        for col in df_safe.columns:
            if df_safe[col].dtype == 'object':
                try:
                    df_safe[col] = df_safe[col].astype(str)
                    logger.debug(f"Final conversion of {col} to string for parquet save")
                except:
                    df_safe[col] = df_safe[col].apply(lambda x: str(x) if pd.notna(x) else 'unknown')
                    logger.debug(f"Force conversion of {col} to string for parquet save")
        
        filepath = self.output_path / filename
        df_safe.to_parquet(filepath)
        logger.debug(f"Saved intermediate result: {filepath}")
    
    def _generate_processing_metrics(self, original_shape: Tuple, final_shape: Tuple,
                                   missing_reduced: int, outliers_handled: int) -> Dict:
        """Generate processing metrics."""
        return {
            'data_reduction': {
                'rows_original': original_shape[0],
                'rows_final': final_shape[0],
                'rows_removed': original_shape[0] - final_shape[0],
                'reduction_percentage': ((original_shape[0] - final_shape[0]) / original_shape[0] * 100) if original_shape[0] > 0 else 0
            },
            'data_quality_improvements': {
                'missing_values_handled': missing_reduced,
                'outliers_handled': outliers_handled
            },
            'feature_info': {
                'features_original': original_shape[1],
                'features_final': final_shape[1],
                'features_added': final_shape[1] - original_shape[1]
            }
        }
    
    def _generate_overall_summary(self, stocks_results: Dict, crypto_results: Dict) -> Dict:
        """Generate overall processing summary."""
        summary = {
            'processing_status': 'completed',
            'datasets_processed': 2,
            'total_data_points': {
                'stocks': stocks_results['final_data'].shape[0] if stocks_results.get('final_data') is not None else 0,
                'crypto': crypto_results['final_data'].shape[0] if crypto_results.get('final_data') is not None else 0
            },
            'data_quality_summary': {
                'stocks_validation': stocks_results['steps'].get('final_validation', {}).get('overall_status', 'unknown'),
                'crypto_validation': crypto_results['steps'].get('final_validation', {}).get('overall_status', 'unknown')
            },
            'normalization_summary': {
                'stocks_normalizer_fitted': 'stocks' in self.normalizer.normalizers,
                'crypto_normalizer_fitted': 'crypto' in self.normalizer.normalizers
            }
        }
        
        # Calculate combined metrics
        total_original_rows = (
            stocks_results.get('metrics', {}).get('data_reduction', {}).get('rows_original', 0) +
            crypto_results.get('metrics', {}).get('data_reduction', {}).get('rows_original', 0)
        )
        
        total_final_rows = (
            stocks_results.get('metrics', {}).get('data_reduction', {}).get('rows_final', 0) +
            crypto_results.get('metrics', {}).get('data_reduction', {}).get('rows_final', 0)
        )
        
        summary['combined_metrics'] = {
            'total_original_rows': total_original_rows,
            'total_final_rows': total_final_rows,
            'overall_data_retention': (total_final_rows / total_original_rows * 100) if total_original_rows > 0 else 0
        }
        
        return summary
    
    def _save_processing_report(self, results: Dict):
        """Save comprehensive processing report."""
        # Remove the actual DataFrames to make the report JSON serializable
        report = {}
        for key, value in results.items():
            if key in ['stocks', 'crypto']:
                asset_report = value.copy()
                if 'final_data' in asset_report:
                    # Replace DataFrame with summary info
                    df = asset_report['final_data']
                    asset_report['final_data_info'] = {
                        'shape': df.shape,
                        'columns': df.columns.tolist(),
                        'dtypes': {str(col): str(dtype) for col, dtype in df.dtypes.items()},
                        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
                    }
                    del asset_report['final_data']
                report[key] = asset_report
            else:
                report[key] = value
        
        # Save report
        report_path = self.output_path / 'processing_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Processing report saved to {report_path}")
        
        # Save normalization summary
        norm_summary = self.normalizer.get_normalization_summary()
        norm_path = self.output_path / 'normalization_summary.json'
        with open(norm_path, 'w') as f:
            json.dump(norm_summary, f, indent=2, default=str)
        
        # Save validation reports
        if self.config['validation']['save_reports']:
            validation_path = self.output_path / 'validation_reports.json'
            self.validator.save_validation_report(str(validation_path))
    
    def save_final_datasets(self):
        """Save the final processed datasets."""
        if 'stocks' in self.normalizer.normalizers:
            # We need to get the final stocks data - this would typically be stored
            logger.info("Final datasets should be saved through the complete processing pipeline")
            
        # Save normalizers for future use
        normalizer_path = self.output_path / 'normalizers'
        self.normalizer.save_normalizers(str(normalizer_path))
        logger.info(f"Normalizers saved to {normalizer_path}")
    
    def load_processed_data(self, asset_type: str) -> pd.DataFrame:
        """
        Load previously processed data.
        
        Args:
            asset_type: 'stocks' or 'crypto'
            
        Returns:
            Processed DataFrame
        """
        filename = f"{asset_type}_04_normalized.parquet"
        filepath = self.output_path / filename
        
        if filepath.exists():
            logger.info(f"Loading processed {asset_type} data from {filepath}")
            return pd.read_parquet(filepath)
        else:
            raise FileNotFoundError(f"Processed {asset_type} data not found at {filepath}")
    
    def get_processing_summary(self) -> Dict:
        """Get summary of all processing activities."""
        summary = {
            'processor_config': self.config,
            'available_datasets': [],
            'normalizers_fitted': list(self.normalizer.normalizers.keys()),
            'validation_summary': self.validator.get_validation_summary()
        }
        
        # Check for available processed datasets
        for asset_type in ['stocks', 'crypto']:
            filename = f"{asset_type}_04_normalized.parquet"
            filepath = self.output_path / filename
            if filepath.exists():
                summary['available_datasets'].append(asset_type)
        
        return summary
    
    def _apply_data_quality_fixes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply comprehensive data quality fixes to normalized data based on validation issues.
        Enhanced version addressing specific validation report findings.
        """
        logger.info("Applying enhanced data quality fixes...")
        
        # 1. Add missing 'close' column if needed (critical for crypto)
        if 'close' not in df.columns:
            close_candidates = [
                'price',
                'santiment_ohlcv_closePriceUsd',
                'exchangePrices.coinbase',
                'exchangePrices.binance',
                'santiment_prices_priceUsd'
            ]
            
            for candidate in close_candidates:
                if candidate in df.columns and df[candidate].notna().sum() > 0:
                    df['close'] = df[candidate].copy()
                    logger.info(f"Added 'close' column using {candidate}")
                    break
            else:
                if 'open' in df.columns:
                    df['close'] = df['open']
                    logger.info("Added 'close' column using 'open' as proxy")
        
        # 2. Remove completely empty columns (found in validation report)
        empty_cols = df.columns[df.isnull().all()].tolist()
        if empty_cols:
            df = df.drop(columns=empty_cols)
            logger.info(f"Removed {len(empty_cols)} completely empty columns: {empty_cols}")
        
        # 3. Fix categorical null values
        categorical_cols = [
            'santiment_exchange_alternative_slug_used',
            'santiment_network_alternative_slug_used', 
            'santiment_financial_alternative_slug_used'
        ]
        
        for col in categorical_cols:
            if col in df.columns:
                null_count = df[col].isnull().sum()
                if null_count > 0:
                    df[col] = df[col].fillna('not_available')
                    logger.info(f"Filled {null_count} null values in {col}")
        
        # 4. Fix RSI values to be within 0-100 range (critical issue from validation)
        rsi_cols = [col for col in df.columns if 'rsi' in col.lower()]
        for col in rsi_cols:
            if col in df.columns:
                # Check for values outside 0-100 range
                invalid_mask = (df[col] < 0) | (df[col] > 100)
                invalid_count = invalid_mask.sum()
                if invalid_count > 0:
                    df[col] = df[col].clip(0, 100)
                    logger.info(f"Fixed {invalid_count} invalid RSI values in {col} (clipped to 0-100)")
        
        # 5. Fix negative volume values (critical financial logic violation)
        volume_cols = [
            'volume', 'volume_alpaca', 'volume_ratio', 'transaction_volume',
            'tx_volume_7d_change', 'tx_volume_sma_7', 'volume_sma_7',
            'santiment_prices_volume', 'santiment_ohlcv_volume',
            'volume_price_momentum'
        ]
        
        for col in volume_cols:
            if col in df.columns:
                negative_mask = df[col] < 0
                negative_count = negative_mask.sum()
                if negative_count > 0:
                    # For financial data, negative volumes should be converted to absolute values
                    df.loc[negative_mask, col] = df.loc[negative_mask, col].abs()
                    logger.info(f"Fixed {negative_count} negative values in {col} (converted to absolute)")
        
        # 6. Fix OHLC logic violations (ensure high >= low)
        if all(col in df.columns for col in ['high', 'low']):
            violations = (df['high'] < df['low'])
            violation_count = violations.sum()
            if violation_count > 0:
                # Swap high and low where high < low
                high_vals = df.loc[violations, 'high'].copy()
                low_vals = df.loc[violations, 'low'].copy()
                df.loc[violations, 'high'] = low_vals
                df.loc[violations, 'low'] = high_vals
                logger.info(f"Fixed {violation_count} OHLC violations where high < low")
        
        # 7. Fix OHLC range logic (open/close should be between high/low)
        if all(col in df.columns for col in ['open', 'close', 'high', 'low']):
            # Ensure open is within high-low range
            open_above_high = df['open'] > df['high']
            open_below_low = df['open'] < df['low']
            open_violations = open_above_high.sum() + open_below_low.sum()
            
            if open_violations > 0:
                df.loc[open_above_high, 'open'] = df.loc[open_above_high, 'high']
                df.loc[open_below_low, 'open'] = df.loc[open_below_low, 'low']
                logger.info(f"Fixed {open_violations} open price violations (clipped to high-low range)")
            
            # Ensure close is within high-low range
            close_above_high = df['close'] > df['high']
            close_below_low = df['close'] < df['low']
            close_violations = close_above_high.sum() + close_below_low.sum()
            
            if close_violations > 0:
                df.loc[close_above_high, 'close'] = df.loc[close_above_high, 'high']
                df.loc[close_below_low, 'close'] = df.loc[close_below_low, 'low']
                logger.info(f"Fixed {close_violations} close price violations (clipped to high-low range)")
        
        # 8. Enhanced numerical null value handling with symbol-aware interpolation
        numeric_cols_to_fix = ['volume_ratio', 'volatility_consistency', 'price_volume_trend']
        
        for col in numeric_cols_to_fix:
            if col in df.columns:
                null_count = df[col].isnull().sum()
                if null_count > 0:
                    if 'symbol' in df.columns:
                        # Group-wise interpolation for better accuracy
                        df[col] = df.groupby('symbol')[col].transform(
                            lambda x: x.interpolate(method='linear').ffill().bfill()
                        )
                    else:
                        # Simple interpolation
                        df[col] = df[col].interpolate(method='linear').ffill().bfill()
                    
                    # Fill remaining nulls with median
                    df[col] = df[col].fillna(df[col].median())
                    logger.info(f"Fixed {null_count} null values in {col}")
        
        # 9. Fix future timestamps (validation issue)
        timestamp_cols = [col for col in df.columns if 'timestamp' in col.lower()]
        for col in timestamp_cols:
            if col in df.columns and pd.api.types.is_datetime64_any_dtype(df[col]):
                current_time = pd.Timestamp.now()
                future_mask = df[col] > current_time
                future_count = future_mask.sum()
                if future_count > 0:
                    df.loc[future_mask, col] = current_time
                    logger.info(f"Fixed {future_count} future timestamps in {col}")
        
        # 10. Ensure chronological timestamp ordering within symbols
        if 'interval_timestamp' in df.columns and 'symbol' in df.columns:
            original_order = df.index.copy()
            try:
                df_sorted = df.groupby('symbol', group_keys=False).apply(
                    lambda x: x.sort_values('interval_timestamp')
                )
                df = df_sorted.reset_index(drop=True)
                logger.info("Sorted timestamps chronologically within each asset")
            except Exception as e:
                logger.warning(f"Could not sort timestamps: {e}")
        
        # 11. Fix price values below minimum threshold (validation issue)
        price_cols = [col for col in df.columns if any(price_term in col.lower() 
                     for price_term in ['price', 'open', 'high', 'low', 'close'])]
        
        threshold = 0.0001  # Minimum threshold from validation report
        for col in price_cols:
            if col in df.columns and df[col].dtype in ['float64', 'float32']:
                # Only fix values that are positive but below threshold
                below_threshold = (df[col] > 0) & (df[col] < threshold)
                below_count = below_threshold.sum()
                if below_count > 0:
                    df.loc[below_threshold, col] = threshold
                    logger.info(f"Fixed {below_count} values below threshold in {col}")
        
        # 12. Address extreme price change consistency issues
        # Apply smoothing to reduce extreme price movements while preserving trends
        price_change_cols = [col for col in df.columns if 'price_change' in col.lower()]
        for col in price_change_cols:
            if col in df.columns and 'symbol' in df.columns:
                # Calculate rolling median to smooth extreme values
                df[f'{col}_smoothed'] = df.groupby('symbol')[col].transform(
                    lambda x: x.rolling(window=3, center=True, min_periods=1).median()
                )
                
                # Identify extreme changes (more than 3 standard deviations)
                symbol_std = df.groupby('symbol')[col].transform('std')
                symbol_mean = df.groupby('symbol')[col].transform('mean')
                extreme_mask = (df[col] - symbol_mean).abs() > 3 * symbol_std
                extreme_count = extreme_mask.sum()
                
                if extreme_count > 0:
                    # Replace extreme values with smoothed values
                    df.loc[extreme_mask, col] = df.loc[extreme_mask, f'{col}_smoothed']
                    logger.info(f"Smoothed {extreme_count} extreme price changes in {col}")
                
                # Drop the temporary smoothed column
                df = df.drop(columns=[f'{col}_smoothed'])
        
        # 13. Fix sentiment-price alignment consistency
        sentiment_cols = [col for col in df.columns if 'sentiment' in col.lower()]
        if sentiment_cols and 'symbol' in df.columns:
            for col in sentiment_cols:
                if col in df.columns:
                    # Apply outlier capping based on IQR
                    Q1 = df.groupby('symbol')[col].transform(lambda x: x.quantile(0.25))
                    Q3 = df.groupby('symbol')[col].transform(lambda x: x.quantile(0.75))
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                    outlier_count = outlier_mask.sum()
                    
                    if outlier_count > 0:
                        df.loc[df[col] < lower_bound, col] = lower_bound[df[col] < lower_bound]
                        df.loc[df[col] > upper_bound, col] = upper_bound[df[col] > upper_bound]
                        logger.info(f"Capped {outlier_count} outlier values in {col}")
        
        # 14. Fix data type inconsistencies for parquet compatibility
        logger.info("Fixing data types for parquet compatibility...")
        
        # Get all columns that might have problematic object types
        problematic_columns = [
            'alpaca_merge_timestamp', 'datetime', 'feature_timestamp', 
            'sentiment_timestamp', 'timestamp', 'timestamp_alpaca', 
            'timestamp_dt', 'interval_timestamp_dt', 'latest_news_timestamp_x',
            'latest_news_timestamp_y'
        ]
        
        # Convert all object columns to string for maximum compatibility
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = df[col].astype(str)
                    logger.debug(f"Converted object column {col} to string for parquet compatibility")
                except Exception as e:
                    logger.warning(f"Error converting column {col}: {e}")
                    # Force conversion by replacing problematic values
                    df[col] = df[col].apply(lambda x: str(x) if pd.notna(x) else 'unknown')
                    logger.debug(f"Force converted problematic column {col} to string")
        
        # 15. Remove non-essential string columns for ML optimization
        logger.info("Removing non-essential string columns for ML optimization...")
        
        # Essential columns to keep (needed for grouping, identification, etc.)
        essential_string_cols = [
            'symbol',  # Essential for grouping and identification
            'interval_timestamp',  # Essential for time series
        ]
        
        # Get all string/object columns
        string_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
        
        # Identify columns to remove (non-essential string columns)
        cols_to_remove = [col for col in string_cols if col not in essential_string_cols]
        
        if cols_to_remove:
            original_shape = df.shape
            df = df.drop(columns=cols_to_remove)
            logger.info(f"Removed {len(cols_to_remove)} non-essential string columns")
            logger.info(f"Shape changed from {original_shape} to {df.shape}")
            logger.debug(f"Removed columns: {cols_to_remove}")
        else:
            logger.info("No non-essential string columns found to remove")
        
        # 16. Final null value cleanup
        remaining_nulls = df.isnull().sum()
        for col in remaining_nulls[remaining_nulls > 0].index:
            if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna('unknown')
        
        # 17. Add data quality metadata (JSON-serializable)
        df.attrs['quality_fixes_applied'] = {
            'timestamp': pd.Timestamp.now().isoformat(),  # Convert to ISO string for JSON compatibility
            'fixes': [
                'Empty columns removed',
                'RSI values normalized to 0-100',
                'Negative volumes converted to absolute',
                'OHLC logic violations fixed',
                'Future timestamps corrected',
                'Extreme price changes smoothed',
                'Sentiment outliers capped',
                'Chronological ordering applied'
            ]
        }
        
        final_nulls = df.isnull().sum().sum()
        logger.info(f"Enhanced data quality fixes completed. Remaining null values: {final_nulls}")
        
        return df


# Convenience function for quick processing
def process_financial_data(data_path: str = "data/features", 
                          output_path: str = "data/processed",
                          config: Dict = None) -> Dict:
    """
    Quick processing function for financial data.
    
    Args:
        data_path: Path to input data
        output_path: Path for output
        config: Processing configuration
        
    Returns:
        Processing results
    """
    processor = DataProcessor(data_path, output_path, config)
    return processor.process_all_data()


if __name__ == "__main__":
    import sys
    import os
    
    # Add project root to path for imports
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    sys.path.insert(0, project_root)
    
    # Run the enhanced data processor
    try:
        results = process_financial_data()
        print("\n=== ENHANCED DATA PROCESSING COMPLETED ===")
        print(f"Results: {results}")
    except Exception as e:
        print(f"Error running enhanced data processor: {e}")
        import traceback
        traceback.print_exc()


