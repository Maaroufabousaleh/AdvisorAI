"""
Data Validator Module

Comprehensive data validation for financial datasets.
Ensures data quality, consistency, and readiness for ML pipelines.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Comprehensive data validator for financial time series data.
    Performs quality checks, consistency validation, and data integrity verification.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize data validator.
        
        Args:
            config: Configuration dictionary with validation rules
        """
        default_config = {
            'required_columns': {
                'stocks': ['symbol', 'close'],
                'crypto': ['symbol', 'close']
            },
            'data_quality_thresholds': {
                'max_missing_percentage': 10.0,
                'min_data_points_per_asset': 50,
                'max_consecutive_missing': 5
            },
            'value_ranges': {
                'price_min': 0.0001,
                'price_max': 1000000,
                'volume_min': 0,
                'rsi_min': 0,
                'rsi_max': 100
            },
            'timestamp_validation': {
                'allow_gaps': True,
                'max_gap_hours': 24,
                'require_chronological': True
            }
        }
        
        self.config = config or default_config
        self.validation_results = {}
    
    def validate_dataset(self, df: pd.DataFrame, asset_type: str = 'unknown', 
                        dataset_name: str = 'unknown') -> Dict:
        """
        Perform comprehensive validation of a dataset.
        
        Args:
            df: DataFrame to validate
            asset_type: Type of asset ('stocks', 'crypto', etc.)
            dataset_name: Name of the dataset for reporting
            
        Returns:
            Comprehensive validation results
        """
        logger.info(f"Validating {dataset_name} dataset ({asset_type})...")
        
        validation_results = {
            'dataset_name': dataset_name,
            'asset_type': asset_type,
            'validation_timestamp': datetime.now().isoformat(),
            'overall_status': 'UNKNOWN',
            'shape': df.shape,
            'checks': {}
        }
        
        # 1. Basic structure validation
        validation_results['checks']['structure'] = self._validate_structure(df, asset_type)
        
        # 2. Data quality validation
        validation_results['checks']['data_quality'] = self._validate_data_quality(df)
        
        # 3. Value range validation
        validation_results['checks']['value_ranges'] = self._validate_value_ranges(df)
        
        # 4. Timestamp validation
        validation_results['checks']['timestamps'] = self._validate_timestamps(df)
        
        # 5. Asset-specific validation
        validation_results['checks']['asset_specific'] = self._validate_asset_specific(df, asset_type)
        
        # 6. Financial logic validation
        validation_results['checks']['financial_logic'] = self._validate_financial_logic(df)
        
        # 7. Data consistency validation
        validation_results['checks']['consistency'] = self._validate_consistency(df)
        
        # Determine overall status
        validation_results['overall_status'] = self._determine_overall_status(validation_results['checks'])
        
        # Generate summary
        validation_results['summary'] = self._generate_validation_summary(validation_results)
        
        logger.info(f"Validation complete. Status: {validation_results['overall_status']}")
        
        self.validation_results[dataset_name] = validation_results
        return validation_results
    
    def _validate_structure(self, df: pd.DataFrame, asset_type: str) -> Dict:
        """Validate basic dataset structure."""
        results = {
            'status': 'PASS',
            'checks': {},
            'issues': []
        }
        
        # Check if DataFrame is empty
        if df.empty:
            results['status'] = 'FAIL'
            results['issues'].append("Dataset is empty")
            return results
        
        results['checks']['empty_check'] = 'PASS'
        
        # Check required columns
        required_columns = self.config.get('required_columns', {}).get(asset_type, [])
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            results['status'] = 'FAIL'
            results['issues'].append(f"Missing required columns: {missing_columns}")
            results['checks']['required_columns'] = 'FAIL'
        else:
            results['checks']['required_columns'] = 'PASS'
        
        # Check column data types
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        object_columns = df.select_dtypes(include=['object']).columns.tolist()
        datetime_columns = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        results['checks']['column_types'] = {
            'numeric_count': len(numeric_columns),
            'object_count': len(object_columns),
            'datetime_count': len(datetime_columns),
            'total_columns': len(df.columns)
        }
        
        # Check for duplicate columns
        duplicate_columns = df.columns[df.columns.duplicated()].tolist()
        if duplicate_columns:
            results['status'] = 'WARN'
            results['issues'].append(f"Duplicate column names: {duplicate_columns}")
            results['checks']['duplicate_columns'] = 'FAIL'
        else:
            results['checks']['duplicate_columns'] = 'PASS'
        
        return results
    
    def _validate_data_quality(self, df: pd.DataFrame) -> Dict:
        """Validate data quality metrics."""
        results = {
            'status': 'PASS',
            'checks': {},
            'issues': [],
            'metrics': {}
        }
        
        # Missing data analysis
        total_missing = df.isnull().sum().sum()
        total_values = df.size
        missing_percentage = (total_missing / total_values) * 100 if total_values > 0 else 0
        
        results['metrics']['missing_percentage'] = missing_percentage
        results['metrics']['total_missing'] = int(total_missing)
        results['metrics']['total_values'] = int(total_values)
        
        max_missing_threshold = self.config.get('data_quality_thresholds', {}).get('max_missing_percentage', 10.0)
        
        if missing_percentage > max_missing_threshold:
            results['status'] = 'FAIL'
            results['issues'].append(f"Missing data percentage ({missing_percentage:.2f}%) exceeds threshold ({max_missing_threshold}%)")
            results['checks']['missing_data'] = 'FAIL'
        else:
            results['checks']['missing_data'] = 'PASS'
        
        # Check for completely empty columns
        empty_columns = df.columns[df.isnull().all()].tolist()
        if empty_columns:
            results['status'] = 'WARN'
            results['issues'].append(f"Completely empty columns: {empty_columns}")
            results['checks']['empty_columns'] = 'FAIL'
        else:
            results['checks']['empty_columns'] = 'PASS'
        
        # Check for duplicate rows
        duplicate_count = df.duplicated().sum()
        results['metrics']['duplicate_rows'] = int(duplicate_count)
        
        if duplicate_count > 0:
            results['status'] = 'WARN'
            results['issues'].append(f"Found {duplicate_count} duplicate rows")
            results['checks']['duplicate_rows'] = 'WARN'
        else:
            results['checks']['duplicate_rows'] = 'PASS'
        
        # Check data density per asset (if symbol column exists)
        if 'symbol' in df.columns:
            asset_counts = df['symbol'].value_counts()
            min_data_points = self.config.get('data_quality_thresholds', {}).get('min_data_points_per_asset', 50)
            
            insufficient_assets = asset_counts[asset_counts < min_data_points]
            results['metrics']['assets_with_insufficient_data'] = len(insufficient_assets)
            
            if len(insufficient_assets) > 0:
                results['status'] = 'WARN'
                results['issues'].append(f"{len(insufficient_assets)} assets have insufficient data points (< {min_data_points})")
                results['checks']['data_density'] = 'WARN'
            else:
                results['checks']['data_density'] = 'PASS'
        
        return results
    
    def _validate_value_ranges(self, df: pd.DataFrame) -> Dict:
        """Validate that values are within expected ranges."""
        results = {
            'status': 'PASS',
            'checks': {},
            'issues': [],
            'outliers': {}
        }
        
        value_ranges = self.config.get('value_ranges', {})
        
        # Check price columns
        price_columns = [col for col in df.columns if any(term in col.lower() 
                        for term in ['price', 'close', 'open', 'high', 'low'])]
        
        for col in price_columns:
            if col in df.columns:
                min_val = df[col].min()
                max_val = df[col].max()
                
                # Skip validation if we have NaN/NaT values or empty data
                if pd.isna(min_val) or pd.isna(max_val) or len(df) == 0:
                    continue
                
                price_min = value_ranges.get('price_min', 0.0001)
                price_max = value_ranges.get('price_max', 1000000)
                
                issues = []
                if min_val < price_min:
                    issues.append(f"Minimum value ({min_val}) below threshold ({price_min})")
                if max_val > price_max:
                    issues.append(f"Maximum value ({max_val}) above threshold ({price_max})")
                
                if issues:
                    results['status'] = 'WARN'
                    results['issues'].extend([f"{col}: {issue}" for issue in issues])
                    results['checks'][f'{col}_range'] = 'WARN'
                else:
                    results['checks'][f'{col}_range'] = 'PASS'
        
        # Check RSI columns (should be 0-100)
        rsi_columns = [col for col in df.columns if 'rsi' in col.lower()]
        
        for col in rsi_columns:
            if col in df.columns:
                min_val = df[col].min()
                max_val = df[col].max()
                
                if min_val < 0 or max_val > 100:
                    results['status'] = 'WARN'
                    results['issues'].append(f"RSI column {col} has values outside 0-100 range")
                    results['checks'][f'{col}_range'] = 'FAIL'
                else:
                    results['checks'][f'{col}_range'] = 'PASS'
        
        # Check for infinite or extremely large values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                results['status'] = 'FAIL'
                results['issues'].append(f"Column {col} contains {inf_count} infinite values")
                results['checks'][f'{col}_infinite'] = 'FAIL'
        
        return results
    
    def _validate_timestamps(self, df: pd.DataFrame) -> Dict:
        """Validate timestamp-related aspects."""
        results = {
            'status': 'PASS',
            'checks': {},
            'issues': [],
            'metrics': {}
        }
        
        # Find timestamp columns
        timestamp_columns = [col for col in df.columns if any(term in col.lower() 
                           for term in ['timestamp', 'time', 'date'])]
        
        if not timestamp_columns:
            results['checks']['timestamp_existence'] = 'WARN'
            results['issues'].append("No timestamp columns found")
            return results
        
        results['checks']['timestamp_existence'] = 'PASS'
        
        for timestamp_col in timestamp_columns:
            if pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
                # Check for chronological order
                if 'symbol' in df.columns:
                    # Check per asset
                    for symbol in df['symbol'].unique():
                        symbol_data = df[df['symbol'] == symbol][timestamp_col]
                        if not symbol_data.is_monotonic_increasing:
                            results['status'] = 'WARN'
                            results['issues'].append(f"Timestamps not chronological for {symbol}")
                            results['checks'][f'{timestamp_col}_chronological'] = 'WARN'
                else:
                    # Check globally
                    if not df[timestamp_col].is_monotonic_increasing:
                        results['status'] = 'WARN'
                        results['issues'].append(f"Timestamps in {timestamp_col} not chronological")
                        results['checks'][f'{timestamp_col}_chronological'] = 'WARN'
                
                # Check for reasonable time range
                min_time = df[timestamp_col].min()
                max_time = df[timestamp_col].max()
                time_span = max_time - min_time
                
                results['metrics'][f'{timestamp_col}_range'] = {
                    'start': min_time.isoformat() if pd.notna(min_time) else None,
                    'end': max_time.isoformat() if pd.notna(max_time) else None,
                    'span_days': time_span.days if pd.notna(time_span) else None
                }
                
                # Check for future dates
                now = pd.Timestamp.now()
                future_dates = df[df[timestamp_col] > now]
                if len(future_dates) > 0:
                    results['status'] = 'WARN'
                    results['issues'].append(f"Found {len(future_dates)} future timestamps in {timestamp_col}")
                    results['checks'][f'{timestamp_col}_future_dates'] = 'WARN'
        
        return results
    
    def _validate_asset_specific(self, df: pd.DataFrame, asset_type: str) -> Dict:
        """Perform asset-type specific validations."""
        results = {
            'status': 'PASS',
            'checks': {},
            'issues': []
        }
        
        if 'symbol' not in df.columns:
            results['issues'].append("No symbol column found for asset-specific validation")
            return results
        
        # Check asset symbols
        unique_symbols = df['symbol'].unique()
        results['checks']['unique_assets'] = len(unique_symbols)
        
        # Asset-type specific checks
        if asset_type == 'stocks':
            # Check for valid stock symbols (basic)
            invalid_symbols = [s for s in unique_symbols if not isinstance(s, str) or len(s) > 10]
            if invalid_symbols:
                results['status'] = 'WARN'
                results['issues'].append(f"Potentially invalid stock symbols: {invalid_symbols}")
        
        elif asset_type == 'crypto':
            # Check for valid crypto symbols
            crypto_patterns = ['BTC', 'ETH', 'USD', 'USDT', 'USDC']
            suspicious_symbols = [s for s in unique_symbols if not any(pattern in str(s).upper() 
                                for pattern in crypto_patterns)]
            if len(suspicious_symbols) == len(unique_symbols):  # All symbols suspicious
                results['status'] = 'WARN'
                results['issues'].append("No recognizable crypto symbols found")
        
        # Check data distribution across assets
        if len(unique_symbols) > 1:
            symbol_counts = df['symbol'].value_counts()
            min_count = symbol_counts.min()
            max_count = symbol_counts.max()
            
            # Check for severely imbalanced data
            if max_count / min_count > 10:  # 10x difference
                results['status'] = 'WARN'
                results['issues'].append(f"Severely imbalanced data across assets (ratio: {max_count/min_count:.1f})")
                results['checks']['data_balance'] = 'WARN'
            else:
                results['checks']['data_balance'] = 'PASS'
        
        return results
    
    def _validate_financial_logic(self, df: pd.DataFrame) -> Dict:
        """Validate financial logic and relationships."""
        results = {
            'status': 'PASS',
            'checks': {},
            'issues': []
        }
        
        # Check OHLC relationships
        ohlc_sets = self._find_ohlc_sets(df)
        
        for ohlc_set in ohlc_sets:
            if all(col in df.columns for col in ohlc_set.values()):
                # High should be >= Low
                high_low_violations = (df[ohlc_set['high']] < df[ohlc_set['low']]).sum()
                if high_low_violations > 0:
                    results['status'] = 'FAIL'
                    results['issues'].append(f"Found {high_low_violations} violations where high < low")
                    results['checks']['high_low_logic'] = 'FAIL'
                else:
                    results['checks']['high_low_logic'] = 'PASS'
                
                # Open and Close should be between High and Low
                if 'open' in ohlc_set and 'close' in ohlc_set:
                    open_violations = ((df[ohlc_set['open']] > df[ohlc_set['high']]) | 
                                     (df[ohlc_set['open']] < df[ohlc_set['low']])).sum()
                    close_violations = ((df[ohlc_set['close']] > df[ohlc_set['high']]) | 
                                      (df[ohlc_set['close']] < df[ohlc_set['low']])).sum()
                    
                    total_violations = open_violations + close_violations
                    if total_violations > 0:
                        results['status'] = 'WARN'
                        results['issues'].append(f"Found {total_violations} OHLC logic violations")
                        results['checks']['ohlc_logic'] = 'WARN'
                    else:
                        results['checks']['ohlc_logic'] = 'PASS'
        
        # Check volume logic (should be non-negative)
        volume_columns = [col for col in df.columns if 'volume' in col.lower()]
        for col in volume_columns:
            # Skip non-numeric columns (like datetime columns)
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                try:
                    negative_volume = (df[col] < 0).sum()
                    if negative_volume > 0:
                        results['status'] = 'FAIL'
                        results['issues'].append(f"Found {negative_volume} negative volume values in {col}")
                        results['checks'][f'{col}_positive'] = 'FAIL'
                    else:
                        results['checks'][f'{col}_positive'] = 'PASS'
                except Exception as e:
                    logger.warning(f"Could not validate volume logic for {col}: {e}")
                    results['checks'][f'{col}_positive'] = 'SKIP'
        
        return results
    
    def _find_ohlc_sets(self, df: pd.DataFrame) -> List[Dict]:
        """Find sets of OHLC columns."""
        ohlc_sets = []
        
        # Look for standard OHLC patterns
        potential_sets = {}
        
        for col in df.columns:
            col_lower = col.lower()
            if 'open' in col_lower:
                key = col.replace('open', '').replace('Open', '')
                if key not in potential_sets:
                    potential_sets[key] = {}
                potential_sets[key]['open'] = col
            elif 'high' in col_lower:
                key = col.replace('high', '').replace('High', '')
                if key not in potential_sets:
                    potential_sets[key] = {}
                potential_sets[key]['high'] = col
            elif 'low' in col_lower:
                key = col.replace('low', '').replace('Low', '')
                if key not in potential_sets:
                    potential_sets[key] = {}
                potential_sets[key]['low'] = col
            elif 'close' in col_lower:
                key = col.replace('close', '').replace('Close', '')
                if key not in potential_sets:
                    potential_sets[key] = {}
                potential_sets[key]['close'] = col
        
        # Filter to complete or near-complete sets
        for key, ohlc_set in potential_sets.items():
            if 'high' in ohlc_set and 'low' in ohlc_set:  # Minimum requirement
                ohlc_sets.append(ohlc_set)
        
        return ohlc_sets
    
    def _validate_consistency(self, df: pd.DataFrame) -> Dict:
        """Validate data consistency across time and assets."""
        results = {
            'status': 'PASS',
            'checks': {},
            'issues': []
        }
        
        # Check for sudden jumps in price data
        price_columns = [col for col in df.columns if any(term in col.lower() 
                        for term in ['close', 'price'])]
        
        if 'symbol' in df.columns:
            for price_col in price_columns:
                if price_col in df.columns:
                    for symbol in df['symbol'].unique():
                        symbol_data = df[df['symbol'] == symbol][price_col].dropna()
                        if len(symbol_data) > 1:
                            # Calculate percentage changes
                            pct_changes = symbol_data.pct_change().dropna()
                            
                            # Flag extreme changes (>50% in one period)
                            extreme_changes = (abs(pct_changes) > 0.5).sum()
                            if extreme_changes > 0:
                                results['status'] = 'WARN'
                                results['issues'].append(f"Found {extreme_changes} extreme price changes for {symbol} in {price_col}")
                                results['checks'][f'{symbol}_{price_col}_consistency'] = 'WARN'
        
        return results
    
    def _determine_overall_status(self, checks: Dict) -> str:
        """Determine overall validation status."""
        has_fail = False
        has_warn = False
        
        def check_nested_status(obj):
            nonlocal has_fail, has_warn
            
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key == 'status':
                        if value == 'FAIL':
                            has_fail = True
                        elif value == 'WARN':
                            has_warn = True
                    else:
                        check_nested_status(value)
        
        check_nested_status(checks)
        
        if has_fail:
            return 'FAIL'
        elif has_warn:
            return 'WARN'
        else:
            return 'PASS'
    
    def _generate_validation_summary(self, validation_results: Dict) -> Dict:
        """Generate a summary of validation results."""
        summary = {
            'total_checks': 0,
            'passed_checks': 0,
            'warning_checks': 0,
            'failed_checks': 0,
            'critical_issues': [],
            'warnings': [],
            'recommendations': []
        }
        
        def count_checks(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key
                    
                    if key == 'status':
                        summary['total_checks'] += 1
                        if value == 'PASS':
                            summary['passed_checks'] += 1
                        elif value == 'WARN':
                            summary['warning_checks'] += 1
                        elif value == 'FAIL':
                            summary['failed_checks'] += 1
                    elif key == 'issues' and isinstance(value, list):
                        for issue in value:
                            if 'FAIL' in str(obj.get('status', '')):
                                summary['critical_issues'].append(f"{path}: {issue}")
                            else:
                                summary['warnings'].append(f"{path}: {issue}")
                    else:
                        count_checks(value, current_path)
        
        count_checks(validation_results['checks'])
        
        # Generate recommendations
        if summary['failed_checks'] > 0:
            summary['recommendations'].append("Address critical validation failures before proceeding")
        
        if summary['warning_checks'] > 0:
            summary['recommendations'].append("Review and address validation warnings")
        
        if validation_results['shape'][0] < 100:
            summary['recommendations'].append("Consider acquiring more data for robust model training")
        
        return summary
    
    def save_validation_report(self, output_path: str):
        """Save validation results to file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(self.validation_results, f, indent=2, default=str)
        
        logger.info(f"Validation report saved to {output_path}")
    
    def get_validation_summary(self) -> Dict:
        """Get summary of all validation results."""
        if not self.validation_results:
            return {"message": "No validation results available"}
        
        summary = {
            'datasets_validated': len(self.validation_results),
            'overall_status': 'PASS',
            'dataset_summaries': {}
        }
        
        for dataset_name, results in self.validation_results.items():
            summary['dataset_summaries'][dataset_name] = {
                'status': results['overall_status'],
                'shape': results['shape'],
                'critical_issues': len(results['summary']['critical_issues']),
                'warnings': len(results['summary']['warnings'])
            }
            
            # Update overall status
            if results['overall_status'] == 'FAIL':
                summary['overall_status'] = 'FAIL'
            elif results['overall_status'] == 'WARN' and summary['overall_status'] != 'FAIL':
                summary['overall_status'] = 'WARN'
        
        return summary
