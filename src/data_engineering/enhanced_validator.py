"""
Updated Data Validator Module

Enhanced data validator that properly handles normalized/scaled data.
Distinguishes between raw and processed data validation.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class EnhancedDataValidator:
    """
    Enhanced data validator that properly handles both raw and normalized data.
    """
    
    def __init__(self, data_type: str = "raw"):
        """
        Initialize enhanced validator.
        
        Args:
            data_type: "raw" for original data, "normalized" for scaled/processed data
        """
        self.data_type = data_type
    
    def validate_dataset(self, df: pd.DataFrame, dataset_name: str, asset_type: str) -> Dict:
        """
        Validate dataset with appropriate rules based on data type.
        """
        results = {
            'dataset_name': dataset_name,
            'asset_type': asset_type,
            'validation_timestamp': datetime.now().isoformat(),
            'overall_status': 'PASS',
            'shape': list(df.shape),
            'checks': {},
            'summary': {
                'total_checks': 0,
                'passed_checks': 0,
                'warning_checks': 0,
                'failed_checks': 0,
                'critical_issues': [],
                'warnings': [],
                'recommendations': []
            }
        }
        
        # Structural validation (always applies)
        results['checks']['structure'] = self._validate_structure(df, asset_type)
        
        # Data quality validation (always applies)
        results['checks']['data_quality'] = self._validate_data_quality(df)
        
        # Value range validation (different for raw vs normalized)
        if self.data_type == "normalized":
            results['checks']['value_ranges'] = self._validate_normalized_ranges(df)
        else:
            results['checks']['value_ranges'] = self._validate_raw_ranges(df)
        
        # Timestamp validation (always applies)
        results['checks']['timestamps'] = self._validate_timestamps(df)
        
        # Asset-specific validation (always applies)
        results['checks']['asset_specific'] = self._validate_asset_specific(df)
        
        # Financial logic validation (different for raw vs normalized)
        if self.data_type == "normalized":
            results['checks']['financial_logic'] = self._validate_normalized_financial_logic(df)
        else:
            results['checks']['financial_logic'] = self._validate_raw_financial_logic(df)
        
        # Consistency validation (adapted for data type)
        results['checks']['consistency'] = self._validate_consistency(df)
        
        # Aggregate results
        self._aggregate_results(results)
        
        return results
    
    def _validate_structure(self, df: pd.DataFrame, asset_type: str) -> Dict:
        """Validate basic structure."""
        results = {
            'status': 'PASS',
            'checks': {
                'empty_check': 'PASS' if not df.empty else 'FAIL',
                'column_types': {
                    'numeric_count': len(df.select_dtypes(include=[np.number]).columns),
                    'object_count': len(df.select_dtypes(include=['object']).columns),
                    'datetime_count': len(df.select_dtypes(include=['datetime']).columns),
                    'total_columns': len(df.columns)
                },
                'duplicate_columns': 'PASS'
            },
            'issues': []
        }
        
        # Check for required columns (flexible for normalized data)
        required_columns = ['symbol']  # Minimal requirement
        if asset_type in ['stocks', 'crypto']:
            # For financial data, we need some price indicator
            price_indicators = ['close', 'price', 'santiment_financial_price']
            has_price = any(col in df.columns for col in price_indicators)
            if not has_price:
                results['status'] = 'FAIL'
                results['checks']['required_columns'] = 'FAIL'
                results['issues'].append("Missing price columns: need at least one of ['close', 'price', 'santiment_financial_price']")
            else:
                results['checks']['required_columns'] = 'PASS'
        
        if df.empty:
            results['status'] = 'FAIL'
            results['issues'].append("Dataset is empty")
        
        return results
    
    def _validate_data_quality(self, df: pd.DataFrame) -> Dict:
        """Validate data quality."""
        results = {
            'status': 'PASS',
            'checks': {},
            'issues': [],
            'metrics': {}
        }
        
        # Missing data check
        total_missing = df.isnull().sum().sum()
        total_values = df.size
        missing_percentage = (total_missing / total_values) * 100
        
        results['metrics']['missing_percentage'] = missing_percentage
        results['metrics']['total_missing'] = int(total_missing)
        results['metrics']['total_values'] = int(total_values)
        
        if missing_percentage > 15:  # More lenient for processed data
            results['status'] = 'WARN'
            results['checks']['missing_data'] = 'WARN'
            results['issues'].append(f"High missing data percentage: {missing_percentage:.1f}%")
        else:
            results['checks']['missing_data'] = 'PASS'
        
        # Empty columns check
        empty_columns = [col for col in df.columns if df[col].isnull().all()]
        if empty_columns:
            results['status'] = 'WARN'
            results['checks']['empty_columns'] = 'WARN'
            results['issues'].append(f"Completely empty columns: {empty_columns}")
        else:
            results['checks']['empty_columns'] = 'PASS'
        
        # Duplicate rows
        duplicate_count = df.duplicated().sum()
        results['metrics']['duplicate_rows'] = int(duplicate_count)
        if duplicate_count > 0:
            results['status'] = 'WARN'
            results['checks']['duplicate_rows'] = 'WARN'
            results['issues'].append(f"Found {duplicate_count} duplicate rows")
        else:
            results['checks']['duplicate_rows'] = 'PASS'
        
        # Data density check
        if 'symbol' in df.columns:
            assets_with_insufficient_data = 0
            for symbol in df['symbol'].unique():
                symbol_data = df[df['symbol'] == symbol]
                if len(symbol_data) < 30:  # More lenient threshold
                    assets_with_insufficient_data += 1
            
            results['metrics']['assets_with_insufficient_data'] = assets_with_insufficient_data
            if assets_with_insufficient_data > 0:
                results['status'] = 'WARN'
                results['checks']['data_density'] = 'WARN'
                results['issues'].append(f"{assets_with_insufficient_data} assets have insufficient data points (< 30)")
            else:
                results['checks']['data_density'] = 'PASS'
        
        return results
    
    def _validate_normalized_ranges(self, df: pd.DataFrame) -> Dict:
        """Validate ranges for normalized data (different expectations)."""
        results = {
            'status': 'PASS',
            'checks': {},
            'issues': [],
            'outliers': {}
        }
        
        # For normalized data, we expect values roughly in range [-3, 3] for standard scaling
        # and [0, 1] for min-max scaling
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col in df.columns and df[col].notna().sum() > 0:
                col_min = df[col].min()
                col_max = df[col].max()
                
                # More lenient checks for normalized data
                if col_min < -10 or col_max > 10:
                    results['status'] = 'WARN'
                    results['checks'][f'{col}_range'] = 'WARN'
                    results['issues'].append(f"{col}: Values outside expected normalized range [{col_min:.3f}, {col_max:.3f}]")
                else:
                    results['checks'][f'{col}_range'] = 'PASS'
        
        return results
    
    def _validate_raw_ranges(self, df: pd.DataFrame) -> Dict:
        """Validate ranges for raw data."""
        results = {
            'status': 'PASS',
            'checks': {},
            'issues': [],
            'outliers': {}
        }
        
        # Original validation logic for raw data
        price_columns = [col for col in df.columns if any(term in col.lower() 
                        for term in ['price', 'open', 'high', 'low', 'close'])]
        
        for col in price_columns:
            if col in df.columns and df[col].notna().sum() > 0:
                col_min = df[col].min()
                if col_min < 0.0001:
                    results['status'] = 'WARN'
                    results['checks'][f'{col}_range'] = 'WARN'
                    results['issues'].append(f"{col}: Minimum value ({col_min}) below threshold (0.0001)")
        
        return results
    
    def _validate_timestamps(self, df: pd.DataFrame) -> Dict:
        """Validate timestamps."""
        results = {
            'status': 'PASS',
            'checks': {},
            'issues': [],
            'metrics': {}
        }
        
        timestamp_cols = [col for col in df.columns if 'timestamp' in col.lower()]
        
        for col in timestamp_cols:
            if col in df.columns and pd.api.types.is_datetime64_any_dtype(df[col]):
                # Check for future dates
                current_time = pd.Timestamp.now()
                future_count = (df[col] > current_time).sum()
                
                results['checks'][f'{col}_future_dates'] = 'PASS'
                if future_count > 0:
                    results['status'] = 'WARN'
                    results['checks'][f'{col}_future_dates'] = 'WARN'
                    results['issues'].append(f"Found {future_count} future timestamps in {col}")
                
                # Add time range info
                if df[col].notna().sum() > 0:
                    results['metrics'][f'{col}_range'] = {
                        'start': df[col].min().isoformat(),
                        'end': df[col].max().isoformat(),
                        'span_days': (df[col].max() - df[col].min()).days
                    }
        
        if timestamp_cols:
            results['checks']['timestamp_existence'] = 'PASS'
        
        return results
    
    def _validate_asset_specific(self, df: pd.DataFrame) -> Dict:
        """Validate asset-specific requirements."""
        results = {
            'status': 'PASS',
            'checks': {},
            'issues': []
        }
        
        if 'symbol' in df.columns:
            unique_assets = df['symbol'].nunique()
            results['checks']['unique_assets'] = unique_assets
            
            # Check data balance
            asset_counts = df['symbol'].value_counts()
            max_count = asset_counts.max()
            min_count = asset_counts.min()
            balance_ratio = max_count / min_count if min_count > 0 else float('inf')
            
            if balance_ratio > 5:  # More lenient threshold
                results['status'] = 'WARN'
                results['checks']['data_balance'] = 'WARN'
                results['issues'].append(f"Imbalanced data across assets (ratio: {balance_ratio:.1f})")
            else:
                results['checks']['data_balance'] = 'PASS'
        
        return results
    
    def _validate_normalized_financial_logic(self, df: pd.DataFrame) -> Dict:
        """Validate financial logic for normalized data (more lenient)."""
        results = {
            'status': 'PASS',
            'checks': {},
            'issues': []
        }
        
        # For normalized data, we can't apply traditional financial logic checks
        # because values have been scaled and may be negative
        
        # We can still check for extreme outliers
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col in df.columns and df[col].notna().sum() > 0:
                # Check for extreme outliers (more than 5 standard deviations)
                std_val = df[col].std()
                mean_val = df[col].mean()
                extreme_outliers = np.abs(df[col] - mean_val) > (5 * std_val)
                extreme_count = extreme_outliers.sum()
                
                if extreme_count > 0:
                    results['status'] = 'WARN'
                    results['checks'][f'{col}_extreme_outliers'] = 'WARN'
                    results['issues'].append(f"Found {extreme_count} extreme outliers in {col}")
                else:
                    results['checks'][f'{col}_extreme_outliers'] = 'PASS'
        
        logger.info("Normalized data: traditional financial logic checks skipped (expected)")
        return results
    
    def _validate_raw_financial_logic(self, df: pd.DataFrame) -> Dict:
        """Validate financial logic for raw data."""
        results = {
            'status': 'PASS',
            'checks': {},
            'issues': []
        }
        
        # Original financial logic validation for raw data
        volume_columns = [col for col in df.columns if 'volume' in col.lower()]
        for col in volume_columns:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                negative_volume = (df[col] < 0).sum()
                if negative_volume > 0:
                    results['status'] = 'FAIL'
                    results['issues'].append(f"Found {negative_volume} negative volume values in {col}")
                    results['checks'][f'{col}_positive'] = 'FAIL'
                else:
                    results['checks'][f'{col}_positive'] = 'PASS'
        
        return results
    
    def _validate_consistency(self, df: pd.DataFrame) -> Dict:
        """Validate data consistency."""
        results = {
            'status': 'PASS',
            'checks': {},
            'issues': []
        }
        
        # Basic consistency checks that work for both raw and normalized data
        if 'symbol' in df.columns:
            # Check for reasonable data distribution per symbol
            for symbol in df['symbol'].unique():
                symbol_data = df[df['symbol'] == symbol]
                if len(symbol_data) < 10:
                    results['status'] = 'WARN'
                    results['issues'].append(f"Insufficient data for {symbol}: {len(symbol_data)} rows")
        
        return results
    
    def _aggregate_results(self, results: Dict):
        """Aggregate validation results."""
        total_checks = 0
        passed_checks = 0
        warning_checks = 0
        failed_checks = 0
        
        for check_category, check_results in results['checks'].items():
            if isinstance(check_results, dict):
                for check_name, status in check_results.get('checks', {}).items():
                    total_checks += 1
                    if status == 'PASS':
                        passed_checks += 1
                    elif status == 'WARN':
                        warning_checks += 1
                    elif status == 'FAIL':
                        failed_checks += 1
                
                # Collect issues
                category_status = check_results.get('status', 'PASS')
                if category_status == 'FAIL':
                    failed_checks += 1
                    results['summary']['critical_issues'].extend(
                        [f"{check_category}: {issue}" for issue in check_results.get('issues', [])]
                    )
                elif category_status == 'WARN':
                    warning_checks += 1
                    results['summary']['warnings'].extend(
                        [f"{check_category}: {issue}" for issue in check_results.get('issues', [])]
                    )
        
        results['summary']['total_checks'] = total_checks
        results['summary']['passed_checks'] = passed_checks
        results['summary']['warning_checks'] = warning_checks
        results['summary']['failed_checks'] = failed_checks
        
        # Set overall status
        if failed_checks > 0:
            results['overall_status'] = 'FAIL'
            results['summary']['recommendations'].append("Address critical validation failures before proceeding")
        elif warning_checks > 0:
            results['overall_status'] = 'WARN'
            results['summary']['recommendations'].append("Review and address validation warnings")
        
        if warning_checks > 0:
            results['summary']['recommendations'].append("Review and address validation warnings")


def validate_fixed_crypto_data():
    """Validate the fixed crypto data."""
    validator = EnhancedDataValidator(data_type="normalized")
    
    # Load fixed crypto data
    df = pd.read_parquet("data/processed/crypto_04_normalized_fixed.parquet")
    
    # Validate
    results = validator.validate_dataset(df, "crypto_final_fixed", "crypto")
    
    # Save results
    with open("data/processed/validation_report_fixed.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Validation completed!")
    print(f"Overall Status: {results['overall_status']}")
    print(f"Passed Checks: {results['summary']['passed_checks']}")
    print(f"Warning Checks: {results['summary']['warning_checks']}")
    print(f"Failed Checks: {results['summary']['failed_checks']}")
    
    if results['summary']['critical_issues']:
        print(f"Critical Issues: {len(results['summary']['critical_issues'])}")
    if results['summary']['warnings']:
        print(f"Warnings: {len(results['summary']['warnings'])}")


if __name__ == "__main__":
    validate_fixed_crypto_data()
