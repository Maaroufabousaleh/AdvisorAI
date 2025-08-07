"""
Data Processing Example Script

Demonstrates how to use the data processing pipeline to clean, validate,
normalize, and prepare your existing stock and crypto feature data for ML models.

This script shows the complete workflow from raw features to ML-ready datasets.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys

# Add the src directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent))

from src.data_engineering.preprocessing import (
    DataProcessor, 
    process_financial_data,
    DataCleaner,
    DataValidator,
    MissingDataHandler,
    OutlierDetector,
    DataNormalizer
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demonstrate_complete_pipeline():
    """
    Demonstrate the complete data processing pipeline.
    """
    logger.info("=== Financial Data Processing Pipeline Demo ===")
    
    # Configuration for processing
    config = {
        'validation': {
            'strict_mode': False,  # Don't fail on warnings
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
                'price_scaling': 'robust',  # More robust for crypto volatility
                'volume_scaling': 'log_standard',
                'indicator_scaling': 'robust'
            }
        },
        'outlier_handling': {
            'method': 'cap',  # Cap outliers rather than remove
            'preserve_extreme_moves': True  # Keep legitimate market moves
        },
        'missing_data': {
            'price_strategy': 'forward_fill',
            'volume_strategy': 'zero_fill',
            'indicator_strategy': 'interpolate'
        }
    }
    
    try:
        # Method 1: Use the simple convenience function
        logger.info("Using convenience function for complete processing...")
        results = process_financial_data(
            data_path="data/features",
            output_path="data/processed", 
            config=config
        )
        
        logger.info("Processing completed successfully!")
        
        # Print summary
        print("\n" + "="*60)
        print("PROCESSING SUMMARY")
        print("="*60)
        
        for asset_type in ['stocks', 'crypto']:
            if asset_type in results and results[asset_type].get('final_data') is not None:
                final_data = results[asset_type]['final_data']
                metrics = results[asset_type]['metrics']
                
                print(f"\n{asset_type.upper()} DATA:")
                print(f"  Final shape: {final_data.shape}")
                print(f"  Data retention: {100 - metrics['data_reduction']['reduction_percentage']:.1f}%")
                print(f"  Missing values handled: {metrics['data_quality_improvements']['missing_values_handled']}")
                print(f"  Outliers handled: {metrics['data_quality_improvements']['outliers_handled']}")
                
                # Show sample of processed data
                print(f"\n  Sample of processed {asset_type} data:")
                print(final_data.head(3).to_string())
        
        print(f"\nOverall Summary:")
        print(f"  Total datasets: {results['summary']['datasets_processed']}")
        print(f"  Processing status: {results['summary']['processing_status']}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in processing pipeline: {e}")
        raise


def demonstrate_step_by_step():
    """
    Demonstrate step-by-step processing for more control.
    """
    logger.info("\n=== Step-by-Step Processing Demo ===")
    
    # Initialize processor
    processor = DataProcessor(
        data_path="data/features",
        output_path="data/processed/manual"
    )
    
    try:
        # Step 1: Load and clean data
        logger.info("Step 1: Loading and cleaning data...")
        stocks_df = processor.cleaner.load_stocks_data()
        crypto_df = processor.cleaner.load_crypto_data()
        
        print(f"\nLoaded data shapes:")
        print(f"  Stocks: {stocks_df.shape}")
        print(f"  Crypto: {crypto_df.shape}")
        
        # Step 2: Validate data quality
        logger.info("Step 2: Validating data quality...")
        stocks_validation = processor.validator.validate_dataset(stocks_df, 'stocks', 'stocks_raw')
        crypto_validation = processor.validator.validate_dataset(crypto_df, 'crypto', 'crypto_raw')
        
        print(f"\nValidation results:")
        print(f"  Stocks: {stocks_validation['overall_status']}")
        print(f"  Crypto: {crypto_validation['overall_status']}")
        
        # Step 3: Handle missing data
        logger.info("Step 3: Handling missing data...")
        stocks_before_missing = stocks_df.isnull().sum().sum()
        crypto_before_missing = crypto_df.isnull().sum().sum()
        
        stocks_df = processor.missing_handler.handle_missing_data(stocks_df, 'stocks')
        crypto_df = processor.missing_handler.handle_missing_data(crypto_df, 'crypto')
        
        stocks_after_missing = stocks_df.isnull().sum().sum()
        crypto_after_missing = crypto_df.isnull().sum().sum()
        
        print(f"\nMissing data handling:")
        print(f"  Stocks: {stocks_before_missing} → {stocks_after_missing}")
        print(f"  Crypto: {crypto_before_missing} → {crypto_after_missing}")
        
        # Step 4: Detect and handle outliers
        logger.info("Step 4: Detecting and handling outliers...")
        stocks_outliers = processor.outlier_detector.detect_outliers(stocks_df, 'stocks')
        crypto_outliers = processor.outlier_detector.detect_outliers(crypto_df, 'crypto')
        
        print(f"\nOutliers detected:")
        print(f"  Stocks: {stocks_outliers['total_outliers']}")
        print(f"  Crypto: {crypto_outliers['total_outliers']}")
        
        stocks_df = processor.outlier_detector.handle_outliers(stocks_df, stocks_outliers, 'cap')
        crypto_df = processor.outlier_detector.handle_outliers(crypto_df, crypto_outliers, 'cap')
        
        # Step 5: Normalize data
        logger.info("Step 5: Normalizing data...")
        processor.normalizer.fit_stocks(stocks_df)
        processor.normalizer.fit_crypto(crypto_df)
        
        stocks_normalized = processor.normalizer.transform_stocks(stocks_df)
        crypto_normalized = processor.normalizer.transform_crypto(crypto_df)
        
        print(f"\nNormalized data shapes:")
        print(f"  Stocks: {stocks_normalized.shape}")
        print(f"  Crypto: {crypto_normalized.shape}")
        
        # Save results
        output_path = Path("data/processed/manual")
        output_path.mkdir(parents=True, exist_ok=True)
        
        stocks_normalized.to_parquet(output_path / "stocks_final_normalized.parquet")
        crypto_normalized.to_parquet(output_path / "crypto_final_normalized.parquet")
        
        logger.info(f"Results saved to {output_path}")
        
        return {
            'stocks': stocks_normalized,
            'crypto': crypto_normalized,
            'validation': {
                'stocks': stocks_validation,
                'crypto': crypto_validation
            },
            'outliers': {
                'stocks': stocks_outliers,
                'crypto': crypto_outliers
            }
        }
        
    except Exception as e:
        logger.error(f"Error in step-by-step processing: {e}")
        raise


def demonstrate_individual_components():
    """
    Demonstrate using individual components for specific tasks.
    """
    logger.info("\n=== Individual Components Demo ===")
    
    try:
        # Just load and examine the data
        cleaner = DataCleaner("data/features")
        stocks_df = cleaner.load_stocks_data()
        
        print(f"\nData overview:")
        print(f"  Shape: {stocks_df.shape}")
        print(f"  Memory usage: {stocks_df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        print(f"  Data types: {dict(stocks_df.dtypes.value_counts())}")
        
        # Get detailed summary
        summary = cleaner.get_data_summary(stocks_df, 'stocks')
        print(f"\nDetailed summary:")
        for key, value in summary.items():
            if key not in ['assets']:  # Skip complex nested data for brevity
                print(f"  {key}: {value}")
        
        # Just check data quality
        validator = DataValidator()
        validation = validator.validate_dataset(stocks_df, 'stocks', 'example')
        
        print(f"\nValidation status: {validation['overall_status']}")
        if validation['summary']['critical_issues']:
            print("Critical issues:")
            for issue in validation['summary']['critical_issues'][:3]:  # Show first 3
                print(f"  - {issue}")
        
        # Just analyze missing data patterns
        missing_handler = MissingDataHandler()
        missing_analysis = missing_handler.analyze_missing_patterns(stocks_df)
        
        print(f"\nMissing data analysis:")
        print(f"  Total missing: {missing_analysis['total_missing']}")
        print(f"  Missing percentage: {missing_analysis['missing_percentage']:.2f}%")
        print(f"  Columns with missing data: {len(missing_analysis['columns_with_missing'])}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in individual components demo: {e}")
        raise


def main():
    """
    Main function that runs all demonstrations.
    """
    print("Financial Data Processing Pipeline Demonstration")
    print("=" * 60)
    
    try:
        # Check if data exists
        data_path = Path("data/features")
        if not data_path.exists():
            print(f"Data path {data_path} does not exist.")
            print("Please ensure you have the stocks_features.parquet and crypto_features.parquet files")
            return
        
        stocks_file = data_path / "stocks_features.parquet"
        crypto_file = data_path / "crypto_features.parquet"
        
        if not stocks_file.exists() or not crypto_file.exists():
            print("Required data files not found:")
            print(f"  {stocks_file} exists: {stocks_file.exists()}")
            print(f"  {crypto_file} exists: {crypto_file.exists()}")
            return
        
        # Run demonstrations
        print("\n1. Running complete pipeline...")
        complete_results = demonstrate_complete_pipeline()
        
        print("\n2. Running step-by-step demonstration...")
        stepwise_results = demonstrate_step_by_step()
        
        print("\n3. Running individual components demonstration...")
        individual_results = demonstrate_individual_components()
        
        print("\n" + "="*60)
        print("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nYour data has been processed and is ready for ML model training.")
        print("Check the 'data/processed' directory for the normalized datasets.")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        print(f"\nAn error occurred: {e}")
        print("Please check the log messages above for more details.")


if __name__ == "__main__":
    main()
