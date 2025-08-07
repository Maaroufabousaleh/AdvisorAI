"""
AdvisorAI Data Processing Main

Main entry point for processing financial data through the complete pipeline.
Handles stocks and crypto feature data, applies cleaning, validation, normalization,
and prepares ML-ready datasets.

Usage:
    python main.py                    # Process with default settings
    python main.py --config custom.json  # Use custom configuration
    python main.py --stocks-only      # Process only stocks data
    python main.py --crypto-only      # Process only crypto data
    python main.py --quick            # Quick processing (skip intermediate saves)
"""

import sys
import argparse
import json
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

# Suppress specific warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')

# Add current directory to path for relative imports
sys.path.append(str(Path(__file__).parent))

from preprocessing import (
    DataProcessor,
    process_financial_data,
    DataCleaner,
    DataValidator,
    DataNormalizer
)


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)
    
    # Set specific logger levels
    logging.getLogger('src.data_engineering').setLevel(logging.INFO)


def load_config(config_path: str) -> Dict:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logging.info(f"Loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        logging.warning(f"Configuration file {config_path} not found, using defaults")
        return {}
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing configuration file: {e}")
        return {}


def get_default_config() -> Dict:
    """
    Get default processing configuration.
    
    Returns:
        Default configuration dictionary
    """
    # Get the current file's directory and calculate paths from project root
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    
    return {
        'data_paths': {
            'input': str(project_root / 'data' / 'features'),
            'output': str(project_root / 'data' / 'processed')
        },
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
                'indicator_scaling': 'standard',
                'return_scaling': 'standard'
            },
            'crypto': {
                'price_scaling': 'robust',
                'volume_scaling': 'log_standard',
                'indicator_scaling': 'robust',
                'return_scaling': 'robust'
            }
        },
        'outlier_handling': {
            'method': 'cap',
            'preserve_extreme_moves': True
        },
        'missing_data': {
            'price_strategy': 'forward_fill',
            'volume_strategy': 'zero_fill',
            'indicator_strategy': 'interpolate',
            'news_strategy': 'neutral_fill'
        },
        'processing': {
            'save_intermediate': True,
            'parallel_processing': False
        }
    }


def validate_data_paths(config: Dict) -> bool:
    """
    Validate that required data paths exist.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if paths are valid
    """
    input_path = Path(config['data_paths']['input'])
    
    if not input_path.exists():
        logging.error(f"Input data path does not exist: {input_path}")
        return False
    
    stocks_file = input_path / "stocks_features.parquet"
    crypto_file = input_path / "crypto_features.parquet"
    
    if not stocks_file.exists():
        logging.warning(f"Stocks data file not found: {stocks_file}")
    
    if not crypto_file.exists():
        logging.warning(f"Crypto data file not found: {crypto_file}")
    
    if not stocks_file.exists() and not crypto_file.exists():
        logging.error("No data files found. Please ensure you have stocks_features.parquet or crypto_features.parquet")
        return False
    
    return True


def print_banner():
    """Print the application banner."""
    banner = """
    =================================================================
    |                       AdvisorAI                               |
    |               Financial Data Processing Pipeline              |
    |                                                               |
    |   Comprehensive data cleaning, validation, and normalization  |
    |   for high-performance financial prediction systems           |
    =================================================================
    """
    print(banner)


def print_processing_summary(results: Dict):
    """
    Print a comprehensive summary of processing results.
    
    Args:
        results: Processing results dictionary
    """
    print("\n" + "="*80)
    print("PROCESSING SUMMARY")
    print("="*80)
    
    summary = results.get('summary', {})
    
    print(f"Processing Status: {summary.get('processing_status', 'Unknown')}")
    print(f"Datasets Processed: {summary.get('datasets_processed', 0)}")
    
    # Dataset-specific summaries
    for asset_type in ['stocks', 'crypto']:
        if asset_type in results and 'final_data' in results[asset_type]:
            asset_results = results[asset_type]
            final_data = asset_results['final_data']
            metrics = asset_results.get('metrics', {})
            
            print(f"\n{asset_type.upper()} DATA:")
            print(f"  Final Shape: {final_data.shape}")
            
            if 'data_reduction' in metrics:
                reduction = metrics['data_reduction']
                retention = 100 - reduction.get('reduction_percentage', 0)
                print(f"  Data Retention: {retention:.1f}%")
                print(f"  Rows: {reduction.get('rows_original', 0)} -> {reduction.get('rows_final', 0)}")
            
            if 'data_quality_improvements' in metrics:
                quality = metrics['data_quality_improvements']
                print(f"  Missing Values Handled: {quality.get('missing_values_handled', 0)}")
                print(f"  Outliers Handled: {quality.get('outliers_handled', 0)}")
            
            # Validation status
            if 'steps' in asset_results and 'final_validation' in asset_results['steps']:
                validation = asset_results['steps']['final_validation']
                print(f"  Final Validation: {validation.get('overall_status', 'Unknown')}")
            
            # Show sample of data columns
            print(f"  Sample Features: {list(final_data.columns[:5])}...")
    
    # Combined metrics
    if 'combined_metrics' in summary:
        combined = summary['combined_metrics']
        print(f"\nCOMBINED METRICS:")
        print(f"  Total Original Rows: {combined.get('total_original_rows', 0):,}")
        print(f"  Total Final Rows: {combined.get('total_final_rows', 0):,}")
        print(f"  Overall Data Retention: {combined.get('overall_data_retention', 0):.1f}%")
    
    # Output location
    print(f"\nOUTPUT LOCATION:")
    print(f"  Processed data saved to: data/processed/")
    print(f"  Reports and logs saved to: data/processed/")
    print(f"  Normalizers saved to: data/processed/normalizers/")


def main():
    """
    Main function that orchestrates the entire data processing pipeline.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="AdvisorAI Financial Data Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # Process all data with defaults
  python main.py --config custom.json     # Use custom configuration
  python main.py --stocks-only             # Process only stocks
  python main.py --crypto-only             # Process only crypto
  python main.py --quick                   # Skip intermediate saves
  python main.py --log-level DEBUG         # Verbose logging
        """
    )
    
    parser.add_argument('--config', '-c', 
                       help='Path to configuration JSON file')
    parser.add_argument('--stocks-only', action='store_true',
                       help='Process only stocks data')
    parser.add_argument('--crypto-only', action='store_true', 
                       help='Process only crypto data')
    parser.add_argument('--quick', action='store_true',
                       help='Quick processing (skip intermediate saves)')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Set logging level')
    parser.add_argument('--log-file',
                       help='Save logs to file')
    parser.add_argument('--output-dir', '-o',
                       help='Output directory for processed data')
    parser.add_argument('--input-dir', '-i', 
                       help='Input directory containing feature data')
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Set up logging
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting AdvisorAI data processing pipeline...")
    logger.info(f"Arguments: {vars(args)}")
    
    try:
        # Load configuration
        if args.config:
            config = load_config(args.config)
            # Merge with defaults
            default_config = get_default_config()
            for key, value in default_config.items():
                if key not in config:
                    config[key] = value
        else:
            config = get_default_config()
        
        # Override config with command line arguments
        if args.input_dir:
            config['data_paths']['input'] = args.input_dir
        if args.output_dir:
            config['data_paths']['output'] = args.output_dir
        if args.quick:
            config['processing']['save_intermediate'] = False
        
        # Validate data paths
        if not validate_data_paths(config):
            logger.error("Data validation failed. Please check your data paths and files.")
            return 1
        
        logger.info(f"Using configuration: {json.dumps(config, indent=2)}")
        
        # Initialize the processor
        processor = DataProcessor(
            data_path=config['data_paths']['input'],
            output_path=config['data_paths']['output'],
            config=config
        )
        
        # Determine what to process
        process_stocks = not args.crypto_only
        process_crypto = not args.stocks_only
        
        # Check which data files exist
        input_path = Path(config['data_paths']['input'])
        stocks_exists = (input_path / "stocks_features.parquet").exists()
        crypto_exists = (input_path / "crypto_features.parquet").exists()
        
        if process_stocks and not stocks_exists:
            logger.warning("Stocks data not found, skipping stocks processing")
            process_stocks = False
        
        if process_crypto and not crypto_exists:
            logger.warning("Crypto data not found, skipping crypto processing")  
            process_crypto = False
        
        if not process_stocks and not process_crypto:
            logger.error("No data to process")
            return 1
        
        # Process the data
        results = {}
        
        if process_stocks and process_crypto:
            logger.info("Processing both stocks and crypto data...")
            results = process_financial_data(
                data_path=config['data_paths']['input'],
                output_path=config['data_paths']['output'],
                config=config
            )
        elif process_stocks:
            logger.info("Processing stocks data only...")
            results['stocks'] = processor.process_stocks_data(
                save_intermediate=config['processing']['save_intermediate']
            )
            results['summary'] = {'datasets_processed': 1, 'processing_status': 'completed'}
        elif process_crypto:
            logger.info("Processing crypto data only...")
            results['crypto'] = processor.process_crypto_data(
                save_intermediate=config['processing']['save_intermediate']
            )
            results['summary'] = {'datasets_processed': 1, 'processing_status': 'completed'}
        
        # Save final datasets and normalizers
        processor.save_final_datasets()
        
        # Print summary
        print_processing_summary(results)
        
        # Success message
        print("\n" + "="*80)
        print("[SUCCESS] DATA PROCESSING COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nYour data is now ready for machine learning!")
        print("\nNext steps:")
        print("  1. Review the processing reports in data/processed/")
        print("  2. Use the normalized datasets for model training")
        print("  3. Load the saved normalizers for real-time inference")
        print("\nExample:")
        print("  import pandas as pd")
        print("  stocks_data = pd.read_parquet('data/processed/stocks_04_normalized.parquet')")
        print("  crypto_data = pd.read_parquet('data/processed/crypto_04_normalized.parquet')")
        
        logger.info("Data processing pipeline completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        print("\n[!] Processing interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"Error in data processing pipeline: {str(e)}", exc_info=True)
        print(f"\n[X] Error: {str(e)}")
        print("Check the logs for more details.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
