# AdvisorAI - High-Performance Financial Data Processing System

A comprehensive AI/ML system for financial prediction with advanced data processing capabilities for stocks and cryptocurrency data.

## Features

- 🔄 **Complete Data Pipeline**: End-to-end processing from raw features to ML-ready datasets
- 🧹 **Data Cleaning**: Intelligent handling of missing values, duplicates, and timestamps
- ✅ **Data Validation**: Comprehensive validation with financial domain logic
- 🎯 **Outlier Detection**: Multi-method outlier detection preserving legitimate market moves
- 📊 **Smart Normalization**: Asset-specific scaling strategies for stocks and crypto
- 📈 **Financial Features**: Specialized handling for OHLC, volume, technical indicators
- 🚀 **Easy Usage**: Single command execution with comprehensive logging

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Your Data
Place your feature files in the `data/features/` directory:
- `stocks_features.parquet` - Stock market features
- `crypto_features.parquet` - Cryptocurrency features

### 3. Run the Processing Pipeline
```bash
# Process all data with default settings
python main.py

# Process only stocks data
python main.py --stocks-only

# Process only crypto data  
python main.py --crypto-only

# Use custom configuration
python main.py --config config.json

# Quick processing (skip intermediate saves)
python main.py --quick

# Verbose logging
python main.py --log-level DEBUG
```

## Usage Examples

### Basic Processing
```bash
python main.py
```

### Advanced Usage
```bash
# Custom input/output directories
python main.py --input-dir "my_data" --output-dir "my_results"

# Save logs to file
python main.py --log-file "processing.log"

# Process with custom configuration
python main.py --config custom_config.json --log-level DEBUG
```

## Configuration

The system uses a JSON configuration file to control processing behavior. See `config.json` for the default configuration:

```json
{
  "normalization": {
    "stocks": {
      "price_scaling": "minmax",
      "volume_scaling": "log_standard"
    },
    "crypto": {
      "price_scaling": "robust",
      "volume_scaling": "log_standard"
    }
  },
  "outlier_handling": {
    "method": "cap",
    "preserve_extreme_moves": true
  }
}
```

## Output Structure

After processing, you'll find:

```
data/processed/
├── stocks_01_cleaned.parquet          # Cleaned stocks data
├── stocks_02_validated.parquet        # Validated stocks data  
├── stocks_03_missing_handled.parquet  # Missing values handled
├── stocks_04_normalized.parquet       # Final normalized stocks data
├── crypto_01_cleaned.parquet          # Cleaned crypto data
├── crypto_02_validated.parquet        # Validated crypto data
├── crypto_03_missing_handled.parquet  # Missing values handled  
├── crypto_04_normalized.parquet       # Final normalized crypto data
├── normalizers/                       # Fitted normalizers for inference
│   ├── stocks_normalizer.joblib
│   └── crypto_normalizer.joblib
└── reports/                           # Processing reports and logs
    ├── stocks_validation_report.json
    └── crypto_validation_report.json
```

## Loading Processed Data

```python
import pandas as pd
from joblib import load

# Load processed datasets
stocks_data = pd.read_parquet('data/processed/stocks_04_normalized.parquet')
crypto_data = pd.read_parquet('data/processed/crypto_04_normalized.parquet')

# Load normalizers for inference  
stocks_normalizer = load('data/processed/normalizers/stocks_normalizer.joblib')
crypto_normalizer = load('data/processed/normalizers/crypto_normalizer.joblib')
```