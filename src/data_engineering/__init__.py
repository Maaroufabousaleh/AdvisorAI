"""
Data Engineering Package

This package contains all data processing, normalization, and feature engineering
components for the financial prediction system.
"""

# Import preprocessing modules (these exist)
from .preprocessing.data_cleaner import DataCleaner
from .preprocessing.data_validator import DataValidator
from .preprocessing.missing_data_handler import MissingDataHandler
from .preprocessing.outlier_detection import OutlierDetector
from .preprocessing.data_normalizer import DataNormalizer
from .preprocessing.data_processor import DataProcessor, process_financial_data

# Feature engineering and pipelines modules will be imported when they're implemented
# from .feature_engineering.technical_indicators import TechnicalIndicators
# from .feature_engineering.volatility_features import VolatilityFeatures
# from .feature_engineering.time_features import TimeFeatures
# from .feature_engineering.cross_asset_features import CrossAssetFeatures
# from .pipelines.training_pipeline import TrainingPipeline
# from .pipelines.inference_pipeline import InferencePipeline

__version__ = "1.0.0"
__all__ = [
    "DataCleaner",
    "DataValidator", 
    "MissingDataHandler",
    "OutlierDetector",
    "TechnicalIndicators",
    "VolatilityFeatures",
    "TimeFeatures",
    "CrossAssetFeatures",
    "TrainingPipeline",
    "InferencePipeline"
]
