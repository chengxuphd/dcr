"""
Configuration settings for DCR framework.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directory
DATA_DIR = PROJECT_ROOT / "data"

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "output"

# Benchmark datasets
BENCHMARKS = {
    'sst2': {
        'name': 'SST-2',
        'file': 'sst2_experimental_data.csv',
        'task': 'Sentiment Analysis'
    },
    'liar2': {
        'name': 'LIAR2', 
        'file': 'liar2_experimental_data.csv',
        'task': 'Fake News Detection'
    },
    'gsm8k': {
        'name': 'GSM8K',
        'file': 'gsm8k_experimental_data.csv', 
        'task': 'Arithmetic Reasoning'
    }
}

# DCR calculation settings
DCR_SETTINGS = {
    'negligible_threshold': 0.02,
    'smoothing_threshold': 0.06,
    'min_membership': 0.001
}

# Output settings
OUTPUT_SETTINGS = {
    'save_csv': True,
    'csv_precision': 4,
    'display_precision': 2
}