"""
Data loading utilities for DCR experiments.
"""

import csv
from typing import List, Dict
import os


class DataLoader:
    """
    Handles loading experimental data from CSV files.
    """
    
    @staticmethod
    def load_csv(filepath: str) -> List[List]:
        """
        Load experimental data from CSV file.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            List of data rows (excluding header)
        """
        data = []
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        with open(filepath, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append([
                    int(row['bdc_level']),
                    row['model'],
                    float(row['test1']),
                    float(row['test2']),
                    float(row['test3']),
                    float(row['test4']),
                    float(row['accuracy'])
                ])
        
        return data
    
    @staticmethod
    def save_results_csv(results: List[Dict], filepath: str):
        """
        Save results to CSV file.
        
        Args:
            results: List of result dictionaries
            filepath: Output file path
        """
        if not results:
            return
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        fieldnames = ['model', 'bdc_level', 'dcr', 'raw_acc', 'adj_acc', 'delta']
        
        with open(filepath, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                writer.writerow({
                    'model': result['model'],
                    'bdc_level': result['bdc_level'],
                    'dcr': f"{result['dcr']:.4f}",
                    'raw_acc': f"{result['raw_acc']:.2f}",
                    'adj_acc': f"{result['adj_acc']:.2f}",
                    'delta': f"{result['delta']:.2f}"
                })