"""
DCR calculation and analysis module.
"""

from typing import List, Tuple, Dict
import numpy as np
from scipy.stats import pearsonr
from .fuzzy_system import FuzzySystem


class DCRCalculator:
    """
    Main calculator for Data Contamination Risk (DCR) analysis.
    """
    
    def __init__(self):
        self.fuzzy_system = FuzzySystem()
        
    def calculate_dcr(self, s1: float, s2: float, s3: float, s4: float) -> float:
        """
        Calculate DCR factor for given contamination scores.
        
        Args:
            s1: Semantic level contamination score
            s2: Information level contamination score
            s3: Data level contamination score  
            s4: Label level contamination score
            
        Returns:
            DCR factor between 0 and 1
        """
        return self.fuzzy_system.calculate_dcr(s1, s2, s3, s4)
    
    def calculate_adjusted_accuracy(self, raw_accuracy: float, dcr_factor: float) -> float:
        """
        Calculate contamination-adjusted accuracy.
        
        Args:
            raw_accuracy: Original accuracy percentage
            dcr_factor: DCR factor between 0 and 1
            
        Returns:
            Adjusted accuracy percentage
        """
        return raw_accuracy * (1 - dcr_factor)
    
    def calculate_delta(self, baseline_adjusted: float, current_adjusted: float) -> float:
        """
        Calculate absolute error between baseline and current adjusted accuracy.
        
        Args:
            baseline_adjusted: Baseline adjusted accuracy
            current_adjusted: Current adjusted accuracy
            
        Returns:
            Absolute difference
        """
        return abs(current_adjusted - baseline_adjusted)
    
    def calculate_average_error(self, ground_truths: List[float], 
                               predictions: List[float]) -> float:
        """
        Calculate average error between ground truth and predictions.
        
        Args:
            ground_truths: List of ground truth values
            predictions: List of predicted values
            
        Returns:
            Average absolute error
        """
        if not ground_truths or not predictions:
            return 0.0
            
        errors = [abs(gt - pred) for gt, pred in zip(ground_truths, predictions)]
        return sum(errors) / len(errors)
    
    def calculate_correlation(self, dcr_values: List[float], 
                            accuracy_values: List[float]) -> Tuple[float, float]:
        """
        Calculate Pearson correlation between DCR and accuracy values.
        
        Args:
            dcr_values: List of DCR factors
            accuracy_values: List of accuracy values
            
        Returns:
            Tuple of (correlation coefficient, p-value)
        """
        if len(dcr_values) < 2 or len(accuracy_values) < 2:
            return 0.0, 1.0
            
        return pearsonr(dcr_values, accuracy_values)
    
    def process_experiment_data(self, data: List[List]) -> Dict:
        """
        Process experimental data and calculate all metrics.
        
        Args:
            data: List of experimental data rows
            
        Returns:
            Dictionary containing results and statistics
        """
        results = []
        ground_truths = {}
        gt_list = []
        pred_list = []
        dcr_values = []
        raw_acc_values = []
        
        for row in data:
            bdc_level, model, t1, t2, t3, t4, acc = row
            
            # Calculate DCR
            dcr = self.calculate_dcr(t1, t2, t3, t4)
            adj_acc = self.calculate_adjusted_accuracy(acc, dcr)
            
            # Track values for correlation
            dcr_values.append(dcr)
            raw_acc_values.append(acc)
            
            # Calculate delta
            if bdc_level == 0:
                ground_truths[model] = adj_acc
                delta = 0.0
            else:
                if model in ground_truths:
                    delta = self.calculate_delta(ground_truths[model], adj_acc)
                    gt_list.append(ground_truths[model])
                    pred_list.append(adj_acc)
                else:
                    delta = 0.0
            
            results.append({
                'model': model,
                'bdc_level': bdc_level,
                'dcr': dcr,
                'raw_acc': acc,
                'adj_acc': adj_acc,
                'delta': delta
            })
        
        # Calculate statistics
        avg_error = self.calculate_average_error(gt_list, pred_list)
        corr_coef, p_value = self.calculate_correlation(dcr_values, raw_acc_values)
        
        return {
            'results': results,
            'average_error': avg_error,
            'correlation': corr_coef,
            'p_value': p_value
        }