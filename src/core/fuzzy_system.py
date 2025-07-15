"""
Fuzzy inference system for DCR calculation.
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import interp_membership, defuzz


class FuzzySystem:
    """
    Implements the fuzzy inference system for calculating Data Contamination Risk (DCR).
    """
    
    def __init__(self):
        self.x = np.linspace(0, 1, 101)
        self.input_terms = self._setup_input_membership()
        self.output_terms = self._setup_output_membership()
    
    def _setup_input_membership(self):
        """Setup input membership functions (trapezoidal for wider coverage)."""
        return {
            'Low': fuzz.trapmf(self.x, [0, 0, 0.1, 0.3]),
            'Medium': fuzz.trapmf(self.x, [0.2, 0.4, 0.5, 0.6]),
            'High': fuzz.trapmf(self.x, [0.5, 0.8, 1.0, 1.0])
        }
    
    def _setup_output_membership(self):
        """Setup output membership functions."""
        return {
            'Negligible': fuzz.trapmf(self.x, [0, 0, 0.1, 0.3]),
            'Minor': fuzz.trimf(self.x, [0.1, 0.3, 0.5]),
            'Moderate': fuzz.trimf(self.x, [0.3, 0.5, 0.7]),
            'Significant': fuzz.trimf(self.x, [0.5, 0.7, 0.9]),
            'Severe': fuzz.trapmf(self.x, [0.7, 0.9, 1.0, 1.0])
        }
    
    def calculate_dcr(self, s1, s2, s3, s4):
        """
        Calculate DCR factor based on four contamination scores.
        
        Args:
            s1: Semantic level contamination score
            s2: Information level contamination score  
            s3: Data level contamination score
            s4: Label level contamination score
            
        Returns:
            float: DCR factor between 0 and 1
        """
        # Input sanitization with proportional minimum
        inputs = [max(0.0, min(float(v), 1.0)) for v in [s1, s2, s3, s4]]
        min_memb = max(0.001, np.mean(inputs))  # Dynamic minimum based on inputs
        
        # Set the threshold for negligible contamination
        if np.mean(inputs) < 0.02:
            return 0.0
        
        # Enhanced fuzzification
        fuzzified = {}
        for i, val in enumerate(inputs):
            fuzzified[f'L{i+1}'] = {
                term: max(interp_membership(self.x, mf, val), min_memb*0.1)
                for term, mf in self.input_terms.items()
            }
        
        try:
            # Apply fuzzy rules
            aggregated = self._apply_rules(fuzzified, min_memb)
            
            # Defuzzification
            dcr = defuzz(self.x, aggregated, 'centroid')
            return round(max(min(dcr, 1.0), 0.0), 4)
        
        except Exception as e:
            return round(min_memb, 3)
    
    def _apply_rules(self, fuzzified, min_memb):
        """Apply fuzzy inference rules."""
        # Rule 0: If all inputs are low, then contamination is Negligible
        rule0 = np.fmin(
            np.fmin(fuzzified['L1']['Low'], fuzzified['L2']['Low']),
            np.fmin(fuzzified['L3']['Low'], fuzzified['L4']['Low'])
        )
        rule0 = np.fmin(rule0, self.output_terms['Negligible'])
        
        # Rule: If L3 or L4 is high then contamination is Severe
        rule_severe = np.fmin(
            np.fmax(fuzzified['L3']['High'], fuzzified['L4']['High']),
            self.output_terms['Severe']
        )
        
        # Rule: If L1 or L2 is high then contamination is Significant
        rule_significant = np.fmin(
            np.fmax(fuzzified['L1']['High'], fuzzified['L2']['High']),
            self.output_terms['Significant']
        )
        
        # Rule: If overall medium membership is high, then consider it Moderate
        medium_avg = np.mean([fuzzified[f'L{i+1}']['Medium'] for i in range(4)])
        rule_moderate = np.fmin(medium_avg, self.output_terms['Moderate'])
        
        # Rule: If lower inputs show slight elevation, then it is Minor
        rule_minor = np.fmin(
            np.fmax(fuzzified['L1']['Medium'], fuzzified['L2']['Low']),
            self.output_terms['Minor']
        )
        
        # Aggregate all rules using maximum operator
        aggregated = np.fmax.reduce([
            rule0,
            rule_severe,
            rule_significant,
            rule_moderate,
            rule_minor,
        ])
        
        # Smoothing for low-value cases
        if np.max(aggregated) < 0.06:
            aggregated += self.output_terms['Negligible'] * min_memb
        
        return aggregated