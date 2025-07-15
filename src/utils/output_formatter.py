"""
Output formatting utilities for DCR results.
"""

from typing import List, Dict


class OutputFormatter:
    """
    Formats and displays DCR analysis results.
    """
    
    @staticmethod
    def print_header():
        """Print the results table header."""
        print(f"{'Model':<15} | {'BDC':<4} | {'DCR':<6} | {'Raw Acc':<7} | {'Adj Acc':<7} | {'Delta':<7}")
        print("-" * 65)
    
    @staticmethod
    def print_result(result: Dict):
        """
        Print a single result row.
        
        Args:
            result: Dictionary containing result data
        """
        print(f"{result['model']:<15} | {result['bdc_level']:4} | "
              f"{result['dcr']:6.4f} | {result['raw_acc']:6.2f}% | "
              f"{result['adj_acc']:6.2f}% | {result['delta']:6.2f}%")
    
    @staticmethod
    def print_statistics(avg_error: float, correlation: float, p_value: float):
        """
        Print statistical summary.
        
        Args:
            avg_error: Average error percentage
            correlation: Pearson correlation coefficient
            p_value: Statistical p-value
        """
        print("-" * 65)
        print(f"Average Error (compared to BDC 0): {avg_error:.2f}%")
        print("-" * 65)
        print(f"Correlation between DCR and Raw Acc (Pearson): {correlation:.4f}, "
              f"p-value: {p_value:.8f}")
    
    @staticmethod
    def format_results_table(results: List[Dict], stats: Dict) -> str:
        """
        Format results as a complete table string.
        
        Args:
            results: List of result dictionaries
            stats: Statistics dictionary
            
        Returns:
            Formatted table string
        """
        output = []
        
        # Header
        output.append(f"{'Model':<15} | {'BDC':<4} | {'DCR':<6} | {'Raw Acc':<7} | {'Adj Acc':<7} | {'Delta':<7}")
        output.append("-" * 65)
        
        # Results
        for result in results:
            output.append(f"{result['model']:<15} | {result['bdc_level']:4} | "
                         f"{result['dcr']:6.4f} | {result['raw_acc']:6.2f}% | "
                         f"{result['adj_acc']:6.2f}% | {result['delta']:6.2f}%")
        
        # Statistics
        output.append("-" * 65)
        output.append(f"Average Error: {stats['average_error']:.2f}%")
        output.append(f"Correlation (Pearson): {stats['correlation']:.4f}, p-value: {stats['p_value']:.8f}")
        
        return '\n'.join(output)