"""
Example usage of the DCR framework API.
"""

from src.core import DCRCalculator
from src.utils import DataLoader, OutputFormatter
from config import DATA_DIR, BENCHMARKS


def example_single_calculation():
    """Example: Calculate DCR for a single set of contamination scores."""
    print("Example 1: Single DCR Calculation")
    print("-" * 40)
    
    calculator = DCRCalculator()
    
    # Example contamination scores (L1, L2, L3, L4)
    s1, s2, s3, s4 = 0.3, 0.2, 0.5, 0.1
    
    dcr = calculator.calculate_dcr(s1, s2, s3, s4)
    raw_accuracy = 85.0
    adj_accuracy = calculator.calculate_adjusted_accuracy(raw_accuracy, dcr)
    
    print(f"Contamination scores: L1={s1}, L2={s2}, L3={s3}, L4={s4}")
    print(f"DCR Factor: {dcr:.4f}")
    print(f"Raw Accuracy: {raw_accuracy:.2f}%")
    print(f"Adjusted Accuracy: {adj_accuracy:.2f}%")
    print()


def example_benchmark_analysis():
    """Example: Analyze a benchmark dataset."""
    print("Example 2: Benchmark Analysis")
    print("-" * 40)
    
    # Load SST-2 data
    data_file = DATA_DIR / BENCHMARKS['sst2']['file']
    data = DataLoader.load_csv(str(data_file))
    
    # Process first 5 entries
    calculator = DCRCalculator()
    sample_data = data[:5]
    
    print("Analyzing first 5 entries from SST-2:")
    print()
    
    for row in sample_data:
        bdc_level, model, t1, t2, t3, t4, acc = row
        dcr = calculator.calculate_dcr(t1, t2, t3, t4)
        adj_acc = calculator.calculate_adjusted_accuracy(acc, dcr)
        
        print(f"{model} (BDC={bdc_level}): DCR={dcr:.4f}, "
              f"Acc={acc:.1f}% -> {adj_acc:.1f}%")
    print()


def example_correlation_analysis():
    """Example: Calculate correlation between DCR and accuracy."""
    print("Example 3: Correlation Analysis")
    print("-" * 40)
    
    # Sample data
    dcr_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    accuracy_values = [70, 75, 85, 92, 95]
    
    calculator = DCRCalculator()
    corr, p_value = calculator.calculate_correlation(dcr_values, accuracy_values)
    
    print(f"DCR values: {dcr_values}")
    print(f"Accuracy values: {accuracy_values}")
    print(f"Pearson correlation: {corr:.4f}")
    print(f"P-value: {p_value:.6f}")
    print()


if __name__ == '__main__':
    print("DCR Framework API Examples")
    print("=" * 40)
    print()
    
    example_single_calculation()
    example_benchmark_analysis()
    example_correlation_analysis()