"""
Main entry point for DCR framework.
"""

import argparse
import os
from pathlib import Path

from src.core import DCRCalculator
from src.utils import DataLoader, OutputFormatter
from config import BENCHMARKS, DATA_DIR, OUTPUT_DIR


def analyze_benchmark(benchmark_name: str, save_csv: bool = False):
    """
    Analyze a specific benchmark dataset.
    
    Args:
        benchmark_name: Name of the benchmark (sst2, liar2, gsm8k)
        save_csv: Whether to save results to CSV
    """
    if benchmark_name not in BENCHMARKS:
        print(f"Error: Unknown benchmark '{benchmark_name}'")
        print(f"Available benchmarks: {', '.join(BENCHMARKS.keys())}")
        return
    
    benchmark_info = BENCHMARKS[benchmark_name]
    data_file = DATA_DIR / benchmark_info['file']
    
    print(f"\n{'='*70}")
    print(f"Analyzing {benchmark_info['name']} ({benchmark_info['task']})")
    print(f"{'='*70}\n")
    
    # Load data
    try:
        data = DataLoader.load_csv(str(data_file))
    except FileNotFoundError:
        print(f"Error: Data file not found: {data_file}")
        return
    
    # Process data
    calculator = DCRCalculator()
    results_dict = calculator.process_experiment_data(data)
    
    # Display results
    OutputFormatter.print_header()
    for result in results_dict['results']:
        OutputFormatter.print_result(result)
    
    OutputFormatter.print_statistics(
        results_dict['average_error'],
        results_dict['correlation'],
        results_dict['p_value']
    )
    
    # Save to CSV if requested
    if save_csv:
        output_file = OUTPUT_DIR / f"{benchmark_name}_results.csv"
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        DataLoader.save_results_csv(results_dict['results'], str(output_file))
        print(f"\nResults saved to: {output_file}")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description='DCR: Data Contamination Risk Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --benchmark sst2
  python main.py --benchmark liar2 --save-csv
  python main.py --all
        """
    )
    
    parser.add_argument(
        '--benchmark', '-b',
        choices=list(BENCHMARKS.keys()),
        help='Benchmark to analyze'
    )
    
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Analyze all benchmarks'
    )
    
    parser.add_argument(
        '--save-csv', '-s',
        action='store_true',
        help='Save results to CSV files'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.benchmark and not args.all:
        parser.error('Either --benchmark or --all must be specified')
    
    # Run analysis
    if args.all:
        for benchmark in BENCHMARKS.keys():
            analyze_benchmark(benchmark, args.save_csv)
    else:
        analyze_benchmark(args.benchmark, args.save_csv)


if __name__ == '__main__':
    main()