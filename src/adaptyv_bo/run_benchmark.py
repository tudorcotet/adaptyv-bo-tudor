import os
import numpy as np
import torch
import pandas as pd
from typing import List
from config.optimization import OptimizationConfig
from utils.load import load_benchmark_data, get_acquisition, get_generator, get_surrogate, get_encoding
from utils.query import BenchmarkQuery
from acquisitions.base import BaseAcquisition
from generator.base import BaseGenerator
from surrogates.base import BaseSurrogate
from encoding.onehot import OneHotEncoding
from encoding.base import BaseEncoding
from utils.plotter import SimplePlotter
from optimization.bayesian_loop import BayesianOptimizationLoop


def run_multiple_seeds(config: OptimizationConfig):
    """
    Run multiple seeds of the Bayesian optimization process and aggregate results.

    This function performs the following steps:
    1. Load benchmark data
    2. For each seed:
        a. Initialize components (acquisition, query, generator, surrogate, encoding, plotter)
        b. Run the Bayesian optimization loop
        c. Plot and save individual seed results
    3. Combine results from all seeds
    4. Save combined results
    5. Plot average results across all seeds

    Args:
        config (OptimizationConfig): Configuration object containing optimization parameters

    Returns:
        None
    """
    benchmark_data = load_benchmark_data(config.benchmark_file)
    all_results: List[pd.DataFrame] = []
    for seed in range(config.n_seeds):
        config.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    
        # Initialize the acquisition function
        acquisition: BaseAcquisition = get_acquisition(config)
        
        # Create a query object for the benchmark data
        query: BenchmarkQuery = BenchmarkQuery(config, benchmark_data)
        
        # Randomly select initial sequences from the benchmark data
        initial_sequences: List[str] = np.random.choice(list(benchmark_data.keys()), size=config.n_initial, replace=False).tolist()
        
        # Initialize the sequence generator
        generator: BaseGenerator = get_generator(config, benchmark_data, initial_sequences)
        
        # Create the surrogate model
        surrogate: BaseSurrogate = get_surrogate(config)
        
        # Initialize the sequence encoding method
        encoding: BaseEncoding = get_encoding(config)
        
        # Create a plotter object for visualizing results
        plotter: SimplePlotter = SimplePlotter(config, encoding)
        
        # Initialize the Bayesian optimization loop
        loop: BayesianOptimizationLoop = BayesianOptimizationLoop(config, acquisition, query, generator, surrogate, encoding, plotter)
        
        # Run the optimization loop and get the results
        sequences, fitness_values, rounds = loop.run()

        loop.save_results()
        loop.plot_results()
        # Save individual seed results
        all_results.append(loop.seed_results_df)
     
    combined_results = pd.concat(all_results, ignore_index=True)
    combined_csv_path = os.path.join(plotter.output_dir, "csv", "combined_results.csv")
    combined_results.to_csv(combined_csv_path, index=False)
    print(f"Combined results saved to {combined_csv_path}")

    # Create a new plotter for the average results
    average_plotter = SimplePlotter(config, encoding)

    # Extract fitness values for each seed
    fitness_by_seed = [combined_results[combined_results['Seed'] == seed]['Fitness'].tolist()
                       for seed in range(config.n_seeds)]

    # Plot average results
    average_plotter.plot_average_results(fitness_by_seed)

if __name__ == "__main__":
    import argparse
    from dataclasses import asdict

    parser = argparse.ArgumentParser()
    for field in OptimizationConfig.__dataclass_fields__.values():
        parser.add_argument(f'--{field.name}', type=field.type, default=field.default)

    args = parser.parse_args()
    config = OptimizationConfig(**{k: v for k, v in vars(args).items() if k in OptimizationConfig.__dataclass_fields__})
    run_multiple_seeds(config)
