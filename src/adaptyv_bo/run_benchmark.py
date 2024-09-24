import os
import numpy as np
import torch
import pandas as pd
from typing import List
from config.optimization import *
from utils.load import load_benchmark_data, get_acquisition, get_generator, get_surrogate, get_encoding
from utils.query import BenchmarkQuery
from acquisitions.base import BaseAcquisition
from generator.base import BaseGenerator
from surrogates.base import BaseSurrogate
from encoding.onehot import OneHotEncoding
from encoding.base import BaseEncoding
from utils.plotter import SimplePlotter
from optimization.bayesian_loop import BayesianOptimizationLoop
import multiprocessing as mp
import time


def run_single_seed(output_dir: str, seed: int, config: OptimizationConfig, benchmark_data) -> pd.DataFrame:
    config.general_config.seed = seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Initialize the acquisition function
    acquisition: BaseAcquisition = get_acquisition(config.acquisition_config)

    # Create a query object for the benchmark data
    query: BenchmarkQuery = BenchmarkQuery(config.query_config, benchmark_data)

    # Randomly select initial sequences from the benchmark data
    initial_sequences: List[str] = np.random.choice(list(benchmark_data.keys()), size=config.general_config.n_initial, replace=False).tolist()

    # Initialize the sequence generator
    generator: BaseGenerator = get_generator(config.generator_config, benchmark_data, initial_sequences)

    # Create the surrogate model
    surrogate: BaseSurrogate = get_surrogate(config.surrogate_config)

    # Initialize the sequence encoding method
    encoding: BaseEncoding = get_encoding(config.encoding_config)

    # Create a plotter object for visualizing results
    plotter: SimplePlotter = SimplePlotter(encoding)

    # Initialize the Bayesian optimization loop
    loop: BayesianOptimizationLoop = BayesianOptimizationLoop(config, acquisition, query, generator, surrogate, encoding, plotter, output_dir, seed)
    
    # Run the optimization loop and get the results
    sequences, fitness_values, rounds = loop.run()
    return loop.seed_results_df

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
    benchmark_data = load_benchmark_data(config.data_config.benchmark_file)
    all_results: List[pd.DataFrame] = []
    output_dir = f"output_benchmark_{int(time.time())}"
    os.makedirs(output_dir, exist_ok=True)

    with mp.Pool(processes=config.general_config.n_seeds) as pool:
        results = [pool.apply_async(run_single_seed, args=(output_dir, seed, config, benchmark_data)) for seed in range(config.general_config.n_seeds)]
        for result in results:
            try:
                seed_result = result.get()
                if not seed_result.empty:
                    all_results.append(seed_result)
            except Exception as e:
                print(f"Error: {e}")

    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)

        # Ensure the output directory exists
        combined_dir = os.path.join(output_dir, "combined")
        combined_csv_dir = os.path.join(combined_dir, "csv")
        os.makedirs(combined_csv_dir, exist_ok=True)
        
        combined_csv_path = os.path.join(combined_csv_dir, "combined_results.csv")
        combined_results.to_csv(combined_csv_path, index=False)
        print(f"Combined results saved to {combined_csv_path}")

        # Create a new plotter for the average results
        average_plotter = SimplePlotter(get_encoding(config.encoding_config))

        # Extract fitness values for each seed
        fitness_by_seed = [combined_results[combined_results['Seed'] == seed]['Fitness'].tolist()
                           for seed in range(config.general_config.n_seeds)]

        # Plot average results
        combined_plots_dir = os.path.join(combined_dir, "plots")
        os.makedirs(combined_plots_dir, exist_ok=True)
        average_plotter.plot_average_results(fitness_by_seed, combined_plots_dir)
    else:
        print("No results to combine.")

if __name__ == "__main__":
    import argparse
    from dataclasses import asdict

    parser = argparse.ArgumentParser()
    for field in OptimizationConfig.__dataclass_fields__.values():
        parser.add_argument(f'--{field.name}', type=field.type, default=field.default)

    args = parser.parse_args()
    #config = OptimizationConfig(**{k: v for k, v in vars(args).items() if k in OptimizationConfig.__dataclass_fields__})
    #run_multiple_seeds(config)

    # Define different configurations
    acquisition_types = ['ucb', 'ei', 'ts', 'greedy', 'random']
    surrogate_types = ['gp']
    kernel_types = ['rbf', 'matern', 'rational_quadratic']

    configs = []
    for acquisition_type in acquisition_types:
        for surrogate_type in surrogate_types:
            for kernel_type in kernel_types:
                configs.append(
                    OptimizationConfig(
                        acquisition_config=AcquisitionConfig(acquisition_type=acquisition_type),
                        surrogate_config=SurrogateConfig(surrogate_type=surrogate_type, kernel_type=kernel_type)
                    )
                )

    # Run multiple seeds for each configuration
    for i, cfg in enumerate(configs):
        print(f"\nRunning configuration {i+1}/{len(configs)}:")
        print(f"Acquisition: {cfg.acquisition_config.acquisition_type}")
        print(f"Surrogate: {cfg.surrogate_config.surrogate_type}")
        
        config_output_dir = os.path.join(output_dir, f"config_{i+1}")
        os.makedirs(config_output_dir, exist_ok=True)
        
        run_multiple_seeds(cfg, config_output_dir)

    print("\nAll configurations completed.")