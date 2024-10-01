import os
import numpy as np
import torch
import pandas as pd
from typing import List, Dict
from config.optimization import *
from utils.load import load_benchmark_data_mlflow, load_benchmark_data, get_acquisition, get_generator, get_surrogate, get_encoding
from utils.query import BenchmarkQuery
from acquisitions.base import BaseAcquisition
from generator.base import BaseGenerator
from surrogates.base import BaseSurrogate
from encoding.onehot import OneHotEncoding
from encoding.base import BaseEncoding
from utils.plotter import SimplePlotter
from optimization.bayesian_loop import BayesianOptimizationLoop
from utils.trackers.mlflow import MLflowTracker
import multiprocessing as mp
import time
import mlflow

def get_parent_run_name(config: OptimizationConfig, dataset_name: str, benchmark: bool = True):
    return (
        f"dataset@{dataset_name.replace('_', '-')}"
        f"_production@{not benchmark}"
        f"_time@{int(time.time())}"
        f"_surrogate@{config.surrogate_config.surrogate_type}"
        f"_acquisition@{config.acquisition_config.acquisition_type}"
        f"_encoding@{config.encoding_config.encoding_type.replace('_', '')}"
        f"_generator@{config.generator_config.generator_type}"
        f"_kernel@{config.surrogate_config.kernel_type}"
    )

def run_single_seed(output_dir: str, seed: int, config: OptimizationConfig, benchmark_data, mlflow_tracker: MLflowTracker) -> pd.DataFrame:
    config.general_config.seed = seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Initialize components
    acquisition = get_acquisition(config.acquisition_config)
    query = BenchmarkQuery(config.query_config, benchmark_data)
    generator = get_generator(config.generator_config, benchmark_data)
    surrogate = get_surrogate(config.surrogate_config)
    encoding = get_encoding(config.encoding_config)
    plotter = SimplePlotter(encoding)

    # Initialize the Bayesian optimization loop
    loop = BayesianOptimizationLoop(config, acquisition, query, generator, surrogate, encoding, plotter, output_dir, seed, mlflow_tracker)
    
    # Run the optimization loop and get the results
    sequences, fitness_values, rounds = loop.run()
    return loop.seed_results_df

def run_multiple_seeds(config: OptimizationConfig):
    #dataset, experiment_name = load_benchmark_data_mlflow(config.data_config.benchmark_file)
    dataset, dataset_name = load_benchmark_data_mlflow(config.data_config.benchmark_file)
    benchmark_data = dataset._df.set_index('sequence')['fitness'].to_dict()

    all_results: List[pd.DataFrame] = []
    output_dir = config.output_dir
    os.makedirs(output_dir, exist_ok=True)

    mlflow_tracker = MLflowTracker(config.mlflow_config)
    parent_run_name = get_parent_run_name(config, dataset_name)

    description = f"Bayesian optimization experiment using {config.surrogate_config.surrogate_type} surrogate, " \
                          f"{config.acquisition_config.acquisition_type} acquisition function, " \
                          f"{config.encoding_config.encoding_type} encoding, and " \
                          f"{config.generator_config.generator_type} generator. " \
                          f"Running for {config.general_config.n_iterations} iterations with {config.general_config.n_seeds} seeds."

    with mlflow_tracker.start_parent_run(parent_run_name, description) as parent_run:

        parent_run_id = mlflow_tracker.parent_run_id
        try:
            #add dataset to mlflow
            mlflow_tracker.log_dataset(dataset, context="benchmark")
            # Add tags
            mlflow_tracker.set_tags({
                "experiment_type": "bayesian_optimization_benchmark",
                "dataset": os.path.basename(config.data_config.benchmark_file),
                "dataset_size": len(benchmark_data),
                "dataset_name": dataset_name,
                "optimization_target": "protein_fitness",
                "surrogate_type": config.surrogate_config.surrogate_type,
                "acquisition_type": config.acquisition_config.acquisition_type,
                "encoding_type": config.encoding_config.encoding_type,
                "generator_type": config.generator_config.generator_type,
                "kernel_type": config.surrogate_config.kernel_type,
                "n_iterations": config.general_config.n_iterations,
                "n_initial": config.general_config.n_initial,
                "n_seeds": config.general_config.n_seeds,
                })

            # Log parameters
            mlflow_tracker.log_params({
                "acquisition_type": config.acquisition_config.acquisition_type,
                "surrogate_type": config.surrogate_config.surrogate_type,
                "encoding_type": config.encoding_config.encoding_type,
                "generator_type": config.generator_config.generator_type,
                "kernel_type": config.surrogate_config.kernel_type,
                "n_iterations": config.general_config.n_iterations,
                "n_initial": config.general_config.n_initial,
                "n_seeds": config.general_config.n_seeds,
            }, run_id=parent_run_id)

            with mp.Pool(processes=config.general_config.n_seeds) as pool:
                results = [pool.apply_async(run_single_seed, args=(output_dir, seed, config, benchmark_data, mlflow_tracker)) 
                           for seed in range(config.general_config.n_seeds)]
                for result in results:
                    try:
                        seed_result = result.get()
                        if not seed_result.empty:
                            all_results.append(seed_result)
                    except Exception as e:
                        print(f"Error in seed run: {e}")

            if all_results:
                combined_results = pd.concat(all_results, ignore_index=True)
                combined_csv_path = os.path.join(output_dir, "combined", "csv", "combined_results.csv")
                os.makedirs(os.path.dirname(combined_csv_path), exist_ok=True)
                combined_results.to_csv(combined_csv_path, index=False)
                print(f"Combined results saved to {combined_csv_path}")

                mlflow_tracker.log_artifact(combined_csv_path, run_id=parent_run_id)

                # Log aggregate metrics across all seeds
                aggregate_metrics = calculate_aggregate_metrics(all_results)
                for metric_name, metric_value in aggregate_metrics.items():
                    mlflow_tracker.log_metric(f"aggregate_{metric_name}", metric_value, run_id=parent_run_id)

        except Exception as e:
            print(f"Error in run_multiple_seeds: {e}")
            raise
    
    mlflow_tracker.end_run()

def calculate_aggregate_metrics(all_results: List[pd.DataFrame]) -> Dict[str, float]:
    # Combine all results into a single DataFrame
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Calculate aggregate metrics
    aggregate_metrics = {
        "max_fitness": combined_df['Fitness'].max(),
        "mean_max_fitness": combined_df.groupby('Round')['Fitness'].max().mean(),
        "std_max_fitness": combined_df.groupby('Round')['Fitness'].max().std(),
        "mean_average_fitness": combined_df['Fitness'].mean(),
        "mean_diversity": combined_df['Diversity'].mean() if 'Diversity' in combined_df.columns else None,
        "mean_coverage": combined_df['Coverage'].mean() if 'Coverage' in combined_df.columns else None,
    }
    
    return {k: v for k, v in aggregate_metrics.items() if v is not None}

if __name__ == "__main__":
    import os
    import argparse
    from dataclasses import asdict

    import mlflow
    mlflow.end_run()

    parser = argparse.ArgumentParser()
    for field in OptimizationConfig.__dataclass_fields__.values():
        parser.add_argument(f'--{field.name}', type=field.type, default=field.default)

    args = parser.parse_args()
    os.environ['MLFLOW_TRACKING_URI'] = 'https://mlflow.internal.adaptyvbio.com/'

    # Define different configurations
    acquisition_types = ['ucb', 'ts', 'greedy', 'random', 'ei']
    surrogate_types = ['gp', 'random_forest']
    kernel_types = ['rbf', 'matern', 'linear']
    output_dir = "output_benchmark_configs"
    configs = []
    for acquisition_type in acquisition_types:
        for surrogate_type in surrogate_types:
            if surrogate_type == 'gp':
                for kernel_type in kernel_types:
                    configs.append(
                        OptimizationConfig(
                            acquisition_config=AcquisitionConfig(acquisition_type=acquisition_type),
                            surrogate_config=SurrogateConfig(surrogate_type=surrogate_type, kernel_type=kernel_type)
                        )
                    )
            else:
                configs.append(
                    OptimizationConfig(
                        acquisition_config=AcquisitionConfig(acquisition_type=acquisition_type),
                        surrogate_config=SurrogateConfig(surrogate_type=surrogate_type, kernel_type='None')
                    )
                )

    # Run multiple seeds for each configuration
    for i, cfg in enumerate(configs):
        print(f"\nRunning configuration {i+1}/{len(configs)}:")
        print(f"Acquisition: {cfg.acquisition_config.acquisition_type}")
        print(f"Surrogate: {cfg.surrogate_config.surrogate_type}")
        if cfg.surrogate_config.surrogate_type == 'gp':
            print(f"Kernel: {cfg.surrogate_config.kernel_type}")
        
        config_output_dir = os.path.join(output_dir, f"acquisition_{cfg.acquisition_config.acquisition_type}_surrogate_{cfg.surrogate_config.surrogate_type}")
        if cfg.surrogate_config.surrogate_type == 'gp':
            config_output_dir += f"_kernel_{cfg.surrogate_config.kernel_type}"
        os.makedirs(config_output_dir, exist_ok=True)
        cfg.output_dir = config_output_dir
        run_multiple_seeds(cfg)

    print("\nAll configurations completed.")

