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

        acquisition: BaseAcquisition = get_acquisition(config)
        query: BenchmarkQuery = BenchmarkQuery(config, benchmark_data)
        initial_sequences: np.ndarray = np.random.choice(list(benchmark_data.keys()), size=config.n_initial, replace=False)

        generator: BaseGenerator = get_generator(config, benchmark_data, initial_sequences)
        surrogate: BaseSurrogate = get_surrogate(config)
        encoding: BaseEncoding = get_encoding(config)
        plotter: SimplePlotter = SimplePlotter(config, encoding)


        loop: BayesianOptimizationLoop = BayesianOptimizationLoop(config, acquisition, query, generator, surrogate, encoding, plotter)
        loop.surrogate.fit(np.array(loop.encoded_sequences), np.array(loop.fitness_values))

        if isinstance(self.generator, (CombinatorialGenerator, BenchmarkGenerator)):
            candidates = loop.generator.generate_all()
        else:  # MutationGenerator
            candidates = self.generator.generate(self.config.n_candidates)

        if not candidates:
            self.logger.warning(f"No new candidates generated in iteration {iteration + 1}. Using existing candidates.")
            candidates = self.sequences  # Use existing sequences as candidates

        encoded_candidates = loop.encoding.encode(candidates)
        mu, sigma = loop.surrogate.batch_predict(encoded_candidates, batch_size = 50)

        acquisition_values = self.acquisition.acquire(mu, sigma, self.max_fitness)

        idx_top = np.argsort(-acquisition_values)[:self.config.batch_size]
        selected_candidates = [candidates[i] for i in idx_top]

        new_fitness_values = self.query.query(selected_candidates)

        self.sequences.extend(selected_candidates)
        self.encoded_sequences.extend(self.encoding.encode(selected_candidates))
        self.fitness_values.extend(new_fitness_values)
        self.rounds.extend([iteration + 1] * len(selected_candidates))

        # Update max fitness
        self.max_fitness = max(self.max_fitness, max(new_fitness_values))

        self.generator.update_sequences(selected_candidates)

        self.logger.info(f"Iteration {iteration + 1} completed. Current max fitness: {self.max_fitness}")

        sequences, fitness_values, rounds = loop.run()
        loop.plot_results()

        seed_results = pd.DataFrame({
            'Seed': seed,
            'Round': rounds,
            'Sequence': sequences,
            'Fitness': fitness_values,
            'Surrogate': config.surrogate_type,
            'Acquisition': config.acquisition_type,
            'Encoding': config.encoding_type,
            'Generator': config.generator_type
        })
        all_results.append(seed_results)

        # Save individual seed results
        seed_csv_path = os.path.join(plotter.output_dir, "csv", f"seed_{seed}_results.csv")
        seed_results.to_csv(seed_csv_path, index=False)
        print(f"Seed {seed} results saved to {seed_csv_path}")

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
    config = OptimizationConfig()

    run_multiple_seeds(config)
