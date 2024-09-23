import logging
import os
import numpy as np
import pandas as pd 
from typing import List, Tuple
from config.optimization import OptimizationConfig
from acquisitions.base import BaseAcquisition
from utils.query import BenchmarkQuery
from generator.generator import BaseGenerator, CombinatorialGenerator, BenchmarkGenerator, MutationGenerator
from surrogates.base import BaseSurrogate
from encoding.base import BaseEncoding
from utils.plotter import SimplePlotter

class BayesianOptimizationLoop:
    """
    Implements the Bayesian Optimization loop for protein sequence optimization.

    This class manages the iterative process of generating candidates, evaluating their fitness,
    updating the surrogate model, and selecting new candidates based on an acquisition function.

    Attributes:
        config (OptimizationConfig): Configuration parameters for the optimization process.
        acquisition (BaseAcquisition): The acquisition function used for candidate selection.
        query (BenchmarkQuery): The query object used to evaluate candidate fitness.
        generator (BaseGenerator): The generator used to create new candidate sequences.
        surrogate (BaseSurrogate): The surrogate model used to approximate the fitness landscape.
        encoding (BaseEncoding): The encoding method used to convert sequences to numerical representations.
        plotter (SimplePlotter): The plotter used for visualizing results.
        sequences (List[str]): List of all evaluated sequences.
        encoded_sequences (List[np.ndarray]): List of encoded representations of evaluated sequences.
        fitness_values (List[float]): List of fitness values for evaluated sequences.
        rounds (List[int]): List of round numbers for each evaluated sequence.
        acquired_sequences (set): Set of all acquired sequences to avoid duplicates.
        logger (logging.Logger): Logger for tracking the optimization process.
        max_fitness (float): The maximum fitness value observed so far.
    """

    def __init__(self, config: OptimizationConfig, acquisition: BaseAcquisition, query: BenchmarkQuery,
                 generator: BaseGenerator, surrogate: BaseSurrogate, encoding: BaseEncoding,
                 plotter: SimplePlotter):
        self.config = config
        self.acquisition = acquisition
        self.query = query
        self.generator = generator
        self.surrogate = surrogate
        self.encoding = encoding
        self.plotter = plotter
        self.sequences: List[str] = []
        self.encoded_sequences: List[np.ndarray] = []
        self.fitness_values: List[float] = []
        self.rounds: List[int] = []
        self.acquired_sequences: set = set()
        self.logger = self._setup_logger()
        self.max_fitness = float('-inf')
        self.seed_results_df: pd.DataFrame = pd.DataFrame()

    def _setup_logger(self) -> logging.Logger:
        """Set up and return a logger for the optimization process."""
        logger = logging.getLogger(f"BayesianOpt_Seed_{self.config.seed}")
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(os.path.join(self.plotter.output_dir, f"log_seed_{self.config.seed}.txt"))
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger

    def initialize(self):
        """Initialize the optimization process with a set of initial sequences."""
        initial_sequences = self.generator.generate(self.config.n_initial)
        initial_fitness = self.query.query(initial_sequences)
        self.sequences.extend(initial_sequences)
        self.encoded_sequences.extend(self.encoding.encode(initial_sequences))
        self.fitness_values.extend(initial_fitness)
        self.rounds.extend([0] * len(initial_sequences))
        self.acquired_sequences.update(initial_sequences)
        self.max_fitness = max(initial_fitness)
        self.logger.info(f"Initialization complete. Max fitness: {self.max_fitness}")
        self.logger.info(f"Initial sequences: {initial_sequences}")
        self.logger.info(f"Initial fitness values: {initial_fitness}")

    def run(self) -> Tuple[List[str], List[float], List[int]]:
        """
        Run the Bayesian Optimization loop.

        Returns:
            Tuple[List[str], List[float], List[int]]: Lists of sequences, fitness values, and rounds.
        """
        self.logger.info("Starting Bayesian Optimization Loop")
        self.initialize()

        for iteration in range(self.config.n_iterations):
            self.logger.info(f"Starting iteration {iteration + 1}")

            self.surrogate.fit(np.array(self.encoded_sequences), np.array(self.fitness_values))
            if isinstance(self.generator, (CombinatorialGenerator, BenchmarkGenerator)):
                candidates = self.generator.generate_all()
            else:  # MutationGenerator
                candidates = self.generator.generate(self.config.n_candidates)

            if not candidates:
                self.logger.warning(f"No new candidates generated in iteration {iteration + 1}. Using existing candidates.")
                candidates = self.sequences  # Use existing sequences as candidates

            encoded_candidates = self.encoding.encode(candidates)
            mu, sigma = self.surrogate.predict(encoded_candidates)

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
            print(selected_candidates)
            self.generator.update_sequences(selected_candidates)

            self.logger.info(f"Iteration {iteration + 1} completed. Current max fitness: {self.max_fitness}")

        self.logger.info("Bayesian Optimization Loop completed")
        return self.sequences, self.fitness_values, self.rounds

    def get_best_sequence(self) -> Tuple[str, float]:
        """
        Get the sequence with the highest fitness value.

        Returns:
            Tuple[str, float]: The best sequence and its fitness value.
        """
        best_idx = np.argmax(self.fitness_values)
        return self.sequences[best_idx], self.fitness_values[best_idx]

    def plot_results(self):
        """Generate plots of the optimization results."""
        self.plotter.plot_embeddings(self.sequences, self.fitness_values, self.rounds)
        self.plotter.plot_max_fitness(self.fitness_values)

    def get_seed_results(self):
        """
        Get the results for the current seed.
        """
        if self.seed_results_df.empty:
            self.seed_results_df = pd.DataFrame({
                'Seed': self.config.seed,
                'Round': self.rounds,
                'Sequence': self.sequences,
                'Fitness': self.fitness_values,
                'Surrogate': self.config.surrogate_type,
                'Acquisition': self.config.acquisition_type,
                'Encoding': self.config.encoding_type,
                'Generator': self.config.generator_type
            })

    def save_results(self):
        """
        Save the optimization results to a CSV file.
        """
        # Create the csv directory if it doesn't exist
        self.get_seed_results()
        if not self.seed_results_df.empty:
            os.makedirs(os.path.join(self.plotter.output_dir, "csv"), exist_ok=True)
            seed_csv_path = os.path.join(self.plotter.output_dir, "csv", f"seed_{self.config.seed}_results.csv")
            self.seed_results_df.to_csv(seed_csv_path, index=False)
            self.logger.info(f"Seed {self.config.seed} results saved to {seed_csv_path}")