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
from utils.trackers.mlflow import MLflowTracker
from config.mlflow import MLflowConfig

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
        self.mlflow_config = config.mlflow_log_config
        self.tracker = MLflowTracker(self, surrogate, mlflow_config)
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

        with self.tracker:
            self.tracker.log_params({
                "acquisition_function": self.acquisition.__class__.__name__,
                "surrogate_model": self.surrogate.__class__.__name__,
                "encoding_method": self.encoding.__class__.__name__,
                "generator_type": self.generator.__class__.__name__
            })

            # Log initial metrics
            initial_fitness_values = self.fitness_values[:self.config.n_initial]
            self._log_metrics(0, initial_fitness_values, self.sequences[:self.config.n_initial], 
                              np.zeros(self.config.n_initial), 0.0, 0.0)

            for iteration in range(self.config.n_iterations):
                start_time = time.time()
                
                # Fit surrogate model
                train_loss, val_loss = self.fit_surrogate()
                
                # Generate and evaluate candidates
                candidates, new_fitness_values, acquisition_values = self.generate_and_evaluate_candidates()

                # Update sequences and metrics
                self.update_sequences_and_metrics(iteration, candidates, new_fitness_values, acquisition_values, train_loss, val_loss, start_time)

                self.logger.info(f"Iteration {iteration + 1} completed. Current max fitness: {self.max_fitness}")

            # Log final results and artifacts
            self.log_final_results()

        self.logger.info("Bayesian Optimization Loop completed")
        return self.sequences, self.fitness_values, self.rounds

    def fit_surrogate(self):
        X = torch.tensor(self.encoded_sequences, dtype=torch.float32)
        y = torch.tensor(self.fitness_values, dtype=torch.float32)
        train_loss, val_loss = self.surrogate.fit(X, y)
        return train_loss, val_loss

    def generate_and_evaluate_candidates(self):
        candidates = self.generator.generate(self.config.n_candidates)
        if not candidates:
            self.logger.warning(f"No new candidates generated. Using existing candidates.")
            candidates = self.sequences

        encoded_candidates = self.encoding.encode(candidates)
        mu, sigma = self.surrogate.predict(encoded_candidates)
        acquisition_values = self.acquisition.acquire(mu, sigma, self.max_fitness)

        idx_top = np.argsort(-acquisition_values)[:self.config.batch_size]
        selected_candidates = [candidates[i] for i in idx_top]
        new_fitness_values = self.query.query(selected_candidates)

        return selected_candidates, new_fitness_values, acquisition_values[idx_top]

    def update_sequences_and_metrics(self, iteration, candidates, new_fitness_values, acquisition_values, train_loss, val_loss, start_time):
        self.sequences.extend(candidates)
        self.encoded_sequences.extend(self.encoding.encode(candidates))
        self.fitness_values.extend(new_fitness_values)
        self.rounds.extend([iteration + 1] * len(candidates))

        iteration_time = time.time() - start_time
        self._log_metrics(iteration + 1, new_fitness_values, candidates, 
                          acquisition_values, train_loss, iteration_time)

        # Update max fitness and generator
        self.max_fitness = max(self.max_fitness, max(new_fitness_values))
        self.generator.update_sequences(candidates)

    def _log_metrics(self, iteration: int, fitness_values: List[float], sequences: List[str], 
                     acquisition_values: List[float], train_loss: float, iteration_time: float):
        metrics = {
            "max_fitness": max(fitness_values),
            "mean_fitness": np.mean(fitness_values),
            "std_fitness": np.std(fitness_values),
            "train_loss": train_loss,
            "iteration_time": iteration_time
        }
        for key, value in metrics.items():
            self.tracker.log_metric(key, value, step=iteration)

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
    
    def log_final_results(self):
        best_sequence, best_fitness = self.get_best_sequence()
        self.tracker.log_metric("final_max_fitness", best_fitness)
        self.tracker.log_params({"best_sequence": best_sequence})

        self.save_results_to_csv(os.path.join(self.plotter.output_dir, "csv", f"seed_{self.config.seed}_results.csv"))
        self.tracker.log_artifact(os.path.join(self.plotter.output_dir, f"log_seed_{self.config.seed}.txt"))
        self.tracker.log_artifact(os.path.join(self.plotter.output_dir, "csv", f"seed_{self.config.seed}_results.csv"))

        # Log figures
        self.tracker.log_figure(self.plotter.plot_embeddings(), "embeddings.png")
        self.tracker.log_figure(self.plotter.plot_max_fitness(), "max_fitness.png")
        self.tracker.log_figure(self.plotter.plot_training_loss(), "training_loss.png")
        self.tracker.log_figure(self.plotter.plot_validation_loss(), "validation_loss.png")
        self.tracker.log_figure(self.plotter.plot_training_metrics(), "training_metrics.png")
        self.tracker.log_figure(self.plotter.plot_validation_metrics(), "validation_metrics.png")
