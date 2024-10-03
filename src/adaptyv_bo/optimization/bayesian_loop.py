import logging
import os
import numpy as np
import pandas as pd 
from typing import List, Tuple, Dict
from config.optimization import OptimizationConfig
from acquisitions.base import BaseAcquisition
from utils.query import BenchmarkQuery
from generator.generator import BaseGenerator, CombinatorialGenerator, BenchmarkGenerator, MutationGenerator
from surrogates.base import BaseSurrogate
from encoding.base import BaseEncoding
from utils.plotter import SimplePlotter
from utils.trackers.mlflow import MLflowTracker
import time
import torch
from utils.metrics import MetricsTracker, MaxFitness, AverageFitness, StandardDeviationFitness, Diversity, Coverage, ExpectedShortfall, ConditionalValueAtRisk
import mlflow
from mlflow.exceptions import MlflowException
from requests.exceptions import RequestException
from scipy.spatial.distance import pdist, squareform

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
                 plotter: SimplePlotter, output_dir: str, seed: int, mlflow_tracker: MLflowTracker):
        self.config = config
        self.acquisition = acquisition
        self.query = query
        self.generator = generator
        self.surrogate = surrogate
        self.encoding = encoding
        self.plotter = plotter
        self.mlflow_config = config.mlflow_config
        self.mlflow_tracker = mlflow_tracker
        self.sequences: List[str] = []
        self.encoded_sequences: List[np.ndarray] = []
        self.fitness_values: List[float] = []
        self.rounds: List[int] = []
        self.acquired_sequences: set = set()
        self.max_fitness = float('-inf')
        self.seed_results_df: pd.DataFrame = pd.DataFrame()
        self.seed = seed
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.seed_output_dir = os.path.join(output_dir, f"seed_{self.seed}")
        os.makedirs(self.seed_output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.seed_output_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(self.seed_output_dir, "csv"), exist_ok=True) 
        self.logger = self._setup_logger()
        self.parent_run_id = self.mlflow_tracker.parent_run_id
        self.child_run_id = None
        self.experiment_id = self.mlflow_tracker.experiment_id
        # Initialize MetricsTracker
        self.metrics_tracker = MetricsTracker({
            'max_fitness': MaxFitness(),
            'average_fitness': AverageFitness(),
            'std_fitness': StandardDeviationFitness(),
            'diversity': Diversity(),
            'coverage': Coverage(len(self.generator.all_candidates)),
            'expected_shortfall': ExpectedShortfall(),
            'cvar': ConditionalValueAtRisk()
        })
        self.loss_fn_name = config.surrogate_config.loss_fn  # Add this line
        self.diversity_values = []  # Add this line

    def _setup_logger(self) -> logging.Logger:
        """Set up and return a logger for the optimization process."""
        logger = logging.getLogger(f"BayesianOpt_Seed_{self.seed}")
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(os.path.join(self.seed_output_dir, f"log_seed_{self.seed}.txt"))
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger

    def initialize(self):
        """Initialize the optimization process with a set of initial sequences."""

        # Generate initial sequences
        initial_sequences = self.generator.generate_initial(self.config.general_config.n_initial)
        initial_fitness = self.query.query(initial_sequences)
        self.sequences.extend(initial_sequences)
        self.encoded_sequences.extend(self.encoding.encode(initial_sequences))
        self.fitness_values.extend(initial_fitness)
        self.rounds.extend([0] * len(initial_sequences))
        self.acquired_sequences.update(initial_sequences)
        self.max_fitness = max(initial_fitness)

        #update the generator with the initial sequences
        self.generator.update_sequences(initial_sequences)

        initial_diversity = self.calculate_diversity(initial_sequences)
        initial_diversity_quantiles = self.calculate_diversity_quantiles(initial_sequences)
        
        self._log_metrics(0, initial_fitness, initial_sequences, 
                          np.zeros(self.config.general_config.n_initial), 0.0, 0.0, 0.0, 
                          np.mean(initial_fitness), np.std(initial_fitness),
                          initial_diversity, initial_diversity_quantiles)

        # Start the MLflow run
        self.logger.info(f"Initialization complete. Max fitness: {self.max_fitness}")
        self.logger.info(f"Initial sequences: {initial_sequences}")
        self.logger.info(f"Initial fitness values: {initial_fitness}")

    def run(self) -> Tuple[List[str], List[float], List[int]]:
        """
        Run the Bayesian Optimization loop.

        Returns:
            Tuple[List[str], List[float], List[int]]: Lists of sequences, fitness values, and rounds.
        """
        
        child_run_name = f'seed_{self.seed}'

        try:
            with self.mlflow_tracker.start_child_run(child_run_name, self.parent_run_id) as child_run:
                self.child_run_id = child_run.info.run_id
                # Log seed run parameters
                self.log_with_retry(self.mlflow_tracker.log_params, {
                    "seed": self.seed,
                    "acquisition_type": self.config.acquisition_config.acquisition_type,
                    "surrogate_type": self.config.surrogate_config.surrogate_type,
                    "encoding_type": self.config.encoding_config.encoding_type,
                    "generator_type": self.config.generator_config.generator_type,
                    "kernel_type": self.config.surrogate_config.kernel_type,
                    "n_iterations": self.config.general_config.n_iterations,
                    "n_initial": self.config.general_config.n_initial,
                    "batch_size": self.config.general_config.batch_size,
                }, run_id=self.child_run_id)
                
                # Log seed run tags
                self.mlflow_tracker.set_tags(tags = {
                   "seed": self.seed,
                   "acquisition_type": self.config.acquisition_config.acquisition_type,
                   "surrogate_type": self.config.surrogate_config.surrogate_type,
                   "encoding_type": self.config.encoding_config.encoding_type,
                   "generator_type": self.config.generator_config.generator_type,
                   "kernel_type": self.config.surrogate_config.kernel_type,
                   "loss_function": self.config.surrogate_config.loss_fn,
                   "n_iterations": self.config.general_config.n_iterations,
                   "n_initial": self.config.general_config.n_initial,
                   "n_seeds": self.config.general_config.n_seeds,
                }, run_id=self.child_run_id)

        except Exception as e:
            self.logger.error(f"Failed to start MLflow run or log parameters: {e}")
            # Continue with the optimization process without MLflow logging

        self.logger.info("Starting Bayesian Optimization Loop")
        self.initialize()

        for iteration in range(self.config.general_config.n_iterations):
            start_time = time.time()
            
            # Fit surrogate model
            train_loss, val_loss = self.fit_surrogate()
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Generate and evaluate candidates
            candidates, new_fitness_values, acquisition_values = self.generate_and_evaluate_candidates()

            if not candidates:
                self.logger.warning(f"No candidates generated in iteration {iteration + 1}")
                continue
            
            # Calculate diversity
            diversity = self.calculate_diversity(self.sequences)
            diversity_quantiles = self.calculate_diversity_quantiles(self.sequences)

            # Update sequences and metrics
            self.update_sequences_and_metrics(iteration, candidates, new_fitness_values, acquisition_values, train_loss, val_loss, start_time)

            # Update metrics
            self.metrics_tracker.update(self.fitness_values, iteration)

            self._log_metrics(iteration + 1, new_fitness_values, candidates, 
                              acquisition_values, train_loss, val_loss, start_time,
                              np.mean(new_fitness_values), np.std(new_fitness_values), diversity, diversity_quantiles)

            self.logger.info(f"Iteration {iteration + 1} completed. Current max fitness: {self.max_fitness}")

        # Log final metrics for this seed
        self.log_final_results()

        return self.sequences, self.fitness_values, self.rounds

    def fit_surrogate(self):
        X = np.array(self.encoded_sequences)
        y = np.array(self.fitness_values)
        train_loss, val_loss = self.surrogate.fit(X, y)
        return train_loss, val_loss

    def generate_and_evaluate_candidates(self):
        candidates = self.generator.generate(self.config.generator_config.n_candidates)
        
        if not candidates:
            self.logger.warning(f"No new candidates generated. Using existing candidates.")
            candidates = self.sequences

        encoded_candidates = self.encoding.encode(candidates)
        mu, sigma = self.surrogate.predict(encoded_candidates)
        acquisition_values = self.acquisition.acquire(mu, sigma, self.max_fitness)

        idx_top = np.argsort(-acquisition_values)[:self.config.general_config.batch_size]
        selected_candidates = [candidates[i] for i in idx_top]
        new_fitness_values = self.query.query(selected_candidates)

        return selected_candidates, new_fitness_values, acquisition_values[idx_top]

    def update_sequences_and_metrics(self, iteration, candidates, new_fitness_values, acquisition_values, train_loss, val_loss, start_time):
        if candidates:  # Only update if there are new candidates
            self.sequences.extend(candidates)
            self.encoded_sequences.extend(self.encoding.encode(candidates))
            self.fitness_values.extend(new_fitness_values)
            self.rounds.extend([iteration + 1] * len(candidates))

            iteration_time = time.time() - start_time

            self.max_fitness = max(self.max_fitness, max(new_fitness_values))
            
            # Calculate diversity only for new candidates
            new_diversity = self.calculate_diversity(candidates)
            self.diversity_values.extend([new_diversity] * len(candidates))

            # Calculate overall diversity and quantiles (for logging purposes)
            overall_diversity = self.calculate_diversity(self.sequences)
            diversity_quantiles = self.calculate_diversity_quantiles(self.sequences)

            if len(new_fitness_values) > 0:
                mean_fitness = np.mean(new_fitness_values)
                std_fitness = np.std(new_fitness_values)
            else:
                mean_fitness = 0
                std_fitness = 0

            self._log_metrics(iteration + 1, new_fitness_values, candidates, 
                              acquisition_values, train_loss, val_loss, iteration_time,
                              mean_fitness, std_fitness, overall_diversity, diversity_quantiles)

            # Update max fitness and generator
            self.generator.update_sequences(candidates)
        else:
            self.logger.warning(f"No new candidates in iteration {iteration + 1}")

    def _log_metrics(self, iteration: int, fitness_values: List[float], sequences: List[str], 
                     acquisition_values: List[float], train_loss: float, val_loss: float, 
                     iteration_time: float, mean_fitness_current: float, std_fitness_current: float,
                     diversity: float, diversity_quantiles: Dict[str, float]):
        metrics = {
            "max_fitness_current": max(fitness_values) if fitness_values else 0,
            "mean_fitness_current": mean_fitness_current,
            "std_fitness_current": std_fitness_current,
            "max_fitness": self.max_fitness,
            "mean_fitness": np.mean(self.fitness_values) if self.fitness_values else 0,
            "std_fitness": np.std(self.fitness_values) if self.fitness_values else 0,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "iteration_time": iteration_time,
            "iteration": iteration,
            "total_evaluated_sequences": len(self.sequences),
            "diversity": diversity,
            "min_diversity": diversity_quantiles['min_diversity'],
            "max_diversity": diversity_quantiles['max_diversity'],
            "median_diversity": diversity_quantiles['median_diversity'],
            "q25_diversity": diversity_quantiles['q25_diversity'],
            "q75_diversity": diversity_quantiles['q75_diversity']
        }
        
        # Add fitness quantiles
        quantiles = self.calculate_fitness_quantiles()
        metrics.update(quantiles)

        try:
            for key, value in metrics.items():
                self.mlflow_tracker.log_metric(key, value, run_id=self.child_run_id, step=iteration)
        except Exception as e:
            self.logger.error(f"Failed to log metrics to MLflow: {e}")
            self.logger.error(f"Metrics: {metrics}")

    def calculate_fitness_quantiles(self):
        fitness_array = np.array(self.fitness_values)
        return {
            'min_fitness': np.min(fitness_array),
            'max_fitness': np.max(fitness_array),
            'median_fitness': np.median(fitness_array),
            'q25_fitness': np.percentile(fitness_array, 25),
            'q75_fitness': np.percentile(fitness_array, 75)
        }

    def calculate_diversity(self, sequences):
        encoded_sequences = self.encoding.encode(sequences)
        distances = pdist(encoded_sequences, metric='hamming')
        return np.mean(distances)

    def calculate_diversity_quantiles(self, sequences):
        encoded_sequences = self.encoding.encode(sequences)
        distances = pdist(encoded_sequences, metric='hamming')
        return {
            'min_diversity': np.min(distances),
            'max_diversity': np.max(distances),
            'median_diversity': np.median(distances),
            'q25_diversity': np.percentile(distances, 25),
            'q75_diversity': np.percentile(distances, 75)
        }

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
        self.get_results()  # Ensure the results DataFrame is populated
        self.plotter.plot_embeddings(self.sequences, self.fitness_values, self.rounds, self.seed_output_dir)
        self.plotter.plot_max_fitness(self.seed_results_df, self.seed_output_dir)
        self.plotter.plot_training_loss(self.train_losses, self.val_losses, self.seed_output_dir)
        self.plotter.plot_validation_loss(self.val_losses, self.seed_output_dir)    

    def get_results(self):
        """
        Get the results for the current seed.
        """
        if self.seed_results_df.empty:
            # Ensure all lists have the same length
            min_length = min(len(self.rounds), len(self.sequences), len(self.fitness_values), len(self.diversity_values))
            
            self.seed_results_df = pd.DataFrame({
                'Seed': [self.seed] * min_length,
                'Round': self.rounds[:min_length],
                'Sequence': self.sequences[:min_length],
                'Fitness': self.fitness_values[:min_length],
                'Diversity': self.diversity_values[:min_length],
                'Surrogate': [self.config.surrogate_config.surrogate_type] * min_length,
                'Kernel': [self.config.surrogate_config.kernel_type] * min_length,
                'Acquisition': [self.config.acquisition_config.acquisition_type] * min_length,
                'Encoding': [self.config.encoding_config.encoding_type] * min_length,
                'Generator': [self.config.generator_config.generator_type] * min_length,
                'Loss_Function': [self.loss_fn_name] * min_length,
            })

            if min_length < len(self.rounds):
                self.logger.warning(f"Some data was truncated. Original lengths: rounds={len(self.rounds)}, "
                                    f"sequences={len(self.sequences)}, fitness={len(self.fitness_values)}, "
                                    f"diversity={len(self.diversity_values)}. Used length: {min_length}")

    def save_results_to_csv(self):
        """
        Save the optimization results to a CSV file.
        """
        # Create the csv directory if it doesn't exist
        self.get_results()
        if not self.seed_results_df.empty:
            seed_csv_path = os.path.join(self.seed_output_dir, "csv", f"seed_{self.seed}_results.csv")
            self.seed_results_df.to_csv(seed_csv_path, index=False)
            self.logger.info(f"Seed {self.seed} results saved to {seed_csv_path}")
    
    def log_final_results(self):
        best_sequence, best_fitness = self.get_best_sequence()
        try:
            self.log_with_retry(self.mlflow_tracker.log_metric, "final_max_fitness", best_fitness, run_id=self.child_run_id)
            self.log_with_retry(self.mlflow_tracker.log_params, {
                "best_sequence": best_sequence,
                "loss_function": self.loss_fn_name  # Add this line
            }, run_id=self.child_run_id)

            # Log final metrics
            final_metrics = self.metrics_tracker.get_final_metrics()
            for metric_name, metric_value in final_metrics.items():
                self.log_with_retry(self.mlflow_tracker.log_metric, f"final_{metric_name}", metric_value, run_id=self.child_run_id)

        except Exception as e:
            self.logger.error(f"Failed to log final results to MLflow: {e}")

        # Save and log the CSV with all metrics and sequences
        self.save_results_to_csv()
        seed_csv_path = os.path.join(self.seed_output_dir, "csv", f"seed_{self.seed}_results.csv")

        if os.path.exists(seed_csv_path):
            self.mlflow_tracker.log_artifact(seed_csv_path, run_id=self.child_run_id)
        else:
            self.logger.error(f"File not found: {seed_csv_path}")

        # Save and log all plots
        self.plot_results()
        plot_dir = os.path.join(self.seed_output_dir, "plots")
        for root, _, files in os.walk(plot_dir):
            for file in files:
                if file.endswith(".png"):
                    self.mlflow_tracker.log_artifact(os.path.join(root, file), run_id=self.child_run_id)

        # Save and log the log file
        log_file_path = os.path.join(self.seed_output_dir, f"log_seed_{self.seed}.txt")
        if os.path.exists(log_file_path):
            self.mlflow_tracker.log_artifact(log_file_path, run_id=self.child_run_id)
        else:
            self.logger.error(f"File not found: {log_file_path}")

    def log_with_retry(self, log_func, *args, max_retries=5, delay=1, run_id: str):
        for attempt in range(max_retries):
            try:
                return log_func(*args, run_id=run_id)
            except (MlflowException, RequestException) as e:
                if attempt == max_retries - 1:
                    self.logger.error(f"Failed to log to MLflow after {max_retries} attempts: {e}")
                    return  # Return without raising an exception
                time.sleep(delay * (2 ** attempt))  # Exponential backoff

    def get_experiment_string(self, seed):
        return f"experiment@{self.config.mlflow_config.experiment_name}_time@{int(time.time())}_surrogate@{self.config.surrogate_config.surrogate_type}_acquisition@{self.config.acquisition_config.acquisition_type}_encoding@{self.config.encoding_config.encoding_type.replace('_','')}_generator@{self.config.generator_config.generator_type}_kernel@{self.config.surrogate_config.kernel_type}_loss@{self.loss_fn_name}_seed@{seed}"  # Add loss function to the experiment string