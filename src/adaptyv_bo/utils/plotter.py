from abc import ABC, abstractmethod
from typing import List
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from encoding.base import BaseEncoding
import pandas as pd

class BasePlotter(ABC):
    """
    Abstract base class for plotting optimization results.

    This class defines the interface for plotting various aspects of the optimization process,
    including sequence embeddings, fitness progression, and average results across multiple runs.

    """

    def __init__(self):
        pass

    @abstractmethod
    def plot_embeddings(self, sequences: List[str], fitness_values: List[float], rounds: List[int]):
        """
        Plot the embeddings of sequences along with their fitness values.

        Args:
            sequences (List[str]): List of sequences to plot.
            fitness_values (List[float]): Corresponding fitness values for each sequence.
            rounds (List[int]): The optimization round for each sequence.
        """
        pass

    @abstractmethod
    def plot_max_fitness(self, fitness_values: List[float]):
        """
        Plot the progression of maximum fitness values over optimization rounds.

        Args:
            fitness_values (List[float]): List of maximum fitness values for each round.
        """
        pass

    @abstractmethod
    def plot_max_average_fitness(self, all_max_fitness: List[List[float]]):
        """
        Plot the maximum average fitness of all acquired variants until and including each round.

        Args:
            all_max_fitness (List[List[float]]): List of maximum fitness progressions for each run.
        """
        pass
    
    @abstractmethod
    def plot_training_loss(self, training_loss: List[float], validation_loss: List[float]):
        """
        Plot the training and validation loss over epochs.

        Args:
            training_loss (List[float]): List of training loss values for each epoch.
            validation_loss (List[float]): List of validation loss values for each epoch.
        """
        pass

    @abstractmethod
    def plot_validation_loss(self, validation_loss: List[float]):
        """
        Plot the validation loss over epochs.

        Args:
            validation_loss (List[float]): List of validation loss values for each epoch.
        """
        pass

    @abstractmethod
    def plot_training_metrics(self, training_metrics: List[float], validation_metrics: List[float]):
        """
        Plot the training and validation metrics over epochs.

        Args:
            training_metrics (List[float]): List of training metric values for each epoch.
            validation_metrics (List[float]): List of validation metric values for each epoch.
        """
        pass

    @abstractmethod
    def plot_validation_metrics(self, validation_metrics: List[float]):
        """
        Plot the validation metrics over epochs.

        Args:
            validation_metrics (List[float]): List of validation metric values for each epoch.
        """
        pass


class SimplePlotter(BasePlotter):
    """
    A simple implementation of the BasePlotter for visualizing optimization results.

    This class provides methods to plot embeddings, max fitness progression, and average results.

    Attributes:
        encoding (BaseEncoding): Encoding object for sequence encoding.
        output_dir (str): Directory path for saving output files.
    """

    def __init__(self, encoding: BaseEncoding):
        """
        Initialize the SimplePlotter object.

        Args:
            encoding (BaseEncoding): Encoding object for sequence encoding.
        """
        super().__init__()
        self.encoding = encoding
    
    def plot_embeddings(self, sequences: List[str], fitness_values: List[float], rounds: List[int], output_dir: str):
        """
        Plot the embeddings of sequences along with their fitness values.

        Args:
            sequences (List[str]): List of sequences to plot.
            fitness_values (List[float]): Corresponding fitness values for each sequence.
            rounds (List[int]): The optimization round for each sequence.
            output_dir (str): Directory path for saving output files.
        """
        encoded = self.encoding.encode(sequences)
        pca = PCA(n_components=2)
        embedded = pca.fit_transform(encoded)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(embedded[:, 0], embedded[:, 1], fitness_values, c=rounds, cmap='viridis')
        ax.set_xlabel('PCA 1')
        ax.set_ylabel('PCA 2')
        ax.set_zlabel('Fitness')
        ax.set_title('Protein Embeddings')
        plt.colorbar(scatter, label='Round')
        plt.savefig(os.path.join(output_dir, "plots", 'embeddings.png'))
        plt.close()

    def plot_max_fitness(self, results_df: pd.DataFrame, output_dir: str):
        """
        Plot the progression of maximum fitness values over optimization rounds.

        Args:
            results_df (pd.DataFrame): DataFrame containing the optimization results.
            output_dir (str): Directory to save the plot.
        """
        # Group by Round and calculate max fitness for each round
        max_fitness_per_round = results_df.groupby('Round')['Fitness'].max()

        # Calculate cumulative max fitness
        cumulative_max_fitness = max_fitness_per_round.cummax()

        plt.figure(figsize=(10, 6))
        plt.plot(cumulative_max_fitness.index, cumulative_max_fitness.values, label='Max Fitness Overall')
        plt.plot(max_fitness_per_round.index, max_fitness_per_round.values, label='Max Fitness per Round')
        plt.xlabel('Round')
        plt.ylabel('Max Fitness')
        plt.title('Max Fitness Progression')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "plots", 'max_fitness.png'))
        plt.close()

    def plot_max_average_fitness(self, all_max_fitness: List[List[float]], output_dir: str):
        """
        Plot the maximum average fitness of all acquired variants until and including each round.

        Args:
            all_max_fitness (List[List[float]]): List of maximum fitness progressions for each run.
            output_dir (str): Directory path for saving output files.
        """
        # Ensure all_max_fitness is a 2D array with equal length sublists
        max_length = max(len(fitness) for fitness in all_max_fitness)
        padded_all_max_fitness = [fitness + [fitness[-1]] * (max_length - len(fitness)) for fitness in all_max_fitness]

        # Calculate the maximum average fitness up to each round
        max_avg_fitness = [np.mean([max(run[:i+1]) for run in padded_all_max_fitness]) for i in range(max_length)]
        
        # Calculate the standard deviation
        std_max_fitness = [np.std([max(run[:i+1]) for run in padded_all_max_fitness]) for i in range(max_length)]

        plt.figure(figsize=(10, 6))
        plt.plot(range(len(max_avg_fitness)), max_avg_fitness, label='Max Average Fitness')
        plt.fill_between(range(len(max_avg_fitness)),
                         np.array(max_avg_fitness) - np.array(std_max_fitness),
                         np.array(max_avg_fitness) + np.array(std_max_fitness),
                         alpha=0.2, label='Standard Deviation')
        plt.xlabel('Round')
        plt.ylabel('Max Average Fitness')
        plt.title('Maximum Average Fitness of All Acquired Variants Until and Including Each Round')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'max_average_fitness.png'))
        plt.close()

    def plot_total_embeddings(self, sequences: List[str], fitness_values: List[float], rounds: List[int], output_dir: str):
        encoded = self.encoding.encode(sequences)
        pca = PCA(n_components=2)
        embedded = pca.fit_transform(encoded)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(embedded[:, 0], embedded[:, 1], fitness_values, c=rounds, cmap='viridis')
        ax.set_xlabel('PCA 1')
        ax.set_ylabel('PCA 2')
        ax.set_zlabel('Fitness')
        ax.set_title('Protein Embeddings')
        plt.colorbar(scatter, label='Fitness')
        plt.savefig(os.path.join(output_dir, 'total_embeddings.png'))
        plt.close()

    def plot_training_loss(self, training_loss: List[float], validation_loss: List[float], output_dir: str):
        """
        Plot the training and validation loss over epochs.

        Args:
            training_loss (List[float]): List of training loss values for each epoch.
            validation_loss (List[float]): List of validation loss values for each epoch.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(training_loss)), training_loss, label='Training Loss')
        plt.plot(range(len(validation_loss)), validation_loss, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.savefig(os.path.join(output_dir, "plots", 'training_loss.png'))
        plt.close()

    def plot_validation_loss(self, validation_loss: List[float], output_dir: str):
        """
        Plot the validation loss over epochs.

        Args:
            validation_loss (List[float]): List of validation loss values for each epoch.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(validation_loss)), validation_loss, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Validation Loss')
        plt.legend()
        plt.savefig(os.path.join(output_dir, "plots", 'validation_loss.png'))
        plt.close()

    def plot_training_metrics(self, training_metrics: List[float], validation_metrics: List[float], output_dir: str):
        """
        Plot the training and validation metrics over epochs.

        Args:
            training_metrics (List[float]): List of training metric values for each epoch.
            validation_metrics (List[float]): List of validation metric values for each epoch.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(training_metrics)), training_metrics, label='Training Metrics')
        plt.plot(range(len(validation_metrics)), validation_metrics, label='Validation Metrics')
        plt.xlabel('Epoch')
        plt.ylabel('Metrics')
        plt.title('Training and Validation Metrics')
        plt.legend()
        plt.savefig(os.path.join(output_dir, "plots", 'training_metrics.png'))
        plt.close()

    def plot_validation_metrics(self, validation_metrics: List[float], output_dir: str):
        """
        Plot the validation metrics over epochs.

        Args:
            validation_metrics (List[float]): List of validation metric values for each epoch.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(validation_metrics)), validation_metrics, label='Validation Metrics')
        plt.xlabel('Epoch')
        plt.ylabel('Metrics')
        plt.title('Validation Metrics')
        plt.legend()
        plt.savefig(os.path.join(output_dir, "plots", 'validation_metrics.png'))
        plt.close()

    def plot_average_max_fitness(self, combined_results_df: pd.DataFrame, output_dir: str):
        """
        Plot the average of the maximum fitness per round across all seeds.

        Args:
            combined_results_df (pd.DataFrame): DataFrame containing results from all seeds.
            output_dir (str): Directory to save the plot.
        """
        # Calculate the maximum fitness for each round and seed
        max_fitness_per_round = combined_results_df.groupby(['Seed', 'Round'])['Fitness'].max().reset_index()

        # Calculate the cumulative maximum fitness for each seed
        max_fitness_per_round['CumulativeMaxFitness'] = max_fitness_per_round.groupby('Seed')['Fitness'].cummax()

        # Calculate the average of max fitness and cumulative max fitness across seeds for each round
        avg_max_fitness = max_fitness_per_round.groupby('Round')['Fitness'].mean()
        avg_cumulative_max_fitness = max_fitness_per_round.groupby('Round')['CumulativeMaxFitness'].mean()

        # Calculate the standard deviation for error bars
        std_max_fitness = max_fitness_per_round.groupby('Round')['Fitness'].std()
        std_cumulative_max_fitness = max_fitness_per_round.groupby('Round')['CumulativeMaxFitness'].std()

        plt.figure(figsize=(10, 6))
        
        # Plot average max fitness per round
        plt.errorbar(avg_max_fitness.index, avg_max_fitness.values, 
                     yerr=std_max_fitness.values, 
                     label='Avg Max Fitness per Round', 
                     color='blue', capsize=5)

        # Plot average cumulative max fitness
        plt.errorbar(avg_cumulative_max_fitness.index, avg_cumulative_max_fitness.values, 
                     yerr=std_cumulative_max_fitness.values, 
                     label='Avg Cumulative Max Fitness', 
                     color='red', capsize=5)

        plt.xlabel('Round')
        plt.ylabel('Average Max Fitness')
        plt.title('Average Maximum Fitness Progression Across Seeds')
        plt.legend()
        plt.grid(True)
        plt.xticks(range(0, 21, 2))  # Set x-axis ticks to show every 2 rounds
        plt.savefig(os.path.join(output_dir, "plots", 'average_max_fitness.png'))
        plt.close()


