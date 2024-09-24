from abc import ABC, abstractmethod
from typing import List
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from config.optimization import OptimizationConfig
from encoding.base import BaseEncoding

class BasePlotter(ABC):
    """
    Abstract base class for plotting optimization results.

    This class defines the interface for plotting various aspects of the optimization process,
    including sequence embeddings, fitness progression, and average results across multiple runs.

    Attributes:
        config (OptimizationConfig): Configuration object containing optimization parameters.
    """

    def __init__(self, config: OptimizationConfig):
        """
        Initialize the BasePlotter object.

        Args:
            config (OptimizationConfig): Configuration object containing optimization parameters.
        """
        self.config = config

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
    def plot_average_results(self, all_max_fitness: List[List[float]]):
        """
        Plot the average results across multiple optimization runs.

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
        config (OptimizationConfig): Configuration object containing optimization parameters.
        encoding (BaseEncoding): Encoding object for sequence encoding.
        output_dir (str): Directory path for saving output files.
    """

    def __init__(self, config: OptimizationConfig, encoding: BaseEncoding):
        """
        Initialize the SimplePlotter object.

        Args:
            config (OptimizationConfig): Configuration object containing optimization parameters.
            encoding (BaseEncoding): Encoding object for sequence encoding.
        """
        super().__init__(config)
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

    def plot_max_fitness(self, fitness_values: List[float], output_dir: str):
        """
        Plot the progression of maximum fitness values over optimization rounds.

        Args:
            fitness_values (List[float]): List of maximum fitness values for each round.
        """
        max_fitness_acquired = [max(fitness_values[:i+1]) for i in range(len(fitness_values))]
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(max_fitness_acquired)), max_fitness_acquired)
        plt.xlabel('Round')
        plt.ylabel('Max Fitness')
        plt.title('Max Fitness of All Acquired Sequences Until and Including the Round')
        plt.savefig(os.path.join(output_dir, "plots", 'max_fitness.png'))
        plt.close()

    def plot_average_results(self, all_max_fitness: List[List[float]], output_dir: str):
        """
        Plot the average max fitness of all acquired sequences until and including the round, averaged across seeds with standard deviation.

        Args:
            all_max_fitness (List[List[float]]): List of maximum fitness progressions for each run.
        """
        max_fitness_acquired = [np.maximum.accumulate(fitness) for fitness in all_max_fitness]
        avg_max_fitness = np.mean(max_fitness_acquired, axis=0)
        std_max_fitness = np.std(max_fitness_acquired, axis=0)

        plt.figure(figsize=(10, 6))
        plt.plot(range(len(avg_max_fitness)), avg_max_fitness, label='Average')
        plt.fill_between(range(len(avg_max_fitness)),
                         avg_max_fitness - std_max_fitness,
                         avg_max_fitness + std_max_fitness,
                         alpha=0.2)
        plt.xlabel('Round')
        plt.ylabel('Max Fitness')
        plt.title('Average Max Fitness of All Acquired Sequences Until and Including the Round (with std dev)')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'average_max_fitness.png'))
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


