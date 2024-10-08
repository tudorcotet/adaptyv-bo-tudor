import pandas as pd
from typing import Dict, List
from config.optimization import OptimizationConfig
from surrogates.gp import GPSurrogate
from surrogates.random_forest import RandomForestSurrogate
from surrogates.cnn import CNN1DDropoutSurrogate
from surrogates.dnn import DropoutLinearSurrogate
from surrogates.bnn import BNNSurrogate
from surrogates.xgboost import XGBoostSurrogate
from surrogates.ensemble import DeepEnsembleSurrogate
from surrogates.dkl import DeepKernelGPSurrogate
from acquisitions.acquisitions import *
from encoding.onehot import *
from generator.generator import *
from config.optimization import *
import mlflow
import os

def load_benchmark_data(file_path: str) -> Dict[str, float]:
    """
    Load benchmark data from a CSV file.

    Args:
        file_path (str): Path to the CSV file containing benchmark data.

    Returns:
        Dict[str, float]: A dictionary mapping sequences to their fitness values.

    Raises:
        ValueError: If the CSV file doesn't contain 'fitness' and 'sequence' columns.
    """
    df = pd.read_csv(file_path)
    if 'fitness' not in df.columns or 'sequence' not in df.columns:
        raise ValueError("CSV file must contain 'fitness' and 'sequence' columns")
    return df

def load_benchmark_data_mlflow(file_path: str):
    """
    Load benchmark data from a CSV file.

    Args:
        mlflow_tracker (MLflowTracker): The MLflow tracker.

    Returns:
        Dict[str, float]: A dictionary mapping sequences to their fitness values.   
    """

    df = pd.read_csv(file_path)
    experiment_name = os.path.basename(file_path).split('.')[0]

    dataset = mlflow.data.from_pandas(df, source=file_path, name=experiment_name, targets="fitness")
    return dataset, experiment_name


def get_surrogate(config: SurrogateConfig):
    """
    Get the surrogate model based on the configuration.

    Args:
        config (OptimizationConfig): The optimization configuration.

    Returns:
        BaseSurrogate: An instance of the specified surrogate model.

    Raises:
        ValueError: If an unknown surrogate type is specified.
    """
    if config.surrogate_type == 'gp':
        return GPSurrogate(config)
    elif config.surrogate_type == 'cnn':
        return CNN1DDropoutSurrogate(config)
    elif config.surrogate_type == 'rf':
        return RandomForestSurrogate(config)
    elif config.surrogate_type == 'ensemble':
        return DeepEnsembleSurrogate(config)
    elif config.surrogate_type == 'dkl':
        return DeepKernelGPSurrogate(config)
    elif config.surrogate_type == 'dnn':
        return DropoutLinearSurrogate(config)
    elif config.surrogate_type == 'bnn':
        return BNNSurrogate(config)
    elif config.surrogate_type == 'xgboost':
        return XGBoostSurrogate(config)
    else:
        raise ValueError(f"Unknown surrogate type: {config.surrogate_type}")

def get_acquisition(config: AcquisitionConfig):
    """
    Get the acquisition function based on the configuration.

    Args:
        config (OptimizationConfig): The optimization configuration.

    Returns:
        BaseAcquisition: An instance of the specified acquisition function.

    Raises:
        ValueError: If an unknown acquisition type is specified.
    """
    if config.acquisition_type == 'ucb':
        return UCBAcquisition(config)
    elif config.acquisition_type == 'ei':
        return ExpectedImprovementAcquisition(config)
    elif config.acquisition_type == 'ts':
        return ThompsonSamplingAcquisition(config)
    elif config.acquisition_type == 'greedy':
        return GreedyAcquisition(config)
    elif config.acquisition_type == 'random':
        return RandomAcquisition(config)
    else:
        raise ValueError(f"Unknown acquisition type: {config.acquisition_type}")

def get_encoding(config: EncodingConfig):
    """
    Get the encoding method based on the configuration.

    Args:
        config (OptimizationConfig): The optimization configuration.

    Returns:
        BaseEncoding: An instance of the specified encoding method.

    Raises:
        ValueError: If an unknown encoding type is specified.
    """
    if config.encoding_type == 'onehot':
        return OneHotEncoding(config)
    # Add other encoding types here
    else:
        raise ValueError(f"Unknown encoding type: {config.encoding_type}")

def get_generator(config: GeneratorConfig, benchmark_data: Dict[str, float]) -> BaseGenerator:
    """
    Get the sequence generator based on the configuration.

    Args:
        config (GeneratorConfig): The generator configuration.
        benchmark_data (Dict[str, float]): The benchmark data.

    Returns:
        BaseGenerator: An instance of the specified generator.

    Raises:
        ValueError: If an unknown generator type is specified.
    """
    if config.generator_type == 'combinatorial':
        return CombinatorialGenerator(config, config.indices_to_mutate)
    elif config.generator_type == 'benchmark':
        return BenchmarkGenerator(config, benchmark_data)
    elif config.generator_type == 'mutation':
        return MutationGenerator(config)
    else:
        raise ValueError(f"Unknown generator type: {config.generator_type}")