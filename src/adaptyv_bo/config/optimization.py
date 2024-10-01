from dataclasses import dataclass
from typing import List, Optional

@dataclass
class SurrogateConfig:
    """
    Configuration class for surrogate model parameters.

    Attributes:
        surrogate_type (str): Type of surrogate model to use. Default is 'gp'.
        kernel_type (str): Type of kernel to use for the surrogate model (if GPs). Default is 'rbf'.
        n_training_iter (int): Number of training iterations for the surrogate model. Default is 10.
        input_dim (int): Input dimension for the surrogate model.
        output_dim (int): Output dimension for the surrogate model.
        hidden_dim (int): Hidden dimension for the surrogate model.
        n_models (int): Number of models for the surrogate model.
        dropout_rate (float): Dropout rate for the surrogate model.
        n_samples (int): Number of samples for the surrogate model.
        mc_samples (int): Number of Monte Carlo samples for the surrogate model.
        learning_rate (float): Learning rate for the surrogate model.
        n_epochs (int): Number of epochs for the surrogate model.
        n_estimators (int): Number of estimators for the surrogate model.
        max_depth (int): Maximum depth for the surrogate model.
        random_state (int): Random state for the surrogate model.
    """
    surrogate_type: str = 'gp'
    kernel_type: str = 'rbf'
    n_training_iter: int = 50
    input_dim: int = 10
    output_dim: int = 1
    hidden_dim: int = 64
    n_models: int = 5
    dropout_rate: float = 0.5
    n_samples: int = 100
    mc_samples: int = 50
    learning_rate: float = 0.001
    n_epochs: int = 100
    n_estimators: int = 100
    max_depth: int = 10
    random_state: int = 42

@dataclass
class EncodingConfig:
    """
    Configuration class for encoding parameters.
    """
    encoding_type: str = 'onehot'
    alphabet: str = 'ACDEFGHIKLMNPQRSTVWY'
    sequence_length: int = 10

@dataclass
class AcquisitionConfig:
    """
    Configuration class for acquisition function parameters.
    """
    acquisition_type: str = 'greedy'
    beta: float = 2.0
    xi: float = 0.01

@dataclass
class GeneratorConfig:
    """
    Configuration class for generator parameters.
    """
    generator_type: str = 'benchmark'
    indices_to_mutate: Optional[List[int]] = None
    n_candidates: str = 'all' #add option to generate all candidates once
    alphabet: str = 'ACDEFGHIKLMNPQRSTVWY'
    sequence_length: int = 10

@dataclass
class QueryConfig:
    """
    Configuration class for query parameters.
    """
    query_method: str = 'identical' 

@dataclass
class GeneralConfig:
    """
    Configuration class for general parameters.
    """
    n_initial: int = 10
    n_iterations: int = 20
    batch_size: int = 5
    n_seeds: int = 5
    use_gpu: bool = False

@dataclass
class ModalConfig:
    """
    Configuration class for modal parameters.
    """
    use_modal: bool = False
    modal_endpoint: str = 'http://localhost:8000'

@dataclass
class MLflowConfig:
    tracking_uri: str = "https://mlflow.internal.adaptyvbio.com/" 
    experiment_name: str = "2024/adaptyv_bo_tudor"
    log_params: bool = True
    log_model_summary: bool = True
    log_model_graph: bool = True
    log_model_weights: bool = True
    log_training_loss: bool = True
    log_validation_loss: bool = True
    log_training_metrics: bool = True
    log_validation_metrics: bool = True 
    log_figures: bool = True
    log_dataset: bool = True

    seed: int = 0

@dataclass
class DataConfig:
    """
    Configuration class for benchmark parameters.
    """
    benchmark_file: str = '/Users/tudorcotet/Documents/Adaptyv/adaptyv-bo-tudor/src/adaptyv_bo/data/datasets/gb1/gb1_3site.csv'


@dataclass
class OptimizationConfig:
    surrogate_config: SurrogateConfig = SurrogateConfig()   
    acquisition_config: AcquisitionConfig = AcquisitionConfig()
    generator_config: GeneratorConfig = GeneratorConfig()
    query_config: QueryConfig = QueryConfig()
    general_config: GeneralConfig = GeneralConfig() 
    mlflow_config: MLflowConfig = MLflowConfig()    
    data_config: DataConfig = DataConfig()
    encoding_config: EncodingConfig = EncodingConfig()

    output_dir = 'output'

