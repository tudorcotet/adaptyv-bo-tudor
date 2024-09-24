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
    """
    surrogate_type: str = 'gp'
    kernel_type: str = 'rbf'
    n_training_iter: int = 50   

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

@dataclass
class GeneratorConfig:
    """
    Configuration class for generator parameters.
    """
    generator_type: str = 'benchmark'
    indices_to_mutate: Optional[List[int]] = None
    n_candidates: int = 1000
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
    n_seeds: int = 1
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
    tracking_uri: str = "http://localhost:5060"
    experiment_name: str = "testing"
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
    benchmark_file: str = '/Users/tudorcotet/Desktop/small_gb1.csv'


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
