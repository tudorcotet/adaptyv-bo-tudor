from dataclasses import dataclass
from typing import List, Optional

@dataclass
class OptimizationConfig:
    """
    Configuration class for optimization parameters.

    Attributes:
        n_iterations (int): Number of optimization iterations. Default is 10.
        n_candidates (int): Number of candidate solutions to generate. Default is 100000.
        n_initial (int): Number of initial random samples. Default is 10.
        seq_length (int): Length of the sequence to optimize. Default is 4.
        beta (float): Exploration-exploitation trade-off parameter for UCB. Default is 2.0.
        n_training_iter (int): Number of training iterations for the surrogate model. Default is 50.
        query_method (str): Method for querying the benchmark data. Default is 'identical'.
        mode (str): Optimization mode. Default is 'benchmark'.
        n_seeds (int): Number of random seeds for multiple runs. Default is 1.
        use_gpu (bool): Whether to use GPU for computations. Default is False.
        use_modal (bool): Whether to use modal computations. Default is False.
        modal_endpoint (str): Endpoint for modal computations. Default is 'http://localhost:8000'.
        alphabet (str): String of valid characters for sequences. Default is 'ACDEFGHIKLMNPQRSTVWY'.
        seed (int): Random seed for reproducibility. Default is 0.
        surrogate_type (str): Type of surrogate model to use. Default is 'gp'.
        acquisition_type (str): Type of acquisition function to use. Default is 'ucb'.
        encoding_type (str): Type of sequence encoding to use. Default is 'one_hot'.
        benchmark_file (str): Path to the benchmark data file. Default is '/Users/tudorcotet/Desktop/small_gb1.csv'.
        batch_size (int): Batch size for optimization. Default is 5.
        generator_type (str): Type of candidate generator to use. Default is 'benchmark'.
        indices_to_mutate (Optional[List[int]]): Specific indices to mutate in the sequence. Default is None.
    """

    n_iterations: int = 10
    n_candidates: int = 10
    n_initial: int = 10
    seq_length: int = 4
    beta: float = 2.0
    n_training_iter: int = 10
    query_method: str = 'identical'
    mode: str = 'benchmark'
    n_seeds: int = 1
    use_gpu: bool = False
    use_modal: bool = False
    modal_endpoint: str = 'http://localhost:8000'
    alphabet: str = 'ACDEFGHIKLMNPQRSTVWY'
    seed: int = 0
    surrogate_type: str = 'gp'
    acquisition_type: str = 'ucb'
    encoding_type: str = 'one_hot'
    benchmark_file: str = '/Users/tudorcotet/Desktop/small_gb1.csv'
    batch_size: int = 5
    generator_type: str = 'benchmark'
    indices_to_mutate: Optional[List[int]] = None
