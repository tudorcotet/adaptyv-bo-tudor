from abc import ABC, abstractmethod
from typing import List, Dict, Any
from config.optimization import QueryConfig
import logging
from scipy.spatial.distance import cityblock
from Levenshtein import distance as levenshtein_distance

class BaseQuery(ABC):
    """
    Abstract base class for query operations.

    This class defines the interface for query operations and provides
    common initialization logic.

    Attributes:
        config (OptimizationConfig): Configuration for optimization.
    """

    def __init__(self, config: QueryConfig):
        """
        Initialize the BaseQuery.

        Args:
            config (OptimizationConfig): Configuration for optimization.
        """
        self.config = config

    @abstractmethod
    def query(self, candidates: List[str]) -> List[float]:
        """
        Abstract method to perform a query operation.

        This method should be implemented by subclasses to define
        specific query behavior.

        Args:
            candidates (List[str]): The candidates to query.

        Returns:
            List[float]: The fitness values for the queried candidates.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        pass

class BenchmarkQuery(BaseQuery):
    """
    Concrete implementation of BaseQuery for benchmark queries.

    This class implements the query method for benchmark data.

    Attributes:
        benchmark_data (Dict[str, float]): The benchmark data mapping sequences to fitness values.
        query_method (str): The method to use for querying (e.g., "identical", "l1_distance", "levenshtein").
        logger (logging.Logger): Logger for this class.
    """

    def __init__(self, config: QueryConfig, benchmark_data: Dict[str, float]):
        """
        Initialize the BenchmarkQuery.

        Args:
            config (QueryConfig): Configuration for query.
            benchmark_data (Dict[str, float]): The benchmark data mapping sequences to fitness values.
        """
        super().__init__(config)
        self.benchmark_data = benchmark_data
        self.query_method = config.query_method
        self.logger = logging.getLogger(f"BenchmarkQuery")
        
    def query(self, candidates: List[str]) -> List[float]:
        """
        Query the fitness values for the given candidates.

        Args:
            candidates (List[str]): The candidates to query.

        Returns:
            List[float]: The fitness values for the queried candidates.

        Raises:
            ValueError: If an invalid query method is specified.
        """
        self.logger.info(f"Querying candidates: {candidates}")
        if self.query_method == "identical":
            fitness_values = [self.benchmark_data.get(c, 0) for c in candidates]
        elif self.query_method == "l1_distance":
            fitness_values = [max(self.benchmark_data.values()) - min(cityblock(list(c), list(k)) for k in self.benchmark_data.keys()) for c in candidates]
        elif self.query_method == "levenshtein":
            fitness_values = [max(self.benchmark_data.values()) - min(levenshtein_distance(c, k) for k in self.benchmark_data.keys()) for c in candidates]
        else:
            raise ValueError(f"Invalid query method: {self.query_method}")

        self.logger.info(f"Fitness values: {fitness_values}")
        return fitness_values
