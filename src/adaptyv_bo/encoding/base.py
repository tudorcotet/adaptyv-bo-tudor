from abc import ABC, abstractmethod
from typing import List
import torch
from config.optimization import OptimizationConfig

class BaseEncoding(ABC):
    """
    Abstract base class for encoding and decoding sequences.

    This class defines the interface for encoding and decoding sequences
    used in the optimization process.

    Attributes:
        config (OptimizationConfig): Configuration object containing
            optimization parameters.
    """

    def __init__(self, config: OptimizationConfig):
        """
        Initialize the BaseEncoding object.

        Args:
            config (OptimizationConfig): Configuration object containing
                optimization parameters.
        """
        self.config = config

    @abstractmethod
    def encode(self, sequences: List[str]) -> torch.Tensor:
        """
        Encode a list of sequences into a tensor representation.

        Args:
            sequences (List[str]): List of sequences to encode.

        Returns:
            torch.Tensor: Encoded representation of the input sequences.
        """
        pass

    @abstractmethod
    def decode(self, encoded_sequences: torch.Tensor) -> List[str]:
        """
        Decode a tensor representation back into a list of sequences.

        Args:
            encoded_sequences (torch.Tensor): Encoded representation of sequences.

        Returns:
            List[str]: Decoded sequences as a list of strings.
        """
        pass
