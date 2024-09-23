"""Example module docstring, this module shows how to type something with a dict and a custom type."""

from typing import Dict, List, Tuple, Union

import torch as pt
from attrs import define


@define
class DatasetStats:
    """A simple dataset stats class, mainly intended to show of how to type things."""

    mu: pt.Tensor
    std: pt.Tensor
    dims: Tuple[int, ...]


@define
class DatasetWrapper:
    """A simple dataset wrapper class, mainly intended to show of how to type things."""

    hyper_parameters: Dict[str, Union[int, float, str]]
    samples: List[Tuple[pt.Tensor, pt.Tensor]]

    @classmethod
    def mk_random(
        cls, hyper_parameters: Dict[str, Union[int, float, str]], num_samples: int
    ) -> "DatasetWrapper":
        """Create a random dataset wrapper."""
        shape = hyper_parameters.get("shape", (10, 5))  # Default shape if not provided
        return cls(
            hyper_parameters=hyper_parameters,
            samples=[(pt.randn(*shape), pt.randn(10)) for _ in range(num_samples)],
        )

    def compute_stats(
        self, over_dims: Tuple[int, ...] = (0,)
    ) -> Tuple[DatasetStats, DatasetStats]:
        """
        Compute the mean and standard deviation of the samples.

        This method first stacks all samples in the dataset, creating an intermediate tensor
        with an additional dimension at the beginning. Then it computes statistics over
        the specified dimensions.

        Args:
            over_dims (Tuple[int, ...]): The dimensions to compute statistics over.
                Default is (0,), which computes stats over the first dimension (samples).

        Returns:
            Tuple[DatasetStats, DatasetStats]: Statistics for X and Y data respectively.

        Example:
            If the shape in hyper_parameters is (10, 5) and there are 100 samples:
            - The intermediate stacked tensor for X will have shape (100, 10, 5)
            - If over_dims=(0,), the resulting stats will have shape (10, 5)
            - If over_dims=(0,1), the resulting stats will have shape (5,)
            - If over_dims=(0,1,2), the resulting stats will be scalars

            For Y data, which always has 10 elements:
            - The intermediate stacked tensor for Y will have shape (100, 10)
            - If over_dims=(0,), the resulting stats will have shape (10,)
            - If over_dims=(0,1), the resulting stats will be scalars
        """
        # stack both X,Y
        stackedX = pt.stack([sample[0] for sample in self.samples])
        stackedY = pt.stack([sample[1] for sample in self.samples])
        # compute mean and std along the specified dimensions
        muX = stackedX.mean(dim=over_dims, keepdim=False)
        stdX = stackedX.std(dim=over_dims, keepdim=False)
        muY = stackedY.mean(dim=over_dims, keepdim=False)
        stdY = stackedY.std(dim=over_dims, keepdim=False)
        dsX = DatasetStats(mu=muX, std=stdX, dims=over_dims)
        dsY = DatasetStats(mu=muY, std=stdY, dims=over_dims)
        return dsX, dsY
