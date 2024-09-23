"""Example of how to use pytests for parametric fixtures and test the DatasetWrapper class."""

import pytest
import torch as pt
from my_package.example import DatasetWrapper


@pytest.fixture(scope="class", params=[(7, 5), (11, 13)])
def dataset(request):
    """
    Parametrized fixture to create DatasetWrapper instances with different shapes.
    
    This fixture creates DatasetWrapper instances with two different shapes:
    (7, 5) and (11, 13).
    
    API Expectation:
    - DatasetWrapper.mk_random should accept a dictionary of hyper_parameters
      and a num_samples parameter to create random datasets.
    """
    shape = request.param
    hyper_parameters = {"shape": shape}
    wrapper = DatasetWrapper.mk_random(hyper_parameters, num_samples=5)
    yield wrapper
    # Clean up code (if needed) can go here


class TestDatasetWrapper:

    def test_dataset_shape(self, dataset):
        """
        Test that the dataset has the correct shape.
        
        Intention:
        Ensure that each sample in the dataset has the shape specified
        in the hyper_parameters.
        
        API Expectation:
        - DatasetWrapper.samples should be a list of tuples (X, Y)
        - X should have the shape specified in hyper_parameters["shape"]
        """
        expected_shape = dataset.hyper_parameters["shape"]
        for sample, _ in dataset.samples:
            assert sample.shape == expected_shape

    @pytest.mark.parametrize("over_dims", [(0,), (0, 1)])
    def test_compute_stats(self, dataset, over_dims):
        """
        Test the compute_stats method.
        
        Intention:
        Verify that the compute_stats method correctly calculates mean and
        standard deviation for both X and Y data with different aggregation dimensions,
        taking into account the stacking operation performed on the samples.
        
        API Expectation:
        - DatasetWrapper.compute_stats should return two DatasetStats objects
        - The shape of mu and std for X should match the input shape excluding the aggregated dimensions,
          considering the extra dimension added by stacking
        - The shape of mu and std for Y should be (10,) when aggregating over just the sample dimension,
          and () (scalar) when aggregating over both sample and Y dimensions
        - The computed statistics should match manual calculations on the stacked data
        """
        stats_x, stats_y = dataset.compute_stats(over_dims=over_dims)
        
        # Calculate expected shapes
        x_shape = dataset.hyper_parameters["shape"]
        y_shape = (10,)
        
        expected_shape_x = tuple(s for i, s in enumerate(x_shape) if i + 1 not in over_dims)
        expected_shape_y = tuple(s for i, s in enumerate(y_shape) if i + 1 not in over_dims)
        
        # Check that the stats have the correct shape
        assert stats_x.mu.shape == expected_shape_x, f"X mu shape mismatch. Expected {expected_shape_x}, got {stats_x.mu.shape}"
        assert stats_x.std.shape == expected_shape_x, f"X std shape mismatch. Expected {expected_shape_x}, got {stats_x.std.shape}"
        assert stats_y.mu.shape == expected_shape_y, f"Y mu shape mismatch. Expected {expected_shape_y}, got {stats_y.mu.shape}"
        assert stats_y.std.shape == expected_shape_y, f"Y std shape mismatch. Expected {expected_shape_y}, got {stats_y.std.shape}"

        # Check that the computed dimensions are correct
        assert stats_x.dims == over_dims
        assert stats_y.dims == over_dims

        # Additional checks to ensure the stats are computed correctly
        stacked_x = pt.stack([x for x, _ in dataset.samples])
        stacked_y = pt.stack([y for _, y in dataset.samples])
        
        assert pt.allclose(stats_x.mu, stacked_x.mean(dim=over_dims))
        assert pt.allclose(stats_x.std, stacked_x.std(dim=over_dims))
        assert pt.allclose(stats_y.mu, stacked_y.mean(dim=over_dims))
        assert pt.allclose(stats_y.std, stacked_y.std(dim=over_dims))

    def test_sample_count(self, dataset):
        """
        Test that the dataset has the correct number of samples.
        
        Intention:
        Ensure that the DatasetWrapper contains the specified number of samples.
        
        API Expectation:
        - DatasetWrapper.samples should be a list with length equal to num_samples
          specified in mk_random
        """
        assert len(dataset.samples) == 5  # As specified in mk_random

    def test_y_shape(self, dataset):
        """
        Test that the Y shape is always (10,).
        
        Intention:
        Verify that the Y component of each sample always has a shape of (10,),
        regardless of the X shape.
        
        API Expectation:
        - Each sample in DatasetWrapper.samples should be a tuple (X, Y)
        - Y should always have a shape of (10,)
        """
        for _, y in dataset.samples:
            assert y.shape == (10,)
