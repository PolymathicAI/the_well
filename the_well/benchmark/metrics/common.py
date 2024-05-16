import numpy as np
import torch


def metric(f):
    """
    Decorator for metrics that standardizes the input arguments and checks the dimensions of the input tensors.
    Parameters
    ----------
    f : function
        Metric function that takes in the following arguments:
        x : torch.Tensor | np.ndarray
            Input tensor.
        y : torch.Tensor | np.ndarray
            Target tensor.
        meta : GenericWellMetadata
            Metadata for the dataset.
        **kwargs : dict
            Additional arguments for the metric.
    """

    def metric_wrapper(*args, **kwargs):
        assert len(args) >= 3, "At least three arguments required (x, y, and meta)"
        x, y, meta = args[:3]

        # Convert x and y to torch.Tensor if they are np.ndarray
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y)
        assert isinstance(x, torch.Tensor), "x must be a torch.Tensor or np.ndarray"
        assert isinstance(y, torch.Tensor), "y must be a torch.Tensor or np.ndarray"

        # Check dimensions
        spatial_ndims = meta.spatial_ndims
        assert (
            x.ndim >= spatial_ndims + 1
        ), "x must have at least spatial_ndims + 1 dimensions"
        assert (
            y.ndim >= spatial_ndims + 1
        ), "y must have at least spatial_ndims + 1 dimensions"

        return f(x, y, meta, **kwargs)

    return metric_wrapper
