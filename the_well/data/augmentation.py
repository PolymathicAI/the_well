"""Data augmentation and transformations."""

from abc import ABC, abstractmethod
from typing import Dict, Tuple

import torch

from .datasets import BoundaryCondition, TrajectoryData, TrajectoryMetadata


class Augmentation(ABC):
    """
    Abstract base class for data augmentation.

    Augmentations are applied to all tensors representing a piece of trajectory (fields,
    scalars, boundary conditions and grids).
    """

    @abstractmethod
    def __call__(
        self, data: TrajectoryData, metadata: TrajectoryMetadata
    ) -> TrajectoryData:
        """

        Args:
            data:
                The input dictionary representing a piece of trajectory. The data dictionary
                always contains the 'variable_fields', 'constant_fields', 'variable_scalars'
                and 'constant_scalars' entries. 'variable_*' means that the content varies
                in time, while 'constant_*' means that the content is constant throughout
                the trajectory.

                - 'variable_fields' and 'constant_fields' entries are dictionaries whose
                entries are themselves name-field dictionaries split by tensor-order. The
                shape of a time-varying scalar field (0th-order) in a system with 2
                spatial dimensions would be (T, D_x, D_y), while a time-constant vector
                field (1st-order) would have a shape (D_x, D_y, 2).

                - 'variable_scalars' and 'constant_scalars' entries are name-scalar
                dictionaries. The shape of a time-varying scalar would be (T), while a
                time-constant scalar would have shape ().

                Additionally, the input dictionary can contain 'boundary_conditions',
                'space_grid' and 'time_grid' entries.

            metadata:
                Additional informations regarding the piece of trajectory, such as the file,
                sample and time indices ('file_idx', 'sample_idx', 'time_idx'), the time
                stride ('time_stride') and the dataset itself ('dataset').

        Returns:
            The updated data dictionary. The dictionary can be updated in-place, but its
            structure should remain the same.
        """
        pass


class Compose(Augmentation):
    r"""Composition of augmentations."""

    def __init__(self, *augmentations: Augmentation):
        super().__init__()

        self.augmentations = augmentations

    def __call__(
        self, data: TrajectoryData, metadata: TrajectoryMetadata
    ) -> TrajectoryData:
        for augmentation in self.augmentations:
            data = augmentation(data, metadata)

        return data


class RandomAxisFlip(Augmentation):
    """Flips the spatial axes randomly.

    Args:
        p:
            The probability of each axis to be flipped.
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(
        self, data: TrajectoryData, metadata: TrajectoryMetadata
    ) -> TrajectoryData:
        spatial = metadata.dataset.n_spatial_dims
        # i-th dim is flipped if mask[i] == True
        mask = torch.rand(spatial) < self.p

        return self.flip(data, mask)

    @staticmethod
    def flip(
        data: TrajectoryData,
        mask: torch.Tensor,  # BoolTensor
    ) -> TrajectoryData:
        mask = mask.long()
        # list of indices to be flipped
        axes: Tuple[int, ...] = tuple(mask.nonzero().flatten().tolist())

        if len(axes) == 0:
            return data

        for key in ("variable_fields", "constant_fields"):
            for order, fields in data[key].items():
                if order > 0:
                    # number of flips for each element of the N-th order tensor
                    masks = (mask for _ in range(order))
                    flips = sum(torch.meshgrid(*masks, indexing="ij"))
                    # an odd number of flips results in a sign change (-1)
                    # an even number of flips results in no sign change (1)
                    signs = 1 - 2 * (flips % 2)

                for name, field in fields.items():
                    if "variable" in key:
                        field = torch.flip(
                            field,
                            dims=tuple(i + 1 for i in axes),
                        )
                    else:
                        field = torch.flip(
                            field,
                            dims=axes,
                        )

                    if order > 0:
                        field = signs * field

                    fields[name] = field

        if "boundary_conditions" in data:
            bcs = data["boundary_conditions"].clone()
            for i in axes:
                bcs[i] = torch.flip(bcs[i], dims=(0,))
            data["boundary_conditions"] = bcs

        if "space_grid" in data:
            data["space_grid"] = torch.flip(
                data["space_grid"],
                dims=axes,
            )

        return data


class RandomAxisPermute(Augmentation):
    """Permutes the spatial axes randomly.

    Args:
        p:
            The probability of axes to be permuted.
    """

    def __init__(self, p: float = 1.0):
        self.p = p

    def __call__(
        self, data: TrajectoryData, metadata: TrajectoryMetadata
    ) -> TrajectoryData:
        spatial = metadata.dataset.n_spatial_dims

        if torch.rand(()) < self.p:
            permutation = torch.randperm(spatial)
        else:
            permutation = torch.arange(spatial)

        return self.permute(data, permutation)

    @staticmethod
    def permute(
        data: TrajectoryData,
        permutation: torch.Tensor,  # LongTensor
    ) -> TrajectoryData:
        spatial = len(permutation)
        src: Tuple[int, ...] = tuple(permutation.tolist())
        dst: Tuple[int, ...] = tuple(range(spatial))

        if src == dst:
            return data

        for key in ("variable_fields", "constant_fields"):
            for order, fields in data[key].items():
                for name, field in fields.items():
                    if "variable" in key:
                        field = torch.movedim(
                            field,
                            source=tuple(i + 1 for i in src),
                            destination=tuple(i + 1 for i in dst),
                        )
                    else:
                        field = torch.movedim(
                            field,
                            source=src,
                            destination=dst,
                        )

                    # permute each axis of the N-th order tensor
                    for i in range(order):
                        field = torch.index_select(
                            field,
                            index=permutation,
                            dim=field.ndim - i - 1,
                        )

                    fields[name] = field

        if "boundary_conditions" in data:
            data["boundary_conditions"] = data["boundary_conditions"][permutation]

        if "space_grid" in data:
            data["space_grid"] = torch.movedim(
                data["space_grid"],
                source=src,
                destination=dst,
            )

        return data


class RandomAxisRoll(Augmentation):
    """Rolls the periodic spatial axes randomly.

    Parameters
    ----------
    p :
        The probability of axes to be rolled.
    """

    def __init__(self, p: float = 1.0):
        self.p = p

    def __call__(
        self, data: TrajectoryData, metadata: TrajectoryMetadata
    ) -> TrajectoryData:
        shape = metadata.dataset.metadata.spatial_resolution

        bc = data["boundary_conditions"]

        periodic = torch.all(bc == BoundaryCondition.PERIODIC.value, dim=-1)
        periodic = periodic.nonzero().flatten().tolist()

        if torch.rand(()) < self.p:
            delta = {i: torch.randint(shape[i], size=()).item() for i in periodic}
        else:
            delta = {}

        return self.roll(data, delta)

    @staticmethod
    def roll(
        data: TrajectoryData,
        delta: Dict[int, int],
    ) -> TrajectoryData:
        axes = tuple(delta.keys())
        shifts = tuple(delta.values())

        if len(axes) == 0:
            return data

        for key in ("variable_fields", "constant_fields"):
            for _, fields in data[key].items():
                for name, field in fields.items():
                    if "variable" in key:
                        field = torch.roll(
                            field,
                            shifts=shifts,
                            dims=tuple(i + 1 for i in axes),
                        )
                    else:
                        field = torch.roll(
                            field,
                            shifts=shifts,
                            dims=axes,
                        )

                    fields[name] = field

        if "space_grid" in data:
            data["space_grid"] = torch.roll(
                data["space_grid"],
                shifts=shifts,
                dims=axes,
            )

        return data
