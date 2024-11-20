from copy import deepcopy
from unittest import TestCase

import torch

from the_well.data.augmentation import (
    RandomAxisFlip,
    RandomAxisPermute,
    RandomAxisRoll,
)


class TestAugmentation(TestCase):
    def test_random_axis_flip(self):
        T, H, W = 3, 5, 7
        mask = torch.tensor([True, False])
        signs = {
            0: torch.tensor(1),
            1: torch.tensor([-1, 1]),
            2: torch.tensor([[1, -1], [-1, 1]]),
        }

        before = {
            "variable_fields": {
                0: {"a": torch.randn(T, H, W)},
                1: {"b": torch.randn(T, H, W, 2)},
                2: {"c": torch.randn(T, H, W, 2, 2)},
            },
            "constant_fields": {
                0: {"d": torch.randn(H, W)},
                1: {"e": torch.randn(H, W, 2)},
                2: {"f": torch.randn(H, W, 2, 2)},
            },
        }

        after = RandomAxisFlip.flip(deepcopy(before), mask=mask)

        for key in ("variable_fields", "constant_fields"):
            for order in (0, 1, 2):
                for name in before[key][order]:
                    self.assertEqual(
                        after[key][order][name].shape,
                        before[key][order][name].shape,
                    )

                    if "variable" in key:
                        expected = before[key][order][name].flip(1) * signs[order]
                    else:
                        expected = before[key][order][name].flip(0) * signs[order]

                    self.assertTrue(
                        torch.allclose(
                            after[key][order][name],
                            expected,
                        )
                    )

    def test_random_axis_permute(self):
        T, L, H, W = 2, 5, 7, 11
        permutation = torch.tensor([2, 0, 1])

        before = {
            "variable_fields": {
                0: {"a": torch.randn(T, L, H, W)},
                1: {"b": torch.randn(T, L, H, W, 3)},
                2: {"c": torch.randn(T, L, H, W, 3, 3)},
            },
            "constant_fields": {
                0: {"d": torch.randn(L, H, W)},
                1: {"e": torch.randn(L, H, W, 3)},
                2: {"f": torch.randn(L, H, W, 3, 3)},
            },
        }

        after = RandomAxisPermute.permute(deepcopy(before), permutation=permutation)

        for key in ("variable_fields", "constant_fields"):
            for order in (0, 1, 2):
                for name in before[key][order]:
                    if "variable" in key:
                        expected = (T, W, L, H) + (3,) * order
                    else:
                        expected = (W, L, H) + (3,) * order

                    self.assertEqual(
                        after[key][order][name].shape,
                        expected,
                    )

                    if "variable" in key:
                        expected = before[key][order][name].movedim(
                            (3, 1, 2),
                            (1, 2, 3),
                        )
                    else:
                        expected = before[key][order][name].movedim(
                            (2, 0, 1),
                            (0, 1, 2),
                        )

                    if order == 0:
                        pass  # a 0-order tensor is not permuted
                    elif order == 1:
                        expected = expected[..., permutation]
                    elif order == 2:
                        expected = expected[..., permutation, :][..., :, permutation]

                    self.assertTrue(torch.allclose(after[key][order][name], expected))

    def test_random_axis_roll(self):
        T, L, H, W = 2, 5, 7, 11
        axes, shifts = [0, 2], [3, 5]

        before = {
            "variable_fields": {
                0: {"a": torch.randn(T, L, H, W)},
                1: {"b": torch.randn(T, L, H, W, 3)},
                2: {"c": torch.randn(T, L, H, W, 3, 3)},
            },
            "constant_fields": {
                0: {"d": torch.randn(L, H, W)},
                1: {"e": torch.randn(L, H, W, 3)},
                2: {"f": torch.randn(L, H, W, 3, 3)},
            },
        }

        after = RandomAxisRoll.roll(deepcopy(before), delta=dict(zip(axes, shifts)))

        for key in ("variable_fields", "constant_fields"):
            for order in (0, 1, 2):
                for name in before[key][order]:
                    self.assertEqual(
                        after[key][order][name].shape,
                        before[key][order][name].shape,
                    )

                    if "variable" in key:
                        expected = torch.roll(
                            before[key][order][name],
                            shifts=shifts,
                            dims=tuple(i + 1 for i in axes),
                        )
                    else:
                        expected = torch.roll(
                            before[key][order][name],
                            shifts=shifts,
                            dims=axes,
                        )

                    self.assertTrue(
                        torch.allclose(
                            after[key][order][name],
                            expected,
                        )
                    )
