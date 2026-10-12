import torch

from the_well.benchmark.trainer.training import (
    accumulate_batch_mean,
    fold_batch_losses,
    long_time_metrics,
)


def test_accumulate_batch_mean_averages_over_batches():
    """Folding N batches with denom=N yields their mean, not the last one."""
    acc = {}
    values = [1.0, 10.0, 100.0]
    for v in values:
        accumulate_batch_mean(acc, {"a": v}, len(values))
    assert acc["a"] == sum(values) / len(values)


def test_accumulate_batch_mean_is_not_an_update():
    """The regression this guards: merging with dict-update keeps only the last batch.

    ``time_logs`` used to be merged with ``|=``, so a per-timestep curve logged over
    several validation batches reflected whichever batch happened to come last, while
    the scalar losses beside it were averaged. This pins the difference.
    """
    curves = [torch.tensor([1.0, 2.0]), torch.tensor([9.0, 10.0])]
    folded, updated = {}, {}
    for c in curves:
        accumulate_batch_mean(folded, {"curve": c}, len(curves))
        updated |= {"curve": c}
    expected_mean = (curves[0] + curves[1]) / 2
    assert torch.allclose(folded["curve"], expected_mean)
    assert torch.allclose(updated["curve"], curves[-1])
    assert not torch.allclose(updated["curve"], expected_mean)


def test_accumulate_batch_mean_handles_tensors_and_new_keys():
    """Tensor values accumulate elementwise, and an unseen key starts from zero."""
    acc = {"seen": torch.tensor([1.0, 1.0])}
    accumulate_batch_mean(acc, {"seen": torch.tensor([3.0, 3.0])}, 1.0)
    accumulate_batch_mean(acc, {"fresh": torch.tensor([2.0, 4.0])}, 2.0)
    assert torch.allclose(acc["seen"], torch.tensor([4.0, 4.0]))
    assert torch.allclose(acc["fresh"], torch.tensor([1.0, 2.0]))


def test_accumulate_batch_mean_returns_the_accumulator():
    """It updates in place and returns the same object, so calls can be chained."""
    acc = {}
    out = accumulate_batch_mean(acc, {"a": 2.0}, 2.0)
    assert out is acc
    assert out["a"] == 1.0


def test_accumulate_batch_mean_leaves_disjoint_keys_untouched():
    """Batches that log different keys do not clobber one another."""
    acc = {}
    accumulate_batch_mean(acc, {"a": 4.0}, 2.0)
    accumulate_batch_mean(acc, {"b": 6.0}, 2.0)
    assert acc == {"a": 2.0, "b": 3.0}


def test_fold_batch_losses_averages_time_curves_over_batches():
    """The regression guard for the bug this PR fixes.

    Unlike the helper tests above, this exercises the ROUTING in validation_loop: which
    accumulator each metric's curve lands in, and whether it is averaged or overwritten.
    Reverting the time-log path to ``time_logs |= new_time_logs`` fails this test; it
    would pass every other test in this file.

    ``split_fn`` is injected, so no Trainer is constructed and the test does not depend
    on which attributes validation_loop happens to touch.
    """
    metric = long_time_metrics[0]
    curves = [torch.tensor([1.0, 2.0]), torch.tensor([9.0, 10.0])]

    def fake_split(sub_loss, k, dset_name, field_names):
        # (scalar losses, time curves) -- shaped like split_up_losses' return
        return {f"{dset_name}/{k}_T=all": sub_loss.mean()}, {
            f"{dset_name}/full_{k}_rollout": sub_loss
        }

    loss_dict, time_logs = {}, {}
    for c in curves:
        fold_batch_losses(
            {metric: c.unsqueeze(0)},
            fake_split,
            "ds",
            ["f"],
            len(curves),
            loss_dict,
            time_logs,
        )

    key = f"ds/full_{metric}_rollout"
    expected = (curves[0] + curves[1]) / 2
    assert (
        key in time_logs
    ), f"a long-horizon metric must reach time_logs; got {list(time_logs)}"
    assert torch.allclose(time_logs[key], expected)
    assert not torch.allclose(
        time_logs[key], curves[-1]
    ), "curve reflects only the last batch"


def test_fold_batch_losses_keeps_short_metrics_out_of_time_logs():
    """A metric that is not long-horizon contributes scalars only, no curve."""

    def fake_split(sub_loss, k, dset_name, field_names):
        return {f"{dset_name}/{k}_T=all": sub_loss.mean()}, {
            f"{dset_name}/{k}_rollout": sub_loss
        }

    loss_dict, time_logs = {}, {}
    fold_batch_losses(
        {"not_a_long_time_metric": torch.tensor([[1.0, 2.0]])},
        fake_split,
        "ds",
        ["f"],
        1.0,
        loss_dict,
        time_logs,
    )
    assert time_logs == {}
    assert loss_dict["ds/not_a_long_time_metric_T=all"] == 1.5
