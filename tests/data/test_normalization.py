import torch

from the_well.data import WellDataset
from the_well.data.normalization import ZScoreNormalization


def compute_mean_std(dataset, n_samples=100, seed=42):
    torch.manual_seed(seed)  # Ensures reproducibility in CI
    total_sum = 0.0
    total_sq_sum = 0.0
    total_count = 0

    n_samples = min(len(dataset), n_samples)
    indices = torch.randperm(len(dataset))[:n_samples]

    for idx in indices:
        sample = dataset[idx]["input_fields"]  # shape: [T, H, W, C]
        total_sum += sample.sum(dim=(0, 1, 2))  # sum over T, H, W
        total_sq_sum += (sample**2).sum(dim=(0, 1, 2))
        total_count += sample.shape[0] * sample.shape[1] * sample.shape[2]

    mean = total_sum / total_count
    var = total_sq_sum / total_count - mean**2
    std = var.sqrt()
    return mean, std


def test_normalization():
    base_path = "hf://datasets/polymathic-ai/"
    dataset_name = "turbulent_radiative_layer_2D"
    n_samples = 100  # Random subset for fast checking

    # Load datasets directly from Hugging Face
    train_dataset = WellDataset(
        well_base_path=base_path,
        well_dataset_name=dataset_name,
        well_split_name="train",
        n_steps_input=4,
        n_steps_output=1,
        use_normalization=True,
        normalization_type=ZScoreNormalization,
    )

    # Compute stats for random subset of train split
    mean_train, std_train = compute_mean_std(train_dataset, n_samples=n_samples)
    print("\n=== Train Set (Random Subset) Statistics After Normalization ===")
    print("Mean:", mean_train)
    print("Std:", std_train)

    # Assert mean ~ 0 and std ~ 1
    assert torch.allclose(mean_train, torch.zeros_like(mean_train), atol=0.1), (
        "Train mean is not close to 0."
    )
    assert torch.allclose(std_train, torch.ones_like(std_train), atol=0.1), (
        "Train std is not close to 1."
    )
