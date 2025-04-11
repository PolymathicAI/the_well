import pytest

from the_well.utils.download import well_download


@pytest.fixture(
    scope="session", params=["active_matter", "turbulent_radiative_layer_2D"]
)
def download_dataset(tmp_path_factory, request):
    """Fixture to download a sample of a dataset from the Well.
    The scope of the fixture is session,
    so the dataset is downloaded only once per test session.
    """
    dataset_name = request.param
    data_dir = tmp_path_factory.mktemp("data")
    well_download(
        base_path=data_dir,
        dataset=dataset_name,
        split="train",
        first_only=True,
    )
    yield data_dir / "datasets" / dataset_name
