import pytest

from the_well.utils.download import well_download


@pytest.fixture(
    scope="session", params=["active_matter", "turbulent_radiative_layer_2D"]
)
def downloaded_dataset(tmp_path_factory, request):
    dataset_name = request.param
    data_dir = tmp_path_factory.mktemp("data")
    well_download(
        base_path=data_dir,
        dataset=dataset_name,
        split="train",
        first_only=True,
    )
    yield data_dir / "datasets" / dataset_name
