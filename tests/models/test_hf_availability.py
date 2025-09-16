import pytest
from huggingface_hub import PyTorchModelHubMixin

from the_well.benchmark.models import FNO, TFNO, UNetClassic, UNetConvNext


@pytest.mark.parametrize("model_cls", [FNO, TFNO, UNetClassic, UNetConvNext])
@pytest.mark.parametrize("dataset", ["active_matter", "turbulent_radiative_layer_2D"])
def test_model_is_available_on_hf(model_cls: PyTorchModelHubMixin, dataset: str):
    model = model_cls.from_pretrained(f"polymathic-ai/{model_cls.__name__}-{dataset}")
    assert isinstance(model, model_cls)
