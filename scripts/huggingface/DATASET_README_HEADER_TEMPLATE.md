---
language:
- en
license: cc-by-4.0
tags:
- physics
{% if tag %}
- {{ tag }}
{% endif %}
task_categories:
- time-series-forecasting
- other
task_ids:
- multivariate-time-series-forecasting
---

# How To Load from HuggingFace Hub

1. Be sure to have `the_well` installed (`pip install the_well`)
2. Use the `WellDataModule` to retrieve data as follows:

```python
from the_well.data import WellDataModule

# The following line may take a couple of minutes to instantiate the datamodule
datamodule = WellDataModule(
    "hf://datasets/polymathic-ai/",
    "{{ dataset_name }}",
)
train_dataloader = datamodule.train_dataloader()

for batch in dataloader:
    # Process training batch
    ...
```
