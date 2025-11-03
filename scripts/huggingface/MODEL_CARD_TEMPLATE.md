---
arxiv: {{ arxiv }}
datasets: polymathic-ai/{{ dataset }}
tags:
- physics
---

# Benchmarking Models on the Well

[The Well](https://github.com/PolymathicAI/the_well) is a 15TB dataset collection of physics simulations. This model is part of the models that have been benchmarked on the Well.


The models have been trained for a fixed time of 12 hours or up to 500 epochs, whichever happens first. The training was performed on a NVIDIA H100 96GB GPU.
In the time dimension, the context length was set to 4. The batch size was set to maximize the memory usage. We experiment with 5 different learning rates for each model on each dataset.
We use the model performing best on the validation set to report test set results.

The reported results are here to provide a simple baseline. **They should not be considered as state-of-the-art**. We hope that the community will build upon these results to develop better architectures for PDE surrogate modeling.

{{ model_readme }}

## Loading the model from Hugging Face

To load the {{ model_name }} model trained on the `{{ dataset }}` of the Well, use the following commands.

```python
from the_well.benchmark.models import {{ model_name }}

model = {{ model_name }}.from_pretrained("polymathic-ai/{{ model_name }}-{{ dataset }}")
```
