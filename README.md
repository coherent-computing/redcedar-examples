# Red Cedar Examples

Example notebooks demonstrating the Red Cedar quantum machine learning pipeline.

## Prerequisites

Install the Red Cedar client:

```bash
cd ../redcedar
pip install -e ".[plotting]"
```

The client needs only `requests`; `[plotting]` adds matplotlib, which the notebooks use for their plots. The client's functions, settings and return values are documented in the [redcedar README](https://github.com/coherent-computing/redcedar#readme).

Set your server connection:

```bash
export REDCEDAR_URL="http://localhost:8000"       # local server
export REDCEDAR_API_KEY="your-api-key"            # API key
```

Or configure directly in the notebook:

```python
import redcedar
client = redcedar.connect(url="http://localhost:8002", api_key="your-key")
```

## Notebooks

Start with `01_quick_start.ipynb`; the others go deeper into one topic each and can be read in any order.

| Notebook | Description |
|----------|-------------|
| **01_quick_start.ipynb** | The basic workflow: train a model, plot the training curves, test it on held-out data, and see which features it relies on (Explainable AI) |
| **02_dataset_analysis.ipynb** | Understand a dataset before training: PCA, mutual information and quantum resource estimation with plots, comparing two datasets, and analyzing your own CSV file |
| **03_training_settings.ipynb** | Training settings, one section per setting: `seed` (repeatable runs) and `op_direction` (order of the operators) |
| **04_training_in_depth.ipynb** | Layers and layer chaining, test accuracy over the course of training, per-class results, choosing the train and test data, reusing trained models, and checking on or stopping a running job |
| **01_full_pipeline.ipynb** | The earlier all-in-one example (analysis, training, inference, Explainable AI in one notebook); covered now by 01 and 02 |

## About

These examples use the Red Cedar Python client to interact with a Red Cedar server. All quantum computation happens on the server — these notebooks require no quantum libraries.
