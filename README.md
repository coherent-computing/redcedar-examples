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
export REDCEDAR_URL="http://localhost:8002"       # local server
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
| **01_quick_start.ipynb** | The basic workflow on your own data, as in the Quick Start guide: train a model on a CSV file, plot the training curves, predict new rows from a second file, and see which features the model relies on (Explainable AI) |
| **02_dataset_analysis.ipynb** | Understand a dataset before training: PCA, mutual information and quantum resource estimation with plots, comparing two datasets, and analyzing your own CSV file |
| **03_training_settings.ipynb** | Training settings, one section per setting: `seed` (repeatable runs) and `op_direction` (order of the operators) |
| **04_training_in_depth.ipynb** | Layers and layer chaining, test accuracy over the course of training, per-class results and testing on the rows the server set aside (`training.evaluate()`), choosing the train and test data, reusing trained models, and checking on or stopping a running job |
| **01_full_pipeline.ipynb** | The earlier all-in-one example (analysis, training, inference, Explainable AI in one notebook); covered now by 01 and 02 |

## Data

`data/wdbc_train.csv` (455 rows) and `data/wdbc_new.csv` (114 rows) are the Breast Cancer Wisconsin (Diagnostic) dataset, split once at random into 80% and 20% with the same mix of diagnoses in both (scikit-learn `train_test_split`, `stratify` on the diagnosis, `random_state=42`). Each row has 30 measurements of a tumor sample and the diagnosis, `M` (malignant) or `B` (benign), in the last column.

Source: W. Wolberg, O. Mangasarian, N. Street and W. Street, *Breast Cancer Wisconsin (Diagnostic)*, UCI Machine Learning Repository, 1993, [doi.org/10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B). Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## About

These examples use the Red Cedar Python client to interact with a Red Cedar server. All quantum computation happens on the server — these notebooks require no quantum libraries.
