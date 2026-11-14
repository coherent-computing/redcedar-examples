# Red Cedar Examples

Example notebooks and scripts that use Red Cedar, through its Python SDK (`redcedar`).

## Prerequisites

Install the SDK, with matplotlib for the plots:

```bash
pip install "redcedar[plotting]"
```

The SDK needs only `requests`; `[plotting]` adds matplotlib. Every SDK function is described in the [User Guide](https://coherent-computing.github.io/redcedar/user_guide/).

Set your server connection before starting Jupyter:

```bash
export REDCEDAR_URL="http://127.0.0.1:8002"       # a server on your own machine
export REDCEDAR_API_KEY="your-api-key"            # the key for that server
```

Or pass them in the notebook:

```python
import redcedar
client = redcedar.connect(url="http://127.0.0.1:8002", api_key="your-key")
```

Which key to use depends on the server: see [Connecting to a Server](https://coherent-computing.github.io/redcedar/user_guide/#connecting-to-a-server) in the User Guide.

## Notebooks

Start with `01_quick_start.ipynb`; the others go deeper into one topic each and can be read in any order. Each notebook is saved with the output of a run, so you can read the results and plots without running it.

| Notebook | Shows |
|----------|-------|
| **01_quick_start.ipynb** | The basic workflow, as in the Quick Start guide: train a quantum model on a CSV file of classified samples, classify the samples in a second file using the trained model, check the predicted classes, and see which features the model relies on (explainable AI) |
| **02_dataset_analysis.ipynb** | How to understand a dataset before training: PCA, mutual information and quantum resource estimation, comparing two datasets, and analyzing your own CSV file |
| **03_training_settings.ipynb** | How training settings change a model, one section per setting: `seed` (repeatable runs) and `op_direction` (order of the operators), the settings saved with each model, and settings kept in a file (`settings/quantum.json` and `settings/forest.json`, for the quantum model and a random forest) |
| **04_training_in_depth.ipynb** | A closer look at training: loss and accuracy during training, layers and layer chaining, test accuracy over the course of training, results per class and per layer, choosing the train and test data, reusing trained models, and checking on or stopping a running job |
| **05_regression.ipynb** | The Quick Start's steps for regression, where the model predicts a number: train on buildings with a known heating load, measure the model on its test rows (mean absolute error), predict the heating load of buildings in a second file, and check the predicted values. Regression does not work well yet in this release: the predictions are currently further off than guessing the average |
| **06_random_forest.ipynb** | The Quick Start's steps with a random forest, the classical model Red Cedar trains as a baseline: train it on the same file, classify the samples, check the predicted classes, and see which columns it relies on. Needs a 2.0 server |
| **07_quantum_and_forest.ipynb** | The quantum model and the random forest side by side, trained and tested on the same rows: test accuracy, the same samples classified by both, the samples where they disagree, and the features each relies on. Needs a 2.0 server |

## Scripts

| Script | Shows |
|--------|-------|
| **quick_start.py** | The same steps as `01_quick_start.ipynb`, from the command line. Run it from this folder (`python quick_start.py`); it prints the results and saves the plots as PNG files here |
| **quantum_and_forest.py** | The same comparison as `07_quantum_and_forest.ipynb`, from the command line (`python quantum_and_forest.py`); it prints the results. Needs a 2.0 server |
| **dataset_analysis.py** | PCA, mutual information and quantum resource estimation for one dataset, as in `02_dataset_analysis.ipynb`: a CSV file of your own (`python dataset_analysis.py my_data.csv`) or a catalog id (`python dataset_analysis.py iris`); WDBC from the catalog without one. It prints the results and saves the plots as PNG files here |

## Settings files

`settings/` holds two example settings files, used by `03_training_settings.ipynb`: `quantum.json` (small, quick settings for the quantum model) and `forest.json` (a random forest). A settings file holds settings only, not the data, so it works with any data file, from Python (`client.training.run("my_data.csv", settings="settings/forest.json")`) or from the command line (`redcedar train my_data.csv --settings settings/forest.json`).

## Data

### Breast cancer (classification)

The `wdbc_` files in `data/` are the Breast Cancer Wisconsin (Diagnostic) dataset, split once at random into 80% and 20% with the same mix of diagnoses in both (scikit-learn `train_test_split`, `stratify` on the diagnosis, `random_state=42`). Each row has 30 measurements of a tumor sample and its diagnosis, `M` (malignant) or `B` (benign).

- `wdbc_train.csv`: 455 classified samples, with their diagnosis in the last column, to train on.
- `wdbc_unclassified.csv`: the other 114 samples, measurements only. They stand in for samples whose class is not yet known.
- `wdbc_unclassified_actual.csv`: the actual diagnoses of those 114 samples, in the same order, used only to check the model's predicted classes.

Notebooks 02 to 04 use the same dataset from the server's catalog (`wdbc`).

Source: W. Wolberg, O. Mangasarian, N. Street and W. Street, *Breast Cancer Wisconsin (Diagnostic)*, UCI Machine Learning Repository, 1993, [doi.org/10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B). Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

### Energy efficiency (regression)

The `energy_` files are the Energy Efficiency dataset, split the same way (80% and 20%, `random_state=42`, not stratified, since the target is a number). Each row is a building design: 8 measurements of its shape (relative compactness, surface area, wall area, roof area, overall height, orientation, glazing area, glazing distribution; the dataset's columns X1 to X8, renamed) and its heating load (Y1, renamed `heating_load`).

- `energy_train.csv`: 614 buildings with their heating load in the last column, to train on.
- `energy_to_predict.csv`: the other 154 buildings, measurements only. They stand in for buildings whose heating load is not yet known.
- `energy_to_predict_actual.csv`: the actual heating loads of those 154 buildings, in the same order, used only to check the model's predicted values.

Source: A. Tsanas and A. Xifara, *Energy Efficiency*, UCI Machine Learning Repository, 2012, [doi.org/10.24432/C51307](https://doi.org/10.24432/C51307). Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## About

These examples use the Red Cedar Python SDK to work with a Red Cedar server. All quantum computation happens on the server, so the examples need no quantum libraries.
