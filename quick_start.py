"""Red Cedar Quick Start, as a script: the same steps as 01_quick_start.ipynb.

A quick way to get started with Red Cedar. It shows how to train a quantum machine learning model
on classified samples (samples whose class is known), and then use that model to classify
samples whose class is not yet known.

1. Train a quantum model on a CSV file
2. Classify samples using the trained model
3. Check the predicted classes
4. See which input features the model relies on

The data is the Wisconsin breast cancer dataset (WDBC). Each row is a tumor sample: 30
measurements (the features) and a diagnosis (the class), M for malignant or B for benign. The
script uses two files in the data/ folder:
  wdbc_train.csv         455 classified samples, with their diagnosis, to train the model
  wdbc_unclassified.csv  114 other samples, with the measurements only, for the trained model
                         to classify

The same steps work for any table of samples with a class, such as equipment readings classified
as normal or faulty, or transactions as legitimate or fraudulent. This is a demonstration on a
public dataset, not a medical tool.

All computation happens on the Red Cedar server; this script sends requests and shows the
results. The User Guide describes every SDK function:
https://coherent-computing.github.io/redcedar/user_guide/

The server comes from REDCEDAR_URL and REDCEDAR_API_KEY. Run the script from this folder, where
the data/ folder is; the results file and the plots (PNG files) are written here too:

    python quick_start.py
"""

import csv

import matplotlib
matplotlib.use("Agg")   # write the plots to files, no windows
import matplotlib.pyplot as plt

import redcedar

# ==============================================================================
# DATA FILES
#
# The two data files described at the top

train_file = "data/wdbc_train.csv"                 # 455 classified samples (diagnosis known), to train on
unclassified_file = "data/wdbc_unclassified.csv"   # 114 samples, measurements only: diagnosis not yet known

# ==============================================================================
# CONNECT TO A RED CEDAR SERVER
#
# The one in REDCEDAR_URL (with the key in REDCEDAR_API_KEY) if those are set, otherwise the one
# on this machine, http://127.0.0.1:8002

client = redcedar.connect()
print(client.status())

# To choose a server in the code instead:
# client = redcedar.connect(url="https://your-server/api", api_key="your-key")

# Developers with the Red Cedar source installed (pip install -e) can use local mode,
# which runs Red Cedar in this process, with no server:
# client = redcedar.connect(local=True)

# ==============================================================================
# 1. TRAIN A MODEL
#
# Training imprints the patterns in your data onto a quantum circuit, optimizing its parameters
# so that the quantum state it produces best links each sample's features to its class.
# training.run() takes the path of a CSV file with the class in the last column; the SDK uploads
# it and the server trains the model.
#
# How the model works: a Red Cedar model is a quantum circuit. A row's features are encoded on
# its qubits, the circuit applies quantum operations whose adjustable settings are the
# parameters, and measuring the circuit gives the class. Training adjusts the parameters step by
# step; each parameter update makes the correct class more likely for the training rows.
#
# Layers: the circuit is built in layers, each adding operations and parameters so the model can
# capture more complex patterns. Layers are trained one at a time, each starting from the
# parameters the one before found, and each finished layer is saved as a model of its own. More
# layers often, but not always, improve the accuracy.
#
# Settings: here 6 qubits and 2 layers, which trains in about a minute; seed makes the run
# repeatable.
#
# After each layer the server reports the test accuracy: the share of the rows it set aside from
# the file (20%), which the model does not train on, that the model classifies correctly.
# More in the User Guide: Training, Settings.

print("\n1. Train a model")
result = client.training.run(train_file, verbose=True, settings={
    "num_qubits": 6,
    "min_layers": 1,
    "max_layers": 2,
    "seed": 123,
})

train_id = result["train_id"]
print(f"\nModel ID: {train_id}")
print(f"Training time: {result['elapsed_seconds']}s")
for layer, v in result["validation"].items():
    print(f"Layer {layer}: test accuracy {v['test_accuracy']:.1%}")

# ==============================================================================
# 2. CLASSIFY SAMPLES USING THE TRAINED MODEL
#
# Now we apply the trained model to the 114 samples in wdbc_unclassified.csv, whose diagnosis is
# not yet known. For each one it predicts the class, M or B, with a confidence: the probability
# the model gives that class.
#
# inference.run() matches the file's columns to the training file's by name (using a trained
# model on samples it hasn't seen is called inference), and output= writes the predictions to
# wdbc_unclassified_predicted.csv. More in the User Guide: Predictions.

CLASS_NAMES = {"M": "Malignant (M)", "B": "Benign (B)"}


def show_predicted_classes(predicted, png):
    """Print how many samples got each class and the first 10 with their confidence; plot the counts."""
    counts = {c: sum(str(r["prediction"]) == c for r in predicted) for c in CLASS_NAMES}
    print(f"\nOf the {len(predicted)} unclassified samples, the model classifies "
          f"{counts['M']} as malignant (M) and {counts['B']} as benign (B).")
    print("\nThe first 10:")
    print(f"  {'Sample':>6}  {'Predicted':<14}  Confidence")
    for i, r in enumerate(predicted[:10], start=1):
        c = str(r["prediction"])
        print(f"  {i:>6}  {CLASS_NAMES[c]:<14}  {r['probabilities'][c]:.0%}")
    plt.figure(figsize=(5, 3))
    plt.bar([CLASS_NAMES[c] for c in counts], list(counts.values()), color=["tab:red", "tab:blue"])
    plt.ylabel("Samples")
    plt.title("Predicted classes of the unclassified samples")
    plt.tight_layout()
    plt.savefig(png)
    print(f"Chart saved to {png}")


# Classify the samples and show the results
print("\n2. Classify samples using the trained model")
results = client.inference.run(train_id, unclassified_file, output="wdbc_unclassified_predicted.csv", verbose=True)
predicted = results["predictions"]
show_predicted_classes(predicted, "quick_start_classes.png")

# ==============================================================================
# 3. CHECK THE PREDICTED CLASSES
#
# To see how well the model does, we compare its predicted classes with the actual ones. These
# samples come from a public dataset, so their actual diagnoses are known: they are in
# data/wdbc_unclassified_actual.csv, which the model never sees. With your own data there would
# be no such file.
#
# Unlike the test accuracy in step 1, which uses rows set aside from the training file, this
# check uses a separate file, as a trained model is used in practice.

print("\n3. Check the predicted classes")
with open("data/wdbc_unclassified_actual.csv", newline="") as f:
    actual = [row["diagnosis"] for row in csv.DictReader(f)]

correct = sum(str(r["prediction"]) == a for r, a in zip(predicted, actual))
print(f"{correct} of {len(actual)} classified correctly ({correct / len(actual):.1%})")
for c in CLASS_NAMES:
    right = sum(str(r["prediction"]) == a == c for r, a in zip(predicted, actual))
    print(f"  {CLASS_NAMES[c]}: {right} of {actual.count(c)} classified correctly")

# ==============================================================================
# 4. WHICH FEATURES THE MODEL RELIES ON
#
# Explainable AI maps each data qubit back to the input feature it encodes and measures how much
# that qubit influences the prediction. More in the User Guide: Explainable AI.


def show_feature_importance(qubit_map, png):
    """Print each data qubit's feature and importance, most important first; plot them."""
    qubit_map = sorted(qubit_map, key=lambda q: q["importance"], reverse=True)
    for q in qubit_map:
        print(f"  Qubit {q['qubit']}: {q['feature_name']} (importance {q['importance']:.1%})")
    plt.figure(figsize=(6, 3))
    plt.barh([f"q{q['qubit']}: {q['feature_name']}" for q in qubit_map], [q["importance"] for q in qubit_map])
    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(png)
    print(f"Chart saved to {png}")


# Explain the model and show the results
print("\n4. Which features the model relies on")
explanation = client.explainable.run(train_id, verbose=True)
show_feature_importance(explanation["qubit_map"], "quick_start_features.png")
