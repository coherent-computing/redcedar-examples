"""Red Cedar Quick Start, as a script: the same steps as 01_quick_start.ipynb.

A quick way to get started with Red Cedar. It shows how to train a quantum machine learning model
on a set of training data, and then apply that model to data whose class is unknown, inferring
the class of each row and reporting on the results.

1. Train a quantum model on a CSV file
2. Check the quality of the training
3. Infer the class of samples whose class is unknown
4. Verify the results
5. See which input features the model relies on

The data is the Wisconsin breast cancer dataset (WDBC). Each row is a tumor sample: 30
measurements (the features) and a diagnosis (the class), M for malignant or B for benign. The
script uses two files in the data/ folder next to it:
  wdbc_train.csv    455 samples with their diagnosis, to train the model
  wdbc_unknown.csv  114 other samples, with the measurements only: their diagnosis is unknown,
                    and the trained model infers it

The server comes from REDCEDAR_URL and REDCEDAR_API_KEY. The plots are saved as PNG files
next to this script.

    python quick_start.py
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")   # write the plots to files, no windows
import matplotlib.pyplot as plt

import redcedar

HERE = os.path.dirname(os.path.abspath(__file__))
train_file = os.path.join(HERE, "data", "wdbc_train.csv")       # 455 samples with their diagnosis, to train on
unknown_file = os.path.join(HERE, "data", "wdbc_unknown.csv")   # 114 samples, measurements only: diagnosis unknown

# Connect to the server
# Reads REDCEDAR_URL and REDCEDAR_API_KEY from environment, or pass explicitly:
# client = redcedar.connect(url="http://127.0.0.1:8002", api_key="your-key")
client = redcedar.connect()
print(f"Red Cedar SDK v{redcedar.__version__}")
status = client.status()
print(f"Server: {client.url or 'local mode, in this process'} (version {status['server_version'] or 'unknown'}, ok={status['ok']})")

# --- 1. Train a model ----------------------------------------------------------
# Training teaches a model to recognize patterns in data: from rows with features and a known
# class, it learns which patterns of features go with which class, so that it can later infer
# the class of rows it has never seen.
# The model is a quantum circuit: a row's features are encoded on its qubits, the circuit applies
# quantum operations with adjustable settings (the parameters), and measuring it gives the class.
# Training adjusts the parameters step by step (parameter updates), each step making the correct
# class more likely. The circuit is built in layers, each adding operations and parameters; the
# layers are trained one at a time, each starting from the parameters the one before found, and
# each finished layer is saved as a model of its own.
# 6 qubits and 2 layers; seed fixes the starting parameters, so the run can be repeated exactly.
# At the end of each layer the server tests the model on rows it set aside (20% by default).
print("\n1. Training")
result = client.training.run(train_file, verbose=True, settings={
    "num_qubits": 6,
    "min_layers": 1,
    "max_layers": 2,
    "seed": 123,
})

train_id = result["train_id"]
print(f"Model ID: {train_id}")
print(f"Training time: {result['elapsed_seconds']}s")
for layer, v in result["validation"].items():
    print(f"Layer {layer}: test accuracy {v['test_accuracy']:.1%}, test loss {v['test_loss']:.4f}")

# --- 2. Training quality -------------------------------------------------------
# While the model trains, the loss and accuracy on the training rows are measured after every
# parameter update, in each layer, and come back in result["history"].
# Training loss: for each row, 1 minus the probability the model gives the correct class,
# averaged over the training rows; lower is better. Training accuracy: the share of training
# rows whose correct class is the one the model finds most likely. The test results above are
# the same two measures on the rows held back. Details: the User Guide, "Training Loss and Accuracy".
fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))
start = 0
for layer, points in result["history"].items():
    steps = range(start, start + len(points))
    ax_loss.plot(steps, [p["train_loss"] for p in points], label=f"layer {layer}")
    ax_acc.plot(steps, [p["train_accuracy"] for p in points], label=f"layer {layer}")
    if start:
        ax_loss.axvline(start, color="gray", ls="--", lw=0.8)
        ax_acc.axvline(start, color="gray", ls="--", lw=0.8)
    start += len(points)

ax_loss.set(xlabel="Parameter update", ylabel="Training loss", title="Training loss")
ax_acc.set(xlabel="Parameter update", ylabel="Training accuracy", title="Training accuracy")
for ax in (ax_loss, ax_acc):
    ax.grid(True)
    ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(HERE, "quick_start_training.png"))
print("\n2. Training curves saved to quick_start_training.png")

# --- 3. Infer the class of the unknown samples --------------------------------
# The trained model infers the diagnosis of the 114 samples in wdbc_unknown.csv, whose diagnosis
# is unknown: for each one the class, M or B, with a confidence (the probability the model gives
# that class). output= writes each sample with its inferred class and the probability of each
# class to wdbc_unknown_inferred.csv, next to this script.
print("\n3. Infer the class of the unknown samples")
inferred_file = os.path.join(HERE, "wdbc_unknown_inferred.csv")
results = client.inference.run(train_id, unknown_file, output=inferred_file, verbose=True)
inferred = results["predictions"]

names = {"M": "Malignant (M)", "B": "Benign (B)"}
counts = {c: sum(str(r["prediction"]) == c for r in inferred) for c in names}
print(f"Of the {len(inferred)} unknown samples, the model infers {counts['M']} malignant (M) and {counts['B']} benign (B).")

print("\nThe first 10:")
print(f"  {'Sample':>6}  {'Inferred':<14}  Confidence")
for i, r in enumerate(inferred[:10], start=1):
    c = str(r["prediction"])
    print(f"  {i:>6}  {names[c]:<14}  {r['probabilities'][c]:.0%}")
print("\nEach sample with its inferred class and probabilities is written to wdbc_unknown_inferred.csv")

plt.figure(figsize=(5, 3))
plt.bar([names[c] for c in counts], list(counts.values()), color=["tab:red", "tab:blue"])
plt.ylabel("Samples")
plt.title("Inferred diagnosis of the unknown samples")
plt.tight_layout()
plt.savefig(os.path.join(HERE, "quick_start_inference.png"))
print("Inferred diagnoses saved to quick_start_inference.png")

# --- 4. Verify the results -----------------------------------------------------
# To see how well the quantum model does, compare its inferences with the true diagnoses of the
# 114 unknown samples. Because the samples come from a public dataset, their true diagnoses are
# known: they are in data/wdbc_unknown_true_diagnosis.csv, which the model never sees. With your
# own unknown data there would be no such file. This is a different check from the test results
# in step 1, which come from rows the server set aside from the training file.
print("\n4. Verify the results")
with open(os.path.join(HERE, "data", "wdbc_unknown_true_diagnosis.csv"), newline="") as f:
    true_diagnosis = [row["diagnosis"] for row in csv.DictReader(f)]

correct = sum(str(r["prediction"]) == t for r, t in zip(inferred, true_diagnosis))
print(f"{correct} of {len(true_diagnosis)} inferred correctly ({correct / len(true_diagnosis):.1%})")
for c in names:
    right = sum(str(r["prediction"]) == t == c for r, t in zip(inferred, true_diagnosis))
    print(f"  {names[c]}: {right} of {true_diagnosis.count(c)} inferred correctly")

# --- 5. Which features the model relies on --------------------------------------
# Explainable AI maps each data qubit back to the input feature it encodes and measures how
# much that qubit influences the prediction.
print("\n5. Explainable AI")
explanation = client.explainable.run(train_id, verbose=True)

qubits = explanation["qubit_map"]
for q in qubits:
    print(f"  Qubit {q['qubit']}: {q['feature_name']} (importance {q['importance']:.1%})")

plt.figure(figsize=(6, 3))
plt.barh([f"q{q['qubit']}: {q['feature_name']}" for q in qubits], [q["importance"] for q in qubits])
plt.gca().invert_yaxis()
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(os.path.join(HERE, "quick_start_features.png"))
print("Feature importance saved to quick_start_features.png")
