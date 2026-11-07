"""Red Cedar Quick Start, as a script: the same steps as 01_quick_start.ipynb.

1. Train a quantum model on a CSV file
2. Check the quality of the training
3. Infer the class of new data the model has never seen
4. See which input features the model relies on

The data is the Wisconsin breast cancer dataset (WDBC). Each row is a tumor sample: 30
measurements (the features) and a diagnosis (the class), M for malignant or B for benign. It is
split once into two files in data/:
  wdbc_train.csv  455 samples with their diagnosis, to train the model
  wdbc_new.csv    114 other samples with the measurements only: new patients whose diagnosis is
                  unknown, which the trained model infers
The real diagnoses of the 114 new samples are kept apart in wdbc_new_diagnosis.csv, used only at
the end of step 3 to check the model.

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
train_file = os.path.join(HERE, "data", "wdbc_train.csv")   # 455 samples with their diagnosis, to train on
new_file = os.path.join(HERE, "data", "wdbc_new.csv")       # 114 new samples: measurements only, no diagnosis

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

# --- 3. Infer the class of new data --------------------------------------------
# The trained model infers the diagnosis of the 114 new samples, whose diagnosis is unknown: for
# each one the class, M or B, with a confidence (the probability the model gives that class).
print("\n3. Infer the class of new data")
results_file = os.path.join(HERE, "wdbc_new_results.csv")
results = client.inference.run(train_id, new_file, output=results_file, verbose=True)
inferred = results["predictions"]

names = {"M": "Malignant (M)", "B": "Benign (B)"}
counts = {c: sum(str(r["prediction"]) == c for r in inferred) for c in names}
print(f"Of {len(inferred)} new patients, the model infers {counts['M']} malignant (M) and {counts['B']} benign (B).")

print("\nThe first 10:")
print(f"  {'Patient':>7}  {'Inferred':<14}  Confidence")
for i, r in enumerate(inferred[:10], start=1):
    c = str(r["prediction"])
    print(f"  {i:>7}  {names[c]:<14}  {r['probabilities'][c]:.0%}")
print("\nEach sample with its inferred class and probabilities is written to wdbc_new_results.csv")

plt.figure(figsize=(5, 3))
plt.bar([names[c] for c in counts], list(counts.values()), color=["tab:red", "tab:blue"])
plt.ylabel("New patients")
plt.title("Inferred diagnosis")
plt.tight_layout()
plt.savefig(os.path.join(HERE, "quick_start_inference.png"))
print("Inferred diagnoses saved to quick_start_inference.png")

# How good were these inferences? The new samples come from the WDBC dataset, so their real
# diagnoses are known; they are kept apart in wdbc_new_diagnosis.csv and the model never sees
# them. With your own new data you would not have them; here they let us check the model.
with open(os.path.join(HERE, "data", "wdbc_new_diagnosis.csv"), newline="") as f:
    actual = [row["diagnosis"] for row in csv.DictReader(f)]

correct = sum(str(r["prediction"]) == a for r, a in zip(inferred, actual))
print(f"\nCheck: {correct} of {len(actual)} inferred correctly ({correct / len(actual):.1%})")
for c in names:
    right = sum(str(r["prediction"]) == a == c for r, a in zip(inferred, actual))
    print(f"  {names[c]}: {right} of {actual.count(c)} inferred correctly")

# --- 4. Which features the model relies on --------------------------------------
# Explainable AI maps each data qubit back to the input feature it encodes and measures how
# much that qubit influences the prediction.
print("\n4. Explainable AI")
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
