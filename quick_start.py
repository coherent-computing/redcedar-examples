"""Red Cedar Quick Start, as a script: the same steps as 01_quick_start.ipynb.

1. Train a quantum model on a CSV file
2. Check the quality of the training
3. Infer the class of new data the model has never seen
4. See which input features the model relies on

The data is the Wisconsin breast cancer dataset (WDBC), split once into two files in data/:
wdbc_train.csv (455 rows) to train on, and wdbc_new.csv (114 rows) to infer. Each row is a
tumor sample: 30 measurements, then the diagnosis in the last column, M (malignant) or B (benign).

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
train_file = os.path.join(HERE, "data", "wdbc_train.csv")   # 455 rows to train on
new_file = os.path.join(HERE, "data", "wdbc_new.csv")       # 114 rows the model never sees in training

# Connect to the server
# Reads REDCEDAR_URL and REDCEDAR_API_KEY from environment, or pass explicitly:
# client = redcedar.connect(url="http://127.0.0.1:8002", api_key="your-key")
client = redcedar.connect()
print(f"Red Cedar SDK v{redcedar.__version__}")
status = client.status()
print(f"Server: {client.url or 'local mode, in this process'} (version {status['server_version'] or 'unknown'}, ok={status['ok']})")

# --- 1. Train a model ----------------------------------------------------------
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
# The model infers the class of each row of a file it never saw in training. This file still
# has the diagnosis column, so the server reports the accuracy; with truly new data you would
# leave that column out.
print("\n3. Inference on new data")
predictions_file = os.path.join(HERE, "wdbc_new_predictions.csv")
predictions = client.inference.run(train_id, new_file, output=predictions_file, verbose=True)

print(f"Accuracy on {predictions['rows']} new rows: {predictions['accuracy']:.1%}")
print("Rows and inferred classes written to wdbc_new_predictions.csv")

# Compare with the diagnosis in the file
with open(new_file, newline="") as f:
    actual = [row["diagnosis"] for row in csv.DictReader(f)]
rows = list(zip(actual, predictions["predictions"]))

print("First 5 rows:")
for a, p in rows[:5]:
    probs = ", ".join(f"{c}: {v:.0%}" for c, v in p["probabilities"].items())
    print(f"  actual {a}, predicted {p['prediction']}  ({probs})")

classes = sorted(set(actual))
accuracy = [sum(str(p["prediction"]) == c for a, p in rows if a == c) / actual.count(c) for c in classes]
plt.figure(figsize=(5, 3))
plt.bar(classes, accuracy)
plt.axhline(predictions["accuracy"], color="gray", ls="--", label="overall")
plt.ylim(0, 1)
plt.xlabel("Class")
plt.ylabel("Accuracy on new rows")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(HERE, "quick_start_inference.png"))
print("Accuracy per class saved to quick_start_inference.png")

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
