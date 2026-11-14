"""Red Cedar: the quantum model beside a random forest, its classical baseline, as a script: the
same comparison as 07_quantum_and_forest.ipynb.

A random forest is a classical model: many decision trees, each trained on a random draw of the
rows, that vote on each prediction. It is a strong, widely used method for tables like this one,
so it is the baseline to compare the quantum model with. Red Cedar trains it with the same calls
as the quantum model, on the same training rows, and tests it on the same held-out rows; one
setting, "model": "random_forest", chooses it.

1. Train the quantum model (the same settings as the Quick Start)
2. Train a random forest on the same file
3. Classify the same unclassified samples with both, and check them against the actual classes
4. Compare the features each model relies on

The data is the Quick Start's: the Wisconsin breast cancer dataset (WDBC), 30 measurements per
tumor sample and a diagnosis, M (malignant) or B (benign).
  wdbc_train.csv                455 classified samples, to train both models
  wdbc_unclassified.csv         114 other samples, for the trained models to classify
  wdbc_unclassified_actual.csv  their actual diagnoses, to check the predictions
This is a demonstration on a public dataset, not a medical tool.

Needs a Red Cedar server of version 0.95 or later (the random forest is new in 0.95). The server
comes from REDCEDAR_URL and REDCEDAR_API_KEY. Run the script from this folder, where the data/
folder is:

    python random_forest_baseline.py
"""

import csv

import redcedar

train_file = "data/wdbc_train.csv"
unclassified_file = "data/wdbc_unclassified.csv"

client = redcedar.connect()
print(client.status())

# ==============================================================================
# 1. TRAIN THE QUANTUM MODEL
#
# The Quick Start's settings: 6 qubits, 2 layers, seed 123, and the original measurements
# (dimension_reduction "none"), so explainable AI names them in step 4.

print("\n1. Train the quantum model")
quantum = client.training.run(train_file, verbose=True, settings={
    "num_qubits": 6,
    "min_layers": 1,
    "max_layers": 2,
    "seed": 123,
    "dimension_reduction": "none",
})
last_layer = max(quantum["validation"])
quantum_test = quantum["validation"][last_layer]["test_accuracy"]
print(f"Quantum model, layer {last_layer}: test accuracy {quantum_test:.1%} "
      f"({quantum['elapsed_seconds']}s)")

# ==============================================================================
# 2. TRAIN A RANDOM FOREST ON THE SAME FILE
#
# The same file and the same split: the forest trains on the same 80% of the rows and is tested
# on the same 20%, so the two test accuracies can be compared directly. The forest uses the
# measurements as they are. Its own settings go in forest_options (here the defaults, written out:
# 300 trees, at least 5 training rows at the end of each branch). The seed makes the run
# repeatable. More in the User Guide: Random Forest.

print("\n2. Train a random forest")
forest = client.training.run(train_file, verbose=True, settings={
    "model": "random_forest",
    "seed": 123,
    "forest_options": {"n_estimators": 300, "min_samples_leaf": 5},
})
print(f"Random forest: test accuracy {forest['test_accuracy']:.1%} ({forest['elapsed_seconds']}s)")

# ==============================================================================
# 3. CLASSIFY THE SAME SAMPLES WITH BOTH
#
# Both models classify the 114 unclassified samples; the actual diagnoses, which neither model
# has seen, show how many each gets right.

print("\n3. Classify the unclassified samples with both models")
with open("data/wdbc_unclassified_actual.csv", newline="") as f:
    actual = [row["diagnosis"] for row in csv.DictReader(f)]

predicted = {
    "Quantum model": client.inference.run(quantum["train_id"], unclassified_file)["predictions"],
    "Random forest": client.inference.run(forest["train_id"], unclassified_file)["predictions"],
}
print(f"\n  {'':14}  {'Correct':>12}  {'Malignant (M)':>14}  {'Benign (B)':>11}")
for name, rows in predicted.items():
    labels = [str(r["prediction"]) for r in rows]
    correct = sum(p == a for p, a in zip(labels, actual))
    per_class = [f"{sum(p == a == c for p, a in zip(labels, actual))} of {actual.count(c)}" for c in ("M", "B")]
    print(f"  {name:14}  {f'{correct} of {len(actual)}':>12}  {per_class[0]:>14}  {per_class[1]:>11}")

agree = sum(str(q["prediction"]) == str(f["prediction"])
            for q, f in zip(predicted["Quantum model"], predicted["Random forest"]))
print(f"\nThe two models agree on {agree} of {len(actual)} samples.")

# ==============================================================================
# 4. WHICH FEATURES EACH MODEL RELIES ON
#
# Explainable AI gives the quantum model's importance for each data qubit. For the forest it gives
# each column's share of the trees' error reduction (they add up to 1). The two measure different
# things, so compare which features come first rather than the numbers themselves.

print("\n4. Which features each model relies on")
quantum_features = sorted(client.explainable.run(quantum["train_id"])["qubit_map"],
                          key=lambda q: q["importance"], reverse=True)
forest_features = client.explainable.run(forest["train_id"])["feature_importances"]

print("\n  Quantum model (one feature per data qubit):")
for q in quantum_features:
    print(f"    {q['feature_name']:<24} {q['importance']:.1%}")
print("\n  Random forest (top 5 of 30 columns):")
for f in forest_features[:5]:
    print(f"    {f['feature_name']:<24} {f['importance']:.1%}")
