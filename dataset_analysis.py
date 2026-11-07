"""Red Cedar Dataset Analysis, as a script: sections 2-4 of 02_dataset_analysis.ipynb, for any dataset.

Shows how to understand a dataset before training a model on it: how many of its dimensions
carry real signal, which of them predict the class, and about how many qubits a model needs.

1. PCA: how many independent dimensions carry real signal
2. Mutual information: which directions in the data predict the class
3. Quantum resource estimation: about how many qubits a model needs

The dataset is either a CSV file of your own (the last column is the class; a header row is
optional), which the script uploads, or the id of a dataset in the server's catalog (the list is
in section 1 of 02_dataset_analysis.ipynb, or from client.datasets.list()). Without one, the
script analyzes the Wisconsin breast cancer dataset (WDBC) from the catalog, the same data as
the Quick Start.

All computation happens on the Red Cedar server; this script sends requests and shows the
results. More in the User Guide: Dataset Analysis,
https://coherent-computing.github.io/redcedar/user_guide/#dataset-analysis

The server comes from REDCEDAR_URL and REDCEDAR_API_KEY. Run the script from this folder; the
plots (PNG files) are written here:

    python dataset_analysis.py                          # WDBC from the catalog
    python dataset_analysis.py data/wdbc_train.csv      # a CSV file of your own
    python dataset_analysis.py iris                     # another catalog dataset
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")   # write the plots to files, no windows
import matplotlib.pyplot as plt

import redcedar

# ==============================================================================
# THE DATASET
#
# From the command line: a CSV file or a catalog id; WDBC from the catalog without one

dataset = sys.argv[1] if len(sys.argv) > 1 else "wdbc"

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

# A CSV file is uploaded first; the upload returns an ID that works with the analysis functions
# (and with client.training.run()). A catalog id is used as it is.
if os.path.isfile(dataset):
    dataset_id = client.datasets.upload(dataset)
    print(f"Uploaded {dataset} as {dataset_id}")
else:
    dataset_id = dataset
print(f"Analyzing: {dataset}")

# ==============================================================================
# 1. PCA: HOW MANY DIMENSIONS MATTER
#
# Principal component analysis finds the directions in which the data varies most. If a few
# components explain most of the variance, the data can be compressed to fewer features (and
# qubits) with little loss.


def plot_pca(pca, png):
    """The variance of each principal component, and the cumulative variance."""
    n = range(1, len(pca["variance_ratios"]) + 1)
    plt.figure(figsize=(7, 4))
    plt.bar(n, pca["variance_ratios"], label="each component")
    plt.plot(n, pca["cumulative_variance"], "o-", color="C1", label="cumulative")
    plt.axhline(0.9, color="gray", ls="--", lw=0.8)
    plt.xlabel("Principal component")
    plt.ylabel("Fraction of variance")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(png)
    print(f"Chart saved to {png}")


print("\n1. PCA: how many dimensions matter")
pca = client.analysis.pca(dataset_id, verbose=True)

print(f"\n{pca['features']} features, {pca['samples']} samples")
print(f"Components for 90% of the variance: {pca['thresholds']['90']}")
print(f"Components for 95% of the variance: {pca['thresholds']['95']}")
plot_pca(pca, "dataset_analysis_pca.png")

# ==============================================================================
# 2. MUTUAL INFORMATION: WHAT PREDICTS THE CLASS
#
# Mutual information measures how much knowing a value tells you about the class label. The
# analysis first applies PCA, the same way training prepares the data, and then scores each
# principal component (PC1, PC2, ...). Components with high scores carry the most signal for
# classification; they are not always the ones with the most variance.


def plot_mi(mi, png, top=10):
    """The mutual information of the top principal components."""
    ranked = mi["ranked_features"][:top]
    plt.figure(figsize=(7, 4))
    plt.barh([f["feature_name"] for f in ranked], [f["mi_score"] for f in ranked])
    plt.gca().invert_yaxis()
    plt.xlabel("Mutual information")
    plt.title(f"Top {len(ranked)} principal components")
    plt.tight_layout()
    plt.savefig(png)
    print(f"Chart saved to {png}")


print("\n2. Mutual information: what predicts the class")
mi = client.analysis.mi(dataset_id, verbose=True)

print(f"\nTotal MI: {mi['total_mi']:.4f}")
print(f"Components for 90% of the total MI: {mi['thresholds']['90']}")
for f in mi["ranked_features"][:5]:
    print(f"  {f['feature_name']}: MI={f['mi_score']:.4f} ({f['mi_percent']:.1f}%)")
plot_mi(mi, "dataset_analysis_mi.png")

# ==============================================================================
# 3. QUANTUM RESOURCE ESTIMATION: HOW MANY QUBITS
#
# The estimate encodes the data with more and more qubits and checks the best accuracy any model
# could reach at each size: once two samples with different classes look identical after
# encoding, no model can tell them apart. The estimate is the number of data qubits where that
# limit reaches 100%, plus the qubits needed for the class label, averaged over 10 random
# train/test splits.


def plot_qre(qre, png):
    """The highest possible train and test accuracy for each number of data qubits."""
    points = qre["plotData"]
    qubits = [p["Number of Qubits"] for p in points]
    plt.figure(figsize=(7, 4))
    plt.errorbar(qubits, [p["Train"] for p in points], yerr=[p["Train sem"] for p in points], marker=".", label="train")
    plt.errorbar(qubits, [p["Test"] for p in points], yerr=[p["Test sem"] for p in points], marker="x", label="test")
    plt.xlabel("Data qubits")
    plt.ylabel("Highest possible accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(png)
    print(f"Chart saved to {png}")


print("\n3. Quantum resource estimation: how many qubits")
qre = client.analysis.qre(dataset_id, verbose=True)

print(f"\nEstimated qubits: {qre['estimatedQubits']} +/- {qre['standardDeviation']:.1f}")
if qre.get("warning_message"):
    print("Note:", qre["warning_message"])
plot_qre(qre, "dataset_analysis_qre.png")
