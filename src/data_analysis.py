"""
Data analysis script for PTR experiments
@author: Lies Hadjadj
"""

import sys
from pathlib import Path
from warnings import simplefilter
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.datasets import fetch_openml, load_digits

# Ignore FutureWarnings
simplefilter(action="ignore", category=FutureWarning)

# Dataset-specific parameters
DATASET_PARAMS: Dict[str, Tuple[float, float, float]] = {
    "default": (0.09, 6.5, 1.0),  # alpha, radian, bias
    "protein": (0.48, 0.29, 2.86),
    "coil20": (0.09, 23.0, 1.0),
    "isolet": (0.09, 23.0, 1.0),
    "statlog": (0.8, 0.07, 6.0),
    "sdd": (1.0, 0.2, 3.3),
    "mnist": (0.09, 23.0, 1.0),
    "banknote": (1.0, 0.3, 3.3),
}


def load_protein_data() -> Tuple[np.ndarray, np.ndarray]:
    """Load and preprocess protein dataset"""
    data = pd.read_csv("../../../Datasets/protein/mice_protein_expression.csv")
    data = data.dropna().drop(["Unnamed: 0"], axis=1)
    y = data.pop("class").values
    class_map = {
        "c-CS-s": 0,
        "c-CS-m": 1,
        "c-SC-s": 2,
        "c-SC-m": 3,
        "t-CS-s": 4,
        "t-CS-m": 5,
        "t-SC-s": 6,
        "t-SC-m": 7,
    }
    y = np.array([class_map[label] for label in y])
    return normalize(data.values, norm="max", axis=0), y.astype(int)


def load_mnist_data() -> Tuple[np.ndarray, np.ndarray]:
    """Load and preprocess MNIST dataset"""
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    return normalize(X.astype(float), norm="max", axis=0), y.astype(int)


def load_digits_data() -> Tuple[np.ndarray, np.ndarray]:
    """Load and preprocess Digits dataset"""
    X, y = load_digits(return_X_y=True)
    return normalize(X, norm="max", axis=0), y


# Dataset loading functions mapping
DATASET_LOADERS = {
    "protein": load_protein_data,
    "mnist": load_mnist_data,
    "digits": load_digits_data,
    # Add other dataset loaders as needed
}


def analyze_split_statistics(
    distance: np.ndarray, density: np.ndarray, split_idx: int
) -> None:
    """Analyze and plot statistics for a given split"""
    print(f"###### split {split_idx} ######")

    # Analyze distances
    bin_dist, h_dist, _ = plt.hist(np.average(distance, axis=1), bins=20)
    print("max occurrence for distance:", h_dist[np.argmax(bin_dist)])
    print("min:", np.min(distance), "max:", np.max(distance))

    # Analyze densities
    bin_dens, h_dens, _ = plt.hist(density, bins=20)
    print("max occurrence for density:", h_dens[np.argmax(bin_dens)])
    print("min:", np.min(density), "max:", np.max(density))


def main():
    """Main analysis function"""
    if len(sys.argv) != 2:
        print("Usage: python data_analysis.py <dataset_name>")
        sys.exit(1)

    dataset_name = sys.argv[1]
    print("dataset:", dataset_name)

    # Load preprocessed data
    data_dir = Path("../../obj")
    try:
        # Load splits
        splits = np.load(data_dir / f"{dataset_name}.npz")
        ids_train, _ = splits["train"], splits["test"]

        # Load density and distance data
        density_data = np.load(data_dir / f"{dataset_name}_density.npz")["density"]
        distance_data = np.load(data_dir / f"{dataset_name}_dist.npz")["distance"]

        # Analyze each split
        for i in range(len(ids_train)):
            analyze_split_statistics(distance_data[i], density_data[i], i)

    except FileNotFoundError:
        print(f"Error: Could not load preprocessed data for {dataset_name}")
        print("Please run data_preprocessing.py first")
        sys.exit(1)


if __name__ == "__main__":
    main()
