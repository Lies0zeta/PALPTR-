"""
Data preprocessing script for PTR experiments
@author: Lies Hadjadj
"""
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from pmlb import fetch_data
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import normalize
from sklearn.datasets import (
    load_digits,
    fetch_openml,
    fetch_20newsgroups
)
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline

def load_protein_data() -> Tuple[np.ndarray, np.ndarray]:
    """Load and preprocess protein dataset"""
    data = pd.read_csv("../../../Datasets/protein/mice_protein_expression.csv")
    data = data.dropna().drop(['Unnamed: 0'], axis=1)
    y = data.pop('class').values
    class_map = {
        'c-CS-s': 0, 'c-CS-m': 1, 'c-SC-s': 2, 'c-SC-m': 3,
        't-CS-s': 4, 't-CS-m': 5, 't-SC-s': 6, 't-SC-m': 7
    }
    y = np.array([class_map[label] for label in y])
    X = data.values
    return normalize(X, norm='max', axis=0), y

def load_coil20_data() -> Tuple[np.ndarray, np.ndarray]:
    """Load and preprocess COIL-20 dataset"""
    data = pd.read_csv("../../../Datasets/COIL-20/coil20.csv").dropna()
    y = data.pop('1025').values.astype(int)
    X = data.values
    return normalize(X, norm='max', axis=0), y

def load_isolet_data() -> Tuple[np.ndarray, np.ndarray]:
    """Load and preprocess ISOLET dataset"""
    data1 = pd.read_csv("../../../Datasets/Isolet/isolet1+2+3+4.data", header=None)
    data2 = pd.read_csv("../../../Datasets/Isolet/isolet5.data", header=None)
    data = pd.concat([data1, data2]).dropna()
    y = data.pop(617).values.astype(int)
    X = data.values
    return normalize(X, norm='max', axis=0), y

def load_statlog_data() -> Tuple[np.ndarray, np.ndarray]:
    """Load and preprocess Statlog dataset"""
    train = pd.read_csv("../../../Datasets/Statlog/sat.trn", sep=' ', header=None)
    test = pd.read_csv("../../../Datasets/Statlog/sat.tst", sep=' ', header=None)
    data = pd.concat([train, test]).dropna()
    y = data.pop(36).values.astype(int)
    X = data.values
    return normalize(X, norm='max', axis=0), y

def load_sdd_data() -> Tuple[np.ndarray, np.ndarray]:
    """Load and preprocess Sensorless Drive Diagnosis dataset"""
    data = pd.read_csv("../../../Datasets/Sensorless_drive_diagnosis.txt",
                       header=None, sep=' ').dropna()
    y = data.pop(48).values.astype(int)
    X = data.values
    return normalize(X, norm='max', axis=0), y

def load_mnist_data() -> Tuple[np.ndarray, np.ndarray]:
    """Load and preprocess MNIST dataset"""
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    return normalize(X.astype(float), norm='max', axis=0), y.astype(int)

def load_20news_data() -> Tuple[np.ndarray, np.ndarray]:
    """Load and preprocess 20 Newsgroups dataset"""
    newsgroups = fetch_20newsgroups(
        subset='all',
        shuffle=False,
        remove=('headers', 'footers', 'quotes')
    )
    vectorizer = Pipeline([
        ('vect', CountVectorizer(
            ngram_range=(1, 2),
            min_df=5,
            max_df=0.8,
            stop_words='english'
        )),
        ('tfidf', TfidfTransformer()),
    ])
    X = vectorizer.fit_transform(newsgroups.data)
    return normalize(X, norm='max', axis=0), newsgroups.target

def load_digits_data() -> Tuple[np.ndarray, np.ndarray]:
    """Load and preprocess Digits dataset"""
    X, y = load_digits(return_X_y=True)
    return normalize(X, norm='max', axis=0), y

def load_spam_data() -> Tuple[np.ndarray, np.ndarray]:
    """Load and preprocess Spambase dataset"""
    data = pd.read_csv("../../../Datasets/Spambase/spambase.data", header=None).dropna()
    y = data.pop(57).values.astype(int)
    X = data.values.astype('float32')
    return normalize(X, norm='max', axis=0), y

def load_banknote_data() -> Tuple[np.ndarray, np.ndarray]:
    """Load and preprocess Banknote dataset"""
    data = pd.read_csv("../../../Datasets/banknote_authentication.txt",
                       sep=',', header=None).dropna()
    y = data.pop(4).values.astype(int)
    X = data.values.astype('float32')
    return normalize(X, norm='max', axis=0), y

# Dataset loading functions mapping
DATASET_LOADERS = {
    'protein': load_protein_data,
    'adult': lambda: fetch_data('adult', return_X_y=True),
    'coil20': load_coil20_data,
    'isolet': load_isolet_data,
    'statlog': load_statlog_data,
    'sdd': load_sdd_data,
    'mnist': load_mnist_data,
    '20news': load_20news_data,
    'digits': load_digits_data,
    'spam': load_spam_data,
    'banknote': load_banknote_data
}

def main():
    """Main preprocessing function"""
    if len(sys.argv) != 2:
        print("Usage: python preprocess_dataset.py <dataset_name>")
        sys.exit(1)

    dataset_name = sys.argv[1]
    print('dataset:', dataset_name)

    if dataset_name not in DATASET_LOADERS:
        print(f"Unknown dataset: {dataset_name}")
        sys.exit(1)

    # Load and preprocess data
    X, y = DATASET_LOADERS[dataset_name]()

    # Prepare cross-validation splits
    sss = StratifiedShuffleSplit(n_splits=20, test_size=.3, random_state=2021)

    ids_train = []
    ids_test = []
    dist = []
    density = []
    idx = []

    # Compute density estimation for each split
    for train_idx, test_idx in sss.split(X, y):
        num_pts = len(train_idx)
        nbrs = NearestNeighbors(n_neighbors=int(num_pts / 10), n_jobs=-1).fit(X[train_idx])
        dist_, idx_ = nbrs.kneighbors()
        density_ = 1/np.sqrt(np.mean(np.square(dist_), axis=1))

        dist.append(dist_)
        density.append(density_)
        idx.append(idx_)
        ids_train.append(train_idx)
        ids_test.append(test_idx)

    # Save results
    save_dir = Path('../../obj')
    save_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(save_dir / f'{dataset_name}_density', density=density)
    np.savez_compressed(save_dir / f'{dataset_name}_dist', distance=dist)
    np.savez_compressed(save_dir / f'{dataset_name}_idx', indices=idx)
    np.savez_compressed(save_dir / dataset_name, train=ids_train, test=ids_test)

if __name__ == "__main__":
    main()
