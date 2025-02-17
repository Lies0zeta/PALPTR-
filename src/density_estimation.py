"""
Density estimation and scale computation utilities for PTR
@author: Lies Hadjadj
"""

from typing import Literal
import numpy as np
from scipy.spatial.distance import directed_hausdorff
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors

InputType = Literal["point cloud", "distance matrix"]
ArrayType = np.ndarray


def estimate_scale(
    X: ArrayType,
    N: int = 100,
    input_type: InputType = "point cloud",
    beta: float = 0.0,
    C: float = 10.0,
) -> float:
    """
    Compute estimated scale of a point cloud or a distance matrix.

    Args:
        X: Input point cloud (n_points × n_coordinates) or distance matrix (n_points × n_points)
        N: Subsampling iterations. Default: 100
        input_type: Type of input data ("point cloud" or "distance matrix")
        beta: Exponent parameter. Default: 0.0
        C: Constant parameter. Default: 10.0

    Returns:
        Estimated scale that can be used with agglomerative clustering

    Reference:
        http://www.jmlr.org/papers/volume19/17-291/17-291.pdf
    """
    num_pts = X.shape[0]
    m = int(num_pts / np.exp((1 + beta) * np.log(np.log(num_pts) / np.log(C))))
    delta = 0.0

    for _ in range(N):
        subpop = np.random.choice(num_pts, size=m, replace=False)
        d = (
            directed_hausdorff(X, X[subpop, :])[0]
            if input_type == "point cloud"
            else np.max(np.min(X[:, subpop], axis=1), axis=0)
        )
        delta += d / N
    return delta


class DistanceToMeasure(BaseEstimator, TransformerMixin):
    """
    Distance-to-measure density estimator.

    Estimates density using k-nearest neighbors distances.
    """

    def __init__(self, n_neighbors: int = 30, input_type: InputType = "point cloud"):
        """
        Initialize the DistanceToMeasure estimator.

        Args:
            n_neighbors: Number of nearest neighbors for density estimation
            input_type: Type of input data ("point cloud" or "distance matrix")
        """
        self.input_type = input_type
        self.n_neighbors = n_neighbors

        metric = "euclidean" if input_type == "point cloud" else "precomputed"
        self.neighb = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)

    def fit(self, X: ArrayType, y: ArrayType = None) -> "DistanceToMeasure":
        """
        Fit the estimator on input data.

        Args:
            X: Input point cloud or distance matrix
            y: Ignored (exists for compatibility)

        Returns:
            self
        """
        self.neighb.fit(X)
        return self

    def score_samples(self, X: ArrayType, y: ArrayType = None) -> ArrayType:
        """
        Compute density estimates for input samples.

        Args:
            X: Input samples to compute density for
            y: Ignored (exists for compatibility)

        Returns:
            Density estimates for each input point
        """
        distances, _ = self.neighb.kneighbors(X)
        return 1 / np.sqrt(np.mean(np.square(distances), axis=1))
