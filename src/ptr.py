"""
Implementation of Proper Topological Regions (PTR) for Initial Data Selection
@author: Lies Hadjadj
All rights reserved
"""

import gc
import time as t
from collections import defaultdict

import numpy as np
import gudhi as gd
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_array
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score


class PTRSelector:
    """
    PTR (Proper Topological Regions) selector for initial point selection in active learning.

    This class implements the method described in "Efficient Initial Data Selection and
    Labeling for Multi-Class Classification Using Topological Analysis" (ECAI 2024).
    """

    def __init__(
        self,
        tau=None,
        n_clusters=None,
        verbose=False,
        density=None,
        dist=None,
        rad=6.5,
        m=0.09,
        bias=1,
    ):
        """
        Initialize PTR selector

        Args:
            tau: Persistence threshold. If None, automatically determined
            n_clusters: Number of clusters/regions to identify
            verbose: Whether to print progress information
            density: Pre-computed density estimates (optional)
            dist: Pre-computed pairwise distances (optional)
            rad: Radius parameter for neighborhood graph
            m: Density scaling parameter
            bias: Bias term for density estimation
        """
        self.rad = rad
        self.tau = tau
        self.n_clusters = n_clusters
        self.density_ = density
        self.dist_ = dist
        self.m = m
        self.bias = bias
        self.verbose = verbose

    def __min_birth_max_death(self, persistence):
        """Get min birth and max death times from persistence diagram"""
        max_death = 0
        min_birth = persistence[0][1][0]
        for interval in reversed(persistence):
            if float(interval[1][1]) != float("inf"):
                if float(interval[1][1]) > max_death:
                    max_death = float(interval[1][1])
            if float(interval[1][0]) > max_death:
                max_death = float(interval[1][0])
            if float(interval[1][0]) < min_birth:
                min_birth = float(interval[1][0])
        return (min_birth, max_death)

    def plot_persistence_diagram(self, persistence, alpha=0.6, band_boot=0):
        """Plot persistence diagram with confidence band"""
        palette = [
            "#ff0000",
            "#00ff00",
            "#0000ff",
            "#00ffff",
            "#ff00ff",
            "#ffff00",
            "#000000",
            "#880000",
            "#008800",
            "#000088",
            "#888800",
            "#880088",
            "#008888",
        ]

        (min_birth, max_death) = self.__min_birth_max_death(persistence)
        delta = (max_death - min_birth) / 10.0
        infinity = max_death + delta
        axis_start = min_birth - delta

        x = np.linspace(axis_start, infinity, 1000)
        plt.plot(x, x, color="k", linewidth=1.0)
        plt.plot(x, [infinity] * len(x), linewidth=1.0, color="k", alpha=alpha)
        plt.fill_between(x, x, x + band_boot, alpha=0.3, facecolor="red")
        plt.text(axis_start, infinity, r"$\infty$", color="k", alpha=alpha)

        for interval in reversed(persistence):
            if float(interval[1][1]) != float("inf"):
                plt.scatter(
                    interval[1][0],
                    interval[1][1],
                    alpha=alpha,
                    color=palette[interval[0]],
                )
            else:
                plt.scatter(
                    interval[1][0], infinity, alpha=alpha, color=palette[interval[0]]
                )

        plt.title("Persistence diagram")
        plt.xlabel("Birth")
        plt.ylabel("Death")
        plt.axis([axis_start, infinity, axis_start, infinity + delta])
        plt.show()

    def rolling_window(self, a, window):
        """Compute rolling window on array"""
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    def find(self, i, parents):
        """Find operation for union-find data structure"""
        if parents[i] == i:
            return i
        return self.find(parents[i], parents)

    def union(self, i, j, parents, f):
        """Union operation for union-find data structure"""
        if f[i] > f[j]:
            parents[j] = i
        else:
            parents[i] = j

    @ignore_warnings(category=ConvergenceWarning)
    def fit(self, X, y=None):
        """
        Fit the PTR selector to data

        Args:
            X: Input features of shape (n_samples, n_features)
            y: Target labels (optional)

        Returns:
            X_train: Selected training points
            y_train: Labels for selected points
        """
        X = check_array(X, accept_sparse="csr")
        np.random.seed(2020)
        num_pts = X.shape[0]

        # Compute density estimation if not provided
        if self.density_ is None:
            tic = t.perf_counter()
            k = int(num_pts / 10)
            nbrs = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(X)
            self.dist_, self.idx_ = nbrs.kneighbors()
            self.density_ = 1 / np.sqrt(np.mean(np.square(self.dist_), axis=1))
            gc.collect()
            if self.verbose:
                print(
                    f"Computing density estimation: {t.perf_counter() - tic:.2f} seconds"
                )

        # Computing underlying graph
        tic = t.perf_counter()
        distance = self.dist_ - np.min(self.dist_)
        density = self.density_ - np.min(self.density_)
        radius = (
            np.power(self.bias - np.reshape(density, (num_pts, 1)), 1 / self.m)
            * self.rad
        )
        G = [list(a[b]) for (a, b) in zip(self.idx_, distance < radius)]
        gc.collect()

        if self.verbose:
            print(
                "Computing underlying rips graph : ", t.perf_counter() - tic, "seconds"
            )

        # Sorting points by density
        sorted_idxs = np.flip(np.argsort(self.density_))
        position = np.arange(num_pts)
        for i in range(num_pts):
            position[sorted_idxs[i]] = i
        d = None
        if self.tau is None:
            # Computing persistence
            tic = t.perf_counter()
            st = gd.SimplexTree()
            for i in range(num_pts):
                st.insert([i], filtration=-self.density_[i])
            for i in range(num_pts):
                for j in G[i]:
                    st.insert(
                        [i, j], filtration=max(-self.density_[i], -self.density_[j])
                    )
            d = st.persistence()  # Assign d
            dgm = st.persistence_intervals_in_dimension(0)
            self.persistences = np.flip(
                np.sort([abs(y - x) if y != float("Inf") else abs(x) for (x, y) in dgm])
            )
            if self.verbose:
                print("Computing persistence : ", t.perf_counter() - tic, "seconds")

            # Finding tau : 2 * persistence = prominence
            if len(self.persistences) <= self.n_clusters:
                self.tau = 0
            ## ToMATo estimation
            # elif len(self.persistences) <= 200:
            #    p = np.argmax(self.persistences[1:-2] - self.persistences[2:-1]) + 2
            #    self.tau = self.persistences[p] / 2
            else:
                w_roll = self.rolling_window(
                    self.persistences, int(len(self.persistences) / 10)
                )
                w_roll_std = np.std(w_roll, 1)
                p = (
                    np.argmax(w_roll_std)
                    + np.argmin(w_roll_std[np.argmax(w_roll_std) :])
                    + int(len(self.persistences) / 20)
                )
                self.tau = self.persistences[p] / 2
            gc.collect()

        if self.verbose:
            self.plot_persistence_diagram(d, band_boot=self.tau)
            print("tau : ", self.tau)

        # Union find hill climbing process
        merge = 0
        tic = t.perf_counter()
        parents = -np.ones(num_pts, dtype=np.int32)
        for i in range(num_pts):
            current_pt = sorted_idxs[i]
            neighbors = G[current_pt]
            higher_neighbors = (
                [n for n in neighbors if position[n] <= i] if len(neighbors) > 0 else []
            )

            if higher_neighbors == []:
                parents[current_pt] = current_pt
            else:
                g = higher_neighbors[
                    np.argmax(self.density_[np.array(higher_neighbors)])
                ]  # highest neighbor
                pg = self.find(g, parents)
                parents[current_pt] = pg  # parent of the highest neighbor
                higher_neighbors.remove(g)

                for neighbor in higher_neighbors:
                    pn = self.find(neighbor, parents)
                    val = min(self.density_[pg], self.density_[pn])
                    if (
                        pg != pn
                        and val < self.density_[current_pt] + self.tau
                        and val > self.tau
                    ):
                        self.union(pg, pn, parents, self.density_)
                        merge += 1
        gc.collect()
        if self.verbose:
            print("MERGES : ", merge)

        # find clusters
        self.parents = np.array([self.find(n, parents) for n in range(num_pts)])
        # print('size of clusters : ', len(np.unique(self.parents)))
        self.clusters_ = np.where(
            self.density_[self.parents] > self.tau,
            self.parents,
            -2 * np.ones(self.parents.shape, dtype=np.int32),
        )
        (reps, sizes) = np.unique(self.clusters_, return_counts=True)

        if len(np.unique(reps)) < 2:
            print("ERROR : one connected component !!")
            return True, True
        if len(np.unique(sizes)) < 2:
            print("ERROR : no connected components !!")
            return True, True

        reps = reps[np.flip(np.argsort(sizes))]
        sizes = np.flip(np.sort(sizes))

        if self.n_clusters is None:
            if sizes[0] < 4:
                self.n_clusters = np.where(sizes >= sizes[0])[0][-1] + 1
            else:
                self.n_clusters = np.where(sizes >= 4)[0][-1] + 1
        else:
            self.n_clusters = min(len(reps) - 1, self.n_clusters)
        if self.verbose:
            print("B_MAX : ", len(reps) - 1)
            print("BUDGET : ", self.n_clusters)

        # Label main clusters and merge if needed
        self.labels_ = -1 * np.ones(self.parents.shape, dtype=np.int32)
        self.seed_ind = reps[: self.n_clusters]
        self.seed = (X[self.seed_ind, :], y[self.seed_ind])
        # self.score = self.rank[reps[:self.n_clusters]]
        labelled = defaultdict(list)
        for i in reps[: self.n_clusters]:
            self.labels_[self.clusters_ == i] = y[i]
            labelled[y[i]].append(i)
        (unique, counts) = np.unique(self.labels_, return_counts=True)
        if self.verbose and len(unique) <= len(np.unique(y)):
            print("WARNING: not all classes !!")
        X_train = X[self.labels_ != -1]
        y_train = self.labels_[self.labels_ != -1]
        return X_train, y_train


class BMeans:
    def __init__(self, n_clusters=None, rank=None, verbose=False):
        self.n_clusters = n_clusters
        self.rank = rank
        self.verbose = verbose

    def fit(self, X, y=None):
        X = check_array(X, accept_sparse="csr")
        y_ind = np.arange(len(y))
        self.clst = KMeans(n_clusters=self.n_clusters, random_state=2020).fit(X)
        self.labels_ = -1 * np.ones(y.shape, dtype=np.int32)

        if self.verbose:
            print("number of clusters k=", self.n_clusters)
            unique, counts = np.unique(self.clst.labels_, return_counts=True)
            print(unique, counts)

        seed_X = []
        seed_y = []
        seed_ind = []
        self.score = []
        for c in range(self.n_clusters):
            medoid = np.argmin(
                pairwise_distances(
                    self.clst.cluster_centers_[c, :].reshape((1, -1)),
                    X[self.clst.labels_ == c, :],
                )[0]
            )
            if self.verbose:
                print(
                    "mediod position : ",
                    medoid,
                    " label : ",
                    y[self.clst.labels_ == c][medoid],
                )
            self.labels_[self.clst.labels_ == c] = y[self.clst.labels_ == c][medoid]
            seed_X.append(X[self.clst.labels_ == c, :][medoid])
            seed_y.append(y[self.clst.labels_ == c][medoid])
            seed_ind.append(y_ind[self.clst.labels_ == c][medoid])
            # self.score.append(self.rank[self.clst.labels_ == c][medoid])

        self.seed = (np.array(seed_X), np.array(seed_y))
        self.seed_ind = np.array(seed_ind)
        if self.verbose:
            unique, counts = np.unique(self.labels_, return_counts=True)
            print(unique, counts)
            for c in unique:
                print(
                    "accuracy_score of class ",
                    c,
                    " :",
                    accuracy_score(
                        y[self.labels_ == c], self.labels_[self.labels_ == c]
                    ),
                )

    def rank_score(self):
        return np.mean(self.score)
