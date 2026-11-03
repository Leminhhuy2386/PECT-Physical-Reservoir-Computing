"""
models.py -- model definitions shared by train.py and evaluate.py.

Not a pipeline stage. visualize.py must never import this module
(01_system.md Rule 3).
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, RidgeClassifier


# ----------------------------------------------------------------------------
# Software Echo State Network baseline
# ----------------------------------------------------------------------------
class SoftwareESN:
    """
    Conventional (software) Echo State Network used as the artificial-reservoir
    baseline.

    FAIRNESS NOTE (01_system.md, "Baseline fairness"):
    `subsample_step` MUST be 1. The reservoir is driven with the complete
    transient, one input sample per time step, so the baseline receives exactly
    the same information as the Physical RC readout. A previous version of this
    project used `X[:, ::20]`, giving the ESN 25 of 500 samples; the resulting
    performance gap measured the decimation, not the method.

    The readout state is the time-average of the reservoir trajectory, giving a
    `size`-dimensional state. With size=500 this matches the 500-D Physical RC
    state dimension exactly.
    """

    def __init__(self, size, spectral_radius=0.95, leak_rate=0.5, input_scaling=0.5,
                 sparsity=0.2, seed=42, subsample_step=1):
        assert subsample_step == 1, (
            "subsample_step must be 1: decimating the baseline's input invalidates "
            "the comparison (01_system.md Rule 10)."
        )
        self.size = int(size)
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate
        self.input_scaling = input_scaling
        self.sparsity = sparsity
        self.seed = seed
        self.subsample_step = subsample_step

        rng = np.random.RandomState(seed)
        self.W_in = (rng.rand(1, self.size) * 2.0 - 1.0) * input_scaling
        W = rng.rand(self.size, self.size) * 2.0 - 1.0
        W[rng.rand(self.size, self.size) > sparsity] = 0.0
        radius = np.max(np.abs(np.linalg.eigvals(W)))
        self.W_res = W * (spectral_radius / radius) if radius > 1e-12 else W

    def states(self, X, block=4096, verbose=False):
        """Drive the reservoir with the full transient. Returns (n, size) float32."""
        n, T = X.shape
        out = np.empty((n, self.size), dtype=np.float32)
        Wr = self.W_res.T
        for s in range(0, n, block):
            e = min(s + block, n)
            Xb = X[s:e]
            x = np.zeros((e - s, self.size))
            acc = np.zeros((e - s, self.size))
            for t in range(T):
                pre = Xb[:, t : t + 1] @ self.W_in + x @ Wr
                x = (1.0 - self.leak_rate) * x + self.leak_rate * np.tanh(pre)
                acc += x
            out[s:e] = (acc / float(T)).astype(np.float32)
            if verbose:
                print(f"      ESN{self.size}: {e}/{n}", flush=True)
        return out

    def input_dim_seen(self, T):
        return T // self.subsample_step


# ----------------------------------------------------------------------------
# feature extractors  (all fitted on the FIT split only)
# ----------------------------------------------------------------------------
class PeakFeature:
    """Peak transient amplitude -- a genuine argmax, not a fixed sample index."""

    name = "peak1"
    state_dim = 1

    def fit(self, X):
        return self

    def transform(self, X):
        return X.max(axis=1, keepdims=True)


class PCAFeature:
    name = "pca10"

    def __init__(self, n_components=10, seed=42):
        self.pca = PCA(n_components=n_components, random_state=seed)
        self.state_dim = n_components

    def fit(self, X):
        self.pca.fit(X)
        return self

    def transform(self, X):
        return self.pca.transform(X)


class RawState:
    """The physical reservoir state itself: the whole normalised transient."""

    name = "rc500"

    def __init__(self, T):
        self.state_dim = T

    def fit(self, X):
        return self

    def transform(self, X):
        return X


# ----------------------------------------------------------------------------
# readouts
# ----------------------------------------------------------------------------
def make_classifier(alpha, seed=42):
    return RidgeClassifier(alpha=alpha, class_weight="balanced",
                           solver="cholesky", random_state=seed)


def make_regressor(alpha):
    return Ridge(alpha=alpha, solver="cholesky")


def sweep_threshold(y, scores, lo, hi, n):
    """
    Return (best_threshold, best_macro_f1, thresholds, f1_curve).

    Must be called on a VALIDATION split. Calling it on test labels fits the test
    set and invalidates every "unseen test data" statement (01_system.md Rule 10).
    """
    from sklearn.metrics import f1_score

    ths = np.linspace(lo, hi, n)
    f1 = np.array([f1_score(y, (scores >= t).astype(int), average="macro",
                            zero_division=0) for t in ths])
    j = int(np.argmax(f1))
    return float(ths[j]), float(f1[j]), ths, f1
