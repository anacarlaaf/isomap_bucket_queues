import time
import numpy as np
import pandas as pd

from sklearn.datasets import make_swiss_roll, make_s_curve
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import Isomap
from scipy.sparse.csgraph import shortest_path


def generate_manifold(n_samples, manifold="swiss_roll"):
    if manifold == "swiss_roll":
        X, _ = make_swiss_roll(
            n_samples=n_samples,
            noise=0.05,
            random_state=42
        )
    elif manifold == "s_curve":
        X, _ = make_s_curve(
            n_samples=n_samples,
            noise=0.05,
            random_state=42
        )
    else:
        raise ValueError(manifold)

    return X


def benchmark_isomap(
    n_samples,
    n_neighbors=10,
    n_components=2
):
    X = generate_manifold(n_samples)

    result = {
        "n_samples": n_samples
    }

    # ------------------------
    # 1. kNN graph
    # ------------------------
    t0 = time.perf_counter()

    nbrs = NearestNeighbors(
        n_neighbors=n_neighbors,
        algorithm="auto"
    )

    nbrs.fit(X)

    graph = nbrs.kneighbors_graph(
        X,
        mode="distance"
    )

    result["knn_time"] = time.perf_counter() - t0

    # ------------------------
    # 2. Geodesic distances
    # ------------------------
    t0 = time.perf_counter()

    D = shortest_path(
        graph,
        directed=False
    )

    result["shortest_path_time"] = (
        time.perf_counter() - t0
    )

    # ------------------------
    # 3. Full Isomap
    # ------------------------
    t0 = time.perf_counter()

    iso = Isomap(
        n_neighbors=n_neighbors,
        n_components=n_components
    )

    iso.fit_transform(X)

    result["full_isomap_time"] = (
        time.perf_counter() - t0
    )

    return result


sizes = [
    500,
    1000,
    2000,
    4000,
    8000,
]

results = []

for n in sizes:
    print(f"Running n={n}")

    try:
        r = benchmark_isomap(n)
        results.append(r)

    except MemoryError:
        print(f"MemoryError em n={n}")
        break

df = pd.DataFrame(results)

print("\nResultados:")
print(df)

df["knn_pct"] = (
    100 * df["knn_time"] /
    df["full_isomap_time"]
)

df["shortest_path_pct"] = (
    100 * df["shortest_path_time"] /
    df["full_isomap_time"]
)

print("\nPercentual do tempo total:")
print(
    df[
        [
            "n_samples",
            "knn_pct",
            "shortest_path_pct"
        ]
    ]
)

