# benchmark_isomap_total.py

import time
import numpy as np

from sklearn.datasets import make_swiss_roll
from sklearn.neighbors import kneighbors_graph

from my_isomap.my_isomap import Isomap as MyIsomap
from isomap.isomap import Isomap as SklearnIsomap


def timed_fit(model, X, n_runs=5):
    times = []

    for _ in range(n_runs):
        t0 = time.perf_counter()
        model.fit_transform(X)
        t1 = time.perf_counter()

        times.append(t1 - t0)

    return np.mean(times), np.std(times)


def benchmark(n_samples, n_neighbors=10):

    X, _ = make_swiss_roll(
        n_samples=n_samples,
        noise=0.05,
        random_state=42,
    )

    G = kneighbors_graph(
        X,
        n_neighbors=n_neighbors,
        mode="distance",
    )

    n = G.shape[0]
    m = G.nnz

    std_time, _ = timed_fit(
        SklearnIsomap(
            n_neighbors=n_neighbors,
            n_components=2,
            path_method="D",
        ),
        X,
        n_runs=5,
    )

    bucket_time, _ = timed_fit(
        MyIsomap(
            n_neighbors=n_neighbors,
            n_components=2,
            path_method="D",
        ),
        X,
        n_runs=5,
    )

    speedup = std_time / bucket_time

    print(
        f"{n},"
        f"{m},"
        f"{std_time:.6f},"
        f"{bucket_time:.6f},"
        f"{speedup:.6f}"
    )


if __name__ == "__main__":

    print("n,m,std_time,bucket_time,speedup")

    for n in [
        500,
        1000,
        2000,
        5000,
        # 10000,
    ]:
        benchmark(
            n_samples=n,
            n_neighbors=10,
        )