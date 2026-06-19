# benchmark_isomap_total.py

import time
import numpy as np

from sklearn.datasets import make_swiss_roll
from sklearn.manifold import Isomap as SklearnIsomap
from sklearn.neighbors import kneighbors_graph

from my_isomap import Isomap as MyIsomap

import cProfile, pstats

def profile_once(n_samples=5000, n_neighbors=10):
    from sklearn.datasets import make_swiss_roll
    X, _ = make_swiss_roll(n_samples=n_samples, noise=0.05, random_state=42)
    model = MyIsomap(n_neighbors=n_neighbors, n_components=2, path_method="D")
    
    pr = cProfile.Profile()
    pr.enable()
    model.fit_transform(X)
    pr.disable()
    
    stats = pstats.Stats(pr)
    stats.sort_stats('cumulative')
    stats.print_stats(15)


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

    # grafo usado pelo Isomap
    G = kneighbors_graph(
        X,
        n_neighbors=n_neighbors,
        mode="distance",
    )

    n_vertices = G.shape[0]
    n_edges = G.nnz
    max_weight = G.data.max()

    print("\n" + "=" * 70)
    print(f"n_samples   : {n_samples}")
    print(f"vertices    : {n_vertices}")
    print(f"arestas     : {n_edges}")
    print(f"peso máximo : {max_weight:.6f}")
    print("=" * 70)

    # --------------------------------------------------
    # sklearn
    # --------------------------------------------------

    sklearn_mean, sklearn_std = timed_fit(
        SklearnIsomap(
            n_neighbors=n_neighbors,
            n_components=2,
            path_method="D",
        ),
        X,
        n_runs=5,
    )

    # --------------------------------------------------
    # bucket queue
    # --------------------------------------------------

    my_mean, my_std = timed_fit(
        MyIsomap(
            n_neighbors=n_neighbors,
            n_components=2,
            path_method="D",
        ),
        X,
        n_runs=5,
    )

    speedup = sklearn_mean / my_mean

    print("\nResultados (5 execuções)")
    print(
        f"sklearn : {sklearn_mean:.4f} ± {sklearn_std:.4f} s"
    )
    print(
        f"bucket  : {my_mean:.4f} ± {my_std:.4f} s"
    )
    print(
        f"speedup : {speedup:.2f}x"
    )


if __name__ == "__main__":

    for n in [
        500,
        1000,
        2000,
        5000,
        #10000,
    ]:
        benchmark(
            n_samples=n,
            n_neighbors=10,
        )