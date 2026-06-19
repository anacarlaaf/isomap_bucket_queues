import argparse
import numpy as np

from sklearn.datasets import make_swiss_roll
from sklearn.neighbors import NearestNeighbors


def generate_graph(
    n_samples,
    k_neighbors,
    output_file
):
    X, _ = make_swiss_roll(
        n_samples=n_samples,
        noise=0.05,
        random_state=42
    )

    nbrs = NearestNeighbors(
        n_neighbors=k_neighbors,
        algorithm="auto"
    )

    nbrs.fit(X)

    distances, indices = nbrs.kneighbors(X)

    edges = []

    for u in range(n_samples):

        for j in range(1, k_neighbors):
            v = int(indices[u, j])
            w = float(distances[u, j])

            if u < v:
                edges.append((u, v, w))

    with open(output_file, "w") as f:

        f.write(f"{n_samples} {len(edges)}\n")

        for u, v, w in edges:
            f.write(f"{u} {v} {w:.15f}\n")

    print(
        f"Grafo salvo em {output_file}"
    )
    print(
        f"n={n_samples}, m={len(edges)}"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n",
        type=int,
        default=10000
    )

    parser.add_argument(
        "--k",
        type=int,
        default=10
    )

    parser.add_argument(
        "--out",
        default="graph.txt"
    )

    args = parser.parse_args()

    generate_graph(
        args.n,
        args.k,
        args.out
    )