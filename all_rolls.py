from pathlib import Path

from sklearn.datasets import make_swiss_roll
from sklearn.neighbors import NearestNeighbors


OUTPUT_DIR = Path("graphs")

SIZES = [
    1000,
    2000,
    5000,
    10000,
    20000,
    50000
]

K = 10


def save_graph(
    n_samples: int,
    k_neighbors: int,
    output_file: Path,
):
    print(f"Generating N={n_samples}")

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

        f.write(
            f"{n_samples} {len(edges)}\n"
        )

        for u, v, w in edges:

            f.write(
                f"{u} {v} {w:.15f}\n"
            )

    print(
        f"saved: {output_file}"
    )
    print(
        f"vertices={n_samples} "
        f"edges={len(edges)}"
    )


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for n in SIZES:

        filename = (
            OUTPUT_DIR
            / f"graph_{n}.txt"
        )

        save_graph(
            n_samples=n,
            k_neighbors=K,
            output_file=filename
        )


if __name__ == "__main__":
    main()