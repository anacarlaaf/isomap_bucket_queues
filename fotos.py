import argparse
import os
import numpy as np

from PIL import Image
from sklearn.neighbors import NearestNeighbors


def load_images(image_dir, max_images=None, size=(128, 128)):
    """
    Lê imagens do dataset e transforma em vetores.
    """

    X = []
    paths = []

    for root, _, files in os.walk(image_dir):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                path = os.path.join(root, f)

                try:
                    img = Image.open(path).convert("L")  # grayscale
                    img = img.resize(size)

                    vec = np.asarray(img, dtype=np.float32).flatten() / 255.0

                    X.append(vec)
                    paths.append(path)

                    if max_images and len(X) >= max_images:
                        return np.array(X), paths

                except Exception as e:
                    print(f"Erro em {path}: {e}")

    return np.array(X), paths


def generate_graph_images(image_dir, k_neighbors, output_file, max_images=None):
    print("Carregando imagens...")

    X, paths = load_images(image_dir, max_images=max_images)

    n_samples = X.shape[0]

    print(f"Imagens carregadas: n={n_samples}, dim={X.shape[1]}")

    print("Construindo kNN...")

    nbrs = NearestNeighbors(
        n_neighbors=k_neighbors,
        algorithm="auto",
        metric="euclidean"
    )

    nbrs.fit(X)

    distances, indices = nbrs.kneighbors(X)

    edges = []

    for u in range(n_samples):
        for j in range(1, k_neighbors):
            v = int(indices[u, j])
            w = float(distances[u, j])

            # evita duplicatas
            if u < v:
                edges.append((u, v, w))

    print("Salvando grafo...")

    with open(output_file, "w") as f:
        f.write(f"{n_samples} {len(edges)}\n")

        for u, v, w in edges:
            f.write(f"{u} {v} {w:.15f}\n")

    print(f"Grafo salvo em {output_file}")
    print(f"n={n_samples}, m={len(edges)}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--data", required=True, help="Path para Multi_Pie")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n", type=int, default=None, help="limite de imagens")
    parser.add_argument("--out", default=f"fotos/graph_img_{parser.parse_args().n}.txt")

    args = parser.parse_args()

    generate_graph_images(
        image_dir=args.data,
        k_neighbors=args.k,
        output_file=args.out,
        max_images=args.n
    )