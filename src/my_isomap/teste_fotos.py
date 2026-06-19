# benchmark_multipie.py

import time
import numpy as np
import os
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap as SklearnIsomap
from src.my_isomap.my_isomap import Isomap as MyIsomap


def load_multipie(folder, illumination="07"):
    files = sorted(os.listdir(folder))
    images, subjects, poses = [], [], []

    for fname in files:
        if not fname.endswith('.png'):
            continue
        parts = fname.replace('_crop_128.png', '').split('_')
        if len(parts) < 5:
            continue
        if parts[4] != illumination:
            continue

        img = Image.open(os.path.join(folder, fname)).convert('RGB')
        images.append(np.array(img, dtype=np.float32).ravel() / 255.0)
        subjects.append(int(parts[0]))
        poses.append(int(parts[3]))

    return np.array(images), np.array(subjects), np.array(poses)


def timed_fit(model, X, n_runs=3):
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model.fit_transform(X)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return np.mean(times), np.std(times)


def benchmark(X_pca, n_samples, n_neighbors=10, n_runs=3):
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_pca), size=n_samples, replace=False)
    X_sub = X_pca[idx]

    print("\n" + "=" * 60)
    print(f"n_samples   : {n_samples}")
    print(f"n_neighbors : {n_neighbors}")
    print(f"n_features  : {X_sub.shape[1]} (pós-PCA)")
    print("=" * 60)

    sklearn_mean, sklearn_std = timed_fit(
        SklearnIsomap(n_neighbors=n_neighbors, n_components=2, path_method="D"),
        X_sub, n_runs,
    )

    my_mean, my_std = timed_fit(
        MyIsomap(n_neighbors=n_neighbors, n_components=2, path_method="D"),
        X_sub, n_runs,
    )

    speedup = sklearn_mean / my_mean

    print(f"\nResultados ({n_runs} execuções)")
    print(f"sklearn : {sklearn_mean:.4f} ± {sklearn_std:.4f} s")
    print(f"bucket  : {my_mean:.4f} ± {my_std:.4f} s")
    print(f"speedup : {speedup:.2f}x")


if __name__ == "__main__":
    FOLDER      = "Multi_Pie/HR_128"
    ILLUMINATION = "07"
    PCA_COMPONENTS = 50
    N_NEIGHBORS = 10

    print("Carregando imagens...")
    X, subjects, poses = load_multipie(FOLDER, ILLUMINATION)
    print(f"Total carregado: {len(X)} imagens, shape={X.shape}")
    print(f"Poses únicas   : {sorted(np.unique(poses))}")
    print(f"Sujeitos únicos: {len(np.unique(subjects))}")

    print(f"\nPCA ({PCA_COMPONENTS} componentes)...")
    pca = PCA(n_components=PCA_COMPONENTS, random_state=42)
    X_pca = pca.fit_transform(X)
    print(f"Variância explicada: {pca.explained_variance_ratio_.sum():.2%}")

    for n in [500, 1000, 2000, len(X)]:
        if n > len(X):
            continue
        benchmark(X_pca, n_samples=n, n_neighbors=N_NEIGHBORS)