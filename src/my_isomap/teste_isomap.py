# teste_isomap.py

import numpy as np

from scipy.spatial.distance import pdist

from sklearn.datasets import make_swiss_roll
from sklearn.manifold import Isomap as SklearnIsomap

from src.my_isomap.my_isomap import Isomap as MyIsomap


# --------------------------------------------------
# Dados
# --------------------------------------------------

X, _ = make_swiss_roll(
    n_samples=500,
    noise=0.05,
    random_state=42
)


# --------------------------------------------------
# Modelos
# --------------------------------------------------

iso_ref = SklearnIsomap(
    n_neighbors=10,
    n_components=2,
    path_method="D",
)

iso_my = MyIsomap(
    n_neighbors=10,
    n_components=2,
    path_method="D",
)


# --------------------------------------------------
# Fit
# --------------------------------------------------

Y_ref = iso_ref.fit_transform(X)
Y_my = iso_my.fit_transform(X)


# --------------------------------------------------
# Distâncias geodésicas
# --------------------------------------------------

D_ref = iso_ref.dist_matrix_
D_my = iso_my.dist_matrix_

geo_abs = np.abs(D_ref - D_my)

geo_max = np.max(geo_abs)
geo_mean = np.mean(geo_abs)

geo_rel_max = np.max(
    geo_abs / np.maximum(np.abs(D_ref), 1e-12)
)

geo_rel_mean = np.mean(
    geo_abs / np.maximum(np.abs(D_ref), 1e-12)
)

print("\n========== MATRIZ GEODÉSICA ==========")

print("distância máxima:", D_ref.max())
print("distância média :", D_ref.mean())

print("\nerro máximo absoluto :", geo_max)
print("erro médio absoluto  :", geo_mean)

print("\nerro máximo relativo :", geo_rel_max)
print("erro médio relativo  :", geo_rel_mean)

fro_error = (
    np.linalg.norm(D_ref - D_my, ord="fro")
    / np.linalg.norm(D_ref, ord="fro")
)

print("\nerro relativo Frobenius:", fro_error)


# --------------------------------------------------
# Embedding
# --------------------------------------------------

PD_ref = pdist(Y_ref)
PD_my = pdist(Y_my)

embed_abs = np.abs(PD_ref - PD_my)

embed_max = np.max(embed_abs)
embed_mean = np.mean(embed_abs)

embed_rel_max = np.max(
    embed_abs / np.maximum(np.abs(PD_ref), 1e-12)
)

embed_rel_mean = np.mean(
    embed_abs / np.maximum(np.abs(PD_ref), 1e-12)
)

print("\n========== EMBEDDING ==========")

print("erro máximo absoluto :", embed_max)
print("erro médio absoluto  :", embed_mean)

print("\nerro máximo relativo :", embed_rel_max)
print("erro médio relativo  :", embed_rel_mean)


# --------------------------------------------------
# Critério simples
# --------------------------------------------------

if fro_error < 1e-3 and embed_rel_mean < 1e-3:
    print("\n✓ Compatibilidade excelente")
elif fro_error < 1e-2 and embed_rel_mean < 1e-2:
    print("\n✓ Compatibilidade muito boa")
else:
    print("\n⚠ Diferenças relevantes detectadas")