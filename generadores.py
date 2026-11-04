import random
import time

import numpy as np
from numba import njit

import censo_3n_ref as sec
import censo_3n2a_gpu as gpu
import datos_reales as dr

TIPOS = ["ba", "aleatorio", "completo"]


@njit
def _ba_aristas(n, k, semilla):
    np.random.seed(semilla)
    m = (n - k) * k
    a = np.empty(m, np.int64)
    b = np.empty(m, np.int64)
    repetidos = np.empty(2 * m + k, np.int64)
    for i in range(k):
        repetidos[i] = i
    tam = k
    e = 0
    for v in range(k, n):
        for _ in range(k):
            a[e] = v
            b[e] = repetidos[np.random.randint(0, tam)]
            e += 1
        for j in range(e - k, e):
            repetidos[tam] = v
            repetidos[tam + 1] = b[j]
            tam += 2
    return a, b


def _aleatorio_aristas(n, m, rng):
    m = min(m, n * (n - 1) // 2)
    claves = np.empty(0, np.int64)
    while len(claves) < m:
        falta = m - len(claves)
        u = rng.integers(0, n, size=int(falta * 1.2) + 16, dtype=np.int64)
        v = rng.integers(0, n, size=len(u), dtype=np.int64)
        ok = u != v
        nuevas = np.minimum(u[ok], v[ok]) * n + np.maximum(u[ok], v[ok])
        claves = np.unique(np.concatenate([claves, nuevas]))
    if len(claves) > m:
        claves = np.sort(rng.choice(claves, m, replace=False))
    return claves // n, claves % n


def csr_sintetico(tipo, n, m=None, k=3, L=3, E=2, semilla=42):
    if tipo not in TIPOS:
        raise ValueError(f"Generador rápido no disponible para '{tipo}' (opciones: {', '.join(TIPOS)})")
    rng = np.random.default_rng(semilla)
    if tipo == "ba":
        k = max(1, k)
        n = max(n, k + 1)
        a, b = _ba_aristas(n, k, semilla)
        nombre = f"ba_np_n{n}_k{k}"
    elif tipo == "aleatorio":
        a, b = _aleatorio_aristas(n, m, rng)
        nombre = f"aleatorio_np_n{n}_m{min(m, n * (n - 1) // 2)}"
    else:
        a, b = np.triu_indices(n, 1)
        nombre = f"completo_np_n{n}"
    a, b, nodos = dr.normalizar_aristas(np.asarray(a, np.int64), np.asarray(b, np.int64))
    et_nodo = rng.integers(0, L, size=nodos, dtype=np.int32)
    et_arista = rng.integers(0, E, size=len(a), dtype=np.int32)
    return gpu.csr_desde_aristas(a, b, nodos, et_nodo, et_arista, sec.alfabeto(L, True), sec.alfabeto(E, False),
                                 f"{nombre}_L{L}_E{E}")


def grafo_etiquetas(L, E, n, m, semilla=2):
    rng = random.Random(semilla)
    en, ea = sec.alfabeto(L, True), sec.alfabeto(E, False)
    g = sec.Grafo(f"etiquetas_L{L}_E{E}_n{n}")
    for i in range(n):
        g.agregar_nodo(i, en[i % L])
    m = min(m, n * (n - 1) // 2)
    k = 0
    while len(g.aristas) < m:
        u, v = rng.randrange(n), rng.randrange(n)
        if u != v and frozenset((u, v)) not in g.pares:
            g.agregar_arista(u, v, ea[k % E])
            k += 1
    return g


def calentar():
    t0 = time.perf_counter()
    _ba_aristas(10, 2, 0)
    return time.perf_counter() - t0
