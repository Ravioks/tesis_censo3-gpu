import os
import sys
import time
import urllib.request

import numpy as np

import censo_3n_ref as sec
import censo_3n2a_gpu as gpu

DATASETS = {
    "livejournal": {
        "nombre": "com-LiveJournal",
        "url": "https://snap.stanford.edu/data/bigdata/communities/com-lj.ungraph.txt.gz",
        "archivo": "com-lj.ungraph.txt.gz",
        "pagina": "https://snap.stanford.edu/data/com-LiveJournal.html",
        "nodos": 3_997_962,
        "aristas": 34_681_189,
        "triangulos": 177_820_130,
    },
    "orkut": {
        "nombre": "com-Orkut",
        "url": "https://snap.stanford.edu/data/bigdata/communities/com-orkut.ungraph.txt.gz",
        "archivo": "com-orkut.ungraph.txt.gz",
        "pagina": "https://snap.stanford.edu/data/com-Orkut.html",
        "nodos": 3_072_441,
        "aristas": 117_185_083,
        "triangulos": 627_584_181,
    },
    "twitter": {
        "nombre": "ego-Twitter",
        "url": "https://snap.stanford.edu/data/twitter_combined.txt.gz",
        "archivo": "twitter_combined.txt.gz",
        "pagina": "https://snap.stanford.edu/data/ego-Twitter.html",
        "nodos": None,
        "aristas": None,
        "triangulos": None,
    },
}


def descargar(clave, carpeta="datasets"):
    info = DATASETS[clave]
    os.makedirs(carpeta, exist_ok=True)
    ruta = os.path.join(carpeta, info["archivo"])
    if os.path.exists(ruta) and os.path.getsize(ruta) > 0:
        return ruta, False
    temporal = ruta + ".parcial"
    inicio = time.perf_counter()

    def progreso(bloques, tam_bloque, total):
        bajado = bloques * tam_bloque
        mb = bajado / 2**20
        vel = mb / max(1e-9, time.perf_counter() - inicio)
        if total > 0:
            sys.stdout.write(f"\r  descargando {info['archivo']}: {min(100, 100 * bajado / total):5.1f}% "
                             f"({mb:,.0f} de {total / 2**20:,.0f} MB, {vel:,.1f} MB/s)")
        else:
            sys.stdout.write(f"\r  descargando {info['archivo']}: {mb:,.0f} MB ({vel:,.1f} MB/s)")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(info["url"], temporal, progreso)
    except Exception as e:
        if os.path.exists(temporal):
            os.remove(temporal)
        raise RuntimeError(
            f"No se pudo descargar {info['url']} ({e}). Descárgalo a mano desde {info['pagina']} "
            f"y déjalo en {ruta}"
        ) from e
    sys.stdout.write("\n")
    os.replace(temporal, ruta)
    return ruta, True


def leer_lista_aristas(ruta, tam_bloque=20_000_000):
    import pandas as pd

    us, vs = [], []
    lector = pd.read_csv(ruta, sep=r"\s+", comment="#", header=None, usecols=[0, 1],
                         dtype=np.int64, engine="c", chunksize=tam_bloque)
    for df in lector:
        us.append(df[0].to_numpy())
        vs.append(df[1].to_numpy())
    if not us:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    return np.concatenate(us), np.concatenate(vs)


def normalizar_aristas(u, v):
    distintos = u != v
    u, v = u[distintos], v[distintos]
    total = len(u)
    if total == 0:
        return np.empty(0, np.int32), np.empty(0, np.int32), 0
    maximo = int(max(u.max(), v.max()))
    if maximo < 8 * total + 1_000_000:
        presentes = np.zeros(maximo + 1, dtype=bool)
        presentes[u] = True
        presentes[v] = True
        mapa = np.cumsum(presentes, dtype=np.int64) - 1
        n = int(mapa[-1]) + 1
        u, v = mapa[u], mapa[v]
    else:
        ids, inversa = np.unique(np.concatenate([u, v]), return_inverse=True)
        n = len(ids)
        u, v = inversa[:total].astype(np.int64), inversa[total:].astype(np.int64)
    claves = np.unique(np.minimum(u, v) * n + np.maximum(u, v))
    return (claves // n).astype(np.int32), (claves % n).astype(np.int32), n


def cargar_dataset(clave, etiquetas_nodo=5, etiquetas_arista=2, semilla=42, carpeta="datasets"):
    info = DATASETS[clave]
    nombres_nodo = sec.alfabeto(etiquetas_nodo, True)
    nombres_arista = sec.alfabeto(etiquetas_arista, False)
    cache = os.path.join(carpeta, f"{clave}_L{len(nombres_nodo)}_E{len(nombres_arista)}_s{semilla}.npz")
    tiempos = {}
    if os.path.exists(cache):
        t0 = time.perf_counter()
        csr = gpu.cargar_npz(cache, validar=False)
        csr.nombre = info["nombre"]
        tiempos["carga_cache_s"] = time.perf_counter() - t0
        return csr, tiempos, {"desde_cache": True, "cache": cache}

    t0 = time.perf_counter()
    ruta, descargado = descargar(clave, carpeta)
    tiempos["descarga_s"] = time.perf_counter() - t0 if descargado else 0.0

    t0 = time.perf_counter()
    u, v = leer_lista_aristas(ruta)
    tiempos["lectura_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    a, b, n = normalizar_aristas(u, v)
    del u, v
    rng = np.random.default_rng(semilla)
    et_nodo = rng.integers(0, len(nombres_nodo), size=n, dtype=np.int32)
    et_arista = rng.integers(0, len(nombres_arista), size=len(a), dtype=np.int32)
    csr = gpu.csr_desde_aristas(a, b, n, et_nodo, et_arista, nombres_nodo, nombres_arista, info["nombre"])
    del a, b
    tiempos["construccion_csr_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    gpu.guardar_npz(csr, cache, comprimir=False)
    tiempos["guardado_cache_s"] = time.perf_counter() - t0
    return csr, tiempos, {"desde_cache": False, "cache": cache, "archivo": ruta}
