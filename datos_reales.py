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
    "ogbn-mag": {
        "nombre": "ogbn-mag",
        "url": "https://snap.stanford.edu/ogb/data/nodeproppred/mag.zip",
        "archivo": os.path.join("ogb", "mag.zip"),
        "pagina": "https://ogb.stanford.edu/docs/nodeprop/#ogbn-mag",
        "nodos": None,
        "aristas": None,
        "triangulos": None,
        "ogb": "mag",
    },
    "ogbn-products": {
        "nombre": "ogbn-products",
        "url": "https://snap.stanford.edu/ogb/data/nodeproppred/products.zip",
        "archivo": os.path.join("ogb", "products.zip"),
        "pagina": "https://ogb.stanford.edu/docs/nodeprop/#ogbn-products",
        "nodos": None,
        "aristas": None,
        "triangulos": None,
        "ogb": "products",
    },
}


def descargar(clave, carpeta="datasets"):
    info = DATASETS[clave]
    ruta = os.path.join(carpeta, info["archivo"])
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
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


def normalizar_aristas_etiquetadas(u, v, et_arista, et_nodo, etiquetas_arista):
    distintos = u != v
    lazos = int(len(u) - np.count_nonzero(distintos))
    u, v, et_arista = u[distintos], v[distintos], et_arista[distintos]
    presentes = np.zeros(len(et_nodo), dtype=bool)
    presentes[u] = True
    presentes[v] = True
    mapa = (np.cumsum(presentes, dtype=np.int64) - 1).astype(np.int32)
    n = int(mapa[-1]) + 1 if len(mapa) else 0
    u, v = mapa[u], mapa[v]
    E = int(etiquetas_arista)
    claves = np.minimum(u, v).astype(np.int64)
    claves *= n
    claves += np.maximum(u, v)
    claves *= E
    claves += et_arista
    del u, v, et_arista, mapa
    claves.sort()
    unico = np.ones(len(claves), dtype=bool)
    unico[1:] = claves[1:] != claves[:-1]
    claves = claves[unico]
    del unico
    pares = claves // E
    primero = np.ones(len(pares), dtype=bool)
    primero[1:] = pares[1:] != pares[:-1]
    estadisticas = {"lazos": lazos, "repetidas": int(len(distintos) - lazos - len(claves)),
                    "conflictos_etiqueta": int(len(claves) - np.count_nonzero(primero)),
                    "nodos_aislados": int(len(et_nodo) - n)}
    pares, et = pares[primero], (claves[primero] % E).astype(np.int32)
    return ((pares // n).astype(np.int32), (pares % n).astype(np.int32), et,
            et_nodo[presentes].astype(np.int32), n, estadisticas)


def _leer_columnas_zip(z, miembro, columnas, tam_bloque=10_000_000):
    import pandas as pd

    partes = [[] for _ in columnas]
    with z.open(miembro) as f:
        for df in pd.read_csv(f, compression="gzip", header=None, usecols=columnas, dtype=np.int32, engine="c",
                              chunksize=tam_bloque):
            for i, c in enumerate(columnas):
                partes[i].append(df[c].to_numpy())
    return [np.concatenate(p) if p else np.empty(0, np.int32) for p in partes]


def _leer_cabecera_zip(z, miembro):
    import pandas as pd

    with z.open(miembro) as f:
        return pd.read_csv(f, compression="gzip")


def leer_ogb_products(ruta_zip):
    import zipfile

    with zipfile.ZipFile(ruta_zip) as z:
        u, v = _leer_columnas_zip(z, "products/raw/edge.csv.gz", [0, 1])
        (et_nodo,) = _leer_columnas_zip(z, "products/raw/node-label.csv.gz", [0])
        categorias = _leer_cabecera_zip(z, "products/mapping/labelidx2productcategory.csv.gz")
    nombres_nodo = [str(x) for x in categorias.sort_values(categorias.columns[0])[categorias.columns[1]]]
    for i in range(len(nombres_nodo), int(et_nodo.max()) + 1 if len(et_nodo) else 0):
        nombres_nodo.append(f"cat{i}")
    return u, v, np.zeros(len(u), np.int32), et_nodo, nombres_nodo, sec.alfabeto(1, False)


def leer_ogb_mag(ruta_zip):
    import zipfile

    with zipfile.ZipFile(ruta_zip) as z:
        cuentas = _leer_cabecera_zip(z, "mag/raw/num-node-dict.csv.gz")
        tipos = list(cuentas.columns)
        tam = {t: int(cuentas[t].iloc[0]) for t in tipos}
        nombres_nodo = sorted(tipos)
        inicio, et_nodo, total = {}, [], 0
        for t in tipos:
            inicio[t] = total
            total += tam[t]
            et_nodo.append(np.full(tam[t], nombres_nodo.index(t), np.int32))
        et_nodo = np.concatenate(et_nodo)
        prefijo = "mag/raw/relations/"
        triples = sorted({n[len(prefijo):].split("/")[0] for n in z.namelist()
                          if n.startswith(prefijo) and n.endswith("/edge.csv.gz")})
        nombres_arista = sorted({t.split("___")[1] for t in triples})
        us, vs, ets = [], [], []
        for t in triples:
            origen, relacion, destino = t.split("___")
            a, b = _leer_columnas_zip(z, f"{prefijo}{t}/edge.csv.gz", [0, 1])
            us.append(a.astype(np.int64) + inicio[origen])
            vs.append(b.astype(np.int64) + inicio[destino])
            ets.append(np.full(len(a), nombres_arista.index(relacion), np.int32))
    return np.concatenate(us), np.concatenate(vs), np.concatenate(ets), et_nodo, nombres_nodo, nombres_arista


LECTORES_OGB = {"products": leer_ogb_products, "mag": leer_ogb_mag}


def cargar_ogb(clave, carpeta="datasets"):
    info = DATASETS[clave]
    cache = os.path.join(carpeta, f"{clave}_real.npz")
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
    u, v, et_arista, et_nodo, nombres_nodo, nombres_arista = LECTORES_OGB[info["ogb"]](ruta)
    crudas, nodos_crudos = len(u), len(et_nodo)
    tiempos["lectura_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    a, b, et, et_nodo, n, estadisticas = normalizar_aristas_etiquetadas(u, v, et_arista, et_nodo, len(nombres_arista))
    del u, v, et_arista
    csr = gpu.csr_desde_aristas(a, b, n, et_nodo, et, nombres_nodo, nombres_arista, info["nombre"])
    del a, b, et
    tiempos["construccion_csr_s"] = time.perf_counter() - t0
    estadisticas.update(nodos_crudos=nodos_crudos, aristas_crudas=crudas, nodos=csr.n, aristas=csr.nnz // 2)
    print(f"  {info['nombre']}: " + " | ".join(f"{k.replace('_', ' ')} {v:,}" for k, v in estadisticas.items()))

    t0 = time.perf_counter()
    gpu.guardar_npz(csr, cache, comprimir=False)
    sec.guardar_json(estadisticas, cache[:-4] + "_limpieza.json")
    tiempos["guardado_cache_s"] = time.perf_counter() - t0
    return csr, tiempos, {"desde_cache": False, "cache": cache, "archivo": ruta, **estadisticas}


def cargar_dataset(clave, etiquetas_nodo=5, etiquetas_arista=2, semilla=42, carpeta="datasets"):
    info = DATASETS[clave]
    if info.get("ogb"):
        return cargar_ogb(clave, carpeta)
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
