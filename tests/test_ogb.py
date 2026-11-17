import gzip
import io
import zipfile

import numpy as np

import censo_3n as c3
import censo_3n2a_gpu as gpu
import censo_3n_cpu_njit as cpu
import censo_3n_ref as sec
import datos_reales as dr


def _gz(texto):
    b = io.BytesIO()
    with gzip.GzipFile(fileobj=b, mode="wb") as f:
        f.write(texto.encode())
    return b.getvalue()


def _aristas_csr(csr):
    filas = np.repeat(np.arange(csr.n), np.diff(csr.offsets))
    return {(int(u), int(v)): csr.nombres_arista[int(e)]
            for u, v, e in zip(filas, csr.indices, csr.et_arista) if u < v}


def _censo_cpu_igual_oraculo(csr):
    ref = sec.censo_completo(gpu.csr_a_grafo(csr))
    return all(c3.censos_iguales(cpu.censo3_njit(csr, None, paralelo=par, estrategia_caminos=e), ref)
               for par in (False, True) for e in cpu.ESTRATEGIAS_CAMINOS)


def test_normalizar_etiquetadas():
    u = np.array([0, 1, 2, 2, 4, 0, 1], dtype=np.int64)
    v = np.array([1, 0, 2, 4, 2, 4, 0], dtype=np.int64)
    et = np.array([1, 1, 0, 0, 2, 1, 0], dtype=np.int32)
    et_nodo = np.array([5, 6, 7, 8, 9], dtype=np.int32)
    a, b, e, etn, n, est = dr.normalizar_aristas_etiquetadas(u, v, et, et_nodo, 3)
    assert n == 4 and etn.tolist() == [5, 6, 7, 9]
    assert sorted(zip(a.tolist(), b.tolist(), e.tolist())) == [(0, 1, 0), (0, 3, 1), (2, 3, 0)]
    assert est == {"lazos": 1, "repetidas": 1, "conflictos_etiqueta": 2, "nodos_aislados": 1}


def test_normalizar_etiquetadas_igual_a_sin_etiquetas():
    rng = np.random.default_rng(7)
    u, v = rng.integers(0, 40, 600), rng.integers(0, 40, 600)
    a0, b0, n0 = dr.normalizar_aristas(u, v)
    a, b, e, etn, n, _ = dr.normalizar_aristas_etiquetadas(u, v, np.zeros(600, np.int32), np.arange(40, dtype=np.int32), 1)
    assert n == n0 and a.tolist() == a0.tolist() and b.tolist() == b0.tolist() and not e.any()


def test_lector_ogb_products(tmp_path):
    carpeta = tmp_path / "datos"
    (carpeta / "ogb").mkdir(parents=True)
    aristas = [(0, 1), (1, 0), (1, 2), (2, 2), (0, 2), (2, 4), (0, 1)]
    with zipfile.ZipFile(carpeta / "ogb" / "products.zip", "w") as z:
        z.writestr("products/raw/edge.csv.gz", _gz("".join(f"{u},{v}\n" for u, v in aristas)))
        z.writestr("products/raw/node-label.csv.gz", _gz("1\n0\n1\n2\n0\n"))
        z.writestr("products/mapping/labelidx2productcategory.csv.gz",
                   _gz("label idx,product category\n0,Cocina\n1,Libros\n2,Otros\n"))
    csr, _, meta = dr.cargar_dataset("ogbn-products", carpeta=str(carpeta))
    assert csr.n == 4 and csr.nnz // 2 == 4
    assert csr.nombres_nodo == ["Cocina", "Libros", "Otros"] and len(csr.nombres_arista) == 1
    assert [csr.nombres_nodo[i] for i in csr.et_nodo] == ["Libros", "Cocina", "Libros", "Cocina"]
    assert set(_aristas_csr(csr)) == {(0, 1), (0, 2), (1, 2), (2, 3)}
    assert meta["lazos"] == 1 and meta["repetidas"] == 2 and meta["nodos_aislados"] == 1
    assert _censo_cpu_igual_oraculo(csr)
    otra, _, meta2 = dr.cargar_dataset("ogbn-products", carpeta=str(carpeta))
    assert meta2["desde_cache"] and np.array_equal(otra.indices, csr.indices) and otra.nombres_nodo == csr.nombres_nodo


def test_lector_ogb_mag(tmp_path):
    carpeta = tmp_path / "datos"
    (carpeta / "ogb").mkdir(parents=True)
    relaciones = {
        "author___affiliated_with___institution": [(0, 0), (1, 0)],
        "author___writes___paper": [(0, 0), (0, 1), (1, 1), (1, 1)],
        "paper___cites___paper": [(0, 1), (1, 0), (2, 0), (2, 2)],
        "paper___has_topic___field_of_study": [(0, 0), (1, 0), (2, 0)],
    }
    with zipfile.ZipFile(carpeta / "ogb" / "mag.zip", "w") as z:
        z.writestr("mag/raw/num-node-dict.csv.gz", _gz("paper,author,institution,field_of_study\n3,2,1,1\n"))
        for nombre, aristas in relaciones.items():
            z.writestr(f"mag/raw/relations/{nombre}/edge.csv.gz", _gz("".join(f"{u},{v}\n" for u, v in aristas)))
            z.writestr(f"mag/raw/relations/{nombre}/num-edge-list.csv.gz", _gz(f"{len(aristas)}\n"))
    csr, _, meta = dr.cargar_dataset("ogbn-mag", carpeta=str(carpeta))
    assert csr.nombres_nodo == ["author", "field_of_study", "institution", "paper"]
    assert csr.nombres_arista == ["affiliated_with", "cites", "has_topic", "writes"]
    assert csr.n == 7 and [csr.nombres_nodo[i] for i in csr.et_nodo] == ["paper"] * 3 + ["author"] * 2 + ["institution", "field_of_study"]
    assert _aristas_csr(csr) == {
        (3, 5): "affiliated_with", (4, 5): "affiliated_with",
        (0, 3): "writes", (1, 3): "writes", (1, 4): "writes",
        (0, 1): "cites", (0, 2): "cites",
        (0, 6): "has_topic", (1, 6): "has_topic", (2, 6): "has_topic",
    }
    assert meta["lazos"] == 1 and meta["repetidas"] == 2 and meta["conflictos_etiqueta"] == 0
    assert _censo_cpu_igual_oraculo(csr)
