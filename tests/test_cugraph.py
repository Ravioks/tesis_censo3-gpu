import importlib.util
import os

import numpy as np
import pytest

import adaptador_cugraph as ad
import censo_3n as c3
import censo_3n_ref as sec
import censo_3n2a_gpu as gpu
from comunes import grafo_etiquetas, grafos_base

SIMULADOR = os.environ.get("NUMBA_ENABLE_CUDASIM") == "1"
HAY_GPU = gpu.cuda.is_available()
HAY_CUPY = importlib.util.find_spec("cupy") is not None
HAY_CUDF = importlib.util.find_spec("cudf") is not None
HAY_CUGRAPH = importlib.util.find_spec("cugraph") is not None
sin_gpu = pytest.mark.skipif(not HAY_GPU or not (SIMULADOR or HAY_CUPY),
                             reason="sin GPU con cupy (o usa NUMBA_ENABLE_CUDASIM=1 para el simulador)")
sin_cugraph = pytest.mark.skipif(SIMULADOR or not (HAY_GPU and HAY_CUGRAPH and HAY_CUDF),
                                 reason="requiere GPU real con cudf y cugraph")
LANZAMIENTO = dict(max_bloques=4, bloque_caminos=32, bloque_tri=32)
CONFIGS = [("entrada", "arista"), ("balanceado", "binaria"), ("combinatorio", "balanceado"), ("nodo", "arista")]


def xp():
    return ad.modulo_arreglos()


def grafos():
    gs = [g for g in grafos_base() if g.aristas]
    return gs + [grafo_etiquetas(13, 1, 200, 900), grafo_etiquetas(4, 3, 150, 1200)]


def aristas_superiores(csr):
    fila = np.repeat(np.arange(csr.n, dtype=np.int64), np.diff(csr.offsets))
    sup = fila < csr.indices
    return fila[sup], csr.indices[sup].astype(np.int64), csr.et_arista[sup]


def tabla(datos):
    if not SIMULADOR and HAY_CUDF:
        import cudf
        return cudf.DataFrame(datos)
    import pandas as pd
    return pd.DataFrame(datos)


def entrada_desde_csr(csr, et_arista=None, indices=None):
    m = xp()
    return ad.desde_arreglos(m.asarray(csr.offsets), m.asarray(csr.indices if indices is None else indices),
                             m.asarray(csr.et_nodo), m.asarray(csr.et_arista if et_arista is None else et_arista),
                             csr.nombres_nodo, csr.nombres_arista, csr.nombre, max_bloques=4)


@sin_gpu
@pytest.mark.parametrize("g", grafos(), ids=lambda g: g.nombre)
def test_residente_igual_oraculo(g):
    csr = gpu.grafo_a_csr(g)
    entrada = entrada_desde_csr(csr)
    res = ad.censo3_residente(entrada, **LANZAMIENTO)
    assert c3.censos_iguales(res, sec.censo_completo(g))
    assert all(c3.invariantes(res).values())
    assert res["tiempos"]["carga_s"] == 0.0
    assert not res["adaptador"]["reordenado"]


@sin_gpu
@pytest.mark.parametrize("e_c,e_t", CONFIGS)
def test_residente_configuraciones(e_c, e_t):
    g = grafos_base()[4]
    csr = gpu.grafo_a_csr(g)
    res = ad.censo3_residente(entrada_desde_csr(csr), estrategia_caminos=e_c, estrategia_tri=e_t, **LANZAMIENTO)
    assert c3.censos_iguales(res, sec.censo_completo(g))
    assert res["tiempos"]["adaptador_s"] > 0


@sin_gpu
@pytest.mark.parametrize("g", grafos()[3:6], ids=lambda g: g.nombre)
def test_filas_desordenadas_y_etiquetas_float(g):
    csr = gpu.grafo_a_csr(g)
    indices = csr.indices.copy()
    eta = csr.et_arista.copy()
    for u in range(csr.n):
        a, b = csr.offsets[u], csr.offsets[u + 1]
        indices[a:b] = indices[a:b][::-1]
        eta[a:b] = eta[a:b][::-1]
    entrada = entrada_desde_csr(csr, et_arista=eta.astype(np.float32), indices=indices)
    assert entrada["info"]["reordenado"]
    res = ad.censo3_residente(entrada, **LANZAMIENTO)
    assert c3.censos_iguales(res, sec.censo_completo(g))


@sin_gpu
def test_vertices_aislados_al_final():
    g = grafos_base()[5]
    for i in range(3):
        g.agregar_nodo(f"aislado{i}", "A")
    csr = gpu.grafo_a_csr(g)
    assert np.all(np.diff(csr.offsets)[-3:] == 0)
    m = xp()
    entrada = ad.desde_arreglos(m.asarray(csr.offsets[:-3]), m.asarray(csr.indices), m.asarray(csr.et_nodo),
                                m.asarray(csr.et_arista), csr.nombres_nodo, csr.nombres_arista, max_bloques=4)
    assert entrada["csr"].n == csr.n
    res = ad.censo3_residente(entrada, **LANZAMIENTO)
    assert c3.censos_iguales(res, sec.censo_completo(g))


@sin_gpu
def test_rechaza_lazos_repetidas_y_asimetria():
    m = xp()
    nombres = (["A"], ["-"])
    with pytest.raises(ValueError, match="lazos"):
        ad.desde_arreglos(m.asarray(np.array([0, 2, 3], np.int64)), m.asarray(np.array([0, 1, 0], np.int32)),
                          m.asarray(np.zeros(2, np.int32)), None, *nombres, max_bloques=4)
    with pytest.raises(ValueError, match="repetidas"):
        ad.desde_arreglos(m.asarray(np.array([0, 2, 4], np.int64)), m.asarray(np.array([1, 1, 0, 0], np.int32)),
                          m.asarray(np.zeros(2, np.int32)), None, *nombres, max_bloques=4)
    with pytest.raises(ValueError, match="simétrica"):
        ad.desde_arreglos(m.asarray(np.array([0, 1, 1], np.int64)), m.asarray(np.array([1], np.int32)),
                          m.asarray(np.zeros(2, np.int32)), None, *nombres, max_bloques=4)
    with pytest.raises(ValueError, match="arista fuera"):
        ad.desde_arreglos(m.asarray(np.array([0, 1, 2], np.int64)), m.asarray(np.array([1, 0], np.int32)),
                          m.asarray(np.zeros(2, np.int32)), m.asarray(np.array([1, 1], np.int32)), *nombres,
                          max_bloques=4)


@sin_gpu
@pytest.mark.parametrize("g", grafos(), ids=lambda g: g.nombre)
def test_desde_tabla_de_aristas_igual_oraculo(g):
    csr = gpu.grafo_a_csr(g)
    a, b, e = aristas_superiores(csr)
    rng = np.random.default_rng(3)
    voltear = rng.random(len(a)) < 0.5
    a, b = np.where(voltear, b, a), np.where(voltear, a, b)
    k = min(5, len(a))
    a = np.concatenate([a, b[:k], [0]])
    b = np.concatenate([b, a[:k], [0]])
    e = np.concatenate([e, e[:k], [0]])
    orden = rng.permutation(len(a))
    df = tabla({"src": a[orden].astype(np.int32), "dst": b[orden].astype(np.int32),
                "etiqueta": e[orden].astype(np.float32)})
    entrada = ad.desde_cudf(df, xp().asarray(csr.et_nodo), csr.nombres_nodo, csr.nombres_arista, etiqueta="etiqueta",
                            nombre=csr.nombre, max_bloques=4)
    assert entrada["info"]["lazos_quitados"] == 1
    assert entrada["info"]["repetidas_quitadas"] == k
    assert entrada["info"]["nnz"] == csr.nnz
    assert np.array_equal(entrada["csr"].offsets, csr.offsets)
    res = ad.censo3_residente(entrada, **LANZAMIENTO)
    assert c3.censos_iguales(res, sec.censo_completo(g))


class GrafoFalso:
    def __init__(self, csr, solo_superior):
        fila = np.repeat(np.arange(csr.n), np.diff(csr.offsets))
        sel = fila < csr.indices if solo_superior else np.ones(csr.nnz, dtype=bool)
        m = xp()
        self.vista = (m.asarray(np.concatenate([[0], np.cumsum(np.bincount(fila[sel], minlength=csr.n))]).astype(np.int32)),
                      m.asarray(csr.indices[sel].astype(np.int32)), m.asarray(csr.et_arista[sel].astype(np.float32)))

    def view_adj_list(self):
        return self.vista

    def is_renumbered(self):
        return False


@sin_gpu
@pytest.mark.parametrize("solo_superior", [True, False], ids=["triangulo_superior", "simetrica"])
@pytest.mark.parametrize("g", grafos()[3:], ids=lambda g: g.nombre)
def test_desde_cugraph_con_vista_simulada(g, solo_superior):
    csr = gpu.grafo_a_csr(g)
    entrada = ad.desde_cugraph(GrafoFalso(csr, solo_superior), xp().asarray(csr.et_nodo), csr.nombres_nodo,
                               csr.nombres_arista, csr.nombre, max_bloques=4)
    assert entrada["info"]["simetrizado"] == solo_superior
    assert entrada["info"]["nnz"] == csr.nnz
    assert np.array_equal(entrada["csr"].offsets, csr.offsets)
    res = ad.censo3_residente(entrada, **LANZAMIENTO)
    assert c3.censos_iguales(res, sec.censo_completo(g))


def grafo_cugraph(csr, renumerar=False):
    import cugraph
    a, b, e = aristas_superiores(csr)
    df = tabla({"src": a.astype(np.int32), "dst": b.astype(np.int32), "etiqueta": e.astype(np.float32)})
    G = cugraph.Graph(directed=False)
    G.from_cudf_edgelist(df, source="src", destination="dst", edge_attr="etiqueta", renumber=renumerar)
    return G


@sin_cugraph
@pytest.mark.parametrize("g", grafos(), ids=lambda g: g.nombre)
def test_desde_cugraph_igual_oraculo(g):
    csr = gpu.grafo_a_csr(g)
    G = grafo_cugraph(csr)
    entrada = ad.desde_cugraph(G, xp().asarray(csr.et_nodo), csr.nombres_nodo, csr.nombres_arista, csr.nombre,
                               max_bloques=4)
    assert entrada["info"]["nnz"] == csr.nnz
    res = ad.censo3_residente(entrada, **LANZAMIENTO)
    assert c3.censos_iguales(res, sec.censo_completo(g))


@sin_cugraph
@pytest.mark.xfail(strict=False, reason="la vista de un grafo dirigido no está documentada en cuGraph")
def test_desde_cugraph_dirigido_simetrico_sin_simetrizar():
    import cugraph
    g = grafos_base()[4]
    csr = gpu.grafo_a_csr(g)
    fila = np.repeat(np.arange(csr.n, dtype=np.int32), np.diff(csr.offsets))
    df = tabla({"src": fila, "dst": csr.indices.astype(np.int32), "etiqueta": csr.et_arista.astype(np.float32)})
    G = cugraph.Graph(directed=True)
    G.from_cudf_edgelist(df, source="src", destination="dst", edge_attr="etiqueta", renumber=False)
    entrada = ad.desde_cugraph(G, xp().asarray(csr.et_nodo), csr.nombres_nodo, csr.nombres_arista, max_bloques=4)
    assert not entrada["info"]["simetrizado"]
    assert c3.censos_iguales(ad.censo3_residente(entrada, **LANZAMIENTO), sec.censo_completo(g))


@sin_cugraph
@pytest.mark.parametrize("g", grafos(), ids=lambda g: g.nombre)
def test_triangle_count_de_cugraph_coincide(g):
    import cugraph
    csr = gpu.grafo_a_csr(g)
    tc = cugraph.triangle_count(grafo_cugraph(csr))
    assert int(tc["counts"].sum()) == 3 * sec.contar_triangulos(g)


@sin_cugraph
def test_rechaza_grafo_renumerado():
    csr = gpu.grafo_a_csr(grafos_base()[4])
    G = grafo_cugraph(csr, renumerar=True)
    if not ad.renumerado(G):
        pytest.skip("cuGraph no marcó el grafo como renumerado")
    with pytest.raises(ValueError, match="renumerado"):
        ad.desde_cugraph(G, xp().asarray(csr.et_nodo), csr.nombres_nodo, csr.nombres_arista)
