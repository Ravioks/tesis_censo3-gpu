import numpy as np
import pytest

import censo_3n as c3
import censo_3n_ref as sec
import censo_3n2a_gpu as gpu
import censo_3n_cpu_njit as cpu
import generadores as gen


@pytest.mark.parametrize("tipo, kw", [("ba", {"n": 300, "k": 4}), ("aleatorio", {"n": 200, "m": 900}),
                                      ("completo", {"n": 25})])
def test_csr_sintetico_valido_y_censo_correcto(tipo, kw):
    csr = gen.csr_sintetico(tipo, kw.get("n"), kw.get("m"), kw.get("k", 3), L=4, E=2, semilla=3)
    assert gpu.filas_ordenadas(csr.offsets, csr.indices)
    csr.id_arista
    g = gpu.csr_a_grafo(csr)
    assert c3.censos_iguales(cpu.censo3_njit(csr), sec.censo_completo(g))


def test_csr_sintetico_reproducible():
    a = gen.csr_sintetico("ba", 500, k=3, semilla=11)
    b = gen.csr_sintetico("ba", 500, k=3, semilla=11)
    c = gen.csr_sintetico("ba", 500, k=3, semilla=12)
    assert np.array_equal(a.indices, b.indices) and np.array_equal(a.et_arista, b.et_arista)
    assert not np.array_equal(a.indices, c.indices)


def test_tamanos_esperados():
    k = gen.csr_sintetico("completo", 30)
    assert k.n == 30 and k.nnz // 2 == 30 * 29 // 2
    er = gen.csr_sintetico("aleatorio", 1000, m=5000)
    assert er.nnz // 2 == 5000
    ba = gen.csr_sintetico("ba", 2000, k=5)
    assert 0.9 * (2000 - 5) * 5 <= ba.nnz // 2 <= (2000 - 5) * 5


def test_grafo_etiquetas_usa_todas_las_etiquetas():
    g = gen.grafo_etiquetas(13, 3, 60, 200)
    csr = gpu.grafo_a_csr(g)
    assert len(csr.nombres_nodo) == 13 and len(csr.nombres_arista) == 3
