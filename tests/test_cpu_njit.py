import numpy as np
import pytest

import censo_3n as c3
import censo_3n_ref as sec
import censo_3n2a_gpu as gpu
import censo_3n_cpu_njit as cpu
from comunes import aleatorio, grafo_etiquetas, grafos_base


@pytest.mark.parametrize("estrategia", cpu.ESTRATEGIAS_CAMINOS)
@pytest.mark.parametrize("paralelo", [False, True])
@pytest.mark.parametrize("g", grafos_base(), ids=lambda g: g.nombre)
def test_njit_igual_oraculo(g, paralelo, estrategia):
    csr = gpu.grafo_a_csr(g)
    r = cpu.censo3_njit(csr, paralelo=paralelo, estrategia_caminos=estrategia)
    assert c3.censos_iguales(r, sec.censo_completo(g))
    assert all(c3.invariantes(r).values())


@pytest.mark.parametrize("semilla", range(15))
def test_njit_aleatorios(semilla):
    g = aleatorio(semilla, n_max=35)
    csr = gpu.grafo_a_csr(g)
    ref = sec.censo_completo(g)
    for paralelo in (False, True):
        for estrategia in cpu.ESTRATEGIAS_CAMINOS:
            assert c3.censos_iguales(cpu.censo3_njit(csr, paralelo=paralelo, estrategia_caminos=estrategia), ref)


@pytest.mark.parametrize("L, E", [(13, 1), (7, 3), (40, 1)])
def test_njit_muchas_etiquetas(L, E):
    g = grafo_etiquetas(L, E, 80, 400)
    csr = gpu.grafo_a_csr(g)
    ref = sec.censo_completo(g)
    for estrategia in cpu.ESTRATEGIAS_CAMINOS:
        assert c3.censos_iguales(cpu.censo3_njit(csr, paralelo=True, estrategia_caminos=estrategia), ref)


def test_njit_serial_igual_paralelo_en_histogramas():
    csr = gpu.grafo_a_csr(grafos_base()[4])
    pre = c3.preparar(csr)
    for estrategia in cpu.ESTRATEGIAS_CAMINOS:
        s = cpu.censo3_njit(csr, pre, paralelo=False, estrategia_caminos=estrategia)["hist"]
        p = cpu.censo3_njit(csr, pre, paralelo=True, estrategia_caminos=estrategia, bloques_por_hilo=3)["hist"]
        assert all(np.array_equal(x, y) for x, y in zip(s, p))


def test_njit_tiempos_coherentes():
    csr = gpu.grafo_a_csr(grafos_base()[4])
    t = cpu.censo3_njit(csr)["tiempos"]
    assert t["preproceso_incluido"]
    assert abs(t["censo_s"] - (t["caminos_s"] + t["triangulos_s"] + t["post_s"])) < 1e-12
    assert abs(t["extremo_s"] - (t["censo_s"] + t["preproceso_s"])) < 1e-12


def test_cortes_balanceados_cubren_todo():
    acum = np.cumsum(np.concatenate([[0], np.array([5, 0, 0, 100, 1, 1, 50, 0, 3], np.int64)]))
    for bloques in (1, 2, 4, 16):
        c = cpu.cortes_balanceados(acum, 9, bloques)
        assert c[0] == 0 and c[-1] == 9 and np.all(np.diff(c) > 0)
