import numpy as np
import pytest

import censo_3n as c3
import censo_3n_ref as sec
import censo_3n2a_gpu as gpu
import censo_3n3a_gpu as tri
import censo_3n_cpu_njit as cpu
import datos_reales as dr
from comunes import grafos_base


def aristas_de(g):
    nn = sorted({str(e) for e in g.etiqueta_nodo.values()})
    na = sorted({str(e[2]) for e in g.aristas}) or ["-"]
    idx_n = {e: i for i, e in enumerate(nn)}
    idx_a = {e: i for i, e in enumerate(na)}
    nodos = list(g.etiqueta_nodo)
    pos = {v: i for i, v in enumerate(nodos)}
    pares = sorted((min(pos[u], pos[v]), max(pos[u], pos[v]), idx_a[str(e)]) for u, v, e in g.aristas)
    a = np.array([p[0] for p in pares], dtype=np.int32)
    b = np.array([p[1] for p in pares], dtype=np.int32)
    et_a = np.array([p[2] for p in pares], dtype=np.int32)
    et_n = np.array([idx_n[str(g.etiqueta_nodo[v])] for v in nodos], dtype=np.int32)
    return a, b, len(nodos), et_n, et_a, nn, na


@pytest.mark.parametrize("g", grafos_base(), ids=lambda g: g.nombre)
def test_csr_ida_y_vuelta(g):
    csr = gpu.grafo_a_csr(g)
    assert gpu.filas_ordenadas(csr.offsets, csr.indices)
    assert sec.censo_completo(gpu.csr_a_grafo(csr)) == sec.censo_completo(g)


@pytest.mark.parametrize("g", grafos_base(), ids=lambda g: g.nombre)
def test_csr_desde_aristas_igual_que_grafo_a_csr(g):
    a, b, n, et_n, et_a, nn, na = aristas_de(g)
    csr = gpu.csr_desde_aristas(a, b, n, et_n, et_a, nn, na, g.nombre)
    ref = gpu.grafo_a_csr(g)
    assert np.array_equal(csr.offsets, ref.offsets)
    assert np.array_equal(csr.indices, ref.indices)
    assert np.array_equal(csr.et_arista, ref.et_arista)
    assert np.array_equal(csr.et_nodo, ref.et_nodo)
    assert c3.censos_iguales(cpu.censo3_njit(csr), sec.censo_completo(g))


@pytest.mark.parametrize("g", grafos_base()[:6], ids=lambda g: g.nombre)
def test_npz_ida_y_vuelta(g, tmp_path):
    csr = gpu.grafo_a_csr(g)
    ruta = tmp_path / "g.npz"
    gpu.guardar_npz(csr, str(ruta))
    otro = gpu.cargar_npz(str(ruta))
    for campo in ("offsets", "indices", "et_nodo", "et_arista"):
        assert np.array_equal(getattr(otro, campo), getattr(csr, campo)), campo
    assert otro.nombres_nodo == csr.nombres_nodo and otro.nombres_arista == csr.nombres_arista
    assert c3.censos_iguales(cpu.censo3_njit(otro), sec.censo_completo(g))


def test_npz_filas_desordenadas_se_ordenan(tmp_path):
    g = grafos_base()[4]
    csr = gpu.grafo_a_csr(g)
    rng = np.random.default_rng(0)
    indices, et = csr.indices.copy(), csr.et_arista.copy()
    for c in range(csr.n):
        i, f = csr.offsets[c], csr.offsets[c + 1]
        orden = rng.permutation(f - i)
        indices[i:f], et[i:f] = indices[i:f][orden], et[i:f][orden]
    ruta = tmp_path / "desordenado.npz"
    np.savez(ruta, offsets=csr.offsets, indices=indices, etiqueta_nodo=csr.et_nodo, etiqueta_arista=et,
             nombres_nodo=np.array(csr.nombres_nodo), nombres_arista=np.array(csr.nombres_arista))
    otro = gpu.cargar_npz(str(ruta))
    assert gpu.filas_ordenadas(otro.offsets, otro.indices)
    assert c3.censos_iguales(cpu.censo3_njit(otro), sec.censo_completo(g))


def test_npz_asimetrico_falla(tmp_path):
    ruta = tmp_path / "malo.npz"
    np.savez(ruta, offsets=np.array([0, 1, 1], np.int64), indices=np.array([1], np.int32),
             etiqueta_nodo=np.zeros(2, np.int32), etiqueta_arista=np.zeros(1, np.int32))
    with pytest.raises(ValueError):
        gpu.cargar_npz(str(ruta))


def test_normalizar_aristas():
    u = np.array([10, 10, 30, 20, 30, 10], dtype=np.int64)
    v = np.array([20, 10, 10, 10, 20, 20], dtype=np.int64)
    a, b, n = dr.normalizar_aristas(u, v)
    assert n == 3
    assert sorted(zip(a.tolist(), b.tolist())) == [(0, 1), (0, 2), (1, 2)]


@pytest.mark.parametrize("g", grafos_base(), ids=lambda g: g.nombre)
def test_orientacion(g):
    csr = gpu.grafo_a_csr(g)
    o = tri.orientar(csr)
    assert o["aristas"] == csr.nnz // 2
    assert gpu.filas_ordenadas(o["off_o"], o["ind_o"])
    grados = np.diff(csr.offsets)
    u, v = o["fila_o"].astype(np.int64), o["ind_o"].astype(np.int64)
    assert np.all((grados[u] < grados[v]) | ((grados[u] == grados[v]) & (u < v)))
    assert o["cota_triangulos"] >= sec.contar_triangulos(g)
    assert np.all(np.diff(o["acum_costo"]) >= 1) and np.all(np.diff(o["acum_binaria"]) >= 1)


def test_lotes_cubren_todo_sin_solaparse():
    csr = gpu.grafo_a_csr(grafos_base()[5])
    prep = gpu.preparar(csr)
    o = tri.orientar(csr)
    for estrategia in ("nodo", "entrada", "combinatorio"):
        r = gpu.lotes(estrategia, csr, prep, 4, 7)
        fin = csr.nnz if estrategia == "entrada" else csr.n
        assert r[0][0] == 0 and r[-1][1] == fin and all(x[1] == y[0] for x, y in zip(r, r[1:]))
    r = gpu.lotes("balanceado", csr, prep, 4, 7)
    assert r[0][0] == 0 and r[-1][1] == (prep["W"] + 3) // 4
    for estrategia in ("arista", "binaria"):
        r = tri.lotes(estrategia, o, 4, 7)
        assert r[0][0] == 0 and r[-1][1] == o["aristas"] and all(x[1] == y[0] for x, y in zip(r, r[1:]))
    r = tri.lotes("balanceado", o, 4, 7)
    assert r[0][0] == 0 and r[-1][1] == (int(o["acum_costo"][-1]) + 3) // 4
