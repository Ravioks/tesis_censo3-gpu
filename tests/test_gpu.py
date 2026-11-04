import pytest

import barrido as bar
import censo_3n as c3
import censo_3n_ref as sec
import censo_3n2a_gpu as gpu
import censo_3n3a_gpu as tri
from comunes import grafo_etiquetas, grafos_base

HAY_GPU = gpu.cuda.is_available()
sin_gpu = pytest.mark.skipif(not HAY_GPU, reason="sin GPU CUDA (usa NUMBA_ENABLE_CUDASIM=1 para el simulador)")
LANZAMIENTO = dict(bloque=32, max_bloques=4)


@sin_gpu
@pytest.mark.parametrize("g", grafos_base()[:6], ids=lambda g: g.nombre)
def test_gpu_caminos_todas_las_configuraciones(g):
    csr = gpu.grafo_a_csr(g)
    prep = gpu.preparar(csr)
    ref, _ = sec.censo_centro(g)
    probadas = 0
    for estrategia in gpu.ESTRATEGIAS:
        for memoria in gpu.MEMORIAS:
            if gpu.config_valida(estrategia, memoria, prep):
                r = gpu.ejecutar_gpu(csr, prep, estrategia, memoria, 32, 4, 4, trabajo_max=500)
                assert r["censo"] == ref, (estrategia, memoria)
                probadas += 1
    assert probadas == len(gpu.ESTRATEGIAS) * len(gpu.MEMORIAS)


@sin_gpu
@pytest.mark.parametrize("g", grafos_base()[:6], ids=lambda g: g.nombre)
def test_gpu_verificar_ids(g):
    csr = gpu.grafo_a_csr(g)
    v = gpu.verificar_ids(csr, gpu.preparar(csr), g, 32, 4, 4, 500)
    assert v["estado"] == "OK" and v["duplicados_gpu"] == 0


@sin_gpu
def test_gpu_emitir_solo_con_balanceado():
    csr = gpu.grafo_a_csr(grafos_base()[0])
    with pytest.raises(ValueError):
        gpu.ejecutar_gpu(csr, gpu.preparar(csr), "entrada", "hash", 32, 4, 4, emitir=True)


@sin_gpu
@pytest.mark.parametrize("g", grafos_base(), ids=lambda g: g.nombre)
def test_gpu_triangulos_todas_las_configuraciones(g):
    csr = gpu.grafo_a_csr(g)
    prep = gpu.preparar(csr)
    o = tri.orientar(csr)
    ref_t, ref_r = sec.censo_triangulos(g)
    for estrategia in tri.ESTRATEGIAS:
        for memoria in tri.MEMORIAS:
            assert tri.config_valida(estrategia, memoria, prep)
            for chunk, trabajo_max in ((tri.CHUNK_TRI, tri.TRABAJO_POR_LANZAMIENTO), (3, 40)):
                r = tri.ejecutar_triangulos_gpu(csr, prep, o, estrategia, memoria, 32, 4, chunk, trabajo_max)
                assert r["triangulos"] == ref_t, (estrategia, memoria, chunk)
                assert r["T"] == sum(ref_t.values())
                assert sec.resta_desde_triangulos(r["triangulos"]) == ref_r


@sin_gpu
@pytest.mark.parametrize("preproceso", ["gpu", "cpu"])
@pytest.mark.parametrize("g", grafos_base(), ids=lambda g: g.nombre)
def test_gpu_censo3_igual_oraculo(g, preproceso):
    csr = gpu.grafo_a_csr(g)
    res = c3.censo3(csr, bloque_caminos=32, bloque_tri=32, max_bloques=4, preproceso=preproceso)
    assert res["preproceso"] == preproceso
    assert c3.censos_iguales(res, sec.censo_completo(g))
    assert all(c3.invariantes(res).values())
    df = c3.a_dataframe(res)
    assert int(df.loc[df.tipo == "camino", "inducido"].sum()) == res["W"] - 3 * res["T"]
    t = res["tiempos"]
    assert t["preproceso_incluido"]
    assert abs(t["censo_s"] - (t["caminos"]["total_s"] + t["triangulos"]["total_s"] + t["inducido_s"])) < 1e-12
    assert abs(t["preparacion_s"] - (t["preproceso_cpu_s"] + t["carga_s"] + t["preproceso_gpu_s"])) < 1e-12
    assert abs(t["extremo_s"] - (t["censo_s"] + t["preparacion_s"])) < 1e-12
    assert (t["preproceso_gpu_s"] > 0) == (preproceso == "gpu")


@sin_gpu
@pytest.mark.parametrize("preproceso", ["gpu", "cpu"])
@pytest.mark.parametrize("cfg", [
    {"estrategia_caminos": "combinatorio", "estrategia_tri": "binaria"},
    {"estrategia_caminos": "balanceado", "memoria_caminos": "hash", "estrategia_tri": "balanceado",
     "memoria_tri": "global", "chunk_tri": 5, "chunk": 3},
    {"estrategia_caminos": "nodo", "memoria_caminos": "global", "estrategia_tri": "arista", "memoria_tri": "hash"},
])
def test_gpu_censo3_configuraciones(cfg, preproceso):
    g = grafos_base()[4]
    csr = gpu.grafo_a_csr(g)
    res = c3.censo3(csr, config=cfg, bloque_caminos=32, bloque_tri=32, max_bloques=4, preproceso=preproceso)
    assert c3.censos_iguales(res, sec.censo_completo(g))
    for clave, valor in cfg.items():
        assert res["config"][clave] == valor


@sin_gpu
@pytest.mark.parametrize("L, E, n, m, esperado", [(13, 1, 60, 300, "global"), (260, 1, 300, 900, "hash")])
def test_gpu_muchas_etiquetas(L, E, n, m, esperado):
    g = grafo_etiquetas(L, E, n, m)
    csr = gpu.grafo_a_csr(g)
    res = c3.censo3(csr, bloque_caminos=32, bloque_tri=32, max_bloques=4)
    assert res["config"]["memoria_caminos"] == esperado and res["config"]["memoria_tri"] == esperado
    assert c3.censos_iguales(res, sec.censo_completo(g))


@sin_gpu
def test_barrido_valida_todas_las_configuraciones():
    g = grafos_base()[4]
    csr = gpu.grafo_a_csr(g)
    pre = c3.preparar(csr)
    ref = sec.censo_completo(g)
    r = bar.barrer(csr, pre, bloques=(32,), max_bloques=4, reps=1, referencia=ref, progreso=lambda s: None)
    medidas = [f for f in r["filas"] if f.get("omitida") is None]
    assert len(medidas) == len(gpu.ESTRATEGIAS) * len(gpu.MEMORIAS) + len(tri.ESTRATEGIAS) * len(tri.MEMORIAS)
    assert all(f["correcto"] for f in medidas)
    assert r["completo"]["igual_referencia"] and all(r["completo"]["invariantes"].values())
    cfg = r["config"]
    assert cfg["estrategia_caminos"] == r["mejor_caminos"]["estrategia"]
    assert cfg["estrategia_tri"] == r["mejor_triangulos"]["estrategia"]


@sin_gpu
def test_barrido_detecta_configuracion_incorrecta():
    g = grafos_base()[4]
    csr = gpu.grafo_a_csr(g)
    pre = c3.preparar(csr)
    ref = sec.censo_completo(g)
    clave = next(iter(ref["triangulos"]))
    falsa = dict(ref, triangulos=dict(ref["triangulos"]))
    falsa["triangulos"][clave] += 1
    r = bar.barrer(csr, pre, parte="triangulos", estrategias_tri=["arista"], memorias_tri=["hash"], bloques=(32,),
                   max_bloques=4, reps=1, referencia=falsa, progreso=lambda s: None)
    assert r["filas"][0]["correcto"] is False and r["mejor_triangulos"] is None


@sin_gpu
@pytest.mark.parametrize("n", [1, 5, 1023, 1024, 1025, 2049, 3000])
def test_scan_en_gpu(n):
    import numpy as np

    x = np.random.default_rng(n).integers(0, 1000, n).astype(np.int64)
    d = gpu.cuda.to_device(x)
    s = gpu.cuda.device_array(n, np.int64)
    gpu.scan_inclusivo(d, s, n)
    assert np.array_equal(s.copy_to_host(), np.cumsum(x))
    gpu.scan_inclusivo(d, d, n)
    assert np.array_equal(d.copy_to_host(), np.cumsum(x))
    e = gpu.prefijo_exclusivo(gpu.cuda.to_device(x), n).copy_to_host()
    assert e[0] == 0 and np.array_equal(e[1:], np.cumsum(x))


@sin_gpu
@pytest.mark.parametrize("g", grafos_base(), ids=lambda g: g.nombre)
def test_orientacion_y_fila_en_gpu_iguales_a_cpu(g):
    import numpy as np

    csr = gpu.grafo_a_csr(g)
    prep = gpu.preparar(csr)
    cpu_o = tri.orientar(csr)
    disp = gpu.subir_grafo(csr)
    gpu_o = tri.orientar_gpu(csr, disp, binaria=True, bloque=32, max_bloques=4)
    m = cpu_o["aristas"]
    assert gpu_o["aristas"] == m and gpu_o["en_gpu"]
    for clave in ("off_o", "ind_o", "eta_o", "fila_o", "acum_costo", "acum_binaria"):
        assert np.array_equal(gpu_o[clave].copy_to_host()[:len(cpu_o[clave])], cpu_o[clave]), clave
    for clave in ("total_costo", "total_binaria", "cota_triangulos", "grado_salida_max"):
        assert gpu_o[clave] == cpu_o[clave], clave
    fila = gpu.fila_en_gpu(disp, csr.n, csr.nnz, 32, 4).copy_to_host()[:csr.nnz]
    assert np.array_equal(fila, gpu.obtener_fila(csr, prep))


@sin_gpu
def test_lotes_y_cortes_con_arreglos_en_gpu():
    import numpy as np

    csr = gpu.grafo_a_csr(grafos_base()[5])
    cpu_o = tri.orientar(csr)
    gpu_o = tri.orientar_gpu(csr, gpu.subir_grafo(csr), bloque=32, max_bloques=4)
    objetivos = np.arange(0, cpu_o["total_costo"] + 5, 7)
    assert np.array_equal(gpu.buscar_cortes(gpu_o["acum_costo"], objetivos),
                          np.searchsorted(cpu_o["acum_costo"], objetivos))
    for estrategia in tri.ESTRATEGIAS:
        assert tri.lotes(estrategia, gpu_o, 3, 7) == tri.lotes(estrategia, cpu_o, 3, 7)


@sin_gpu
@pytest.mark.parametrize("preproceso", ["gpu", "cpu"])
def test_dispositivo_compartido_entre_configuraciones(preproceso):
    g = grafos_base()[4]
    csr = gpu.grafo_a_csr(g)
    ref = sec.censo_completo(g)
    pre = c3.preparar(csr, orientar=preproceso == "cpu")
    dispositivo = c3.preparar_dispositivo(csr, pre, gpu.ESTRATEGIAS, tri.ESTRATEGIAS, preproceso, max_bloques=4)
    assert {"fila", "acum"} <= set(dispositivo["disp"])
    for e_c, e_t in (("entrada", "arista"), ("balanceado", "binaria"), ("combinatorio", "balanceado"),
                     ("nodo", "arista")):
        res = c3.censo3(csr, pre, max_bloques=4, dispositivo=dispositivo, estrategia_caminos=e_c, estrategia_tri=e_t,
                        bloque_caminos=32, bloque_tri=32)
        assert c3.censos_iguales(res, ref), (e_c, e_t)
        assert res["tiempos"]["carga_s"] == 0.0
    assert {"fila", "acum"} <= set(dispositivo["disp"])
