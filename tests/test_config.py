import json

import pytest

import censo_3n as c3
import censo_3n2a_gpu as gpu
import censo_3n3a_gpu as tri
from comunes import grafo_etiquetas


def test_config_por_defecto_y_cambios():
    cfg = c3.resolver_config()
    assert cfg == c3.CONFIG_POR_DEFECTO
    cfg = c3.resolver_config({"estrategia_caminos": "balanceado", "bloque_tri": 128},
                             estrategia_caminos="combinatorio", memoria_tri=None)
    assert cfg["estrategia_caminos"] == "combinatorio" and cfg["bloque_tri"] == 128 and cfg["memoria_tri"] is None


def test_config_rechaza_valores_invalidos():
    with pytest.raises(TypeError):
        c3.resolver_config(estrategia="entrada")
    with pytest.raises(ValueError):
        c3.resolver_config(estrategia_caminos="arista")
    with pytest.raises(ValueError):
        c3.resolver_config(estrategia_tri="entrada")
    with pytest.raises(ValueError):
        c3.resolver_config(memoria_tri="local")


def test_cargar_config_de_barrido(tmp_path):
    ruta = tmp_path / "mejor_config.json"
    ruta.write_text(json.dumps({"config": {"estrategia_caminos": "combinatorio", "memoria_caminos": "global",
                                           "estrategia_tri": "binaria"}, "grafo": "x", "criterio": "total"}))
    cfg = c3.resolver_config(c3.cargar_config(str(ruta)))
    assert (cfg["estrategia_caminos"], cfg["memoria_caminos"], cfg["estrategia_tri"]) == ("combinatorio", "global", "binaria")


@pytest.mark.parametrize("L, E, n, m, esperado", [(3, 2, 30, 60, "shared"), (13, 1, 40, 80, "global"),
                                                  (260, 1, 300, 600, "hash")])
def test_memoria_por_defecto_segun_etiquetas(L, E, n, m, esperado):
    prep = gpu.preparar(gpu.grafo_a_csr(grafo_etiquetas(L, E, n, m)))
    assert prep["L"] == L and prep["E"] == E
    cfg = c3.completar_memorias(c3.resolver_config(), prep)
    assert cfg["memoria_caminos"] == esperado and cfg["memoria_tri"] == esperado


def test_motivos_de_configuraciones_invalidas():
    prep = gpu.preparar(gpu.grafo_a_csr(grafo_etiquetas(260, 1, 300, 600)))
    assert gpu.motivo_invalida("combinatorio", "hash", prep)
    assert gpu.motivo_invalida("entrada", "shared", prep) and gpu.motivo_invalida("entrada", "global", prep)
    assert gpu.motivo_invalida("entrada", "hash", prep) is None
    assert tri.motivo_invalida("arista", "global", prep) and tri.motivo_invalida("arista", "hash", prep) is None
    falso = dict(prep, max_cunas=gpu.LIMITE_TRABAJO_HILO + 1, grado_max=10_001)
    assert gpu.motivo_invalida("nodo", "hash", falso)
