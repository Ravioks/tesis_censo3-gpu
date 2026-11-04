import random
from math import comb

import pytest

import censo_3n_ref as sec
from comunes import MOLECULAS, aleatorio, grafos_base
from fuerza_bruta import censo_fuerza_bruta


@pytest.mark.parametrize("semilla", range(40))
def test_oraculo_igual_fuerza_bruta(semilla):
    g = aleatorio(semilla)
    assert sec.censo_completo(g) == censo_fuerza_bruta(g)


def test_oraculo_moleculas():
    g = sec.cargar_gspan(MOLECULAS)
    assert sec.censo_completo(g) == censo_fuerza_bruta(g)


@pytest.mark.parametrize("semilla", range(40))
def test_resta_desde_triangulos_igual_resta_directa(semilla):
    g = aleatorio(semilla, n_max=30)
    triangulos, resta = sec.censo_triangulos(g)
    assert sec.resta_desde_triangulos(triangulos) == resta


def test_exploracion_igual_centro():
    for semilla in range(10):
        g = aleatorio(semilla)
        assert sec.censo_exploracion(g)[0] == sec.censo_centro(g)[0]


def test_casos_borde():
    t = sec.censo_completo(sec.generar_triangulo())
    assert sum(t["caminos_no_inducidos"].values()) == 3
    assert t["caminos_inducidos"] == {}
    assert sum(t["triangulos"].values()) == 1

    e = sec.censo_completo(sec.generar_estrella(12))
    assert e["triangulos"] == {}
    assert e["caminos_inducidos"] == e["caminos_no_inducidos"]
    assert sum(e["caminos_no_inducidos"].values()) == comb(11, 2)

    n = 9
    k = sec.censo_completo(sec.generar_completo(n, random.Random(1), sec.alfabeto(2, True), sec.alfabeto(2, False)))
    assert k["caminos_inducidos"] == {}
    assert sum(k["triangulos"].values()) == comb(n, 3)
    assert sum(k["caminos_no_inducidos"].values()) == n * comb(n - 1, 2)

    vacio = sec.censo_completo(sec.generar_aleatorio(6, 0, random.Random(0), ["A"], ["a"]))
    assert vacio == {"caminos_no_inducidos": {}, "caminos_inducidos": {}, "triangulos": {}}


@pytest.mark.parametrize("g", grafos_base(), ids=lambda g: g.nombre)
def test_invariantes_oraculo(g):
    ref = sec.censo_completo(g)
    W = sum(comb(len(a), 2) for a in g.adyacencia.values())
    T = sec.contar_triangulos(g)
    assert sum(ref["caminos_no_inducidos"].values()) == W
    assert sum(ref["triangulos"].values()) == T
    assert sum(ref["caminos_inducidos"].values()) == W - 3 * T


def test_censo_inducido_detecta_inconsistencias():
    clave = ("A", ("a", "B"), ("a", "B"))
    with pytest.raises(ValueError):
        sec.censo_inducido({clave: 1}, {clave: 2})
    with pytest.raises(ValueError):
        sec.censo_inducido({}, {clave: 1})


def test_estadisticas():
    e = sec.estadisticas([1.0, 2.0, 3.0])
    assert e["media"] == 2.0 and e["min"] == 1.0 and e["max"] == 3.0 and e["n"] == 3
    assert abs(e["desv"] - (2 / 3) ** 0.5) < 1e-12
    assert sec.estadisticas([5.0])["desv"] == 0.0
    assert sec.aplanar("t_kernel", e) == {"t_kernel_s": 2.0, "t_kernel_desv_s": e["desv"], "t_kernel_min_s": 1.0}
