import os
import random

import censo_3n2a as sec
from generadores import grafo_etiquetas

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOLECULAS = os.path.join(RAIZ, "ejemplos", "moleculas.txt")


def aleatorio(semilla, n_max=25):
    rng = random.Random(semilla)
    n = rng.randint(3, n_max)
    m = rng.randint(0, n * (n - 1) // 2)
    return sec.generar_aleatorio(n, m, rng, sec.alfabeto(rng.randint(1, 4), True),
                                 sec.alfabeto(rng.randint(1, 3), False))


def grafos_base():
    rng = random.Random(7)
    return [
        sec.generar_ejemplo(),
        sec.generar_triangulo(),
        sec.generar_estrella(15),
        sec.generar_completo(10, rng, sec.alfabeto(3, True), sec.alfabeto(2, False)),
        sec.generar_ba(120, 3, rng, sec.alfabeto(3, True), sec.alfabeto(2, False)),
        sec.generar_aleatorio(60, 300, rng, sec.alfabeto(4, True), sec.alfabeto(3, False)),
        sec.generar_aleatorio(6, 0, rng, sec.alfabeto(2, True), sec.alfabeto(1, False)),
    ]
