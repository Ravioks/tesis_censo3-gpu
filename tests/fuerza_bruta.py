from collections import defaultdict
from itertools import combinations


def censo_fuerza_bruta(g):
    arista = {frozenset((u, v)): eid for eid, (u, v, _) in enumerate(g.aristas)}

    def etq(x, y):
        return g.aristas[arista[frozenset((x, y))]][2]

    def camino(centro, x, y):
        r1 = (etq(centro, x), g.etiqueta_nodo[x])
        r2 = (etq(centro, y), g.etiqueta_nodo[y])
        return (g.etiqueta_nodo[centro],) + tuple(sorted((r1, r2)))

    no_inducidos = defaultdict(int)
    inducidos = defaultdict(int)
    triangulos = defaultdict(int)
    for a, b, c in combinations(list(g.etiqueta_nodo), 3):
        ab, ac, bc = (frozenset(p) in arista for p in ((a, b), (a, c), (b, c)))
        presentes = ab + ac + bc
        if presentes == 2:
            centro = a if not bc else b if not ac else c
            x, y = (v for v in (a, b, c) if v != centro)
            clave = camino(centro, x, y)
            no_inducidos[clave] += 1
            inducidos[clave] += 1
        elif presentes == 3:
            for centro in (a, b, c):
                x, y = (v for v in (a, b, c) if v != centro)
                no_inducidos[camino(centro, x, y)] += 1
            triangulos[tuple(sorted(((etq(b, c), g.etiqueta_nodo[a]), (etq(a, c), g.etiqueta_nodo[b]),
                                     (etq(a, b), g.etiqueta_nodo[c]))))] += 1
    return {"caminos_no_inducidos": dict(no_inducidos), "caminos_inducidos": dict(inducidos),
            "triangulos": dict(triangulos)}
