import argparse
import heapq
import json
import os
import random
import re
import statistics
import string
import time
import tracemalloc
from collections import Counter, defaultdict
from math import comb


class Grafo:
    def __init__(self, nombre="grafo"):
        self.nombre = nombre
        self.etiqueta_nodo = {}
        self.transaccion = {}
        self.aristas = []
        self.adyacencia = {}
        self.pares = set()
        self.lazos_ignorados = 0
        self.multiaristas_ignoradas = 0

    @property
    def num_transacciones(self):
        return len(set(self.transaccion.values()))

    def agregar_nodo(self, nid, etiqueta, tid="0"):
        self.etiqueta_nodo[nid] = etiqueta
        self.transaccion[nid] = tid
        self.adyacencia.setdefault(nid, [])

    def agregar_arista(self, u, v, etiqueta):
        if u not in self.etiqueta_nodo or v not in self.etiqueta_nodo:
            raise ValueError(f"Arista ({u}, {v}) usa un nodo inexistente")
        if u == v:
            self.lazos_ignorados += 1
            return None
        par = frozenset((u, v))
        if par in self.pares:
            self.multiaristas_ignoradas += 1
            return None
        self.pares.add(par)
        eid = len(self.aristas)
        self.aristas.append((u, v, etiqueta))
        self.adyacencia[u].append((v, eid))
        self.adyacencia[v].append((u, eid))
        return eid


def clave_patron(g, centro, extremo1, eid1, extremo2, eid2):
    rama1 = (g.aristas[eid1][2], g.etiqueta_nodo[extremo1])
    rama2 = (g.aristas[eid2][2], g.etiqueta_nodo[extremo2])
    return (g.etiqueta_nodo[centro],) + tuple(sorted((rama1, rama2)))


def clave_triangulo(g, u, v, w, e_uv, e_uw, e_vw):
    return tuple(sorted((
        (g.aristas[e_vw][2], g.etiqueta_nodo[u]),
        (g.aristas[e_uw][2], g.etiqueta_nodo[v]),
        (g.aristas[e_uv][2], g.etiqueta_nodo[w]),
    )))


def formato_patron(clave):
    lc, (le1, l1), (le2, l2) = clave
    return f"{l1} -[{le1}]- {lc} -[{le2}]- {l2}"


def formato_triangulo(clave):
    (e1, l1), (e2, l2), (e3, l3) = clave
    return f"{l1} -[{e3}]- {l2} -[{e1}]- {l3} -[{e2}]- {l1}"


def formato_nodo(n):
    return f"g{n[0]}:v{n[1]}" if isinstance(n, tuple) else str(n)


def dibujar_grafo(g, ruta, max_nodos=300, semilla=42):
    n = len(g.etiqueta_nodo)
    if n > max_nodos:
        print(f"\nDibujo omitido: {n} nodos supera el límite de {max_nodos} (ajústalo con --max-dibujo)")
        return False
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx
    from matplotlib.lines import Line2D

    G = nx.Graph()
    for nid, etq in g.etiqueta_nodo.items():
        G.add_node(nid, etiqueta=str(etq))
    for u, v, etq in g.aristas:
        G.add_edge(u, v, etiqueta=str(etq))

    etq_nodo = sorted({G.nodes[v]["etiqueta"] for v in G})
    etq_arista = sorted({d["etiqueta"] for _, _, d in G.edges(data=True)})
    paleta_n = plt.get_cmap("tab10" if len(etq_nodo) <= 10 else "tab20")
    color_n = {e: paleta_n(i % paleta_n.N) for i, e in enumerate(etq_nodo)}
    paleta_a = plt.get_cmap("Dark2")
    estilos = ["solid", "dashed", "dotted", "dashdot"]
    estilo_a = {e: (paleta_a(i % paleta_a.N), estilos[i % len(estilos)]) for i, e in enumerate(etq_arista)}

    pos = nx.spring_layout(G, seed=semilla, k=1.5 / max(1, n) ** 0.5)
    grados = dict(G.degree())
    gmax = max(grados.values(), default=1) or 1
    base, extra = (400, 600) if n <= 40 else (60, 440)
    tamanos = [base + extra * grados[v] / gmax for v in G]

    fig, ax = plt.subplots(figsize=(11, 9))
    for e, (color, estilo) in estilo_a.items():
        aristas = [(u, v) for u, v, d in G.edges(data=True) if d["etiqueta"] == e]
        nx.draw_networkx_edges(G, pos, edgelist=aristas, edge_color=[color], style=estilo,
                               width=1.4 if n <= 60 else 0.6, alpha=0.8, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color=[color_n[G.nodes[v]["etiqueta"]] for v in G],
                           node_size=tamanos, edgecolors="black", linewidths=0.5, ax=ax)
    if n <= 40:
        nx.draw_networkx_labels(G, pos, labels={v: f"{formato_nodo(v)}\n{G.nodes[v]['etiqueta']}" for v in G},
                                font_size=8, ax=ax)
    if G.number_of_edges() <= 60:
        nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): d["etiqueta"] for u, v, d in G.edges(data=True)},
                                     font_size=7, ax=ax)

    leyenda = [Line2D([0], [0], marker="o", color="w", markerfacecolor=color_n[e], markeredgecolor="black",
                      markersize=9, label=f"nodo {e}") for e in etq_nodo[:20]]
    leyenda += [Line2D([0], [0], color=c, linestyle=s, linewidth=2, label=f"arista {e}")
                for e, (c, s) in list(estilo_a.items())[:8]]
    ax.legend(handles=leyenda, loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8, frameon=False)
    ax.set_title(f"{g.nombre} — {n} nodos, {len(g.aristas)} aristas (tamaño del nodo ∝ grado)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def analizar_conexidad(g):
    padre = {v: v for v in g.etiqueta_nodo}

    def raiz(x):
        while padre[x] != x:
            padre[x] = padre[padre[x]]
            x = padre[x]
        return x

    for u, v, _ in g.aristas:
        ru, rv = raiz(u), raiz(v)
        if ru != rv:
            padre[ru] = rv
    tamanos = Counter(raiz(v) for v in g.etiqueta_nodo)
    raices_por_transaccion = defaultdict(set)
    for v, t in g.transaccion.items():
        raices_por_transaccion[t].add(raiz(v))
    mayor = max(tamanos.values(), default=0)
    return {
        "conexo": len(tamanos) <= 1,
        "componentes": len(tamanos),
        "componente_mayor": mayor,
        "pct_componente_mayor": 100 * mayor / len(g.etiqueta_nodo) if g.etiqueta_nodo else 0.0,
        "transacciones_conexas": sum(1 for r in raices_por_transaccion.values() if len(r) == 1),
    }


def imprimir_conexidad(gr):
    if gr["transacciones"] > 1:
        print(f"Transacciones conexas: {gr['transacciones_conexas']} de {gr['transacciones']} | componentes totales: {gr['componentes']}")
    else:
        print(f"Conexo: {'sí' if gr['conexo'] else 'no'} | componentes: {gr['componentes']} | "
              f"componente mayor: {gr['componente_mayor']} nodos ({gr['pct_componente_mayor']:.1f}%)")


def preparar_salida(carpeta, nombre, sufijo=""):
    os.makedirs(carpeta, exist_ok=True)
    numeros = [int(m.group(1)) for f in os.listdir(carpeta) if (m := re.match(r"(\d+)_", f))]
    base = re.sub(r"[^\w.-]+", "_", os.path.splitext(os.path.basename(str(nombre)))[0])
    base = re.sub(r"_csr$", "", re.sub(r"^\d+_(gpu_)?", "", base))
    return os.path.join(carpeta, f"{max(numeros, default=0) + 1:04d}_{sufijo}{base}")


def guardar_json(resultado, ruta):
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)


def censo_exploracion(g):
    censo = defaultdict(int)
    vistos = set()
    generados = 0
    descartados = 0
    for u in g.etiqueta_nodo:
        for v, e1 in g.adyacencia[u]:
            for centro, otro in ((u, v), (v, u)):
                for w, e2 in g.adyacencia[centro]:
                    if e2 == e1:
                        continue
                    generados += 1
                    id_embedding = (e1, e2) if e1 < e2 else (e2, e1)
                    if id_embedding in vistos:
                        descartados += 1
                        continue
                    vistos.add(id_embedding)
                    censo[clave_patron(g, centro, otro, e1, w, e2)] += 1
    return dict(censo), {"generados": generados, "descartados": descartados, "tam_hash_ids": len(vistos)}


def censo_centro(g):
    censo = defaultdict(int)
    generados = 0
    for c, vecinos in g.adyacencia.items():
        for i in range(len(vecinos)):
            v, e1 = vecinos[i]
            for j in range(i + 1, len(vecinos)):
                w, e2 = vecinos[j]
                generados += 1
                censo[clave_patron(g, c, v, e1, w, e2)] += 1
    return dict(censo), {"generados": generados, "descartados": 0, "tam_hash_ids": 0}


METODOS = {"exploracion": censo_exploracion, "centro": censo_centro}


def calcular_soporte(g):
    soporte = defaultdict(set)
    for c, vecinos in g.adyacencia.items():
        t = g.transaccion[c]
        for i in range(len(vecinos)):
            v, e1 = vecinos[i]
            for j in range(i + 1, len(vecinos)):
                w, e2 = vecinos[j]
                soporte[clave_patron(g, c, v, e1, w, e2)].add(t)
    return {k: len(s) for k, s in soporte.items()}


def censo_triangulos(g):
    orden = {nid: i for i, nid in enumerate(g.etiqueta_nodo)}
    vecinos = {n: dict(ady) for n, ady in g.adyacencia.items()}
    triangulos = defaultdict(int)
    resta = defaultdict(int)
    for u, v, _ in g.aristas:
        if orden[u] > orden[v]:
            u, v = v, u
        nu, nv = vecinos[u], vecinos[v]
        if len(nu) > len(nv):
            comunes = [w for w in nv if w in nu]
        else:
            comunes = [w for w in nu if w in nv]
        e_uv = nu[v]
        for w in comunes:
            if orden[w] <= orden[v]:
                continue
            e_uw, e_vw = nu[w], nv[w]
            triangulos[clave_triangulo(g, u, v, w, e_uv, e_uw, e_vw)] += 1
            resta[clave_patron(g, u, v, e_uv, w, e_uw)] += 1
            resta[clave_patron(g, v, u, e_uv, w, e_vw)] += 1
            resta[clave_patron(g, w, u, e_uw, v, e_vw)] += 1
    return dict(triangulos), dict(resta)


def resta_desde_triangulos(triangulos):
    resta = defaultdict(int)
    for clave, cant in triangulos.items():
        for i in range(3):
            j, k = (x for x in range(3) if x != i)
            (_, n_i), (e_j, n_j), (e_k, n_k) = clave[i], clave[j], clave[k]
            resta[(n_i,) + tuple(sorted(((e_k, n_j), (e_j, n_k))))] += cant
    return dict(resta)


def censo_inducido(no_inducido, resta):
    inducido = {}
    for clave, cant in no_inducido.items():
        quedan = cant - resta.get(clave, 0)
        if quedan < 0:
            raise ValueError(f"Conteo inducido negativo para {formato_patron(clave)}")
        if quedan:
            inducido[clave] = quedan
    sobrantes = set(resta) - set(no_inducido)
    if sobrantes:
        raise ValueError(f"{len(sobrantes)} patrones de la resta no existen en el censo no inducido")
    return inducido


def censo_completo(g):
    no_inducido, _ = censo_centro(g)
    triangulos, resta = censo_triangulos(g)
    return {
        "caminos_no_inducidos": no_inducido,
        "caminos_inducidos": censo_inducido(no_inducido, resta),
        "triangulos": triangulos,
    }


def contar_triangulos(g):
    vecinos = {n: {v for v, _ in ady} for n, ady in g.adyacencia.items()}
    return sum(len(vecinos[u] & vecinos[v]) for u, v, _ in g.aristas) // 3


def estadisticas_grafo(g):
    grados = [len(a) for a in g.adyacencia.values()]
    n, m = len(grados), len(g.aristas)
    caminos_nodo = [comb(d, 2) for d in grados]
    esperado = sum(caminos_nodo)
    top = max(1, n // 100)
    concentracion = 100 * sum(heapq.nlargest(top, caminos_nodo)) / esperado if esperado else 0.0
    return {
        "nombre": g.nombre,
        "transacciones": g.num_transacciones,
        "nodos": n,
        "aristas": m,
        "densidad": (2 * m / (n * (n - 1)) if n > 1 else 0.0) if g.num_transacciones == 1 else None,
        "grado_min": min(grados, default=0),
        "grado_max": max(grados, default=0),
        "grado_prom": statistics.fmean(grados) if grados else 0.0,
        "grado_desv": statistics.pstdev(grados) if grados else 0.0,
        "nodos_aislados": sum(1 for d in grados if d == 0),
        "lazos_ignorados": g.lazos_ignorados,
        "multiaristas_ignoradas": g.multiaristas_ignoradas,
        "etiquetas_nodo": dict(Counter(g.etiqueta_nodo.values()).most_common()),
        "etiquetas_arista": dict(Counter(e[2] for e in g.aristas).most_common()),
        "caminos_esperados": esperado,
        "triangulos": contar_triangulos(g),
        "pct_caminos_top1pct_nodos": concentracion,
        **analizar_conexidad(g),
    }


def nodos_hub(g, k, total):
    orden = heapq.nlargest(k, g.adyacencia, key=lambda n: len(g.adyacencia[n]))
    salida = []
    for n in orden:
        d = len(g.adyacencia[n])
        salida.append({
            "nodo": formato_nodo(n),
            "etiqueta": g.etiqueta_nodo[n],
            "grado": d,
            "caminos": comb(d, 2),
            "porcentaje": 100 * comb(d, 2) / total if total else 0.0,
        })
    return salida


def ejecutar(g, metodo, reps, medir_memoria):
    tiempos = []
    censo, info = {}, {}
    for _ in range(reps):
        t0 = time.perf_counter()
        censo, info = metodo(g)
        tiempos.append(time.perf_counter() - t0)
    pico = None
    if medir_memoria:
        tracemalloc.start()
        metodo(g)
        pico = tracemalloc.get_traced_memory()[1] / 2**20
        tracemalloc.stop()
    return censo, info, tiempos, pico


def resumen_tiempos(tiempos, unicos, pico):
    prom = statistics.fmean(tiempos)
    return {
        "repeticiones": len(tiempos),
        "t_censo_prom_s": prom,
        "t_censo_min_s": min(tiempos),
        "t_censo_max_s": max(tiempos),
        "t_censo_desv_s": statistics.pstdev(tiempos),
        "caminos_por_s": unicos / prom if prom > 0 else None,
        "memoria_pico_mb": pico,
    }


def alfabeto(k, mayusculas):
    base = string.ascii_uppercase if mayusculas else string.ascii_lowercase
    k = max(1, k)
    return list(base[:k]) if k <= 26 else [f"{base[0]}{i}" for i in range(k)]


def generar_ejemplo():
    g = Grafo("ejemplo")
    for nid, etq in [(0, "C"), (1, "C"), (2, "O"), (3, "N"), (4, "C"), (5, "O")]:
        g.agregar_nodo(nid, etq)
    for u, v, etq in [(0, 1, "s"), (1, 2, "d"), (0, 3, "s"), (1, 4, "s"),
                      (4, 5, "s"), (0, 4, "s"), (3, 4, "d")]:
        g.agregar_arista(u, v, etq)
    return g


def generar_triangulo():
    g = Grafo("triangulo")
    for i in range(3):
        g.agregar_nodo(i, "A")
    for u, v in [(0, 1), (1, 2), (0, 2)]:
        g.agregar_arista(u, v, "a")
    return g


def generar_estrella(n):
    g = Grafo(f"estrella_n{n}")
    g.agregar_nodo(0, "A")
    for i in range(1, n):
        g.agregar_nodo(i, "B")
        g.agregar_arista(0, i, "a")
    return g


def generar_completo(n, rng, en, ea):
    g = Grafo(f"completo_n{n}")
    for i in range(n):
        g.agregar_nodo(i, rng.choice(en))
    for u in range(n):
        for v in range(u + 1, n):
            g.agregar_arista(u, v, rng.choice(ea))
    return g


def generar_aleatorio(n, m, rng, en, ea):
    m = min(m, n * (n - 1) // 2)
    g = Grafo(f"aleatorio_n{n}_m{m}")
    for i in range(n):
        g.agregar_nodo(i, rng.choice(en))
    while len(g.aristas) < m:
        u, v = rng.randrange(n), rng.randrange(n)
        if u != v and frozenset((u, v)) not in g.pares:
            g.agregar_arista(u, v, rng.choice(ea))
    return g


def generar_ba(n, k, rng, en, ea):
    k = max(1, k)
    n = max(n, k + 1)
    g = Grafo(f"ba_n{n}_k{k}")
    for i in range(n):
        g.agregar_nodo(i, rng.choice(en))
    objetivos = set(range(k))
    repetidos = []
    for nuevo in range(k, n):
        for t in objetivos:
            g.agregar_arista(nuevo, t, rng.choice(ea))
            repetidos.extend((nuevo, t))
        objetivos = set()
        while len(objetivos) < k:
            objetivos.add(rng.choice(repetidos))
    return g


def cargar_gspan(ruta, ids=None, max_grafos=None):
    g = Grafo(ruta)
    ids = set(ids) if ids else None
    tid = "0"
    incluir = ids is None or tid in ids
    cargados = 0
    with open(ruta, encoding="utf-8") as f:
        for linea in f:
            p = linea.split()
            if not p:
                continue
            if p[0] == "t":
                if p[-1] == "-1":
                    break
                tid = p[-1]
                incluir = ids is None or tid in ids
                if incluir:
                    if max_grafos and cargados >= max_grafos:
                        break
                    cargados += 1
            elif incluir and p[0] == "v":
                g.agregar_nodo((tid, p[1]), p[2] if len(p) > 2 else "-", tid)
            elif incluir and p[0] == "e":
                g.agregar_arista((tid, p[1]), (tid, p[2]), p[3] if len(p) > 3 else "-")
    return g


def construir_grafo(a):
    rng = random.Random(a.semilla)
    en = alfabeto(a.etiquetas_nodo, True)
    ea = alfabeto(a.etiquetas_arista, False)
    if a.archivo:
        return cargar_gspan(a.archivo, a.ids, a.max_grafos)
    if a.grafo == "ejemplo":
        return generar_ejemplo()
    if a.grafo == "triangulo":
        return generar_triangulo()
    if a.grafo == "estrella":
        return generar_estrella(a.n)
    if a.grafo == "completo":
        return generar_completo(a.n, rng, en, ea)
    if a.grafo == "aleatorio":
        return generar_aleatorio(a.n, a.m, rng, en, ea)
    return generar_ba(a.n, a.k, rng, en, ea)


def armar_resultado(g, est, args, t_construccion, censo, info, tiempos, pico, soporte, comparacion, tri=None):
    total = sum(censo.values())
    unicos = info["generados"] - info["descartados"]
    inducido = tri["caminos_inducidos"] if tri else None
    filas = []
    for clave, freq in sorted(censo.items(), key=lambda x: (-x[1], formato_patron(x[0]))):
        fila = {"patron": formato_patron(clave), "frecuencia": freq, "porcentaje": 100 * freq / total if total else 0.0}
        if inducido is not None:
            fila["inducido"] = inducido.get(clave, 0)
        if soporte is not None:
            fila["soporte"] = soporte.get(clave, 0)
        filas.append(fila)
    filas_tri = []
    if tri:
        total_tri = sum(tri["triangulos"].values())
        for clave, freq in sorted(tri["triangulos"].items(), key=lambda x: (-x[1], formato_triangulo(x[0]))):
            filas_tri.append({"patron": formato_triangulo(clave), "frecuencia": freq,
                              "porcentaje": 100 * freq / total_tri if total_tri else 0.0})
    conteo = {
        "no_inducido": total,
        "caminos_en_triangulos": 3 * est["triangulos"],
        "inducido": total - 3 * est["triangulos"],
        "triangulos": est["triangulos"],
        "patrones_distintos": len(censo),
        "patrones_frecuencia_1": sum(1 for f in censo.values() if f == 1),
    }
    if tri:
        conteo.update({
            "t_triangulos_s": tri["t_s"],
            "patrones_triangulo": len(tri["triangulos"]),
            "patrones_inducidos": len(inducido),
            "inducido_por_patron_ok": sum(inducido.values()) == total - 3 * est["triangulos"],
            "triangulos_por_patron_ok": sum(tri["triangulos"].values()) == est["triangulos"],
        })
    ejecucion = {"metodo": args.metodo, "t_construccion_s": t_construccion}
    ejecucion.update(resumen_tiempos(tiempos, unicos, pico))
    return {
        "grafo": est,
        "ejecucion": ejecucion,
        "exploracion": {
            "generados": info["generados"],
            "unicos": unicos,
            "descartados": info["descartados"],
            "pct_descartados": 100 * info["descartados"] / info["generados"] if info["generados"] else 0.0,
            "redundancia": info["generados"] / unicos if unicos else 0.0,
            "tam_hash_ids": info["tam_hash_ids"],
            "validacion_ok": total == est["caminos_esperados"],
        },
        "conteo": conteo,
        "censo": filas,
        "triangulos": filas_tri,
        "hubs": nodos_hub(g, args.hubs, est["caminos_esperados"]),
        "comparacion": comparacion,
    }


def seg(t):
    return f"{t * 1000:.3f} ms" if t < 1 else f"{t:.3f} s"


def estadisticas(valores):
    v = [float(x) for x in valores]
    return {"media": statistics.fmean(v), "desv": statistics.pstdev(v) if len(v) > 1 else 0.0,
            "min": min(v), "max": max(v), "n": len(v)}


def aplanar(prefijo, est):
    return {f"{prefijo}_s": est["media"], f"{prefijo}_desv_s": est["desv"], f"{prefijo}_min_s": est["min"]}


def seg_est(est):
    if est["n"] < 2:
        return seg(est["media"])
    return f"{seg(est['media'])} ± {seg(est['desv'])}"


def imprimir_reporte(r, top, min_freq):
    gr, ej, ex, co = r["grafo"], r["ejecucion"], r["exploracion"], r["conteo"]
    densidad = "n/a" if gr["densidad"] is None else f"{gr['densidad']:.6f}"
    print("\n=== GRAFO ===")
    print(f"Nombre: {gr['nombre']} | transacciones: {gr['transacciones']}")
    print(f"Nodos: {gr['nodos']} | aristas: {gr['aristas']} | densidad: {densidad}")
    print(f"Grado min/max/prom/desv: {gr['grado_min']} / {gr['grado_max']} / {gr['grado_prom']:.2f} / {gr['grado_desv']:.2f}")
    print(f"Nodos aislados: {gr['nodos_aislados']} | lazos ignorados: {gr['lazos_ignorados']} | multiaristas ignoradas: {gr['multiaristas_ignoradas']}")
    print(f"Etiquetas nodo ({len(gr['etiquetas_nodo'])}): {dict(list(gr['etiquetas_nodo'].items())[:10])}")
    print(f"Etiquetas arista ({len(gr['etiquetas_arista'])}): {dict(list(gr['etiquetas_arista'].items())[:10])}")
    imprimir_conexidad(gr)
    print(f"Triángulos: {gr['triangulos']}")
    print(f"El 1% de nodos con mayor grado genera el {gr['pct_caminos_top1pct_nodos']:.2f}% de los caminos")

    print("\n=== EJECUCIÓN ===")
    print(f"Método: {ej['metodo']} | repeticiones: {ej['repeticiones']}")
    print(f"Construcción/carga del grafo: {seg(ej['t_construccion_s'])}")
    print(f"Censo prom/min/max/desv: {seg(ej['t_censo_prom_s'])} / {seg(ej['t_censo_min_s'])} / {seg(ej['t_censo_max_s'])} / {seg(ej['t_censo_desv_s'])}")
    if ej["caminos_por_s"]:
        print(f"Rendimiento: {ej['caminos_por_s']:,.0f} caminos/s")
    if ej["memoria_pico_mb"] is not None:
        print(f"Memoria pico del censo: {ej['memoria_pico_mb']:.2f} MB")

    print("\n=== EXPLORACIÓN ===")
    print(f"Candidatos generados: {ex['generados']} | únicos: {ex['unicos']} | descartados: {ex['descartados']} ({ex['pct_descartados']:.1f}%)")
    print(f"Redundancia (generados/únicos): {ex['redundancia']:.2f}x | IDs en hash de vistos: {ex['tam_hash_ids']}")
    print(f"Validación contra Σ C(grado,2) = {gr['caminos_esperados']}: {'OK' if ex['validacion_ok'] else 'ERROR'}")

    print("\n=== CONTEO ===")
    print(f"Caminos no inducidos: {co['no_inducido']} | en triángulos: {co['caminos_en_triangulos']} | "
          f"caminos inducidos: {co['inducido']} | triángulos: {co['triangulos']}")
    print(f"Patrones de camino: {co['patrones_distintos']} | con frecuencia 1: {co['patrones_frecuencia_1']}")
    if "patrones_triangulo" in co:
        print(f"Censo de triángulos por patrón: {seg(co['t_triangulos_s'])} | patrones de triángulo: {co['patrones_triangulo']} | "
              f"patrones de camino inducido: {co['patrones_inducidos']}")
        print(f"Σ inducido por patrón = no inducido − 3·triángulos: {'OK' if co['inducido_por_patron_ok'] else 'ERROR'} | "
              f"Σ triángulos por patrón = triángulos: {'OK' if co['triangulos_por_patron_ok'] else 'ERROR'}")

    filas = [f for f in r["censo"] if f["frecuencia"] >= min_freq]
    mostradas = filas[:top] if top else filas
    print(f"\n=== CENSO DE CAMINOS (mostrando {len(mostradas)} de {len(filas)} con frecuencia >= {min_freq}) ===")
    if mostradas:
        ancho = max(len("Patrón"), max(len(f["patron"]) for f in mostradas))
        con_soporte = "soporte" in mostradas[0]
        con_inducido = "inducido" in mostradas[0]
        encabezado = f"{'#':>4}  {'Patrón':<{ancho}}  {'No inducido':>12}  {'%':>7}"
        print(encabezado + (f"  {'Inducido':>12}" if con_inducido else "") + (f"  {'Soporte':>8}" if con_soporte else ""))
        for i, f in enumerate(mostradas, 1):
            linea = f"{i:>4}  {f['patron']:<{ancho}}  {f['frecuencia']:>12}  {f['porcentaje']:>6.2f}%"
            linea += f"  {f['inducido']:>12}" if con_inducido else ""
            print(linea + (f"  {f['soporte']:>8}" if con_soporte else ""))

    tri = r["triangulos"][:top] if top else r["triangulos"]
    if tri:
        print(f"\n=== CENSO DE TRIÁNGULOS (mostrando {len(tri)} de {len(r['triangulos'])}) ===")
        ancho = max(len("Patrón"), max(len(f["patron"]) for f in tri))
        print(f"{'#':>4}  {'Patrón':<{ancho}}  {'Frecuencia':>12}  {'%':>7}")
        for i, f in enumerate(tri, 1):
            print(f"{i:>4}  {f['patron']:<{ancho}}  {f['frecuencia']:>12}  {f['porcentaje']:>6.2f}%")

    if r["hubs"]:
        print("\n=== NODOS QUE MÁS APORTAN ===")
        for h in r["hubs"]:
            print(f"{h['nodo']:>12} ({h['etiqueta']})  grado {h['grado']:>6}  caminos {h['caminos']:>10}  {h['porcentaje']:6.2f}%")

    c = r["comparacion"]
    if c:
        print("\n=== COMPARACIÓN DE MÉTODOS ===")
        for nombre, t in c["tiempos_prom_s"].items():
            mem = c["memoria_pico_mb"].get(nombre)
            print(f"{nombre:>12}: {seg(t)}" + (f" | {mem:.2f} MB" if mem is not None else ""))
        print(f"Censos idénticos: {'sí' if c['censos_iguales'] else 'NO'} | aceleración de 'centro' sobre 'exploracion': {c['aceleracion_centro']:.2f}x")


def crear_parser():
    p = argparse.ArgumentParser(description="Censo de subgrafos de 3 nodos: caminos (inducidos y no inducidos) y triángulos")
    p.add_argument("--grafo", choices=["ejemplo", "triangulo", "estrella", "completo", "aleatorio", "ba"])
    p.add_argument("--archivo", help="Archivo en formato gSpan (t/v/e)")
    p.add_argument("--ids", nargs="+", help="IDs de transacciones a cargar del archivo")
    p.add_argument("--max-grafos", type=int, default=None)
    p.add_argument("--n", type=int, default=1000, help="Número de nodos")
    p.add_argument("--m", type=int, default=5000, help="Número de aristas (aleatorio)")
    p.add_argument("--k", type=int, default=3, help="Conexiones por nodo nuevo (ba)")
    p.add_argument("--etiquetas-nodo", type=int, default=3)
    p.add_argument("--etiquetas-arista", type=int, default=2)
    p.add_argument("--semilla", type=int, default=42)
    p.add_argument("--metodo", choices=list(METODOS), default="exploracion")
    p.add_argument("--comparar", action="store_true", help="Ejecuta ambos métodos y los compara")
    p.add_argument("--reps", type=int, default=1)
    p.add_argument("--memoria", action="store_true", help="Mide memoria pico (ejecución extra)")
    p.add_argument("--top", type=int, default=20, help="Filas del censo a mostrar (0 = todas)")
    p.add_argument("--min-freq", type=int, default=1)
    p.add_argument("--hubs", type=int, default=5)
    p.add_argument("--sin-triangulos", action="store_true",
                   help="No calcula el censo de triángulos ni el inducido por patrón (grafos grandes)")
    p.add_argument("--json", action="store_true", help="Exporta el resultado completo a JSON")
    p.add_argument("--dibujar", action="store_true", help="Guarda un dibujo del grafo (.png)")
    p.add_argument("--salida", default="resultados", help="Carpeta para los archivos generados")
    p.add_argument("--max-dibujo", type=int, default=300, help="Máximo de nodos para dibujar")
    return p


def preguntar(texto, defecto, tipo=str):
    r = input(f"{texto} [{defecto}]: ").strip()
    if not r:
        return defecto
    try:
        return tipo(r)
    except ValueError:
        return defecto


def si_no(s):
    return s.lower().startswith("s")


def menu_interactivo(parser):
    a = parser.parse_args([])
    opciones = [
        ("ejemplo", "grafo pequeño de prueba (6 nodos)"),
        ("triangulo", "K3, muestra la diferencia inducido/no inducido"),
        ("estrella", "un centro con n-1 hojas, caso extremo de hub"),
        ("completo", "K_n con etiquetas aleatorias"),
        ("aleatorio", "Erdős–Rényi G(n, m)"),
        ("ba", "Barabási–Albert, libre de escala"),
        ("archivo", "cargar archivo gSpan"),
    ]
    print("Grafos disponibles:")
    for i, (nombre, desc) in enumerate(opciones, 1):
        print(f"  {i}. {nombre:<10} {desc}")
    op = min(max(preguntar("Elige un grafo", 1, int), 1), len(opciones))
    eleccion = opciones[op - 1][0]
    if eleccion == "archivo":
        a.archivo = preguntar("Ruta del archivo", "datos.txt")
        ids = preguntar("IDs de transacciones separados por espacio (vacío = todas)", "")
        a.ids = ids.split() if ids else None
        a.max_grafos = preguntar("Máximo de grafos (0 = sin límite)", 0, int) or None
    else:
        a.grafo = eleccion
    if eleccion in ("estrella", "completo", "aleatorio", "ba"):
        a.n = preguntar("Nodos", a.n, int)
    if eleccion == "aleatorio":
        a.m = preguntar("Aristas", a.m, int)
    if eleccion == "ba":
        a.k = preguntar("Conexiones por nodo nuevo", a.k, int)
    if eleccion in ("completo", "aleatorio", "ba"):
        a.etiquetas_nodo = preguntar("Cantidad de etiquetas de nodo", a.etiquetas_nodo, int)
        a.etiquetas_arista = preguntar("Cantidad de etiquetas de arista", a.etiquetas_arista, int)
        a.semilla = preguntar("Semilla", a.semilla, int)
    a.metodo = preguntar("Método (exploracion/centro)", a.metodo)
    if a.metodo not in METODOS:
        a.metodo = "exploracion"
    a.reps = max(1, preguntar("Repeticiones", a.reps, int))
    a.comparar = preguntar("¿Comparar ambos métodos? (s/n)", "n", si_no)
    a.memoria = preguntar("¿Medir memoria pico? (s/n)", "n", si_no)
    a.top = preguntar("Filas del censo a mostrar (0 = todas)", a.top, int)
    a.json = preguntar("¿Exportar a JSON? (s/n)", "n", si_no)
    a.dibujar = preguntar("¿Guardar dibujo del grafo? (s/n)", "n", si_no)
    return a


def main():
    parser = crear_parser()
    args = parser.parse_args()
    if args.grafo is None and args.archivo is None:
        args = menu_interactivo(parser)
    args.reps = max(1, args.reps)

    t0 = time.perf_counter()
    g = construir_grafo(args)
    t_construccion = time.perf_counter() - t0

    est = estadisticas_grafo(g)
    censo, info, tiempos, pico = ejecutar(g, METODOS[args.metodo], args.reps, args.memoria)
    soporte = calcular_soporte(g) if g.num_transacciones > 1 else None

    comparacion = None
    if args.comparar:
        otro = "centro" if args.metodo == "exploracion" else "exploracion"
        censo2, _, tiempos2, pico2 = ejecutar(g, METODOS[otro], args.reps, args.memoria)
        t = {args.metodo: statistics.fmean(tiempos), otro: statistics.fmean(tiempos2)}
        comparacion = {
            "tiempos_prom_s": t,
            "memoria_pico_mb": {k: v for k, v in ((args.metodo, pico), (otro, pico2)) if v is not None},
            "censos_iguales": censo == censo2,
            "aceleracion_centro": t["exploracion"] / t["centro"] if t["centro"] > 0 else 0.0,
        }

    tri = None
    if not args.sin_triangulos:
        t0 = time.perf_counter()
        triangulos, resta = censo_triangulos(g)
        tri = {"triangulos": triangulos, "caminos_inducidos": censo_inducido(censo, resta),
               "t_s": time.perf_counter() - t0}

    resultado = armar_resultado(g, est, args, t_construccion, censo, info, tiempos, pico, soporte, comparacion, tri)
    imprimir_reporte(resultado, args.top, args.min_freq)

    generados = []
    if args.json or args.dibujar:
        prefijo = preparar_salida(args.salida, g.nombre)
        if args.json:
            guardar_json(resultado, prefijo + ".json")
            generados.append(prefijo + ".json")
        if args.dibujar and dibujar_grafo(g, prefijo + ".png", args.max_dibujo, args.semilla):
            generados.append(prefijo + ".png")
    if generados:
        print("\nArchivos generados:")
        for ruta in generados:
            print(f"  {ruta}")


if __name__ == "__main__":
    main()
