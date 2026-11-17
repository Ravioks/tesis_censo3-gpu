import argparse
import csv
import glob
import json
import os
import statistics

import numpy as np

AZUL, NARANJA, AQUA, VIOLETA = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
TINTA, TINTA_2, TENUE, GRILLA, BASE, SUPERFICIE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#ffffff"
SECUENCIAL = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
              "#1c5cab", "#184f95", "#104281", "#0d366b"]
ANCHO = 4.8

NOMBRES = {
    "ba_np_n1000000_k8_L3_E2": "BA",
    "aleatorio_np_n100000_m5000000_L3_E2": "ER",
    "completo_np_n1000_L3_E2": "$K_{1000}$",
    "com-LiveJournal": "LiveJournal",
    "ogbn-mag": "ogbn-mag",
    "ogbn-products": "ogbn-products",
}
REALES = ["com-LiveJournal", "ogbn-mag", "ogbn-products"]
SINTETICOS = ["ba_np_n1000000_k8_L3_E2", "aleatorio_np_n100000_m5000000_L3_E2", "completo_np_n1000_L3_E2"]
ESTRATEGIAS = {"caminos": ["nodo", "entrada", "balanceado", "combinatorio"], "triangulos": ["arista", "binaria", "balanceado"]}
NOMBRE_ESTRATEGIA = {"nodo": "node", "entrada": "entry", "balanceado": "balanced", "combinatorio": "combinatorial",
                     "arista": "edge", "binaria": "binary"}
MEMORIAS = ["global", "shared", "hash"]
MACROS = {"com-LiveJournal": "LJ", "ogbn-mag": "MAG", "ogbn-products": "PROD", "ba_np_n1000000_k8_L3_E2": "BA",
          "aleatorio_np_n100000_m5000000_L3_E2": "ER", "completo_np_n1000_L3_E2": "KN"}


def cargar(ruta):
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def demos(res):
    return [(os.path.basename(os.path.dirname(r)), cargar(r)) for r in sorted(glob.glob(os.path.join(res, "demo_*", "resultados.json")))]


def corridas_reales(res):
    reales = {}
    for nombre, d in demos(res):
        g = d.get("grafo_real")
        if g and g.get("metodos"):
            reales[g["grafo"]] = {"demo": nombre, "commit": d["entorno"].get("commit"), **g}
    return reales


def demo_sinteticos(res):
    elegido = None
    for nombre, d in demos(res):
        if d.get("barrido") and d.get("cpu_gpu"):
            elegido = (nombre, d)
    return elegido


def sesiones_escalabilidad(res):
    return [(nombre, d["entorno"].get("commit"), d["escalabilidad"]) for nombre, d in demos(res) if d.get("escalabilidad")]


def barridos(res):
    filas = {}
    for ruta in sorted(glob.glob(os.path.join(res, "barrido_*", "barrido.json"))):
        d = cargar(ruta)
        filas[d["grafo"]["nombre"]] = {"filas": d["filas"], "config": d["config"], "commit": d.get("commit"),
                                       "completo": d.get("completo"), "grafo": d["grafo"]}
    sint = demo_sinteticos(res)
    if sint:
        for b in sint[1]["barrido"]:
            filas.setdefault(b["grafo"], {"filas": b["filas"], "config": b["config"],
                                          "commit": sint[1]["entorno"].get("commit"), "completo": b.get("completo"),
                                          "grafo": {"nombre": b["grafo"], "nodos": b["nodos"], "aristas": b["aristas"],
                                                    "W": b["W"]}})
    return filas


def censos(res):
    salida = []
    for ruta in sorted(glob.glob(os.path.join(res, "*", "*_censo3_*.json"))):
        d = cargar(ruta)
        d["_ruta"] = ruta
        salida.append(d)
    return salida


def misma_config(a, b):
    claves = ("estrategia_caminos", "memoria_caminos", "bloque_caminos", "estrategia_tri", "memoria_tri", "bloque_tri")
    return all(a.get(k) == b.get(k) for k in claves)


def corridas_fases(res, barr):
    fases = {}
    for d in censos(res):
        nombre = d["grafo"]["nombre"]
        if nombre not in barr or d.get("reps", 0) < 5 or not misma_config(d["config"], barr[nombre]["config"]):
            continue
        fases.setdefault(nombre, {})[d["preproceso"]] = d
    return fases


def grafo_censo(res, nombre):
    elegido = None
    for d in censos(res):
        if d["grafo"]["nombre"] == nombre and (elegido is None or d.get("reps", 0) >= elegido.get("reps", 0)):
            elegido = d
    return elegido


def csv_patrones(res, nombre):
    rutas = [r for r in sorted(glob.glob(os.path.join(res, "*", "*_censo3_*.csv")))
             if os.path.basename(r).split("_censo3_")[1][:-4] == nombre]
    if not rutas:
        return None
    filas = {"camino": [], "triangulo": []}
    with open(rutas[-1], encoding="utf-8") as f:
        for fila in csv.DictReader(f):
            filas[fila["tipo"]].append(int(fila["no_inducido"]))
    return filas


def metodo(metodos, prefijo, contiene=None, excluye=None):
    for nombre, valores in metodos.items():
        if nombre.startswith(prefijo) and (contiene is None or contiene in nombre) and (excluye is None or excluye not in nombre):
            return valores
    return None


def metodos_reales(g):
    m = g["metodos"]
    cfg = g["config"]
    principal = f"GPU · {cfg['estrategia_caminos']} + "
    otro = "entrada" if cfg["estrategia_caminos"] != "entrada" else None
    return {
        "njit1_enum": metodo(m, "njit 1 hilo · enumeracion"),
        "njit1_comb": metodo(m, "njit 1 hilo · combinatorio"),
        "njitN_enum": metodo(m, "njit ", "hilos · enumeracion"),
        "njitN_comb": metodo(m, "njit ", "hilos · combinatorio"),
        "gpu": metodo(m, principal, "prep. gpu"),
        "gpu_prep_cpu": metodo(m, principal, "prep. cpu"),
        "gpu_entrada": metodo(m, "GPU · entrada + ", "prep. gpu") if otro else None,
    }


def estilo(plt):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.edgecolor": BASE, "axes.linewidth": 0.6,
        "axes.labelcolor": TINTA_2, "xtick.color": TINTA_2, "ytick.color": TINTA_2, "axes.titlecolor": TINTA,
        "text.color": TINTA, "figure.facecolor": SUPERFICIE, "axes.facecolor": SUPERFICIE, "savefig.facecolor": SUPERFICIE,
        "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "legend.frameon": False,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    })


def grilla(ax, eje="y"):
    ax.grid(True, axis=eje, which="major", color=GRILLA, linewidth=0.5)
    ax.set_axisbelow(True)


def texto_tiempo(s):
    if s >= 100:
        return f"{s:,.0f} s"
    if s >= 1:
        return f"{s:.2f} s"
    ms = s * 1e3
    return f"{ms:.0f} ms" if ms >= 100 else f"{ms:.1f} ms"


def veces(x):
    return f"{x:.0f}×" if x >= 100 else f"{x:.1f}×"


def guardar(fig, carpeta, nombre, generados):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(carpeta, f"{nombre}.{ext}"), dpi=200, bbox_inches="tight", pad_inches=0.02)
    generados.append(nombre)


def mejores_por_celda(filas, parte):
    celdas = {}
    for f in filas:
        if f["parte"] != parte:
            continue
        clave = (f["estrategia"], f["memoria"])
        if f.get("omitida"):
            celdas.setdefault(clave, None)
            continue
        if f.get("correcto") and (celdas.get(clave) is None or f["t_total_s"] < celdas[clave]):
            celdas[clave] = f["t_total_s"]
    return celdas


def figura_heatmap(plt, barr, carpeta, generados):
    from matplotlib.colors import ListedColormap, LogNorm

    orden = [g for g in SINTETICOS + REALES if g in barr]
    cmap = ListedColormap(SECUENCIAL)
    norm = LogNorm(1, 1000)
    fig, ejes = plt.subplots(2, 1, figsize=(ANCHO, 2.75), gridspec_kw={"height_ratios": [4, 3], "hspace": 0.55})
    for ax, parte, titulo in zip(ejes, ("caminos", "triangulos"), ("(a) Paths", "(b) Triangles")):
        filas_e = ESTRATEGIAS[parte]
        matriz = np.full((len(filas_e), 3 * len(orden)), np.nan)
        omitidas = np.zeros_like(matriz, dtype=bool)
        for j, g in enumerate(orden):
            celdas = mejores_por_celda(barr[g]["filas"], parte)
            validos = [v for v in celdas.values() if v]
            minimo = min(validos) if validos else None
            for i, e in enumerate(filas_e):
                for k, mem in enumerate(MEMORIAS):
                    v = celdas.get((e, mem), "ausente")
                    if v is None:
                        omitidas[i, 3 * j + k] = True
                    elif v != "ausente":
                        matriz[i, 3 * j + k] = v / minimo
        ax.imshow(np.ma.masked_invalid(matriz), cmap=cmap, norm=norm, aspect="auto")
        for i in range(matriz.shape[0]):
            for c in range(matriz.shape[1]):
                if omitidas[i, c]:
                    ax.add_patch(plt.Rectangle((c - 0.5, i - 0.5), 1, 1, facecolor="#f0efec", edgecolor=BASE, hatch="////",
                                               linewidth=0))
                elif not np.isnan(matriz[i, c]):
                    v = matriz[i, c]
                    ax.text(c, i, "1" if v < 1.05 else (f"{v:.1f}" if v < 10 else f"{v:.0f}"), ha="center", va="center",
                            fontsize=5, color="#ffffff" if v >= 8 else TINTA)
        for j in range(1, len(orden)):
            ax.axvline(3 * j - 0.5, color=SUPERFICIE, linewidth=2.5)
        ax.set_yticks(range(len(filas_e)), [NOMBRE_ESTRATEGIA[e] for e in filas_e])
        ax.set_xticks(range(matriz.shape[1]), [m[0] for m in MEMORIAS] * len(orden), fontsize=5.5)
        ax.tick_params(length=0)
        for lado in ("left", "bottom"):
            ax.spines[lado].set_visible(False)
        for j, g in enumerate(orden):
            ax.text(3 * j + 1, -0.75, NOMBRES[g], ha="center", va="bottom", fontsize=6.5, color=TINTA)
        ax.set_title(titulo, loc="left", pad=16, x=-0.13)
    guardar(fig, carpeta, "fig_heatmap", generados)
    plt.close(fig)


def figura_desbalance(plt, barr, carpeta, generados):
    colores = {"nodo": NARANJA, "entrada": AZUL, "balanceado": AQUA}
    marcas = {"nodo": "s", "entrada": "o", "balanceado": "^"}
    fig, ax = plt.subplots(figsize=(ANCHO * 0.62, 2.1))
    for e in ("nodo", "entrada", "balanceado"):
        xs, ys = [], []
        for g in SINTETICOS + REALES:
            if g not in barr:
                continue
            filas = [f for f in barr[g]["filas"] if f["parte"] == "caminos" and f.get("correcto")
                     and f["estrategia"] in colores and f.get("desbalance")]
            if not filas:
                continue
            minimo = min(f["t_kernel_s"] for f in filas)
            for f in filas:
                if f["estrategia"] == e:
                    xs.append(f["desbalance"])
                    ys.append(f["t_kernel_s"] / minimo)
        ax.scatter(xs, ys, s=14, marker=marcas[e], color=colores[e], edgecolor=SUPERFICIE, linewidth=0.5,
                   label=NOMBRE_ESTRATEGIA[e], zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Analytical load imbalance (heaviest thread / mean)")
    ax.set_ylabel("Kernel time / best enumerating\nconfiguration of the same graph")
    grilla(ax, "both")
    ax.legend(loc="upper left", handletextpad=0.3)
    guardar(fig, carpeta, "fig_imbalance", generados)
    plt.close(fig)


def figura_cpu_gpu(plt, reales, carpeta, generados):
    etiquetas = [("njit1_enum", "CPU 1 thr., enum."), ("njit1_comb", "CPU 1 thr., comb."), ("njitN_enum", "CPU 6 thr., enum."),
                 ("njitN_comb", "CPU 6 thr., comb."), ("gpu_entrada", "GPU, entry"), ("gpu", "GPU, best")]
    orden = [g for g in REALES if g in reales]
    fig, ejes = plt.subplots(1, len(orden), figsize=(ANCHO, 1.9), sharey=True, sharex=True)
    ejes = np.atleast_1d(ejes)
    for ax, g in zip(ejes, orden):
        m = metodos_reales(reales[g])
        filas = [(t, m[k]["censo_s"]["t_s"], k.startswith("gpu")) for k, t in etiquetas if m.get(k)]
        ys = np.arange(len(filas))[::-1]
        for y, (t, v, es_gpu) in zip(ys, filas):
            ax.barh(y, v, height=0.62, color=AZUL if es_gpu else NARANJA, left=1e-3)
            ax.text(v * 1.25, y, texto_tiempo(v), va="center", fontsize=5.5, color=TINTA_2)
        ax.set_yticks(ys, [t for t, _, _ in filas])
        ax.set_xscale("log")
        ax.set_xlim(1e-2, 1e4)
        ax.set_xticks([1e-2, 1, 1e2, 1e4])
        ax.tick_params(axis="y", length=0)
        grilla(ax, "x")
        acel = m["njitN_comb"]["censo_s"]["t_s"] / m["gpu"]["censo_s"]["t_s"]
        ax.set_title(f"{NOMBRES[g]}\nGPU: {veces(acel)} vs. CPU 6 thr.", loc="left", fontsize=6.3)
    ejes[len(ejes) // 2].set_xlabel("Census time (s, log scale)")
    guardar(fig, carpeta, "fig_cpu_gpu", generados)
    plt.close(fig)


def figura_fases(plt, fases, carpeta, generados):
    partes = [("cpu", "CPU preparation", NARANJA), ("carga", "Graph load (H2D)", VIOLETA),
              ("gpu", "GPU preparation", AQUA), ("censo", "Census", AZUL)]
    orden = [g for g in REALES if g in fases and {"gpu", "cpu"} <= set(fases[g])]
    fig, ejes = plt.subplots(len(orden), 1, figsize=(ANCHO, 0.62 * len(orden) + 0.45))
    ejes = np.atleast_1d(ejes)
    for ax, g in zip(ejes, orden):
        for y, modo in ((1, "gpu"), (0, "cpu")):
            t = fases[g][modo]["tiempos"]
            valores = {"cpu": t["preproceso_cpu"]["t_s"], "carga": t["carga"]["t_s"], "gpu": t["preproceso_gpu"]["t_s"],
                       "censo": t["censo"]["t_s"]}
            izquierda = 0.0
            for clave, _, color in partes:
                v = valores[clave]
                if v > 0:
                    ax.barh(y, v, left=izquierda, height=0.62, color=color, edgecolor=SUPERFICIE, linewidth=0.8)
                izquierda += v
            ax.text(izquierda, y, f"  {texto_tiempo(t['extremo']['t_s'])}", va="center", fontsize=5.5, color=TINTA_2)
        ax.set_yticks([1, 0], ["prep. on GPU", "prep. on CPU"])
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(0, ax.get_xlim()[1] * 1.12)
        grilla(ax, "x")
        ax.set_title(NOMBRES[g], loc="left", fontsize=6.5, pad=2)
    ejes[-1].set_xlabel("Time (s)")
    manijas = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in partes]
    fig.legend(manijas, [n for _, n, _ in partes], loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02),
               handlelength=1, columnspacing=1)
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=0.4)
    guardar(fig, carpeta, "fig_phases", generados)
    plt.close(fig)


def series_escalabilidad(sesiones):
    datos = {}
    for nombre, commit, series in sesiones:
        for serie, puntos in series.items():
            if serie == "etiquetas" and len({p["config"]["estrategia_caminos"] for p in puntos}) > 1:
                continue
            for i, p in enumerate(puntos):
                datos.setdefault(serie, {}).setdefault(i, []).append((nombre, p))
    return datos


def figura_escalabilidad(plt, sesiones, carpeta, generados):
    datos = series_escalabilidad(sesiones)
    paneles = [("tamano", "aristas", "(a) BA: size", "Edges $m$"), ("densidad", "aristas", "(b) ER: density", "Edges $m$"),
               ("etiquetas", "P", "(c) BA: label alphabet", "Path codes $P$")]
    curvas = [("cpu_censo_s", "CPU 6 thr.", NARANJA, "s", "-"), ("gpu_censo_s", "GPU census", AZUL, "o", "-"),
              ("gpu_calculo_s", "GPU kernels", AQUA, "^", "--")]
    fig, ejes = plt.subplots(1, 3, figsize=(ANCHO, 1.85))
    from matplotlib.ticker import LogLocator, NullFormatter

    for ax, (serie, eje_x, titulo, nombre_x) in zip(ejes, paneles):
        if serie not in datos:
            ax.set_visible(False)
            continue
        puntos = [datos[serie][i] for i in sorted(datos[serie])]
        xs = [p[0][1][eje_x] for p in puntos]
        for clave, nombre, color, marca, linea in curvas:
            if clave == "gpu_calculo_s" and serie != "etiquetas":
                continue
            med = [statistics.median(q[1][clave] for q in p) for p in puntos]
            bajo = [med[i] - min(q[1][clave] for q in p) for i, p in enumerate(puntos)]
            alto = [max(q[1][clave] for q in p) - med[i] for i, p in enumerate(puntos)]
            ax.errorbar(xs, med, yerr=[bajo, alto], color=color, marker=marca, markersize=3.5, linewidth=1.2,
                        linestyle=linea, elinewidth=0.6, capsize=1.5, label=nombre)
        ax.set_xscale("log")
        ax.set_yscale("log")
        if eje_x == "P":
            ax.set_xticks([1, 1e2, 1e4, 1e6])
        else:
            ax.xaxis.set_major_locator(LogLocator(base=10, numticks=8))
        ax.xaxis.set_minor_formatter(NullFormatter())
        grilla(ax, "both")
        n = len(puntos[0])
        ax.set_xlabel(nombre_x)
        ax.set_title(f"{titulo} ({n} session{'s' if n > 1 else ''})", loc="left", fontsize=6.3)
    ejes[0].set_ylabel("Time (s, log scale)")
    manijas, nombres = ejes[2].get_legend_handles_labels()
    fig.legend(manijas, nombres, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08), handlelength=1.8)
    fig.tight_layout(w_pad=0.6)
    guardar(fig, carpeta, "fig_scalability", generados)
    plt.close(fig)


def figura_distribucion(plt, res, carpeta, generados):
    colores = {"com-LiveJournal": AZUL, "ogbn-mag": NARANJA, "ogbn-products": AQUA}
    marcas = {"com-LiveJournal": "o", "ogbn-mag": "s", "ogbn-products": "^"}
    fig, ejes = plt.subplots(1, 2, figsize=(ANCHO, 1.9), sharey=True)
    for ax, tipo, titulo in zip(ejes, ("camino", "triangulo"), ("(a) Non-induced paths", "(b) Triangles")):
        for g in REALES:
            pat = csv_patrones(res, g)
            if not pat or not pat[tipo]:
                continue
            v = np.sort(np.array(pat[tipo], dtype=np.float64))[::-1]
            rango = np.arange(1, len(v) + 1)
            cada = max(1, len(v) // 400)
            ax.plot(rango, 100 * v / v.sum(), color=colores[g], linewidth=1.2, marker=marcas[g], markersize=2.8,
                    markevery=slice(0, None, max(1, len(v) // 12)) if cada > 1 or len(v) > 30 else None,
                    label=f"{NOMBRES[g]} ({len(v):,} patterns)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        grilla(ax, "both")
        ax.set_xlabel("Pattern rank")
        ax.set_title(titulo, loc="left")
        ax.legend(loc="lower left", fontsize=5.5, handlelength=1.4)
    ejes[0].set_ylabel("Share of the census (%)")
    fig.tight_layout(w_pad=0.8)
    guardar(fig, carpeta, "fig_distribution", generados)
    plt.close(fig)


def entero(x):
    return f"{int(x):,}"


def ms(s):
    v = s * 1e3
    if v >= 1000:
        return f"{s:.3f}\\,s"
    return f"{v:.1f}\\,ms" if v >= 100 else f"{v:.2f}\\,ms"


def cifras(res, reales, sint, barr, fases, sesiones):
    m = {}

    def poner(nombre, valor):
        m[nombre] = valor

    for g, r in reales.items():
        p = MACROS[g]
        mr = metodos_reales(r)
        cfg = r["config"]
        poner(f"{p}n", entero(r["nodos"]))
        poner(f"{p}m", entero(r["aristas"]))
        poner(f"{p}Dmax", entero(r["grado_max"]))
        poner(f"{p}W", entero(r["W"]))
        poner(f"{p}T", entero(r["T"]))
        poner(f"{p}Ind", entero(r["W"] - 3 * r["T"]))
        for clave, nombre in (("njit1_enum", "CpuOneEnum"), ("njit1_comb", "CpuOneComb"), ("njitN_enum", "CpuSixEnum"),
                              ("njitN_comb", "CpuSixComb"), ("gpu", "Gpu"), ("gpu_prep_cpu", "GpuPrepCpu"),
                              ("gpu_entrada", "GpuEntry")):
            if not mr.get(clave):
                continue
            v = mr[clave]
            poner(f"{p}{nombre}Census", ms(v["censo_s"]["t_s"]))
            poner(f"{p}{nombre}CensusSd", ms(v["censo_s"]["t_desv_s"]))
            poner(f"{p}{nombre}EndToEnd", ms(v["extremo_s"]["t_s"]))
            if "caminos_s" in v:
                poner(f"{p}{nombre}Paths", ms(v["caminos_s"]["t_s"]))
                poner(f"{p}{nombre}Triangles", ms(v["triangulos_s"]["t_s"]))
        gpu = mr["gpu"]["censo_s"]["t_s"]
        poner(f"{p}SpeedupSix", veces(mr["njitN_comb"]["censo_s"]["t_s"] / gpu).replace("×", "$\\times$"))
        poner(f"{p}SpeedupOne", veces(mr["njit1_comb"]["censo_s"]["t_s"] / gpu).replace("×", "$\\times$"))
        poner(f"{p}SpeedupSixEndToEnd",
              veces(mr["njitN_comb"]["extremo_s"]["t_s"] / mr["gpu"]["extremo_s"]["t_s"]).replace("×", "$\\times$"))
        poner(f"{p}Config", f"{NOMBRE_ESTRATEGIA[cfg['estrategia_caminos']]}+{cfg['memoria_caminos']} / "
                            f"{NOMBRE_ESTRATEGIA[cfg['estrategia_tri']]}+{cfg['memoria_tri']}")
        poner(f"{p}Commit", r.get("commit") or "?")
        c = grafo_censo(res, g)
        if c:
            poner(f"{p}Dout", entero(c["grafo"]["grado_salida_max"]))
            poner(f"{p}PatPaths", entero(len(c["caminos_no_inducidos"])))
            poner(f"{p}PatTriangles", entero(len(c["triangulos"])))
    for g in SINTETICOS:
        c = grafo_censo(res, g)
        if not c:
            continue
        p = MACROS[g]
        gr = c["grafo"]
        for k, v in (("n", gr["nodos"]), ("m", gr["aristas"]), ("Dmax", gr["grado_max"]), ("Dout", gr["grado_salida_max"]),
                     ("W", c["totales"]["W"]), ("T", c["totales"]["T"])):
            poner(f"{p}{k}", entero(v))
    if sint:
        for f in sint[1]["cpu_gpu"]:
            if f["grafo"] not in MACROS:
                continue
            p = MACROS[f["grafo"]]
            poner(f"{p}CpuSixBest", ms(f["njit_n_s"]))
            poner(f"{p}GpuBest", ms(f["gpu_censo_s"]))
            poner(f"{p}SpeedupSix", veces(f["njit_n_s"] / f["gpu_censo_s"]).replace("×", "$\\times$"))
    for g, modos in fases.items():
        p = MACROS[g]
        for modo, nombre in (("gpu", "PrepGpu"), ("cpu", "PrepCpu")):
            if modo not in modos:
                continue
            t = modos[modo]["tiempos"]
            poner(f"{p}{nombre}Preparation", ms(t["preparacion"]["t_s"]))
            poner(f"{p}{nombre}Load", ms(t["carga"]["t_s"]))
            poner(f"{p}{nombre}Census", ms(t["censo"]["t_s"]))
            poner(f"{p}{nombre}Kernels", ms(t["calculo"]["t_s"]))
            poner(f"{p}{nombre}Induced", ms(t["inducido"]["t_s"]))
            poner(f"{p}{nombre}EndToEnd", ms(t["extremo"]["t_s"]))
            poner(f"{p}{nombre}Jit", f"{modos[modo]['t_jit_s']:.2f}\\,s")
    for g, b in barr.items():
        p = MACROS.get(g)
        if not p:
            continue
        medidas = [f for f in b["filas"] if f.get("correcto") is not None]
        poner(f"{p}SweepMeasured", str(len(medidas)))
        poner(f"{p}SweepSkipped", str(sum(1 for f in b["filas"] if f.get("omitida"))))
        poner(f"{p}SweepWrong", str(sum(1 for f in medidas if not f.get("correcto"))))
    for nombre, clave in (("ogbn-mag_real_limpieza.json", "MAG"), ("ogbn-products_real_limpieza.json", "PROD")):
        ruta = os.path.join(res, nombre)
        if os.path.exists(ruta):
            l = cargar(ruta)
            for k, n in (("nodos_crudos", "RawN"), ("aristas_crudas", "RawM"), ("lazos", "SelfLoops"),
                         ("repetidas", "Duplicates"), ("nodos_aislados", "Isolated")):
                poner(f"{clave}{n}", entero(l[k]))
    h = sorted(glob.glob(os.path.join(res, "hash", "resumen_*.json")))
    if h:
        d = cargar(h[-1])
        poner("HashW", entero(d["totales"]["W"]))
        poner("HashPatterns", entero(d["patrones"]["caminos_no_inducidos"]))
        poner("HashKernels", ms(d["tiempos"]["calculo"]["t_s"]))
        poner("HashCensus", ms(d["tiempos"]["censo"]["t_s"]))
    poner("ScalSessions", str(len(sesiones)))
    return m


def escribir_cifras(m, ruta, commits):
    with open(ruta, "w", encoding="utf-8") as f:
        f.write("% Generado por figuras_articulo.py a partir de resultados/v1. No editar a mano.\n")
        f.write(f"% Commits de las corridas: {', '.join(sorted(c for c in commits if c))}\n")
        for k in sorted(m):
            f.write(f"\\newcommand{{\\num{k}}}{{{m[k]}}}\n")


def main():
    p = argparse.ArgumentParser(description="Figuras (PDF) y cifras.tex del artículo a partir de resultados/v1")
    p.add_argument("--resultados", default=os.path.join("resultados", "v1"))
    p.add_argument("--salida", default=os.path.join("resultados", "v1", "figuras"))
    a = p.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    estilo(plt)
    os.makedirs(a.salida, exist_ok=True)
    reales = corridas_reales(a.resultados)
    sint = demo_sinteticos(a.resultados)
    barr = barridos(a.resultados)
    fases = corridas_fases(a.resultados, barr)
    sesiones = sesiones_escalabilidad(a.resultados)
    generados = []
    figura_heatmap(plt, barr, a.salida, generados)
    figura_desbalance(plt, barr, a.salida, generados)
    figura_cpu_gpu(plt, reales, a.salida, generados)
    figura_fases(plt, fases, a.salida, generados)
    figura_escalabilidad(plt, sesiones, a.salida, generados)
    figura_distribucion(plt, a.resultados, a.salida, generados)
    m = cifras(a.resultados, reales, sint, barr, fases, sesiones)
    commits = {r.get("commit") for r in reales.values()} | {b.get("commit") for b in barr.values()}
    commits |= {c for _, c, _ in sesiones}
    escribir_cifras(m, os.path.join(a.salida, "cifras.tex"), commits)
    print(f"Grafos reales: {', '.join(reales)} | barridos: {len(barr)} | fases: {', '.join(fases)} | "
          f"sesiones de escalabilidad: {', '.join(n for n, _, _ in sesiones)}")
    print(f"Figuras: {', '.join(generados)} | cifras: {len(m)} macros")
    print(f"Salida: {a.salida}")


if __name__ == "__main__":
    main()
