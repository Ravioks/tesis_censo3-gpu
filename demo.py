import argparse
import gc
import json
import os
import platform
import random
import time
import traceback
import tracemalloc
from datetime import datetime

import numpy as np

import censo_3n2a as sec

try:
    from numba import cuda, get_num_threads
    import barrido as bar
    import censo_3n as c3
    import censo_3n2a_gpu as gpu
    import censo_3n3a_gpu as tri
    import censo_3n_cpu_njit as cpu
    import generadores as gen
    ERROR_NUMBA = None
except Exception as e:
    cuda = get_num_threads = bar = c3 = gpu = tri = cpu = gen = None
    ERROR_NUMBA = f"{type(e).__name__}: {e}"

EXPERIMENTOS = ["validacion", "estrategias", "barrido", "cpu_gpu", "escalabilidad", "real"]
LIMITE_PYTHON_DEMO = 5_000_000
COLOR_GPU, COLOR_NJIT, COLOR_PYTHON = "#2a78d6", "#eb6834", "#1baf7a"
TINTA, TINTA_2, SUPERFICIE, GRILLA = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"

ESCALAS = {
    "completa": {
        "validacion": [("ejemplo", {}), ("triangulo", {}), ("estrella", {"n": 50}), ("ba", {"n": 300, "k": 2}),
                       ("aleatorio", {"n": 200, "m": 600, "L": 4, "E": 3}), ("completo", {"n": 30}),
                       ("etiquetas", {"L": 13, "E": 1, "n": 200, "m": 800}),
                       ("etiquetas", {"L": 260, "E": 1, "n": 600, "m": 2400})],
        "estrategias": [("estrella", {"n": 1000}), ("aleatorio", {"n": 10000, "m": 50000}), ("ba", {"n": 20000, "k": 3})],
        "barrido": [("ba", {"n": 1_000_000, "k": 8}), ("aleatorio", {"n": 100_000, "m": 5_000_000}),
                    ("completo", {"n": 1000})],
        "bloques_barrido": [128, 256, 512],
        "cpu_gpu": [("py:ba", {"n": 50000, "k": 3}), ("ba", {"n": 1_000_000, "k": 8}),
                    ("aleatorio", {"n": 100_000, "m": 5_000_000}), ("completo", {"n": 1000})],
        "escalabilidad": {
            "tamano": [("ba", {"n": n, "k": 8}) for n in (100_000, 300_000, 1_000_000, 3_000_000)],
            "densidad": [("aleatorio", {"n": 100_000, "m": m}) for m in (500_000, 1_000_000, 2_000_000, 5_000_000)],
            "etiquetas": [("ba", {"n": 300_000, "k": 8, "L": L, "E": E})
                          for L, E in ((1, 1), (2, 1), (4, 2), (8, 2), (16, 2), (64, 2))],
        },
        "dataset": "livejournal",
    },
    "rapida": {
        "validacion": [("ejemplo", {}), ("triangulo", {}), ("ba", {"n": 300, "k": 2}),
                       ("etiquetas", {"L": 13, "E": 1, "n": 200, "m": 800})],
        "estrategias": [("aleatorio", {"n": 3000, "m": 15000}), ("ba", {"n": 5000, "k": 3})],
        "barrido": [("ba", {"n": 200_000, "k": 8}), ("completo", {"n": 500})],
        "bloques_barrido": [256],
        "cpu_gpu": [("py:ba", {"n": 10000, "k": 3}), ("ba", {"n": 200_000, "k": 8}), ("completo", {"n": 500})],
        "escalabilidad": {
            "tamano": [("ba", {"n": n, "k": 8}) for n in (30_000, 100_000, 300_000)],
            "densidad": [("aleatorio", {"n": 30_000, "m": m}) for m in (100_000, 300_000, 1_000_000)],
            "etiquetas": [("ba", {"n": 100_000, "k": 8, "L": L, "E": E}) for L, E in ((1, 1), (4, 2), (8, 2), (64, 2))],
        },
        "dataset": "ninguno",
    },
    "prueba": {
        "validacion": [("ejemplo", {}), ("triangulo", {}), ("estrella", {"n": 15}),
                       ("etiquetas", {"L": 13, "E": 1, "n": 40, "m": 120})],
        "estrategias": [("aleatorio", {"n": 200, "m": 600})],
        "barrido": [("ba", {"n": 60, "k": 3})],
        "bloques_barrido": [32],
        "cpu_gpu": [("py:ba", {"n": 50, "k": 3}), ("ba", {"n": 60, "k": 3})],
        "escalabilidad": {
            "tamano": [("ba", {"n": n, "k": 3}) for n in (40, 80)],
            "densidad": [("aleatorio", {"n": 40, "m": m}) for m in (80, 160)],
            "etiquetas": [("ba", {"n": 60, "k": 3, "L": L, "E": E}) for L, E in ((1, 1), (13, 1))],
        },
        "dataset": "ninguno",
        "bloque": 32,
        "max_bloques": 4,
    },
}


def num(x, dec=0):
    return f"{x:,.{dec}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def veces(x):
    if x is None:
        return "-"
    return f"{num(x, 2 if x < 1 else 1 if x < 100 else 0)}x"


def mb(x):
    return f"{num(x, 2)} MB"


def est_txt(e):
    return sec.seg_est(e) if e else "-"


class Informe:
    def __init__(self):
        self.md = []
        self.datos = {}

    def titulo(self, texto):
        linea = "=" * 78
        print(f"\n{linea}\n {texto}\n{linea}")
        self.md.append(f"\n## {texto}\n")

    def subtitulo(self, texto):
        print(f"\n--- {texto} ---")
        self.md.append(f"\n### {texto}\n")

    def texto(self, t, consola=True):
        if consola:
            print(t)
        self.md.append(t + "\n")

    def tabla(self, encabezados, filas, consola=True):
        celdas = [[str(c) for c in f] for f in filas]
        anchos = [max(len(str(e)), *(len(f[i]) for f in celdas)) if celdas else len(str(e))
                  for i, e in enumerate(encabezados)]
        if consola:
            print("  ".join(str(e).ljust(a) if i == 0 else str(e).rjust(a)
                            for i, (e, a) in enumerate(zip(encabezados, anchos))))
            print("  ".join("-" * a for a in anchos))
            for f in celdas:
                print("  ".join(c.ljust(a) if i == 0 else c.rjust(a) for i, (c, a) in enumerate(zip(f, anchos))))
        self.md.append("| " + " | ".join(str(e) for e in encabezados) + " |")
        self.md.append("|" + "|".join(["---"] + ["---:"] * (len(encabezados) - 1)) + "|")
        for f in celdas:
            self.md.append("| " + " | ".join(f) + " |")
        self.md.append("")


def grafo_python(tipo, kw, semilla):
    rng = random.Random(semilla)
    L, E = kw.get("L", 3), kw.get("E", 2)
    en, ea = sec.alfabeto(L, True), sec.alfabeto(E, False)
    if tipo == "ejemplo":
        return sec.generar_ejemplo()
    if tipo == "triangulo":
        return sec.generar_triangulo()
    if tipo == "estrella":
        return sec.generar_estrella(kw["n"])
    if tipo == "completo":
        return sec.generar_completo(kw["n"], rng, en, ea)
    if tipo == "aleatorio":
        return sec.generar_aleatorio(kw["n"], kw["m"], rng, en, ea)
    if tipo == "etiquetas":
        return gen.grafo_etiquetas(L, E, kw["n"], kw["m"], semilla)
    return sec.generar_ba(kw["n"], kw["k"], rng, en, ea)


def grafo_csr(tipo, kw, semilla):
    if tipo.startswith("py:"):
        g = grafo_python(tipo[3:], kw, semilla)
        return gpu.grafo_a_csr(g), g
    return gen.csr_sintetico(tipo, kw.get("n"), kw.get("m"), kw.get("k", 3), kw.get("L", 3), kw.get("E", 2), semilla), None


def cronometrar(fn, *args):
    gc.collect()
    gc.disable()
    t0 = time.perf_counter()
    try:
        resultado = fn(*args)
    finally:
        t = time.perf_counter() - t0
        gc.enable()
    return resultado, t


def detectar_gpu(sin_gpu):
    if sin_gpu:
        return False, "desactivada con --sin-gpu"
    if gpu is None:
        return False, f"Numba / numba-cuda no disponible ({ERROR_NUMBA})"
    try:
        if not cuda.is_available():
            return False, "Numba no detecta una GPU CUDA (revisa el driver y la instalación de numba-cuda)"
        cuda.to_device(np.zeros(1))
        cuda.synchronize()
        return True, ""
    except Exception as e:
        return False, f"error al inicializar CUDA: {e}"


def describir_entorno(usar_gpu):
    d = {"python": platform.python_version(), "sistema": platform.platform(), "numpy": np.__version__}
    for modulo, clave in (("numba", "numba"), ("numba_cuda", "numba_cuda"), ("pandas", "pandas")):
        try:
            d[clave] = __import__(modulo).__version__
        except Exception:
            d[clave] = "no instalado"
    try:
        with open("/proc/cpuinfo") as f:
            d["cpu"] = next(l.split(":", 1)[1].strip() for l in f if l.startswith("model name"))
    except Exception:
        d["cpu"] = platform.processor() or "desconocida"
    d["hilos_numba"] = get_num_threads() if get_num_threads else None
    try:
        d["ram_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except Exception:
        d["ram_gb"] = None
    if usar_gpu:
        try:
            dev = cuda.get_current_device()
            d["gpu"] = dev.name.decode() if isinstance(dev.name, bytes) else str(dev.name)
            d["capacidad_computo"] = ".".join(str(x) for x in dev.compute_capability)
        except Exception:
            d["gpu"] = "desconocida"
        libre, total = gpu.memoria_gpu()
        if total:
            d["vram_total_gb"] = round(total / 2**30, 2)
            d["vram_libre_gb"] = round(libre / 2**30, 2)
        try:
            v = cuda.cudadrv.driver.driver.get_version()
            d["cuda_driver"] = f"{v[0]}.{v[1]}"
        except Exception:
            pass
    return d


def config_valida_para(cfg, prep):
    cfg = dict(cfg)
    if gpu.motivo_invalida(cfg["estrategia_caminos"], cfg["memoria_caminos"] or "hash", prep):
        cfg["estrategia_caminos"] = c3.CONFIG_POR_DEFECTO["estrategia_caminos"]
    if cfg["memoria_caminos"] and gpu.motivo_invalida(cfg["estrategia_caminos"], cfg["memoria_caminos"], prep):
        cfg["memoria_caminos"] = None
    if cfg["memoria_tri"] and tri.motivo_invalida(cfg["estrategia_tri"], cfg["memoria_tri"], prep):
        cfg["memoria_tri"] = None
    return c3.completar_memorias(cfg, prep)


def medir_gpu_completo(csr, pre, cfg, p, reps, preproceso=None):
    kw = dict(max_bloques=p.max_bloques, trabajo_max=p.trabajo_por_lanzamiento, preproceso=preproceso or p.preproceso)
    c3.censo3(csr, pre, cfg, **kw)
    gpu.liberar_gpu()
    corridas = []
    for _ in range(reps):
        corridas.append(c3.censo3(csr, None, cfg, **kw))
        gpu.liberar_gpu()
    return corridas[-1], c3.estadisticas_corridas(corridas), corridas


def medir_njit(csr, paralelo, estrategia, reps):
    corridas = [cpu.censo3_njit(csr, None, paralelo=paralelo, estrategia_caminos=estrategia) for _ in range(reps)]
    est = {k: sec.estadisticas(c["tiempos"][k] for c in corridas)
           for k in ("caminos_s", "triangulos_s", "post_s", "censo_s", "preproceso_s", "extremo_s")}
    return corridas[-1], est


def texto_config(cfg):
    cfg = dict(cfg, memoria_caminos=cfg["memoria_caminos"] or "auto", memoria_tri=cfg["memoria_tri"] or "auto")
    caminos, triangulos = c3.describir_config(cfg)
    return f"{caminos.split(' (')[0]} / {triangulos.split(' (')[0]}"


def exp_validacion(inf, escala, usar_gpu, p, estado):
    inf.titulo("1. Validación de correctitud")
    inf.texto("Referencia: censo en Python (oráculo), validado contra fuerza bruta sobre cada trío de nodos en los tests.\n"
              "Por grafo: exploración = centro; njit (enumeración y combinatoria, 1 hilo y multihilo) = referencia; "
              "todas las configuraciones GPU de caminos (estrategia × memoria) y de triángulos = referencia; "
              "censo completo GPU = referencia; IDs de cada camino emitidos por la GPU = conjunto de la CPU.")
    filas, datos = [], []
    for tipo, kw in escala["validacion"]:
        g = grafo_python(tipo, kw, p.semilla)
        ref = sec.censo_completo(g)
        W, T = sum(ref["caminos_no_inducidos"].values()), sum(ref["triangulos"].values())
        inv_ok = sum(ref["caminos_inducidos"].values()) == W - 3 * T
        expl_ok = sec.censo_exploracion(g)[0] == ref["caminos_no_inducidos"]
        csr = gpu.grafo_a_csr(g)
        pre = c3.preparar(csr)
        prep = pre["prep"]
        njit_ok = sum(c3.censos_iguales(cpu.censo3_njit(csr, pre, paralelo=par, estrategia_caminos=e), ref)
                      for par in (False, True) for e in cpu.ESTRATEGIAS_CAMINOS)
        fila = {"grafo": g.nombre, "W": W, "T": T, "L": prep["L"], "E": prep["E"], "P": prep["P"],
                "invariantes": inv_ok, "exploracion_igual_centro": expl_ok, "njit": f"{njit_ok}/4"}
        txt_c = txt_t = completo = ids = "-"
        if usar_gpu:
            ok = total = 0
            for e in gpu.ESTRATEGIAS:
                for m in gpu.MEMORIAS:
                    if gpu.config_valida(e, m, prep):
                        r = gpu.ejecutar_gpu(csr, prep, e, m, p.bloque, p.max_bloques, 64,
                                             trabajo_max=p.trabajo_por_lanzamiento)
                        gpu.liberar_gpu()
                        total += 1
                        ok += r["censo"] == ref["caminos_no_inducidos"]
            txt_c = f"{ok}/{total}"
            ok = total = 0
            for e in tri.ESTRATEGIAS:
                for m in tri.MEMORIAS:
                    if tri.config_valida(e, m, prep):
                        r = tri.ejecutar_triangulos_gpu(csr, prep, pre["orient"], e, m, p.bloque, p.max_bloques,
                                                        tri.CHUNK_TRI, p.trabajo_por_lanzamiento)
                        gpu.liberar_gpu()
                        total += 1
                        ok += r["triangulos"] == ref["triangulos"]
            txt_t = f"{ok}/{total}"
            cfg = config_valida_para(estado["config_base"], prep)
            ok_modos = [c3.censos_iguales(c3.censo3(csr, pre, cfg, p.max_bloques, p.trabajo_por_lanzamiento,
                                                    preproceso=modo), ref) for modo in c3.PREPROCESOS]
            completo = f"{sum(ok_modos)}/{len(ok_modos)}"
            v = gpu.verificar_ids(csr, prep, g, p.bloque, p.max_bloques, 64, p.trabajo_por_lanzamiento)
            ids = f"{v['estado']} ({v['duplicados_gpu']} dup.)" if "duplicados_gpu" in v else v["estado"]
        fila.update({"gpu_caminos": txt_c, "gpu_triangulos": txt_t, "gpu_censo3": completo, "ids": ids})
        datos.append(fila)
        filas.append([g.nombre, num(W), num(T), num(W - 3 * T), f"{prep['L']}×{prep['E']}", num(prep["P"]),
                      "sí" if inv_ok and expl_ok else "NO", fila["njit"], txt_c, txt_t, completo, ids])
    inf.tabla(["Grafo", "W", "T", "Inducidos", "Etiq. L×E", "P caminos", "Invariantes", "njit", "GPU caminos",
               "GPU triángulos", "GPU censo3 (prep. GPU/CPU)", "IDs GPU = CPU"], filas)
    inf.datos["validacion"] = datos


def exp_estrategias(inf, escala, p):
    inf.titulo("2. Estrategias secuenciales en Python: exploración (hash de IDs) vs centro (regla de pertenencia)")
    inf.texto(f"Tiempo = media ± desviación de {p.reps} repeticiones. Memoria pico medida aparte con tracemalloc.")
    filas, datos = [], []
    for tipo, kw in escala["estrategias"]:
        g = grafo_python(tipo, kw, p.semilla)
        W = sum(len(a) * (len(a) - 1) // 2 for a in g.adyacencia.values())
        tiempos = {}
        censos = {}
        info = None
        for metodo in ("exploracion", "centro"):
            ts = []
            for _ in range(p.reps):
                (censo, inf_m), t = cronometrar(sec.METODOS[metodo], g)
                ts.append(t)
            tiempos[metodo] = sec.estadisticas(ts)
            censos[metodo] = censo
            info = inf_m if metodo == "exploracion" else info
        memoria = {}
        for metodo in ("exploracion", "centro"):
            gc.collect()
            tracemalloc.start()
            sec.METODOS[metodo](g)
            memoria[metodo] = tracemalloc.get_traced_memory()[1] / 2**20
            tracemalloc.stop()
        acel = tiempos["exploracion"]["media"] / tiempos["centro"]["media"] if tiempos["centro"]["media"] else None
        pct = 100 * info["descartados"] / info["generados"] if info["generados"] else 0
        filas.append([g.nombre, num(W), est_txt(tiempos["exploracion"]), est_txt(tiempos["centro"]), veces(acel),
                      mb(memoria["exploracion"]), mb(memoria["centro"]), f"{num(info['descartados'])} ({num(pct, 1)}%)",
                      "sí" if censos["exploracion"] == censos["centro"] else "NO"])
        datos.append({"grafo": g.nombre, "W": W, "exploracion": sec.aplanar("t", tiempos["exploracion"]),
                      "centro": sec.aplanar("t", tiempos["centro"]), "memoria_mb": memoria,
                      "generados": info["generados"], "descartados": info["descartados"],
                      "iguales": censos["exploracion"] == censos["centro"]})
        print(f"  listo: {g.nombre}")
    inf.tabla(["Grafo", "Caminos", "Exploración", "Centro", "Aceleración", "Mem. exploración", "Mem. centro",
               "Descartados (expl.)", "Mismo censo"], filas)
    inf.datos["estrategias"] = datos


def tabla_barrido(inf, filas, parte, mejor_fila):
    filas = [f for f in filas if f["parte"] == parte]
    datos = []
    for f in filas:
        marca = "**" if f is mejor_fila else ""
        if f.get("omitida") or f.get("error"):
            datos.append([f["estrategia"], f["memoria"], f["bloque"], "-", "-", "-",
                          ("omitida: " + f["omitida"]) if f.get("omitida") else ("ERROR: " + f["error"])])
            continue
        datos.append([f"{marca}{f['estrategia']}{marca}", f["memoria"], f["bloque"],
                      f"{sec.seg(f['t_kernel_s'])} ± {sec.seg(f['t_kernel_desv_s'])}",
                      f"{sec.seg(f['t_total_s'])} ± {sec.seg(f['t_total_desv_s'])}",
                      num(f["desbalance"], 2) if f.get("desbalance") is not None else "-",
                      "sí" if f["correcto"] else "NO"])
    inf.tabla(["Estrategia", "Memoria", "Bloque", "Cálculo", "Total", "Desbal.", "OK"], datos, consola=False)


def exp_barrido(inf, escala, p, estado, carpeta):
    inf.titulo("3. Barrido de configuraciones GPU: estrategia × memoria × bloque")
    inf.texto(f"Cada configuración: 1 ejecución de calentamiento (incluye la compilación JIT) y {p.reps} medidas; "
              f"media ± desviación. Bloques probados: {escala['bloques_barrido']}. Cada resultado se compara con la "
              "CPU compilada (njit). La mejor configuración (menor tiempo total medio) se usa en los experimentos 4 y 5.\n"
              "Las tablas completas quedan en resumen.md; en la consola se muestra el avance y la mejor configuración.")
    datos = []
    for tipo, kw in escala["barrido"]:
        csr, _ = grafo_csr(tipo, kw, p.semilla)
        pre = c3.preparar(csr)
        prep = pre["prep"]
        inf.subtitulo(f"{csr.nombre}: {num(csr.n)} nodos, {num(csr.nnz // 2)} aristas, W = {num(prep['W'])}, "
                      f"grado máx. {num(prep['grado_max'])}")
        try:
            ref = cpu.censo3_njit(csr, pre, paralelo=True)
        except ValueError:
            ref = None
        r = bar.barrer(csr, pre, bloques=escala["bloques_barrido"], max_bloques=p.max_bloques, reps=p.reps,
                       trabajo_max=p.trabajo_por_lanzamiento, referencia=ref, progreso=print, preproceso=p.preproceso)
        tabla_barrido(inf, r["filas"], "caminos", r["mejor_caminos"])
        tabla_barrido(inf, r["filas"], "triangulos", r["mejor_triangulos"])
        cfg = c3.completar_memorias(r["config"], prep)
        estado["mejores"][csr.nombre] = cfg
        if estado["config_escalabilidad"] is None and not p.config:
            estado["config_escalabilidad"] = cfg
        ruta = os.path.join(carpeta, f"mejor_config_{csr.nombre}.json")
        sec.guardar_json({"config": cfg, "grafo": csr.nombre, "criterio": "total"}, ruta)
        malas = [f for f in r["filas"] if f.get("correcto") is False]
        comp = r["completo"]
        inf.texto(f"Mejor: {texto_config(cfg)} (guardada en {os.path.basename(ruta)}) | configuraciones incorrectas: "
                  f"{len(malas)}" + (f" | censo completo con la mejor: {est_txt(comp['est']['censo'])}, "
                                    f"igual a njit: {comp['igual_referencia']}" if comp else ""))
        datos.append({"grafo": csr.nombre, "nodos": csr.n, "aristas": csr.nnz // 2, "W": prep["W"],
                      "filas": r["filas"], "config": cfg,
                      "completo": {k: v for k, v in comp.items() if k != "est"} if comp else None})
        del csr, pre
        gpu.liberar_gpu()
    inf.datos["barrido"] = datos


def comparar_cpu_gpu(inf, csr, g, p, estado, usar_gpu, reps_1hilo):
    pre = c3.preparar(csr)
    prep = pre["prep"]
    filas = {}
    ref = None
    for paralelo in (False, True):
        for estrategia in cpu.ESTRATEGIAS_CAMINOS:
            r, est = medir_njit(csr, paralelo, estrategia, p.reps if paralelo else reps_1hilo)
            ref = ref or r
            hilos = get_num_threads() if paralelo else 1
            filas[f"njit {hilos} hilo{'s' if hilos > 1 else ''} · {estrategia}"] = {
                "tipo": "njit", "paralelo": paralelo, "est": est, "igual": c3.censos_iguales(r, ref)}
            filas[next(reversed(filas))]["est"]["preparacion_s"] = est["preproceso_s"]
    if prep["W"] <= LIMITE_PYTHON_DEMO:
        g = g if g is not None else gpu.csr_a_grafo(csr)
        ref_py, t = cronometrar(sec.censo_completo, g)
        e = sec.estadisticas([t])
        filas["Python (referencia)"] = {"tipo": "python", "est": {"censo_s": e, "extremo_s": e},
                                        "igual": c3.censos_iguales(ref_py, ref)}
    cfg = None
    if usar_gpu:
        cfg = config_valida_para(estado["mejores"].get(csr.nombre) or estado["config_base"], prep)
        for modo in [p.preproceso] + [m for m in c3.PREPROCESOS if m != p.preproceso]:
            r, est, corridas = medir_gpu_completo(csr, pre, cfg, p, p.reps, modo)
            principal = modo == p.preproceso
            filas[f"GPU · {texto_config(cfg)} · prep. {modo}"] = {
                "tipo": "gpu" if principal else "gpu_alt", "preproceso": modo,
                "est": {"caminos_s": est["caminos_total"], "triangulos_s": est["triangulos_total"],
                        "censo_s": est["censo"], "extremo_s": est["extremo"], "calculo_s": est["calculo"],
                        "preparacion_s": est["preparacion"]},
                "igual": all(c3.censos_iguales(c, ref) for c in corridas)}
    njit_1 = min(f["est"]["censo_s"]["media"] for f in filas.values() if f["tipo"] == "njit" and not f["paralelo"])
    njit_n = min(f["est"]["censo_s"]["media"] for f in filas.values() if f["tipo"] == "njit" and f["paralelo"])
    tabla = []
    for nombre, f in filas.items():
        e = f["est"]
        t = e["censo_s"]["media"]
        tabla.append([nombre, est_txt(e.get("caminos_s")), est_txt(e.get("triangulos_s")), est_txt(e["censo_s"]),
                      est_txt(e.get("calculo_s")), est_txt(e.get("preparacion_s", e.get("preproceso_s"))),
                      est_txt(e["extremo_s"]), veces(njit_1 / t), veces(njit_n / t), "sí" if f["igual"] else "NO"])
    inf.tabla(["Método", "Caminos", "Triángulos", "Censo", "Solo cálculo", "Preparación", "Extremo a extremo",
               "vs njit 1 hilo", "vs njit N hilos", "= njit"], tabla)
    return filas, ref, cfg, njit_1, njit_n


def exp_cpu_gpu(inf, escala, usar_gpu, p, estado):
    inf.titulo("4. Censo completo: CPU (Python, njit 1 hilo, njit multihilo) vs GPU")
    inf.texto("Censo = caminos + triángulos + resta e inducido, con el grafo ya preparado. Preparación = en CPU, suma "
              "prefija y orientación (njit); en GPU, suma prefija en CPU + copia del grafo + orientación y fila, en GPU "
              "o en CPU según el preproceso. Extremo a extremo = preparación + censo; todo se mide en cada repetición. "
              f"Media ± desviación de {p.reps} repeticiones (njit de 1 hilo: 1 repetición en grafos grandes). La GPU usa "
              "la mejor configuración del barrido para ese grafo, si existe; si no, la configuración base, y se mide "
              "con los dos preprocesos. Aceleraciones sobre la media del censo contra el njit más rápido de cada tipo.")
    datos = []
    for tipo, kw in escala["cpu_gpu"]:
        csr, g = grafo_csr(tipo, kw, p.semilla)
        prep = gpu.preparar(csr)
        inf.subtitulo(f"{csr.nombre}: {num(csr.n)} nodos, {num(csr.nnz // 2)} aristas, W = {num(prep['W'])}")
        reps_1 = p.reps if prep["W"] <= 50_000_000 else 1
        filas, ref, cfg, njit_1, njit_n = comparar_cpu_gpu(inf, csr, g, p, estado, usar_gpu, reps_1)
        gpu_f = next((f for f in filas.values() if f["tipo"] == "gpu"), None)
        fila = {"grafo": csr.nombre, "nodos": csr.n, "aristas": csr.nnz // 2, "W": ref["W"], "T": ref["T"],
                "njit_1_s": njit_1, "njit_n_s": njit_n, "config": cfg,
                "metodos": {k: {c: sec.aplanar("t", e) for c, e in f["est"].items()} for k, f in filas.items()},
                "todos_iguales": all(f["igual"] for f in filas.values())}
        if gpu_f:
            fila.update({"gpu_censo_s": gpu_f["est"]["censo_s"]["media"], "gpu_extremo_s": gpu_f["est"]["extremo_s"]["media"],
                         "gpu_calculo_s": gpu_f["est"]["calculo_s"]["media"]})
        alt = next((f for f in filas.values() if f["tipo"] == "gpu_alt"), None)
        if alt:
            fila["gpu_extremo_otro_preproceso_s"] = alt["est"]["extremo_s"]["media"]
        estado["resumen"].append(fila)
        datos.append(fila)
        del csr, g
        if usar_gpu:
            gpu.liberar_gpu()
    inf.datos["cpu_gpu"] = datos


def exp_escalabilidad(inf, escala, usar_gpu, p, estado):
    inf.titulo("5. Escalabilidad: tamaño, densidad y número de etiquetas")
    base = estado["config_escalabilidad"] or estado["config_base"]
    inf.texto(f"CPU = njit multihilo ({get_num_threads()} hilos), la más rápida entre enumeración y combinatoria. GPU = "
              f"configuración fija {texto_config(base)} ('auto' o memoria no válida para un punto: se usa la primera "
              "válida entre shared, global y hash; si la estrategia no es válida, la estrategia por defecto). "
              f"Media de {p.reps} repeticiones; tiempos de censo (sin la preparación).")
    series = {}
    for serie, puntos in escala["escalabilidad"].items():
        inf.subtitulo({"tamano": "Tamaño (Barabási–Albert, k fijo)", "densidad": "Densidad (Erdős–Rényi, n fijo)",
                       "etiquetas": "Número de etiquetas (Barabási–Albert fijo)"}[serie])
        filas, datos = [], []
        for tipo, kw in puntos:
            csr, _ = grafo_csr(tipo, kw, p.semilla)
            pre = c3.preparar(csr)
            prep = pre["prep"]
            mejor_cpu, ref = None, None
            for estrategia in cpu.ESTRATEGIAS_CAMINOS:
                try:
                    r, est = medir_njit(csr, True, estrategia, p.reps)
                except ValueError:
                    continue
                ref = ref or r
                if mejor_cpu is None or est["censo_s"]["media"] < mejor_cpu[1]["censo_s"]["media"]:
                    mejor_cpu = (estrategia, est)
            fila = {"tipo": tipo, **kw, "nodos": csr.n, "aristas": csr.nnz // 2, "W": prep["W"],
                    "L": prep["L"], "E": prep["E"], "P": prep["P"], "PT": tri.espacio_triangulos(prep),
                    "T": ref["T"] if ref else None,
                    "cpu_estrategia": mejor_cpu[0] if mejor_cpu else None,
                    "cpu_censo_s": mejor_cpu[1]["censo_s"]["media"] if mejor_cpu else None}
            txt_gpu = txt_calc = txt_cfg = acel = "-"
            if usar_gpu:
                cfg = config_valida_para(base, prep)
                r, est, corridas = medir_gpu_completo(csr, pre, cfg, p, p.reps)
                fila.update({"config": cfg, "gpu_censo_s": est["censo"]["media"], "gpu_censo_desv_s": est["censo"]["desv"],
                             "gpu_calculo_s": est["calculo"]["media"], "gpu_extremo_s": est["extremo"]["media"],
                             "gpu_igual_njit": all(c3.censos_iguales(c, ref) for c in corridas) if ref else None})
                txt_gpu, txt_calc, txt_cfg = est_txt(est["censo"]), est_txt(est["calculo"]), texto_config(cfg)
                if fila["cpu_censo_s"]:
                    acel = veces(fila["cpu_censo_s"] / est["censo"]["media"])
            datos.append(fila)
            filas.append([csr.nombre, num(csr.nnz // 2), num(prep["W"]), num(fila["T"]) if fila["T"] is not None else "-",
                          f"{prep['L']}×{prep['E']}", f"{num(prep['P'])} / {num(fila['PT'])}",
                          est_txt(mejor_cpu[1]["censo_s"]) if mejor_cpu else "-", mejor_cpu[0] if mejor_cpu else "-",
                          txt_gpu, txt_calc, txt_cfg, acel,
                          {True: "sí", False: "NO", None: "-"}[fila.get("gpu_igual_njit")]])
            print(f"  listo: {csr.nombre}")
            del csr, pre
            if usar_gpu:
                gpu.liberar_gpu()
        inf.tabla(["Grafo", "Aristas", "W", "T", "Etiq.", "P / PT", "CPU njit", "Estrategia CPU", "GPU censo",
                   "GPU cálculo", "Config. GPU", "Aceleración", "= njit"], filas)
        series[serie] = datos
    inf.datos["escalabilidad"] = series
    estado["escalabilidad"] = series


def exp_real(inf, clave, usar_gpu, p, estado, carpeta):
    import datos_reales as dr

    info = dr.DATASETS[clave]
    inf.titulo(f"6. Grafo real masivo: {info['nombre']} (SNAP)")
    inf.texto(f"Fuente: {info['pagina']}\nLos grafos de SNAP no traen etiquetas: se asignan {p.etiquetas_nodo} etiquetas "
              f"de nodo y {p.etiquetas_arista} de arista al azar (uniformes, semilla {p.semilla}).")
    csr, tiempos, meta = dr.cargar_dataset(clave, p.etiquetas_nodo, p.etiquetas_arista, p.semilla, p.datos)
    for k, v in tiempos.items():
        inf.texto(f"- {k.replace('_s', '').replace('_', ' ')}: {sec.seg(v)}")
    pre = c3.preparar(csr)
    prep = pre["prep"]
    m = csr.nnz // 2
    coincide = ""
    if info["nodos"]:
        coincide = " (coincide con SNAP)" if (csr.n, m) == (info["nodos"], info["aristas"]) else \
            f" (SNAP reporta {num(info['nodos'])} nodos y {num(info['aristas'])} aristas)"
    inf.texto(f"Nodos: {num(csr.n)} | aristas: {num(m)}{coincide}\nGrado máximo: {num(prep['grado_max'])} | "
              f"W = {num(prep['W'])} | P caminos: {num(prep['P'])} | P triángulos: {num(tri.espacio_triangulos(prep))} | "
              f"preproceso: suma prefija {sec.seg(pre['t_prefijo_s'])}, orientación {sec.seg(pre['t_orientacion_s'])}")
    if usar_gpu:
        libre, total = gpu.memoria_gpu()
        if libre:
            inf.texto(f"Memoria GPU libre: {num(libre / 2**30, 2)} GB de {num(total / 2**30, 2)} GB")
    datos = {"dataset": clave, "nombre": info["nombre"], "nodos": csr.n, "aristas": m, "grado_max": prep["grado_max"],
             "W": prep["W"], "tiempos_carga_s": tiempos, "cache": meta.get("cache")}
    if usar_gpu and not p.sin_barrido_real:
        inf.subtitulo("Barrido de configuraciones en el grafo real")
        ref = cpu.censo3_njit(csr, pre, paralelo=True)
        if info.get("triangulos"):
            inf.texto(f"T (njit) = {num(ref['T'])} | SNAP publica {num(info['triangulos'])}: "
                      f"{'coincide' if ref['T'] == info['triangulos'] else 'NO coincide'}")
            datos["T_igual_SNAP"] = ref["T"] == info["triangulos"]
        r = bar.barrer(csr, pre, bloques=[256], max_bloques=p.max_bloques, reps=min(p.reps, 2),
                       trabajo_max=p.trabajo_por_lanzamiento, referencia=ref, completo=False, progreso=print,
                       preproceso=p.preproceso)
        tabla_barrido(inf, r["filas"], "caminos", r["mejor_caminos"])
        tabla_barrido(inf, r["filas"], "triangulos", r["mejor_triangulos"])
        cfg = c3.completar_memorias(r["config"], prep)
        estado["mejores"][csr.nombre] = cfg
        sec.guardar_json({"config": cfg, "grafo": csr.nombre, "criterio": "total"},
                         os.path.join(carpeta, f"mejor_config_{clave}.json"))
        inf.texto(f"Mejor configuración: {texto_config(cfg)}")
        datos["barrido"] = r["filas"]
        gpu.liberar_gpu()
    inf.subtitulo("Censo completo: CPU vs GPU")
    filas, ref, cfg, njit_1, njit_n = comparar_cpu_gpu(inf, csr, None, p, estado, usar_gpu, 1)
    if info.get("triangulos") and "T_igual_SNAP" not in datos:
        datos["T_igual_SNAP"] = ref["T"] == info["triangulos"]
        inf.texto(f"T = {num(ref['T'])} | SNAP publica {num(info['triangulos'])}: "
                  f"{'coincide' if datos['T_igual_SNAP'] else 'NO coincide'}")
    gpu_f = next((f for f in filas.values() if f["tipo"] == "gpu"), None)
    fila = {"grafo": info["nombre"], "nodos": csr.n, "aristas": m, "W": ref["W"], "T": ref["T"],
            "njit_1_s": njit_1, "njit_n_s": njit_n, "config": cfg,
            "metodos": {k: {c: sec.aplanar("t", e) for c, e in f["est"].items()} for k, f in filas.items()},
            "todos_iguales": all(f["igual"] for f in filas.values())}
    if gpu_f:
        fila.update({"gpu_censo_s": gpu_f["est"]["censo_s"]["media"], "gpu_extremo_s": gpu_f["est"]["extremo_s"]["media"],
                     "gpu_calculo_s": gpu_f["est"]["calculo_s"]["media"]})
    alt = next((f for f in filas.values() if f["tipo"] == "gpu_alt"), None)
    if alt:
        fila["gpu_extremo_otro_preproceso_s"] = alt["est"]["extremo_s"]["media"]
    estado["resumen"].append(fila)
    W, T = ref["W"], ref["T"]
    for titulo, censo, formato, total in (("caminos no inducidos", ref["caminos_no_inducidos"], sec.formato_patron, W),
                                          ("caminos inducidos", ref["caminos_inducidos"], sec.formato_patron, W - 3 * T),
                                          ("triángulos", ref["triangulos"], sec.formato_triangulo, T)):
        top = sorted(censo.items(), key=lambda x: -x[1])[:5]
        inf.texto(f"\nTop 5 {titulo} ({len(censo)} patrones distintos):")
        inf.tabla(["Patrón", "Frecuencia", "%"],
                  [[formato(k), num(v), f"{num(100 * v / total, 2) if total else 0}%"] for k, v in top])
    datos.update(fila)
    inf.datos["grafo_real"] = datos


def _estilo_ejes(ax):
    ax.set_facecolor(SUPERFICIE)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(GRILLA)
    ax.tick_params(colors=TINTA_2, labelsize=8)
    ax.yaxis.label.set_color(TINTA_2)
    ax.xaxis.label.set_color(TINTA_2)
    ax.grid(True, which="major", color=GRILLA, linewidth=0.6)
    ax.set_axisbelow(True)


def graficar_cpu_gpu(resumen, ruta):
    filas = [r for r in resumen if r.get("gpu_censo_s") and r.get("njit_n_s")]
    if not filas:
        return False
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    x = np.arange(len(filas))
    ancho = 0.36
    fig, ax = plt.subplots(figsize=(max(7, 1.9 * len(filas)), 5), facecolor=SUPERFICIE)
    _estilo_ejes(ax)
    ax.bar(x - ancho / 2 - 0.01, [r["njit_n_s"] for r in filas], ancho, color=COLOR_NJIT, label="CPU njit multihilo (mejor)")
    ax.bar(x + ancho / 2 + 0.01, [r["gpu_censo_s"] for r in filas], ancho, color=COLOR_GPU, label="GPU (mejor configuración)")
    for i, r in enumerate(filas):
        ax.annotate(veces(r["njit_n_s"] / r["gpu_censo_s"]), (x[i] + ancho / 2, r["gpu_censo_s"]),
                    textcoords="offset points", xytext=(0, 4), ha="center", fontsize=9, color=TINTA)
    ax.set_yscale("log")
    ax.set_ylabel("Tiempo del censo completo (s, escala log)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['grafo']}\nW = {num(r['W'])}\nT = {num(r['T'])}" for r in filas], fontsize=7, color=TINTA_2)
    ax.legend(loc="best", fontsize=8, frameon=False)
    ax.set_title("Censo completo de 3 nodos (inducido y no inducido): CPU compilada vs GPU", fontsize=10, color=TINTA)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150, facecolor=SUPERFICIE)
    plt.close(fig)
    return True


def graficar_escalabilidad(series, carpeta):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    generados = []
    ejes_x = {"tamano": ("aristas", "Aristas (escala log)", True), "densidad": ("aristas", "Aristas (escala log)", True),
              "etiquetas": ("P", "Patrones de camino posibles P = L·(E·L)² (escala log)", True)}
    for serie, datos in series.items():
        puntos = [d for d in datos if d.get("cpu_censo_s")]
        if len(puntos) < 2:
            continue
        clave, etiqueta_x, log_x = ejes_x[serie]
        fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=SUPERFICIE)
        _estilo_ejes(ax)
        xs = [d[clave] for d in puntos]
        ax.plot(xs, [d["cpu_censo_s"] for d in puntos], color=COLOR_NJIT, linewidth=2, marker="o", markersize=6,
                label="CPU njit multihilo")
        con_gpu = [d for d in puntos if d.get("gpu_censo_s")]
        if con_gpu:
            ax.plot([d[clave] for d in con_gpu], [d["gpu_censo_s"] for d in con_gpu], color=COLOR_GPU, linewidth=2,
                    marker="o", markersize=6, label="GPU")
            if serie == "etiquetas":
                for d in con_gpu:
                    ax.annotate(d["config"]["memoria_caminos"], (d[clave], d["gpu_censo_s"]), textcoords="offset points",
                                xytext=(0, -12), ha="center", fontsize=7, color=TINTA_2)
        ultimo = puntos[-1]
        ax.annotate("CPU", (ultimo[clave], ultimo["cpu_censo_s"]), textcoords="offset points", xytext=(6, 0),
                    fontsize=8, color=TINTA, va="center")
        if con_gpu:
            ax.annotate("GPU", (con_gpu[-1][clave], con_gpu[-1]["gpu_censo_s"]), textcoords="offset points",
                        xytext=(6, 0), fontsize=8, color=TINTA, va="center")
        if log_x:
            ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(etiqueta_x)
        ax.set_ylabel("Tiempo del censo (s, escala log)")
        ax.legend(loc="best", fontsize=8, frameon=False)
        ax.set_title({"tamano": "Escalabilidad con el tamaño", "densidad": "Escalabilidad con la densidad",
                      "etiquetas": "Efecto del número de etiquetas (bajo GPU: memoria de caminos usada)"}[serie],
                     fontsize=10, color=TINTA)
        fig.tight_layout()
        ruta = os.path.join(carpeta, f"escalabilidad_{serie}.png")
        fig.savefig(ruta, dpi=150, facecolor=SUPERFICIE)
        plt.close(fig)
        generados.append(os.path.basename(ruta))
    return generados


def ejecutar(nombre, fn, inf, errores, *args):
    try:
        fn(inf, *args)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        errores.append({"experimento": nombre, "error": f"{type(e).__name__}: {e}"})
        inf.texto(f"\n[ERROR] {nombre}: {type(e).__name__}: {e}\n(se continúa con el siguiente experimento)")
        traceback.print_exc()


def crear_parser():
    p = argparse.ArgumentParser(description="Demo y experimentos: censo de 3 nodos (caminos inducidos y no inducidos, "
                                            "triángulos) en CPU y GPU")
    p.add_argument("--escala", choices=list(ESCALAS), default="completa",
                   help="completa (incluye LiveJournal), rapida (minutos, sin grafo real) o prueba (segundos; sirve con "
                        "el simulador)")
    p.add_argument("--rapido", action="store_true", help="Atajo de --escala rapida")
    p.add_argument("--solo", nargs="+", choices=EXPERIMENTOS, default=None, help="Corre solo estos experimentos")
    p.add_argument("--dataset", choices=["livejournal", "orkut", "twitter", "ninguno"], default=None,
                   help="Grafo real para el experimento 6 (por defecto livejournal en escala completa)")
    p.add_argument("--sin-barrido-real", action="store_true", help="No hace el barrido de configuraciones en el grafo real")
    p.add_argument("--config", help="JSON de configuración GPU (mejor_config.json); fija la configuración base y la de "
                                    "escalabilidad")
    p.add_argument("--preproceso", choices=["gpu", "cpu"], default="gpu",
                   help="Preproceso de la configuración principal (en CPU vs GPU también se mide el otro)")
    p.add_argument("--sin-gpu", action="store_true", help="Ejecuta solo la parte de CPU")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--etiquetas-nodo", type=int, default=5, help="Etiquetas de nodo para el grafo real")
    p.add_argument("--etiquetas-arista", type=int, default=2, help="Etiquetas de arista para el grafo real")
    p.add_argument("--semilla", type=int, default=42)
    p.add_argument("--salida", default="resultados")
    p.add_argument("--datos", default="datasets", help="Carpeta para descargas y caché de grafos reales")
    p.add_argument("--max-bloques", type=int, default=None)
    p.add_argument("--trabajo-por-lanzamiento", type=int, default=1_000_000_000)
    return p


def main():
    p = crear_parser().parse_args()
    if p.rapido:
        p.escala = "rapida"
    p.reps = max(1, p.reps)
    escala = ESCALAS[p.escala]
    p.bloque = escala.get("bloque", 256)
    p.max_bloques = p.max_bloques or escala.get("max_bloques", 4096)
    dataset = p.dataset or escala["dataset"]
    experimentos = p.solo or EXPERIMENTOS
    carpeta = os.path.join(p.salida, "demo_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(carpeta, exist_ok=True)
    inicio = time.perf_counter()

    inf = Informe()
    print("=" * 78)
    print(" DEMO — Censo de 3 nodos (caminos inducidos/no inducidos y triángulos), CPU vs GPU")
    print("=" * 78)
    inf.md.append("# Demo: censo de 3 nodos — caminos inducidos y no inducidos, triángulos (CPU vs GPU)\n")
    inf.md.append(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')} | escala: {p.escala} | repeticiones: {p.reps} | "
                  f"experimentos: {', '.join(experimentos)}\n")

    usar_gpu, motivo = detectar_gpu(p.sin_gpu)
    inf.titulo("0. Entorno")
    entorno = describir_entorno(usar_gpu)
    for k, v in entorno.items():
        inf.texto(f"- {k}: {v}")
    if not usar_gpu:
        inf.texto(f"GPU: no se usará ({motivo}). Se ejecutan solo las partes de CPU.")
    inf.datos["entorno"] = entorno
    inf.datos["opciones"] = {k: v for k, v in vars(p).items()}

    errores = []
    if c3 is None:
        inf.texto(f"Numba no está disponible ({ERROR_NUMBA}): solo se ejecuta el experimento 2.")
        ejecutar("estrategias", exp_estrategias, inf, errores, escala, p)
    else:
        base = c3.resolver_config(c3.cargar_config(p.config) if p.config else None)
        if "bloque" in escala and not p.config:
            base.update(bloque_caminos=escala["bloque"], bloque_tri=escala["bloque"])
        estado = {"config_base": base, "config_escalabilidad": base if p.config else None, "mejores": {},
                  "resumen": [], "escalabilidad": {}}
        gen.calentar()
        cpu.calentar()
        if "validacion" in experimentos:
            ejecutar("validacion", exp_validacion, inf, errores, escala, usar_gpu, p, estado)
        if "estrategias" in experimentos:
            ejecutar("estrategias", exp_estrategias, inf, errores, escala, p)
        if "barrido" in experimentos:
            if usar_gpu:
                ejecutar("barrido", exp_barrido, inf, errores, escala, p, estado, carpeta)
            else:
                inf.titulo("3. Barrido de configuraciones GPU")
                inf.texto("Omitido: requiere GPU.")
        if "cpu_gpu" in experimentos:
            ejecutar("cpu_gpu", exp_cpu_gpu, inf, errores, escala, usar_gpu, p, estado)
        if "escalabilidad" in experimentos:
            ejecutar("escalabilidad", exp_escalabilidad, inf, errores, escala, usar_gpu, p, estado)
        if "real" in experimentos and dataset != "ninguno":
            ejecutar("real", exp_real, inf, errores, dataset, usar_gpu, p, estado, carpeta)

        inf.titulo("7. Resumen")
        filas = [[r["grafo"], num(r["W"]), num(r["T"]), sec.seg(r["njit_1_s"]), sec.seg(r["njit_n_s"]),
                  sec.seg(r["gpu_censo_s"]) if r.get("gpu_censo_s") else "-",
                  sec.seg(r["gpu_extremo_s"]) if r.get("gpu_extremo_s") else "-",
                  sec.seg(r["gpu_extremo_otro_preproceso_s"]) if r.get("gpu_extremo_otro_preproceso_s") else "-",
                  veces(r["njit_n_s"] / r["gpu_censo_s"]) if r.get("gpu_censo_s") else "-",
                  texto_config(r["config"]) if r.get("config") else "-",
                  "sí" if r["todos_iguales"] else "NO"] for r in estado["resumen"]]
        if filas:
            inf.texto("Censo completo (media del censo, sin la preparación; aceleración contra el njit multihilo más rápido):")
            otro = "cpu" if p.preproceso == "gpu" else "gpu"
            inf.tabla(["Grafo", "W", "T", "njit 1 hilo", f"njit {get_num_threads()} hilos", "GPU censo",
                       f"GPU extremo (prep. {p.preproceso})", f"GPU extremo (prep. {otro})", "Aceleración",
                       "Config. GPU", "Todos iguales"], filas)
        inf.datos["resumen"] = estado["resumen"]
        inf.datos["mejores_configuraciones"] = estado["mejores"]
        figuras = []
        if graficar_cpu_gpu(estado["resumen"], os.path.join(carpeta, "cpu_vs_gpu.png")):
            figuras.append("cpu_vs_gpu.png")
        figuras += graficar_escalabilidad(estado["escalabilidad"], carpeta)
        for f in figuras:
            inf.md.append(f"\n![{f}]({f})\n")

    inf.datos["errores"] = errores
    duracion = time.perf_counter() - inicio
    inf.texto(f"\nDuración total de la demo: {sec.seg(duracion)}")
    if errores:
        inf.texto(f"Experimentos con error: {', '.join(e['experimento'] for e in errores)}")
    with open(os.path.join(carpeta, "resumen.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(inf.md))
    with open(os.path.join(carpeta, "resultados.json"), "w", encoding="utf-8") as f:
        json.dump(inf.datos, f, ensure_ascii=False, indent=2, default=str)
    print("\nArchivos generados:")
    for nombre in sorted(os.listdir(carpeta)):
        print(f"  {os.path.join(carpeta, nombre)}")


if __name__ == "__main__":
    main()
