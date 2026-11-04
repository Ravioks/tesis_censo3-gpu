import argparse
import csv
import json
import os
import re
import time
from datetime import datetime

import numpy as np

import censo_3n as c3
import censo_3n2a as sec
import censo_3n2a_gpu as gpu
import censo_3n3a_gpu as tri

REFERENCIAS = ["njit", "python", "ninguna"]
CRITERIOS = {"total": "t_total_s", "calculo": "t_kernel_s"}
CAMPOS_CSV = ["parte", "estrategia", "memoria", "bloque", "chunk", "correcto", "omitida", "error", "reps",
              "t_h2d_s", "t_h2d_desv_s", "t_h2d_min_s", "t_kernel_s", "t_kernel_desv_s", "t_kernel_min_s",
              "t_d2h_s", "t_d2h_desv_s", "t_d2h_min_s", "t_total_s", "t_total_desv_s", "t_total_min_s",
              "t_jit_s", "hilos", "lanzamientos", "cap_hash", "trabajo", "tasa_kernel", "desbalance"]


def motivo_vram(necesario):
    libre, _ = gpu.memoria_gpu()
    if libre is not None and necesario > 0.9 * libre:
        return f"requiere {necesario / 2**30:.2f} GB y hay {libre / 2**30:.2f} GB libres"
    return None


def medir(fn, reps):
    primera = fn()
    gpu.liberar_gpu()
    corridas = []
    for _ in range(reps):
        corridas.append(fn())
        gpu.liberar_gpu()
    est = {k: sec.estadisticas(c[f"t_{k}"] for c in corridas) for k in ("h2d", "kernel", "d2h")}
    est["total"] = sec.estadisticas(c["t_h2d"] + c["t_kernel"] + c["t_d2h"] for c in corridas)
    return primera, corridas, est


def fila_medida(base, primera, corridas, est, trabajo, correcto, desbalance):
    calculo = est["kernel"]["media"]
    ultima = corridas[-1]
    return {**base, "omitida": None, "error": None, "reps": len(corridas), "hilos": ultima["hilos"],
            "lanzamientos": ultima["lanzamientos"], "cap_hash": ultima["cap_hash"],
            "t_jit_s": max(0.0, primera["t_kernel"] - calculo),
            **sec.aplanar("t_h2d", est["h2d"]), **sec.aplanar("t_kernel", est["kernel"]),
            **sec.aplanar("t_d2h", est["d2h"]), **sec.aplanar("t_total", est["total"]),
            "trabajo": trabajo, "tasa_kernel": trabajo / calculo if calculo > 0 else None,
            "desbalance": desbalance, "correcto": bool(correcto)}


def _barrer(parte, combinaciones, validar, ejecutar, comparar, trabajo, desbalance, reps, progreso):
    filas = []
    ref = None
    for base in combinaciones:
        base = {"parte": parte, **base}
        motivo = validar(base)
        if motivo:
            filas.append({**base, "omitida": motivo, "error": None, "correcto": None})
            progreso(f"  {parte:<10} {base['estrategia']:<13} {base['memoria']:<7} bloque {base['bloque']:<4} omitida: {motivo}")
            continue
        try:
            primera, corridas, est = medir(lambda: ejecutar(base), reps)
        except Exception as e:
            gpu.liberar_gpu()
            filas.append({**base, "omitida": None, "error": f"{type(e).__name__}: {e}", "correcto": False})
            progreso(f"  {parte:<10} {base['estrategia']:<13} {base['memoria']:<7} bloque {base['bloque']:<4} ERROR: {e}")
            continue
        correcto, ref = comparar(corridas, ref)
        fila = fila_medida(base, primera, corridas, est, trabajo, correcto,
                           desbalance(base, corridas[-1]["hilos"]))
        filas.append(fila)
        progreso(f"  {parte:<10} {base['estrategia']:<13} {base['memoria']:<7} bloque {base['bloque']:<4} "
                 f"cálculo {sec.seg_est(est['kernel']):>22}  total {sec.seg_est(est['total']):>22}  "
                 f"{'OK' if correcto else 'NO COINCIDE'}")
    return filas


def barrer_caminos(csr, pre, estrategias, memorias, bloques, chunk, max_bloques, reps, trabajo_max, ref=None,
                   progreso=print, disp=None):
    prep = pre["prep"]
    W = prep["W"]
    combinaciones = [{"estrategia": e, "memoria": m, "bloque": b, "chunk": chunk if e == "balanceado" else None}
                     for e in estrategias for m in memorias for b in bloques]

    def validar(base):
        return (gpu.motivo_invalida(base["estrategia"], base["memoria"], prep)
                or motivo_vram(gpu.bytes_necesarios(csr, prep, base["estrategia"], base["memoria"], disp=disp)))

    def ejecutar(base):
        return gpu.ejecutar_gpu(csr, prep, base["estrategia"], base["memoria"], base["bloque"], max_bloques, chunk,
                                trabajo_max=trabajo_max, disp=disp)

    def comparar(corridas, ref_actual):
        objetivo = ref if ref is not None else (ref_actual if ref_actual is not None else corridas[-1]["censo"])
        ok = all(c["censo"] == objetivo and sum(c["censo"].values()) == W for c in corridas)
        return ok, objetivo

    def desbalance(base, hilos):
        return gpu.desbalance(csr, prep, base["estrategia"], hilos, chunk)

    return _barrer("caminos", combinaciones, validar, ejecutar, comparar, W, desbalance, reps, progreso)


def barrer_triangulos(csr, pre, estrategias, memorias, bloques, chunk_tri, max_bloques, reps, trabajo_max, ref=None,
                      progreso=print, orient=None, et_nodo=None):
    prep = pre["prep"]
    orient = orient if orient is not None else pre["orient"]
    combinaciones = [{"estrategia": e, "memoria": m, "bloque": b, "chunk": chunk_tri if e == "balanceado" else None}
                     for e in estrategias for m in memorias for b in bloques]

    def validar(base):
        return (tri.motivo_invalida(base["estrategia"], base["memoria"], prep)
                or motivo_vram(tri.bytes_necesarios(csr, prep, orient, base["estrategia"], base["memoria"],
                                                    et_nodo_en_gpu=et_nodo is not None)))

    def ejecutar(base):
        return tri.ejecutar_triangulos_gpu(csr, prep, orient, base["estrategia"], base["memoria"], base["bloque"],
                                           max_bloques, chunk_tri, trabajo_max, et_nodo=et_nodo)

    def comparar(corridas, ref_actual):
        objetivo = ref if ref is not None else (ref_actual if ref_actual is not None else corridas[-1]["triangulos"])
        T = sum(objetivo.values())
        ok = all(c["triangulos"] == objetivo and c["T"] == T for c in corridas)
        return ok, objetivo

    def desbalance(base, hilos):
        return None

    return _barrer("triangulos", combinaciones, validar, ejecutar, comparar, orient["aristas"], desbalance, reps,
                   progreso)


def mejor(filas, criterio):
    campo = CRITERIOS[criterio]
    validas = [f for f in filas if f.get("correcto")]
    return min(validas, key=lambda f: f[campo]) if validas else None


def config_desde_mejores(mejor_caminos, mejor_tri, chunk, chunk_tri):
    cfg = dict(c3.CONFIG_POR_DEFECTO)
    cfg["chunk"], cfg["chunk_tri"] = chunk, chunk_tri
    if mejor_caminos:
        cfg.update(estrategia_caminos=mejor_caminos["estrategia"], memoria_caminos=mejor_caminos["memoria"],
                   bloque_caminos=mejor_caminos["bloque"])
    if mejor_tri:
        cfg.update(estrategia_tri=mejor_tri["estrategia"], memoria_tri=mejor_tri["memoria"], bloque_tri=mejor_tri["bloque"])
    return cfg


def calcular_referencia(csr, g, pre, tipo):
    if tipo == "ninguna":
        return None, "ninguna (las configuraciones se comparan entre sí)", None
    if tipo == "python":
        if pre["prep"]["W"] > c3.LIMITE_PYTHON:
            raise ValueError(f"W = {pre['prep']['W']:,} supera el límite de la referencia Python ({c3.LIMITE_PYTHON:,})")
        g = g if g is not None else gpu.csr_a_grafo(csr)
        t0 = time.perf_counter()
        return sec.censo_completo(g), "Python (oráculo)", time.perf_counter() - t0
    import censo_3n_cpu_njit as cpu

    cpu.calentar()
    r = cpu.censo3_njit(csr, pre, paralelo=True)
    return r, "CPU compilada (njit, multihilo, enumeración)", r["tiempos"]["censo_s"]


def barrer(csr, pre, parte="ambas", estrategias_caminos=None, memorias_caminos=None, estrategias_tri=None,
           memorias_tri=None, bloques=(256,), chunk=64, chunk_tri=tri.CHUNK_TRI, max_bloques=4096, reps=5,
           trabajo_max=gpu.TRABAJO_POR_LANZAMIENTO, criterio="total", referencia=None, completo=True, progreso=print,
           preproceso="gpu"):
    estrategias_caminos = estrategias_caminos or gpu.ESTRATEGIAS
    memorias_caminos = memorias_caminos or list(gpu.MEMORIAS)
    estrategias_tri = estrategias_tri or tri.ESTRATEGIAS
    memorias_tri = memorias_tri or list(tri.MEMORIAS)
    dispositivo = c3.preparar_dispositivo(csr, pre, estrategias_caminos if parte != "triangulos" else [],
                                          estrategias_tri if parte != "caminos" else [], preproceso, max_bloques)
    preparacion = {"preproceso": preproceso, "t_prefijo_s": pre["t_prefijo_s"],
                   "t_orientacion_cpu_s": pre["t_orientacion_s"] if preproceso == "cpu" else 0.0,
                   **dispositivo["tiempos"]}
    filas = []
    mejor_c = mejor_t = None
    if parte in ("ambas", "caminos"):
        ref_c = referencia["caminos_no_inducidos"] if referencia else None
        f = barrer_caminos(csr, pre, estrategias_caminos, memorias_caminos, bloques, chunk, max_bloques, reps,
                           trabajo_max, ref_c, progreso, disp=dispositivo["disp"])
        filas += f
        mejor_c = mejor(f, criterio)
    if parte in ("ambas", "triangulos"):
        ref_t = referencia["triangulos"] if referencia else None
        f = barrer_triangulos(csr, pre, estrategias_tri, memorias_tri, bloques, chunk_tri, max_bloques, reps,
                              trabajo_max, ref_t, progreso, orient=dispositivo["orient"],
                              et_nodo=dispositivo["disp"]["et_nodo"])
        filas += f
        mejor_t = mejor(f, criterio)
    del dispositivo
    gpu.liberar_gpu()
    config = config_desde_mejores(mejor_c, mejor_t, chunk, chunk_tri)

    resumen_completo = None
    if completo and parte == "ambas" and mejor_c and mejor_t:
        kw = dict(max_bloques=max_bloques, trabajo_max=trabajo_max, preproceso=preproceso)
        c3.censo3(csr, None, config, **kw)
        corridas = [c3.censo3(csr, None, config, **kw) for _ in range(max(1, reps))]
        res = corridas[-1]
        est = c3.estadisticas_corridas(corridas)
        resumen_completo = {
            "config": res["config"],
            "tiempos": {k: sec.aplanar("t", v) for k, v in est.items()},
            "est": est,
            "W": res["W"],
            "T": res["T"],
            "invariantes": c3.invariantes(res),
            "repeticiones_identicas": all(c3.censos_iguales(c, res) for c in corridas),
            "igual_referencia": c3.censos_iguales(res, referencia) if referencia else None,
        }
    return {"filas": filas, "mejor_caminos": mejor_c, "mejor_triangulos": mejor_t, "config": config,
            "completo": resumen_completo, "preparacion": preparacion}


def imprimir_tabla(filas, parte, mejor_fila, criterio):
    filas = [f for f in filas if f["parte"] == parte]
    if not filas:
        return
    titulo = "CAMINOS NO INDUCIDOS" if parte == "caminos" else "TRIÁNGULOS"
    unidad = "caminos/s" if parte == "caminos" else "aristas orient./s"
    print(f"\n=== {titulo}: estrategia × memoria × bloque (media ± desviación; * = mejor por {criterio}) ===")
    print(f"  {'Estrategia':<13}{'Memoria':<8}{'Bloque':>7}{'Cálculo (kernel)':>26}{'Total (H2D+cálc.+D2H)':>26}"
          f"{'Tasa ' + unidad:>24}{'Desbal':>8}  OK")
    for f in filas:
        marca = "*" if mejor_fila is not None and f is mejor_fila else " "
        if f.get("omitida") or f.get("error"):
            print(f"{marca} {f['estrategia']:<13}{f['memoria']:<8}{f['bloque']:>7}  "
                  f"{'omitida: ' + f['omitida'] if f.get('omitida') else 'ERROR: ' + f['error']}")
            continue
        calc = f"{sec.seg(f['t_kernel_s'])} ± {sec.seg(f['t_kernel_desv_s'])}"
        total = f"{sec.seg(f['t_total_s'])} ± {sec.seg(f['t_total_desv_s'])}"
        tasa = f"{f['tasa_kernel']:,.3g}" if f["tasa_kernel"] else "-"
        des = f"{f['desbalance']:.2f}" if f.get("desbalance") is not None else "-"
        print(f"{marca} {f['estrategia']:<13}{f['memoria']:<8}{f['bloque']:>7}{calc:>26}{total:>26}{tasa:>24}{des:>8}"
              f"  {'sí' if f['correcto'] else 'NO'}")


def guardar(carpeta, datos, filas, config, criterio, grafo):
    os.makedirs(carpeta, exist_ok=True)
    with open(os.path.join(carpeta, "barrido.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS_CSV, extrasaction="ignore")
        w.writeheader()
        for fila in filas:
            w.writerow({k: fila.get(k) for k in CAMPOS_CSV})
    sec.guardar_json(datos, os.path.join(carpeta, "barrido.json"))
    sec.guardar_json({"config": config, "grafo": grafo, "criterio": criterio,
                      "fecha": datetime.now().strftime("%Y-%m-%d %H:%M")}, os.path.join(carpeta, "mejor_config.json"))
    return [os.path.join(carpeta, n) for n in ("barrido.csv", "barrido.json", "mejor_config.json")]


def crear_parser():
    p = argparse.ArgumentParser(description="Barrido de configuraciones GPU (estrategia × memoria × bloque) para caminos "
                                            "y triángulos, con validación de cada configuración")
    c3.agregar_args_grafo(p)
    b = p.add_argument_group("barrido")
    b.add_argument("--parte", choices=["ambas", "caminos", "triangulos"], default="ambas")
    b.add_argument("--estrategias-caminos", nargs="+", choices=gpu.ESTRATEGIAS, default=gpu.ESTRATEGIAS)
    b.add_argument("--memorias-caminos", nargs="+", choices=list(gpu.MEMORIAS), default=list(gpu.MEMORIAS))
    b.add_argument("--estrategias-tri", nargs="+", choices=tri.ESTRATEGIAS, default=tri.ESTRATEGIAS)
    b.add_argument("--memorias-tri", nargs="+", choices=list(tri.MEMORIAS), default=list(tri.MEMORIAS))
    b.add_argument("--bloques", nargs="+", type=int, default=[256], help="Hilos por bloque a probar (ej.: 128 256 512)")
    b.add_argument("--chunk", type=int, default=64, help="Caminos por ítem (estrategia balanceado de caminos)")
    b.add_argument("--chunk-tri", type=int, default=tri.CHUNK_TRI, help="Costo por ítem (balanceado de triángulos)")
    b.add_argument("--preproceso", choices=c3.PREPROCESOS, default="gpu",
                   help="Dónde se construyen la orientación y el arreglo fila (se hace una vez para todo el barrido)")
    b.add_argument("--max-bloques", type=int, default=4096)
    b.add_argument("--trabajo-por-lanzamiento", type=int, default=gpu.TRABAJO_POR_LANZAMIENTO)
    b.add_argument("--reps", type=int, default=5, help="Repeticiones medidas por configuración, tras 1 de calentamiento")
    b.add_argument("--criterio", choices=list(CRITERIOS), default="total",
                   help="Qué tiempo medio decide la mejor configuración")
    b.add_argument("--referencia", choices=REFERENCIAS, default="njit",
                   help="Contra qué se valida cada configuración (njit por defecto; python solo en grafos chicos)")
    b.add_argument("--sin-completo", action="store_true", help="No corre el censo completo con la mejor configuración")
    b.add_argument("--salida", default="resultados")
    return p


def main():
    a = crear_parser().parse_args()
    a.reps = max(1, a.reps)
    if not gpu.cuda.is_available():
        raise SystemExit("No se detectó GPU CUDA. Para depurar sin GPU: NUMBA_ENABLE_CUDASIM=1 con grafos muy pequeños.")
    csr, g, info, t_carga = c3.cargar_grafo(a)
    tri.calentar()
    pre = c3.preparar(csr, orientar=a.preproceso == "cpu")
    prep = pre["prep"]
    print("\n=== GRAFO ===")
    print(f"{csr.nombre} | nodos {csr.n:,} | aristas {csr.nnz // 2:,} | grado máx. {prep['grado_max']:,} | "
          f"W = {prep['W']:,} | carga {sec.seg(t_carga)}")
    print(f"Etiquetas nodo {prep['L']} | arista {prep['E']} | patrones posibles: caminos {prep['P']:,}, "
          f"triángulos {tri.espacio_triangulos(prep):,}")
    gpu.cuda.to_device(np.zeros(1))
    gpu.cuda.synchronize()

    try:
        referencia, desc_ref, t_ref = calcular_referencia(csr, g, pre, a.referencia)
    except ValueError as e:
        print(f"Referencia '{a.referencia}' no disponible ({e}); se comparan las configuraciones entre sí.")
        referencia, desc_ref, t_ref = calcular_referencia(csr, g, pre, "ninguna")
    print(f"Referencia de correctitud: {desc_ref}" + (f" ({sec.seg(t_ref)})" if t_ref is not None else ""))
    if referencia and info and info.get("triangulos"):
        print(f"T de la referencia = {referencia['T']:,} | SNAP publica {info['triangulos']:,}: "
              f"{'coincide' if referencia['T'] == info['triangulos'] else 'NO coincide'}")

    print(f"\nMidiendo ({a.reps} repeticiones por configuración, tras 1 de calentamiento que incluye la compilación JIT)...")
    t0 = time.perf_counter()
    r = barrer(csr, pre, a.parte, a.estrategias_caminos, a.memorias_caminos, a.estrategias_tri, a.memorias_tri,
               a.bloques, a.chunk, a.chunk_tri, a.max_bloques, a.reps, a.trabajo_por_lanzamiento, a.criterio,
               referencia, not a.sin_completo, preproceso=a.preproceso)
    duracion = time.perf_counter() - t0
    pr = r["preparacion"]
    print(f"\nPreparación compartida por todas las configuraciones (preproceso en {a.preproceso.upper()}, una vez): "
          f"suma prefija {sec.seg(pr['t_prefijo_s'])}"
          + (f" | orientación CPU {sec.seg(pr['t_orientacion_cpu_s'])} | fila CPU {sec.seg(pr['preproceso_cpu_s'])}"
             if a.preproceso == "cpu" else "")
          + f" | carga a GPU {sec.seg(pr['carga_s'])}"
          + (f" | orientación y fila en GPU {sec.seg(pr['preproceso_gpu_s'])}" if a.preproceso == "gpu" else ""))
    print("En las tablas, H2D = reservar el histograma o la tabla; el grafo ya está en la GPU.")

    imprimir_tabla(r["filas"], "caminos", r["mejor_caminos"], a.criterio)
    imprimir_tabla(r["filas"], "triangulos", r["mejor_triangulos"], a.criterio)
    malas = [f for f in r["filas"] if f.get("correcto") is False]
    print(f"\nConfiguraciones medidas: {sum(1 for f in r['filas'] if f.get('correcto') is not None)} | "
          f"omitidas: {sum(1 for f in r['filas'] if f.get('omitida'))} | incorrectas o con error: {len(malas)} | "
          f"duración del barrido: {sec.seg(duracion)}")
    txt_c, txt_t = c3.describir_config(c3.completar_memorias(r["config"], prep))
    print(f"Mejor configuración ({a.criterio}): caminos {txt_c} | triángulos {txt_t}")
    comp = r["completo"]
    if comp:
        est = comp["est"]
        print(f"\n=== CENSO COMPLETO CON LA MEJOR CONFIGURACIÓN ({a.reps} repeticiones) ===")
        print(f"Censo (caminos + triángulos + inducido): {sec.seg_est(est['censo'])} | solo cálculo: "
              f"{sec.seg_est(est['calculo'])} | preparación: {sec.seg_est(est['preparacion'])} | "
              f"extremo a extremo: {sec.seg_est(est['extremo'])}")
        print(f"W = {comp['W']:,} | T = {comp['T']:,} | inducidos = {comp['W'] - 3 * comp['T']:,}")
        print(f"Invariantes: {all(comp['invariantes'].values())} | repeticiones idénticas: {comp['repeticiones_identicas']}"
              f" | igual a la referencia: {comp['igual_referencia'] if comp['igual_referencia'] is not None else '-'}")

    nombre_seguro = re.sub(r"[^\w.-]+", "_", os.path.splitext(os.path.basename(str(csr.nombre)))[0])
    carpeta = os.path.join(a.salida, f"barrido_{nombre_seguro}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    grafo = {"nombre": csr.nombre, "nodos": csr.n, "aristas": csr.nnz // 2, "grado_max": prep["grado_max"],
             "W": prep["W"], "L": prep["L"], "E": prep["E"],
             "P_caminos": prep["P"], "P_triangulos": tri.espacio_triangulos(prep)}
    datos = {"grafo": grafo, "opciones": {k: v for k, v in vars(a).items()}, "referencia": desc_ref,
             "t_referencia_s": t_ref, "filas": r["filas"], "mejor_caminos": r["mejor_caminos"],
             "mejor_triangulos": r["mejor_triangulos"], "config": r["config"],
             "completo": {k: v for k, v in comp.items() if k != "est"} if comp else None,
             "preparacion": r["preparacion"], "duracion_s": duracion}
    print("\nArchivos generados:")
    for ruta in guardar(carpeta, datos, r["filas"], r["config"], a.criterio, csr.nombre):
        print(f"  {ruta}")
    print(f"Para usar la mejor configuración: python censo_3n.py <grafo> --config {os.path.join(carpeta, 'mejor_config.json')}")


if __name__ == "__main__":
    main()
