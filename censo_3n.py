import argparse
import json
import time

import numpy as np

import censo_3n2a as sec
import censo_3n2a_gpu as gpu
import censo_3n3a_gpu as tri

LIMITE_PYTHON = 20_000_000
PREPROCESOS = ["gpu", "cpu"]
CLAVES = ("caminos_no_inducidos", "caminos_inducidos", "triangulos")
CONFIG_POR_DEFECTO = {
    "estrategia_caminos": "entrada",
    "memoria_caminos": None,
    "bloque_caminos": 256,
    "chunk": 64,
    "estrategia_tri": "arista",
    "memoria_tri": None,
    "bloque_tri": 256,
    "chunk_tri": tri.CHUNK_TRI,
}


def preparar(csr, orientar=True):
    t0 = time.perf_counter()
    prep = gpu.preparar(csr)
    t_prep = time.perf_counter() - t0
    orient = tri.orientar(csr) if orientar else None
    return {"prep": prep, "orient": orient, "t_prefijo_s": t_prep,
            "t_orientacion_s": orient["t_s"] if orient else 0.0}


def resolver_config(config=None, **cambios):
    desconocidas = set(cambios) - set(CONFIG_POR_DEFECTO)
    if desconocidas:
        raise TypeError(f"Opciones de configuración desconocidas: {', '.join(sorted(desconocidas))}")
    cfg = dict(CONFIG_POR_DEFECTO)
    cfg.update({k: v for k, v in (config or {}).items() if k in CONFIG_POR_DEFECTO and v is not None})
    cfg.update({k: v for k, v in cambios.items() if v is not None})
    if cfg["estrategia_caminos"] not in gpu.ESTRATEGIAS:
        raise ValueError(f"Estrategia de caminos desconocida: {cfg['estrategia_caminos']}")
    if cfg["estrategia_tri"] not in tri.ESTRATEGIAS:
        raise ValueError(f"Estrategia de triángulos desconocida: {cfg['estrategia_tri']}")
    for clave in ("memoria_caminos", "memoria_tri"):
        if cfg[clave] is not None and cfg[clave] not in gpu.MEMORIAS:
            raise ValueError(f"Modo de memoria desconocido: {cfg[clave]}")
    return cfg


def completar_memorias(cfg, prep):
    cfg = dict(cfg)
    cfg["memoria_caminos"] = cfg["memoria_caminos"] or gpu.memoria_por_defecto(prep["P"])
    cfg["memoria_tri"] = cfg["memoria_tri"] or tri.memoria_por_defecto(prep)
    return cfg


def cargar_config(ruta):
    with open(ruta, encoding="utf-8") as f:
        datos = json.load(f)
    return datos.get("config", datos)


def preparar_dispositivo(csr, pre, estrategias_caminos, estrategias_tri, preproceso="gpu", max_bloques=4096):
    if preproceso not in PREPROCESOS:
        raise ValueError(f"Preproceso desconocido: {preproceso} (opciones: {', '.join(PREPROCESOS)})")
    prep = pre["prep"]
    orient = pre["orient"]
    necesita_fila = "entrada" in estrategias_caminos
    t_cpu = 0.0
    fila = None
    if preproceso == "cpu":
        t0 = time.perf_counter()
        if orient is None or orient.get("en_gpu"):
            orient = tri.orientar(csr)
        if necesita_fila:
            fila = gpu.obtener_fila(csr, prep)
        t_cpu = time.perf_counter() - t0

    gpu.cuda.synchronize()
    t0 = time.perf_counter()
    disp = gpu.subir_grafo(csr)
    if "balanceado" in estrategias_caminos:
        disp["acum"] = gpu.cuda.to_device(prep["acum"])
    if preproceso == "cpu":
        if fila is not None:
            disp["fila"] = gpu.cuda.to_device(fila)
        orient = tri.subir_orientacion(orient, acum_en_gpu="balanceado" in estrategias_tri)
    gpu.cuda.synchronize()
    t_carga = time.perf_counter() - t0

    t_gpu = 0.0
    if preproceso == "gpu":
        t0 = time.perf_counter()
        if necesita_fila:
            disp["fila"] = gpu.fila_en_gpu(disp, csr.n, csr.nnz, max_bloques=max_bloques)
        orient = tri.orientar_gpu(csr, disp, binaria="binaria" in estrategias_tri, max_bloques=max_bloques)
        gpu.cuda.synchronize()
        t_gpu = time.perf_counter() - t0
    return {"disp": disp, "orient": orient, "preproceso": preproceso,
            "tiempos": {"preproceso_cpu_s": t_cpu, "carga_s": t_carga, "preproceso_gpu_s": t_gpu}}


def censo3(csr, pre=None, config=None, max_bloques=4096, trabajo_max=gpu.TRABAJO_POR_LANZAMIENTO, preproceso="gpu",
           dispositivo=None, **cambios):
    preproceso_incluido = pre is None
    pre = pre or preparar(csr, orientar=preproceso == "cpu")
    prep = pre["prep"]
    cfg = completar_memorias(resolver_config(config, **cambios), prep)
    propio = dispositivo is None
    if propio:
        dispositivo = preparar_dispositivo(csr, pre, [cfg["estrategia_caminos"]], [cfg["estrategia_tri"]],
                                           preproceso, max_bloques)
        t_disp = dispositivo["tiempos"]
    else:
        preproceso = dispositivo["preproceso"]
        t_disp = {"preproceso_cpu_s": 0.0, "carga_s": 0.0, "preproceso_gpu_s": 0.0}
    disp, orient = dispositivo["disp"], dispositivo["orient"]

    rc = gpu.ejecutar_gpu(csr, prep, cfg["estrategia_caminos"], cfg["memoria_caminos"], cfg["bloque_caminos"],
                          max_bloques, cfg["chunk"], trabajo_max=trabajo_max, disp=disp)
    if propio:
        disp.pop("fila", None)
        disp.pop("acum", None)
    gpu.liberar_gpu()
    rt = tri.ejecutar_triangulos_gpu(csr, prep, orient, cfg["estrategia_tri"], cfg["memoria_tri"], cfg["bloque_tri"],
                                     max_bloques, cfg["chunk_tri"], trabajo_max, et_nodo=disp["et_nodo"])
    info_orient = {"aristas_orientadas": orient["aristas"], "grado_salida_max": orient["grado_salida_max"]}
    if propio:
        del disp, orient, dispositivo
    gpu.liberar_gpu()

    t0 = time.perf_counter()
    inducido = sec.censo_inducido(rc["censo"], sec.resta_desde_triangulos(rt["triangulos"]))
    t_inducido = time.perf_counter() - t0

    partes = {}
    for nombre, r in (("caminos", rc), ("triangulos", rt)):
        partes[nombre] = {"h2d_s": r["t_h2d"], "kernel_s": r["t_kernel"], "d2h_s": r["t_d2h"],
                          "total_s": r["t_h2d"] + r["t_kernel"] + r["t_d2h"], "lanzamientos": r["lanzamientos"],
                          "hilos": r["hilos"]}
    t_pre_cpu = pre["t_prefijo_s"] + (pre["t_orientacion_s"] if preproceso == "cpu" else 0.0) + t_disp["preproceso_cpu_s"]
    t_preparacion = t_pre_cpu + t_disp["carga_s"] + t_disp["preproceso_gpu_s"]
    t_censo = partes["caminos"]["total_s"] + partes["triangulos"]["total_s"] + t_inducido
    return {
        "caminos_no_inducidos": rc["censo"],
        "caminos_inducidos": inducido,
        "triangulos": rt["triangulos"],
        "W": prep["W"],
        "T": rt["T"],
        "config": cfg,
        "preproceso": preproceso,
        **info_orient,
        "tiempos": {
            **partes,
            "inducido_s": t_inducido,
            "preproceso_cpu_s": t_pre_cpu,
            "carga_s": t_disp["carga_s"],
            "preproceso_gpu_s": t_disp["preproceso_gpu_s"],
            "preparacion_s": t_preparacion,
            "preproceso_incluido": preproceso_incluido,
            "calculo_s": partes["caminos"]["kernel_s"] + partes["triangulos"]["kernel_s"],
            "censo_s": t_censo,
            "extremo_s": t_preparacion + t_censo,
        },
    }


CAMPOS_TIEMPO = ("inducido_s", "preproceso_cpu_s", "carga_s", "preproceso_gpu_s", "preparacion_s", "calculo_s",
                 "censo_s", "extremo_s")


def estadisticas_corridas(corridas):
    t = [c["tiempos"] for c in corridas]
    est = {}
    for parte in ("caminos", "triangulos"):
        for campo in ("h2d_s", "kernel_s", "d2h_s", "total_s"):
            est[f"{parte}_{campo[:-2]}"] = sec.estadisticas(x[parte][campo] for x in t)
    for campo in CAMPOS_TIEMPO:
        est[campo[:-2]] = sec.estadisticas(x[campo] for x in t)
    return est


def invariantes(res):
    W, T = res["W"], res["T"]
    return {
        "suma_no_inducido_igual_W": sum(res["caminos_no_inducidos"].values()) == W,
        "suma_triangulos_igual_T": sum(res["triangulos"].values()) == T,
        "suma_inducido_igual_W_menos_3T": sum(res["caminos_inducidos"].values()) == W - 3 * T,
    }


def censos_iguales(a, b):
    return all(a[k] == b[k] for k in CLAVES)


def a_dataframe(res):
    import pandas as pd

    filas = [{"tipo": "camino", "patron": sec.formato_patron(k), "no_inducido": v,
              "inducido": res["caminos_inducidos"].get(k, 0)} for k, v in res["caminos_no_inducidos"].items()]
    filas += [{"tipo": "triangulo", "patron": sec.formato_triangulo(k), "no_inducido": v, "inducido": v}
              for k, v in res["triangulos"].items()]
    df = pd.DataFrame(filas, columns=["tipo", "patron", "no_inducido", "inducido"])
    return df.sort_values(["tipo", "no_inducido", "patron"], ascending=[True, False, True], ignore_index=True)


def agregar_args_grafo(p):
    g = p.add_argument_group("grafo de entrada")
    g.add_argument("--grafo", choices=["ejemplo", "triangulo", "estrella", "completo", "aleatorio", "ba"])
    g.add_argument("--archivo", help="Archivo gSpan (t/v/e)")
    g.add_argument("--csr", help="Archivo .npz en CSR")
    g.add_argument("--dataset", choices=["livejournal", "orkut", "twitter"])
    g.add_argument("--datos", default="datasets", help="Carpeta de descargas y caché de grafos reales")
    g.add_argument("--ids", nargs="+")
    g.add_argument("--max-grafos", type=int, default=None)
    g.add_argument("--generador", choices=["python", "numpy"], default="python",
                   help="numpy: genera ba, aleatorio y completo directo en CSR (para millones de aristas)")
    g.add_argument("--n", type=int, default=1000)
    g.add_argument("--m", type=int, default=5000)
    g.add_argument("--k", type=int, default=3)
    g.add_argument("--etiquetas-nodo", type=int, default=None, help="Por defecto 3 (sintéticos) o 5 (dataset)")
    g.add_argument("--etiquetas-arista", type=int, default=2)
    g.add_argument("--semilla", type=int, default=42)


def agregar_args_config(p, sufijo_ayuda=""):
    g = p.add_argument_group("configuración GPU" + sufijo_ayuda)
    g.add_argument("--config", help="JSON con la configuración (por ejemplo, mejor_config.json de barrido.py)")
    g.add_argument("--estrategia-caminos", choices=gpu.ESTRATEGIAS, default=None,
                   help=f"Reparto del trabajo de caminos (por defecto {CONFIG_POR_DEFECTO['estrategia_caminos']})")
    g.add_argument("--memoria-caminos", choices=list(gpu.MEMORIAS), default=None,
                   help="Histograma de caminos (por defecto, el primer modo válido: shared, global, hash)")
    g.add_argument("--bloque-caminos", type=int, default=None, help="Hilos por bloque, caminos (por defecto 256)")
    g.add_argument("--chunk", type=int, default=None, help="Caminos por ítem en la estrategia balanceado (por defecto 64)")
    g.add_argument("--estrategia-tri", choices=tri.ESTRATEGIAS, default=None,
                   help=f"Reparto del trabajo de triángulos (por defecto {CONFIG_POR_DEFECTO['estrategia_tri']})")
    g.add_argument("--memoria-tri", choices=list(tri.MEMORIAS), default=None,
                   help="Histograma de triángulos (por defecto, el primer modo válido: shared, global, hash)")
    g.add_argument("--bloque-tri", type=int, default=None, help="Hilos por bloque, triángulos (por defecto 256)")
    g.add_argument("--chunk-tri", type=int, default=None,
                   help=f"Costo por ítem en la estrategia balanceado de triángulos (por defecto {tri.CHUNK_TRI})")
    g.add_argument("--preproceso", choices=PREPROCESOS, default="gpu",
                   help="Dónde se construyen la orientación y el arreglo fila: gpu (por defecto) o cpu")
    g.add_argument("--max-bloques", type=int, default=4096)
    g.add_argument("--trabajo-por-lanzamiento", type=int, default=gpu.TRABAJO_POR_LANZAMIENTO,
                   help="Trabajo máximo por lanzamiento de kernel (evita el watchdog de Windows/WSL)")


def config_desde_args(a):
    base = cargar_config(a.config) if a.config else None
    return resolver_config(base, estrategia_caminos=a.estrategia_caminos, memoria_caminos=a.memoria_caminos,
                           bloque_caminos=a.bloque_caminos, chunk=a.chunk, estrategia_tri=a.estrategia_tri,
                           memoria_tri=a.memoria_tri, bloque_tri=a.bloque_tri, chunk_tri=a.chunk_tri)


def cargar_grafo(a):
    if a.etiquetas_nodo is None:
        a.etiquetas_nodo = 5 if a.dataset else 3
    if not (a.grafo or a.archivo or a.csr or a.dataset):
        a.grafo = "ejemplo"
    t0 = time.perf_counter()
    g = None
    info = None
    if a.dataset:
        import datos_reales as dr
        info = dr.DATASETS[a.dataset]
        csr, _, _ = dr.cargar_dataset(a.dataset, a.etiquetas_nodo, a.etiquetas_arista, a.semilla, a.datos)
    elif a.csr:
        csr = gpu.cargar_npz(a.csr)
    elif a.generador == "numpy" and a.grafo in ("ba", "aleatorio", "completo"):
        import generadores
        csr = generadores.csr_sintetico(a.grafo, a.n, a.m, a.k, a.etiquetas_nodo, a.etiquetas_arista, a.semilla)
    else:
        g = sec.construir_grafo(a)
        csr = gpu.grafo_a_csr(g)
    return csr, g, info, time.perf_counter() - t0


def describir_config(cfg):
    caminos = f"{cfg['estrategia_caminos']} + {cfg['memoria_caminos']} (bloque {cfg['bloque_caminos']}"
    caminos += f", chunk {cfg['chunk']})" if cfg["estrategia_caminos"] == "balanceado" else ")"
    tri_txt = f"{cfg['estrategia_tri']} + {cfg['memoria_tri']} (bloque {cfg['bloque_tri']}"
    tri_txt += f", chunk {cfg['chunk_tri']})" if cfg["estrategia_tri"] == "balanceado" else ")"
    return caminos, tri_txt


def resumen_censo(censo, formato, top):
    filas = sorted(censo.items(), key=lambda x: (-x[1], formato(x[0])))
    return [{"patron": formato(k), "frecuencia": v} for k, v in (filas[:top] if top else filas)]


def imprimir_top(titulo, filas, total):
    print(f"\n=== {titulo} (top {len(filas)}) ===")
    if not filas:
        print("(vacío)")
        return
    ancho = max(len("Patrón"), max(len(f["patron"]) for f in filas))
    print(f"{'#':>4}  {'Patrón':<{ancho}}  {'Frecuencia':>15}  {'%':>7}")
    for i, f in enumerate(filas, 1):
        pct = 100 * f["frecuencia"] / total if total else 0.0
        print(f"{i:>4}  {f['patron']:<{ancho}}  {f['frecuencia']:>15,}  {pct:>6.2f}%")


def verificar_contra_python(csr, g, res):
    if g is None:
        g = gpu.csr_a_grafo(csr)
    t0 = time.perf_counter()
    ref = sec.censo_completo(g)
    return censos_iguales(ref, res), time.perf_counter() - t0


def crear_parser():
    p = argparse.ArgumentParser(description="Censo completo de 3 nodos en GPU: caminos (no inducidos e inducidos) y triángulos")
    agregar_args_grafo(p)
    agregar_args_config(p)
    s = p.add_argument_group("medición, verificación y salida")
    s.add_argument("--reps", type=int, default=3, help="Repeticiones medidas, tras 1 de calentamiento")
    s.add_argument("--verificar", action="store_true", help="Compara con la referencia Python (oráculo)")
    s.add_argument("--forzar-python", action="store_true", help=f"Permite --verificar con más de {LIMITE_PYTHON:,} caminos")
    s.add_argument("--verificar-njit", action="store_true", help="Compara con el censo de CPU compilada (njit, multihilo)")
    s.add_argument("--top", type=int, default=10)
    s.add_argument("--json", action="store_true", help="Exporta configuración, tiempos, verificación y censos a JSON")
    s.add_argument("--csv", action="store_true", help="Exporta la tabla tipo, patrón, no inducido, inducido a CSV")
    s.add_argument("--salida", default="resultados")
    return p


def main():
    a = crear_parser().parse_args()
    a.reps = max(1, a.reps)
    if not gpu.cuda.is_available():
        raise SystemExit("No se detectó GPU CUDA. Para depurar sin GPU: NUMBA_ENABLE_CUDASIM=1 con grafos muy pequeños.")
    cfg_pedida = config_desde_args(a)

    csr, g, info, t_carga = cargar_grafo(a)
    tri.calentar()
    pre = preparar(csr, orientar=a.preproceso == "cpu")
    prep = pre["prep"]
    cfg = completar_memorias(cfg_pedida, prep)
    for parte, (estrategia, memoria, motivo) in {
        "caminos": (cfg["estrategia_caminos"], cfg["memoria_caminos"],
                    gpu.motivo_invalida(cfg["estrategia_caminos"], cfg["memoria_caminos"], prep)),
        "triángulos": (cfg["estrategia_tri"], cfg["memoria_tri"],
                       tri.motivo_invalida(cfg["estrategia_tri"], cfg["memoria_tri"], prep)),
    }.items():
        if motivo:
            raise SystemExit(f"Configuración de {parte} no válida ({estrategia} + {memoria}): {motivo}")

    print("\n=== GRAFO ===")
    print(f"{csr.nombre} | nodos {csr.n:,} | aristas {csr.nnz // 2:,} | grado máx. {prep['grado_max']:,} | carga {sec.seg(t_carga)}")
    print(f"Etiquetas nodo {prep['L']} | etiquetas arista {prep['E']} | patrones posibles: caminos {prep['P']:,}, "
          f"triángulos {tri.espacio_triangulos(prep):,}")
    txt_c, txt_t = describir_config(cfg)
    print(f"Configuración: caminos {txt_c} | triángulos {txt_t} | preproceso en {a.preproceso.upper()}")

    gpu.cuda.to_device(np.zeros(1))
    gpu.cuda.synchronize()
    kw = dict(max_bloques=a.max_bloques, trabajo_max=a.trabajo_por_lanzamiento, preproceso=a.preproceso)
    t0 = time.perf_counter()
    censo3(csr, pre, cfg, **kw)
    t_jit = time.perf_counter() - t0
    corridas = [censo3(csr, None, cfg, **kw) for _ in range(a.reps)]
    res = corridas[-1]
    est = estadisticas_corridas(corridas)
    print(f"Aristas orientadas {res['aristas_orientadas']:,} | grado de salida máx. {res['grado_salida_max']:,}")

    print(f"\n=== TIEMPOS (media ± desviación de {a.reps} repeticiones; calentamiento con JIT: {sec.seg(t_jit)}) ===")
    vacio = ""
    print(f"{'Fase':<38}{'H2D':>26}{'Cálculo (kernel)':>26}{'D2H + decodificación':>26}{'Total':>26}{'Lanz.':>7}")
    print(f"{'Preproceso CPU':<38}{vacio:>78}{sec.seg_est(est['preproceso_cpu']):>26}")
    print(f"{'Carga del grafo a la GPU':<38}{vacio:>78}{sec.seg_est(est['carga']):>26}")
    print(f"{'Preproceso GPU':<38}{vacio:>78}{sec.seg_est(est['preproceso_gpu']):>26}")
    print(f"{'= Preparación':<38}{vacio:>78}{sec.seg_est(est['preparacion']):>26}")
    for parte, nombre in (("caminos", f"Caminos ({txt_c.split(' (')[0]})"),
                          ("triangulos", f"Triángulos ({txt_t.split(' (')[0]})")):
        print(f"{nombre:<38}{sec.seg_est(est[parte + '_h2d']):>26}{sec.seg_est(est[parte + '_kernel']):>26}"
              f"{sec.seg_est(est[parte + '_d2h']):>26}{sec.seg_est(est[parte + '_total']):>26}"
              f"{res['tiempos'][parte]['lanzamientos']:>7}")
    print(f"{'Resta e inducido (CPU)':<38}{vacio:>78}{sec.seg_est(est['inducido']):>26}")
    print(f"{'= Censo':<38}{vacio:>26}{sec.seg_est(est['calculo']):>26}{vacio:>26}{sec.seg_est(est['censo']):>26}")
    print(f"{'Extremo a extremo (preparación + censo)':<38}{vacio:>78}{sec.seg_est(est['extremo']):>26}")
    print("Preparación: CPU (suma prefija" + (", orientación y fila" if a.preproceso == "cpu" else "") +
          "), copia del grafo a la GPU" + (" y orientación y fila en GPU" if a.preproceso == "gpu" else "") +
          ". H2D de cada parte = reservar el histograma o la tabla.")

    W, T = res["W"], res["T"]
    print("\n=== TOTALES ===")
    print(f"Caminos no inducidos W = Σ C(d,2): {W:,}")
    print(f"Triángulos T: {T:,}")
    print(f"Caminos inducidos = W − 3T: {W - 3 * T:,}")
    print(f"Patrones distintos: caminos no inducidos {len(res['caminos_no_inducidos'])} | "
          f"caminos inducidos {len(res['caminos_inducidos'])} | triángulos {len(res['triangulos'])}")

    verif = invariantes(res)
    verif["repeticiones_identicas"] = all(censos_iguales(c, res) for c in corridas)
    if info and info.get("triangulos"):
        verif["T_igual_SNAP"] = T == info["triangulos"]
    if a.verificar:
        if W > LIMITE_PYTHON and not a.forzar_python:
            verif["igual_CPU_python"] = f"omitida (W > {LIMITE_PYTHON:,}; usa --forzar-python)"
        else:
            verif["igual_CPU_python"], verif["t_cpu_python_s"] = verificar_contra_python(csr, g, res)
    if a.verificar_njit:
        import censo_3n_cpu_njit as cpu
        cpu.calentar()
        r_cpu = cpu.censo3_njit(csr, pre, paralelo=True)
        verif["igual_CPU_njit"] = censos_iguales(r_cpu, res)
        verif["t_cpu_njit_s"] = r_cpu["tiempos"]["censo_s"]
    print("\n=== VERIFICACIÓN ===")
    for k, v in verif.items():
        print(f"{k}: {sec.seg(v) if k.startswith('t_') else v}")
    if "T_igual_SNAP" in verif:
        print(f"(SNAP publica {info['triangulos']:,} triángulos para {info['nombre']})")

    imprimir_top("CAMINOS NO INDUCIDOS", resumen_censo(res["caminos_no_inducidos"], sec.formato_patron, a.top), W)
    imprimir_top("CAMINOS INDUCIDOS", resumen_censo(res["caminos_inducidos"], sec.formato_patron, a.top), W - 3 * T)
    imprimir_top("TRIÁNGULOS", resumen_censo(res["triangulos"], sec.formato_triangulo, a.top), T)

    generados = []
    if a.json or a.csv:
        prefijo = sec.preparar_salida(a.salida, csr.nombre, "censo3_")
        if a.json:
            sec.guardar_json({
                "grafo": {"nombre": csr.nombre, "nodos": csr.n, "aristas": csr.nnz // 2, "grado_max": prep["grado_max"],
                          "L": prep["L"], "E": prep["E"], "P_caminos": prep["P"],
                          "P_triangulos": tri.espacio_triangulos(prep), "t_carga_s": t_carga},
                "config": cfg,
                "preproceso": a.preproceso,
                "reps": a.reps,
                "t_jit_s": t_jit,
                "tiempos": {k: sec.aplanar("t", v) for k, v in est.items()},
                "lanzamientos": {p: res["tiempos"][p]["lanzamientos"] for p in ("caminos", "triangulos")},
                "totales": {"W": W, "T": T, "inducidos": W - 3 * T},
                "verificacion": verif,
                "caminos_no_inducidos": resumen_censo(res["caminos_no_inducidos"], sec.formato_patron, 0),
                "caminos_inducidos": resumen_censo(res["caminos_inducidos"], sec.formato_patron, 0),
                "triangulos": resumen_censo(res["triangulos"], sec.formato_triangulo, 0),
            }, prefijo + ".json")
            generados.append(prefijo + ".json")
        if a.csv:
            a_dataframe(res).to_csv(prefijo + ".csv", index=False)
            generados.append(prefijo + ".csv")
    if generados:
        print("\nArchivos generados:")
        for ruta in generados:
            print(f"  {ruta}")


if __name__ == "__main__":
    main()
