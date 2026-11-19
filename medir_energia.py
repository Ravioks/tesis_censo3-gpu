import argparse
import json
import os
import subprocess
import time
from datetime import datetime

import numpy as np

import censo_3n as c3
import censo_3n_cpu_njit as cpu
import censo_3n_ref as sec
import censo_3n2a_gpu as gpu
import medidores_energia as me

MODOS = ["gpu_censo", "gpu_extremo", "cpu6_censo", "cpu6_extremo", "cpu1_censo"]


def commit():
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        sucio = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                               text=True).stdout.strip()
        return h + ("-sucio" if sucio else "")
    except Exception:
        return None


def preparar_modo(modo, csr, cfg, a):
    kw = dict(max_bloques=a.max_bloques, trabajo_max=a.trabajo_por_lanzamiento)
    if modo == "gpu_censo":
        pre = c3.preparar(csr, orientar=False)
        cfgc = c3.completar_memorias(c3.resolver_config(cfg), pre["prep"])
        disp = c3.preparar_dispositivo(csr, pre, [cfgc["estrategia_caminos"]], [cfgc["estrategia_tri"]], "gpu",
                                       a.max_bloques)
        return lambda: c3.censo3(csr, pre, config=cfg, dispositivo=disp, **kw), (pre, disp)
    if modo == "gpu_extremo":
        return lambda: c3.censo3(csr, None, config=cfg, preproceso="gpu", **kw), None
    hilos = 1 if modo.startswith("cpu1") else a.hilos
    paralelo = hilos > 1
    if paralelo:
        cpu.set_num_threads(hilos)
    if modo.endswith("_censo"):
        pre = c3.preparar(csr)
        return lambda: cpu.censo3_njit(csr, pre, paralelo=paralelo, estrategia_caminos="combinatorio"), pre
    return lambda: cpu.censo3_njit(csr, None, paralelo=paralelo, estrategia_caminos="combinatorio"), None


def liberar_ligero():
    try:
        gpu.cuda.current_context().deallocations.clear()
    except Exception:
        pass


def tiempo_corrida(modo, res):
    t = res["tiempos"]
    return t["censo_s"] if modo.endswith("_censo") else t["extremo_s"]


def medir_modo(modo, csr, cfg, ref, med, a, ronda):
    print(f"  [{modo}] ronda {ronda}: reposo {a.reposo:.0f} s", flush=True)
    clave = f"{modo}_r{ronda}"
    ti0 = med.marcar(clave + "_reposo0")
    time.sleep(a.reposo)
    ti1 = med.marcar(clave + "_reposo1")
    fn, vivo = preparar_modo(modo, csr, cfg, a)
    cal = fn()
    if not c3.censos_iguales(cal, ref):
        raise RuntimeError(f"{modo}: el censo de calentamiento no coincide con la referencia")
    gpu.cuda.synchronize()
    original = gpu.liberar_gpu
    if a.sin_gc and modo.startswith("gpu"):
        gpu.liberar_gpu = liberar_ligero
    try:
        t0 = med.marcar(clave + "_bucle0")
        reps, tiempos = 0, []
        while reps < a.reps_min or me.ahora() - t0 < a.duracion:
            res = fn()
            tiempos.append(tiempo_corrida(modo, res))
            reps += 1
        gpu.cuda.synchronize()
        t1 = med.marcar(clave + "_bucle1")
    finally:
        gpu.liberar_gpu = original
    iguales = c3.censos_iguales(res, ref)
    del vivo, fn, res
    gpu.liberar_gpu()
    print(f"  [{modo}] {reps} repeticiones en {t1 - t0:.1f} s | enfriamiento {a.enfriamiento:.0f} s", flush=True)
    time.sleep(a.enfriamiento)
    out = {"modo": modo, "ronda": ronda, "reps": reps, "duracion_s": t1 - t0, "reposo_s": ti1 - ti0,
           "tiempo_por_corrida": sec.estadisticas(tiempos), "iguales_a_referencia": iguales, "componentes": {}}
    for comp in med.componentes():
        e = med.energia(comp, clave + "_bucle0", clave + "_bucle1", t0, t1)
        r = med.energia(comp, clave + "_reposo0", clave + "_reposo1", ti0, ti1)
        if e["julios"] is None or r["julios"] is None:
            out["componentes"][comp] = {"error": "sin muestras"}
            continue
        dinamica = e["julios"] - r["watts"] * (t1 - t0)
        out["componentes"][comp] = {"julios_bucle": e["julios"], "watts_bucle": e["watts"], "watts_reposo": r["watts"],
                                    "julios_por_corrida": e["julios"] / reps,
                                    "julios_dinamicos_por_corrida": dinamica / reps,
                                    "muestras_bucle": e["muestras"], "muestras_reposo": r["muestras"]}
    c = out["componentes"]
    if "julios_por_corrida" in c.get("cpu", {}) and "julios_por_corrida" in c.get("gpu", {}):
        out["total_julios_por_corrida"] = c["cpu"]["julios_por_corrida"] + c["gpu"]["julios_por_corrida"]
        out["total_julios_dinamicos_por_corrida"] = (c["cpu"]["julios_dinamicos_por_corrida"] +
                                                     c["gpu"]["julios_dinamicos_por_corrida"])
        out["watts_total_bucle"] = c["cpu"]["watts_bucle"] + c["gpu"]["watts_bucle"]
        out["watts_total_reposo"] = c["cpu"]["watts_reposo"] + c["gpu"]["watts_reposo"]
        print(f"  [{modo}] {out['total_julios_por_corrida']:.3f} J por corrida "
              f"(dinámica {out['total_julios_dinamicos_por_corrida']:.3f} J) | CPU {c['cpu']['watts_bucle']:.1f} W, "
              f"GPU {c['gpu']['watts_bucle']:.1f} W (reposo {out['watts_total_reposo']:.1f} W)", flush=True)
    return out


def razon(a, b):
    return a / b if b and b > 0 and a is not None else None


def resumir(mediciones, modos):
    resumen = {}
    for modo in modos:
        ms = [m for m in mediciones if m["modo"] == modo and "total_julios_por_corrida" in m]
        if not ms:
            continue
        resumen[modo] = {
            "rondas": len(ms),
            "tiempo_por_corrida_s": sec.estadisticas(m["duracion_s"] / m["reps"] for m in ms),
            "julios_por_corrida": sec.estadisticas(m["total_julios_por_corrida"] for m in ms),
            "julios_dinamicos_por_corrida": sec.estadisticas(m["total_julios_dinamicos_por_corrida"] for m in ms),
            "watts_total_bucle": sec.estadisticas(m["watts_total_bucle"] for m in ms),
            "watts_total_reposo": sec.estadisticas(m["watts_total_reposo"] for m in ms),
            "watts_cpu_bucle": sec.estadisticas(m["componentes"]["cpu"]["watts_bucle"] for m in ms),
            "watts_gpu_bucle": sec.estadisticas(m["componentes"]["gpu"]["watts_bucle"] for m in ms),
            "julios_cpu_por_corrida": sec.estadisticas(m["componentes"]["cpu"]["julios_por_corrida"] for m in ms),
            "julios_gpu_por_corrida": sec.estadisticas(m["componentes"]["gpu"]["julios_por_corrida"] for m in ms),
            "todas_iguales": all(m["iguales_a_referencia"] for m in ms),
        }
        if all("gpu_lhm" in m["componentes"] and "watts_bucle" in m["componentes"]["gpu_lhm"] for m in ms):
            resumen[modo]["watts_gpu_lhm_bucle"] = sec.estadisticas(m["componentes"]["gpu_lhm"]["watts_bucle"]
                                                                   for m in ms)
    razones = {}
    for cpu_m, gpu_m in (("cpu6_censo", "gpu_censo"), ("cpu1_censo", "gpu_censo"), ("cpu6_extremo", "gpu_extremo")):
        if cpu_m in resumen and gpu_m in resumen:
            rc, rg = resumen[cpu_m], resumen[gpu_m]
            razones[f"{cpu_m}/{gpu_m}"] = {
                "energia": razon(rc["julios_por_corrida"]["media"], rg["julios_por_corrida"]["media"]),
                "energia_dinamica": razon(rc["julios_dinamicos_por_corrida"]["media"],
                                          rg["julios_dinamicos_por_corrida"]["media"]),
                "tiempo": razon(rc["tiempo_por_corrida_s"]["media"], rg["tiempo_por_corrida_s"]["media"]),
            }
    return resumen, razones


def crear_parser():
    p = argparse.ArgumentParser(description="Energía del censo: CPU compilada (njit) vs GPU, misma sesión")
    c3.agregar_args_grafo(p)
    c3.agregar_args_config(p)
    p.add_argument("--modos", nargs="+", choices=MODOS, default=MODOS)
    p.add_argument("--rondas", type=int, default=3, help="Rondas por modo; el orden se invierte en rondas impares")
    p.add_argument("--duracion", type=float, default=30.0, help="Duración mínima del bucle medido (s)")
    p.add_argument("--reps-min", type=int, default=3)
    p.add_argument("--reposo", type=float, default=20.0, help="Ventana de reposo antes de cada bucle (s)")
    p.add_argument("--enfriamiento", type=float, default=15.0)
    p.add_argument("--hilos", type=int, default=6)
    p.add_argument("--fuente-cpu", choices=["auto", "rapl", "lhm", "falsa"], default="auto")
    p.add_argument("--fuente-gpu", choices=["auto", "nvml_energia", "nvml", "smi", "lhm", "falsa"], default="auto")
    p.add_argument("--url-lhm", default=None)
    p.add_argument("--sensor-cpu", default=None, help="Texto o SensorId del sensor de CPU en LibreHardwareMonitor")
    p.add_argument("--sensor-gpu", default=None)
    p.add_argument("--sin-gc", action="store_true",
                   help="En los modos GPU, el bucle medido no llama a gc.collect() entre corridas")
    p.add_argument("--salida", default="resultados/v3_energia")
    return p


def main():
    a = crear_parser().parse_args()
    falsa = {"cpu": 40.0, "gpu": 60.0} if "falsa" in (a.fuente_cpu, a.fuente_gpu) else None
    med = me.Medidores(a.fuente_cpu, a.fuente_gpu, a.url_lhm, a.sensor_cpu, a.sensor_gpu, potencia_falsa=falsa)
    print("Fuentes:", json.dumps(med.info()["descripcion"], ensure_ascii=False, default=str)[:600], flush=True)
    csr, _, _, t_carga = c3.cargar_grafo(a)
    cfg = c3.config_desde_args(a)
    print(f"Grafo {csr.nombre}: n={csr.n:,} nnz={csr.nnz:,} | configuración GPU "
          f"{c3.describir_config(c3.completar_memorias(cfg, gpu.preparar(csr)))}", flush=True)
    ref = c3.censo3(csr, config=cfg, max_bloques=a.max_bloques, trabajo_max=a.trabajo_por_lanzamiento)
    cpu.calentar()
    ref_cpu = cpu.censo3_njit(csr, None, paralelo=True, estrategia_caminos="combinatorio")
    if not c3.censos_iguales(ref, ref_cpu):
        raise SystemExit("La CPU y la GPU no coinciden en el censo de referencia")
    gpu.liberar_gpu()
    med.iniciar()
    mediciones = []
    t_inicio = me.ahora()
    try:
        for r in range(a.rondas):
            orden = a.modos if r % 2 == 0 else list(reversed(a.modos))
            for modo in orden:
                try:
                    mediciones.append(medir_modo(modo, csr, cfg, ref, med, a, r))
                except Exception as e:
                    print(f"  [{modo}] ERROR: {e!r}", flush=True)
                    mediciones.append({"modo": modo, "ronda": r, "error": repr(e)})
                    gpu.liberar_gpu()
    finally:
        med.parar()
    resumen, razones = resumir(mediciones, a.modos)
    for modo, b in resumen.items():
        print(f"[{modo}] {b['julios_por_corrida']['media']:.3f} ± {b['julios_por_corrida']['desv']:.3f} J por corrida | "
              f"dinámica {b['julios_dinamicos_por_corrida']['media']:.3f} J | {b['tiempo_por_corrida_s']['media']:.4f} s | "
              f"{b['watts_total_bucle']['media']:.1f} W (reposo {b['watts_total_reposo']['media']:.1f} W) | "
              f"iguales {b['todas_iguales']}")
    for k, v in razones.items():
        txt = " | ".join(f"{n} {v[n]:.1f}x" if v[n] is not None else f"{n} n/d" for n in ("energia", "energia_dinamica",
                                                                                         "tiempo"))
        print(f"[{k}] {txt}")
    os.makedirs(a.salida, exist_ok=True)
    marca = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(a.salida, f"energia_{csr.nombre}{'_singc' if a.sin_gc else ''}_{marca}")
    out = {"grafo": csr.nombre, "n": csr.n, "nnz": csr.nnz, "fecha": datetime.now().isoformat(timespec="seconds"),
           "commit": commit(), "config_gpu": c3.completar_memorias(cfg, gpu.preparar(csr)), "hilos_cpu": a.hilos,
           "parametros": {k: getattr(a, k) for k in ("rondas", "duracion", "reps_min", "reposo", "enfriamiento",
                                                      "modos", "sin_gc")},
           "medidores": med.info(), "t_inicio_monotonic": t_inicio, "mediciones": mediciones, "resumen": resumen,
           "razones": razones}
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    np.savez_compressed(base + "_muestras.npz", **{k: v for k, v in med.muestras().items()})
    print(f"Resultados en {base}.json (muestras de potencia en {base}_muestras.npz)")
    ok = resumen and all(b["todas_iguales"] for b in resumen.values()) and not any("error" in m for m in mediciones)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
