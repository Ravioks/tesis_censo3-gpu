import argparse
import cProfile
import gc
import io
import json
import pstats
import time

import censo_3n as c3
import censo_3n2a_gpu as gpu


def liberar_ligero():
    try:
        gpu.cuda.current_context().deallocations.clear()
    except Exception:
        pass


def preparar(csr, cfg, a):
    pre = c3.preparar(csr, orientar=False)
    cfgc = c3.completar_memorias(c3.resolver_config(cfg), pre["prep"])
    disp = c3.preparar_dispositivo(csr, pre, [cfgc["estrategia_caminos"]], [cfgc["estrategia_tri"]], "gpu",
                                   a.max_bloques)
    return lambda: c3.censo3(csr, pre, config=cfg, dispositivo=disp, max_bloques=a.max_bloques,
                             trabajo_max=a.trabajo_por_lanzamiento), (pre, disp)


def serie(fn, reps, ref):
    pared, censo = [], []
    ok = True
    for _ in range(reps):
        t0 = time.perf_counter()
        r = fn()
        gpu.cuda.synchronize()
        pared.append(time.perf_counter() - t0)
        censo.append(r["tiempos"]["censo_s"])
        ok = ok and c3.censos_iguales(r, ref)
    m = lambda x: sum(x) / len(x)
    return {"pared_s": m(pared), "censo_s": m(censo), "fuera_s": m(pared) - m(censo), "iguales": ok}


def main():
    p = argparse.ArgumentParser(description="Dónde se va el tiempo de una llamada completa a censo3 (GPU)")
    c3.agregar_args_grafo(p)
    c3.agregar_args_config(p)
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--salida", default=None)
    a = p.parse_args()
    csr, _, _, _ = c3.cargar_grafo(a)
    cfg = c3.config_desde_args(a)
    ref = c3.censo3(csr, config=cfg, max_bloques=a.max_bloques, trabajo_max=a.trabajo_por_lanzamiento)
    fn, vivo = preparar(csr, cfg, a)
    fn()
    out = {"grafo": csr.nombre, "reps": a.reps}

    t = []
    for _ in range(5):
        t0 = time.perf_counter()
        gc.collect()
        t.append(time.perf_counter() - t0)
    out["gc_collect_s"] = sum(t) / len(t)
    out["objetos_gc"] = len(gc.get_objects())

    out["normal"] = serie(fn, a.reps, ref)
    original = gpu.liberar_gpu
    gpu.liberar_gpu = liberar_ligero
    try:
        out["sin_gc"] = serie(fn, a.reps, ref)
    finally:
        gpu.liberar_gpu = original

    prof = cProfile.Profile()
    prof.enable()
    for _ in range(5):
        fn()
    gpu.cuda.synchronize()
    prof.disable()
    s = io.StringIO()
    pstats.Stats(prof, stream=s).sort_stats("cumulative").print_stats(25)
    out["perfil"] = s.getvalue()

    for k in ("normal", "sin_gc"):
        r = out[k]
        print(f"[{k:6s}] pared {r['pared_s'] * 1e3:8.1f} ms | censo {r['censo_s'] * 1e3:8.1f} ms | "
              f"fuera del censo {r['fuera_s'] * 1e3:8.1f} ms | iguales {r['iguales']}")
    print(f"gc.collect(): {out['gc_collect_s'] * 1e3:.1f} ms con {out['objetos_gc']:,} objetos rastreados")
    print(out["perfil"])
    if a.salida:
        with open(a.salida, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    del vivo


if __name__ == "__main__":
    main()
