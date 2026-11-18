import argparse
import importlib
import json
import os
import subprocess
import time
from datetime import datetime

import numpy as np

import adaptador_cugraph as ad
import censo_3n as c3
import censo_3n_ref as sec
import censo_3n2a_gpu as gpu

MODOS = ["host", "cudf", "cugraph"]
CAMPOS = ("adaptador_s", "preproceso_cpu_s", "carga_s", "preproceso_gpu_s", "preparacion_s", "inducido_s", "censo_s",
          "extremo_s", "calculo_s")


def versiones():
    v = {}
    for mod in ("numpy", "numba", "numba_cuda", "cupy", "cudf", "cugraph", "pylibcugraph", "rmm"):
        try:
            v[mod] = importlib.import_module(mod).__version__
        except Exception:
            v[mod] = None
    try:
        salida = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.used,memory.total",
                                 "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
        v["nvidia_smi"] = salida
    except Exception:
        v["nvidia_smi"] = None
    return v


def commit():
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        sucio = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                               text=True).stdout.strip()
        return h + ("-sucio" if sucio else "")
    except Exception:
        return None


def memoria():
    libre, total = gpu.memoria_gpu()
    return {"libre_mb": None if libre is None else libre / 2**20, "total_mb": None if total is None else total / 2**20}


def resumen(corridas):
    est = {}
    for campo in CAMPOS:
        valores = [c["tiempos"][campo] for c in corridas if campo in c["tiempos"]]
        if valores:
            est[campo[:-2]] = sec.estadisticas(valores)
    for parte in ("caminos", "triangulos"):
        for campo in ("h2d_s", "kernel_s", "d2h_s", "total_s"):
            est[f"{parte}_{campo[:-2]}"] = sec.estadisticas(c["tiempos"][parte][campo] for c in corridas)
    return est


def medir(funcion, reps):
    calentamiento = funcion()
    corridas = [funcion() for _ in range(reps)]
    return calentamiento, corridas


def verificar(corridas, ref):
    return {"iguales_a_referencia": all(c3.censos_iguales(c, ref) for c in corridas),
            "invariantes": all(all(c3.invariantes(c).values()) for c in corridas)}


def bloque_modo(calentamiento, corridas, ref):
    est = resumen(corridas)
    return {"tiempos": est, "calentamiento": {k: calentamiento["tiempos"][k] for k in CAMPOS
                                              if k in calentamiento["tiempos"]},
            "exceso_jit_s": calentamiento["tiempos"]["extremo_s"] - est["extremo"]["media"],
            "W": corridas[-1]["W"], "T": corridas[-1]["T"], "config": corridas[-1]["config"],
            "lanzamientos": {p: corridas[-1]["tiempos"][p]["lanzamientos"] for p in ("caminos", "triangulos")},
            "adaptador": corridas[-1].get("adaptador"), **verificar(corridas, ref)}


def tabla_aristas(csr):
    try:
        import cudf
    except ImportError:
        if ad.modulo_arreglos() is not np:
            raise
        import pandas as cudf
    fila = np.repeat(np.arange(csr.n, dtype=np.int32), np.diff(csr.offsets))
    sup = fila < csr.indices
    datos = {"src": fila[sup], "dst": csr.indices[sup].astype(np.int32),
             "etiqueta": csr.et_arista[sup].astype(np.float32)}
    del fila, sup
    ad.sincronizar()
    t0 = time.perf_counter()
    df = cudf.DataFrame(datos)
    ad.sincronizar()
    return df, time.perf_counter() - t0


def tiempo_triangle_count(G, reps):
    import cugraph
    tiempos = []
    total = None
    for _ in range(reps + 1):
        ad.sincronizar()
        t0 = time.perf_counter()
        r = cugraph.triangle_count(G)
        total = int(r["counts"].sum())
        ad.sincronizar()
        tiempos.append(time.perf_counter() - t0)
        del r
    return {"calentamiento_s": tiempos[0], "tiempos": sec.estadisticas(tiempos[1:]), "suma_counts": total}


def imprimir(nombre, b):
    t = b["tiempos"]
    partes = [f"{k} {sec.seg_est(t[k])}" for k in ("adaptador", "preproceso_cpu", "carga", "preproceso_gpu",
                                                   "preparacion", "censo", "extremo") if k in t]
    print(f"[{nombre}] " + " | ".join(partes))
    print(f"[{nombre}] iguales a la referencia: {b['iguales_a_referencia']} | invariantes: {b['invariantes']} | "
          f"exceso JIT {sec.seg(max(0.0, b['exceso_jit_s']))}")
    if b.get("adaptador"):
        a = b["adaptador"]
        print(f"[{nombre}] adaptador: reordenado={a.get('reordenado')} sin_copia={a.get('sin_copia')} "
              f"vista={a.get('t_vista_s')}")


def crear_parser():
    p = argparse.ArgumentParser(description="Censo con el grafo ya residente en GPU (cuDF/cuGraph) frente a la "
                                            "entrada desde el host: correctitud y overhead del adaptador")
    c3.agregar_args_grafo(p)
    c3.agregar_args_config(p)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--modos", nargs="+", choices=MODOS, default=MODOS)
    p.add_argument("--sin-triangle-count", action="store_true", help="No mide cugraph.triangle_count")
    p.add_argument("--sin-etiquetas", action="store_true",
                   help="Mide además el censo con una sola etiqueta de nodo y arista sobre el grafo de cuGraph")
    p.add_argument("--salida", default="resultados/v2_cugraph")
    return p


def main():
    a = crear_parser().parse_args()
    if a.preproceso != "gpu":
        raise SystemExit("Este script compara contra la entrada desde el host con preproceso en GPU (--preproceso gpu)")
    rmm_en_cupy = ad.usar_rmm_en_cupy()
    csr, _, _, t_carga_grafo = c3.cargar_grafo(a)
    cfg = c3.config_desde_args(a)
    kw = dict(config=cfg, max_bloques=a.max_bloques, trabajo_max=a.trabajo_por_lanzamiento)
    print(f"Grafo {csr.nombre}: n={csr.n:,} nnz={csr.nnz:,} | carga desde disco {sec.seg(t_carga_grafo)}")
    print(f"Configuración pedida: {c3.describir_config(c3.completar_memorias(cfg, gpu.preparar(csr)))}")
    out = {"grafo": csr.nombre, "n": csr.n, "nnz": csr.nnz, "L": len(csr.nombres_nodo), "E": len(csr.nombres_arista),
           "reps": a.reps, "fecha": datetime.now().isoformat(timespec="seconds"), "commit": commit(),
           "versiones": versiones(), "rmm_en_cupy": rmm_en_cupy, "memoria_inicial": memoria(), "modos": {},
           "extra": {}}
    ref = None

    if "host" in a.modos:
        cal, corr = medir(lambda: c3.censo3(csr, preproceso="gpu", **kw), a.reps)
        ref = corr[-1]
        out["modos"]["host"] = bloque_modo(cal, corr, ref)
        imprimir("host", out["modos"]["host"])
        del cal, corr
        ad.liberar()

    xp = ad.modulo_arreglos()
    et_nodo = xp.asarray(csr.et_nodo)
    df = None
    if "cudf" in a.modos or "cugraph" in a.modos:
        df, t_subida = tabla_aristas(csr)
        out["extra"]["subida_tabla_aristas_s"] = t_subida
        out["extra"]["aristas_tabla"] = int(len(df))
        out["extra"]["memoria_con_tabla"] = memoria()
        print(f"Tabla de aristas en cuDF: {len(df):,} filas, subida {sec.seg(t_subida)}")

    if "cudf" in a.modos:
        try:
            cal, corr = medir(lambda: ad.censo3_residente(
                ad.desde_cudf(df, et_nodo, csr.nombres_nodo, csr.nombres_arista, etiqueta="etiqueta",
                              nombre=csr.nombre, max_bloques=a.max_bloques), **kw), a.reps)
            ref = ref or corr[-1]
            out["modos"]["cudf"] = bloque_modo(cal, corr, ref)
            imprimir("cudf", out["modos"]["cudf"])
            del cal, corr
        except Exception as e:
            out["modos"]["cudf"] = {"error": repr(e)}
            print(f"[cudf] ERROR: {e!r}")
        ad.liberar()

    if "cugraph" in a.modos:
        G = None
        try:
            import cugraph
            m0 = memoria()
            ad.sincronizar()
            t0 = time.perf_counter()
            G = cugraph.Graph(directed=False)
            G.from_cudf_edgelist(df, source="src", destination="dst", edge_attr="etiqueta", renumber=False)
            ad.sincronizar()
            out["extra"]["construccion_cugraph_s"] = time.perf_counter() - t0
            del df
            df = None
            ad.liberar()
            out["extra"]["memoria_antes_cugraph"] = m0
            out["extra"]["memoria_con_cugraph"] = memoria()
            print(f"Grafo de cuGraph construido en {sec.seg(out['extra']['construccion_cugraph_s'])}")
            cal, corr = medir(lambda: ad.censo3_residente(
                ad.desde_cugraph(G, et_nodo, csr.nombres_nodo, csr.nombres_arista, csr.nombre,
                                 max_bloques=a.max_bloques), **kw), a.reps)
            ref = ref or corr[-1]
            out["modos"]["cugraph"] = bloque_modo(cal, corr, ref)
            out["modos"]["cugraph"]["vista_primera_llamada_s"] = (cal.get("adaptador") or {}).get("t_vista_s")
            imprimir("cugraph", out["modos"]["cugraph"])
            del cal, corr
            ad.liberar()
        except Exception as e:
            out["modos"]["cugraph"] = {"error": repr(e)}
            print(f"[cugraph] ERROR: {e!r}")
            ad.liberar()

        if G is not None and not a.sin_triangle_count:
            try:
                tc = tiempo_triangle_count(G, a.reps)
                tc["coincide_con_T"] = ref is not None and tc["suma_counts"] == 3 * ref["T"]
                out["extra"]["triangle_count_cugraph"] = tc
                print(f"[triangle_count] {sec.seg_est(tc['tiempos'])} | coincide con T: {tc['coincide_con_T']}")
            except Exception as e:
                out["extra"]["triangle_count_cugraph"] = {"error": repr(e)}
                print(f"[triangle_count] ERROR: {e!r}")
            ad.liberar()

        if G is not None and a.sin_etiquetas:
            try:
                def sin_etiquetas():
                    off, ind, _ = ad.vista_adyacencia(G)
                    entrada = ad.desde_arreglos(off, ind, xp.zeros(csr.n, dtype=np.int32), None, ["-"], ["-"],
                                                csr.nombre + "_sin_etiquetas", a.max_bloques)
                    return ad.censo3_residente(entrada, **kw)
                cal, corr = medir(sin_etiquetas, a.reps)
                b = {"tiempos": resumen(corr), "T": corr[-1]["T"], "W": corr[-1]["W"], "config": corr[-1]["config"],
                     "T_igual": ref is not None and corr[-1]["T"] == ref["T"]}
                out["extra"]["censo_sin_etiquetas"] = b
                print(f"[sin etiquetas] triángulos kernel {sec.seg_est(b['tiempos']['triangulos_kernel'])} | "
                      f"censo {sec.seg_est(b['tiempos']['censo'])} | T igual: {b['T_igual']}")
                del cal, corr
            except Exception as e:
                out["extra"]["censo_sin_etiquetas"] = {"error": repr(e)}
                print(f"[sin etiquetas] ERROR: {e!r}")
            ad.liberar()
        del G
        ad.liberar()
    del df

    modos = out["modos"]
    if "host" in modos and "tiempos" in modos["host"]:
        h = modos["host"]["tiempos"]
        for nombre in ("cudf", "cugraph"):
            if nombre in modos and "tiempos" in modos[nombre]:
                r = modos[nombre]["tiempos"]
                out["extra"][f"comparacion_{nombre}"] = {
                    "carga_host_s": h["carga"]["media"], "adaptador_s": r["adaptador"]["media"],
                    "preparacion_host_s": h["preparacion"]["media"], "preparacion_residente_s": r["preparacion"]["media"],
                    "censo_host_s": h["censo"]["media"], "censo_residente_s": r["censo"]["media"],
                    "extremo_host_s": h["extremo"]["media"], "extremo_residente_s": r["extremo"]["media"],
                    "ahorro_extremo_s": h["extremo"]["media"] - r["extremo"]["media"],
                }
                c = out["extra"][f"comparacion_{nombre}"]
                print(f"[{nombre} vs host] carga evitada {sec.seg(c['carga_host_s'])} | adaptador "
                      f"{sec.seg(c['adaptador_s'])} | extremo {sec.seg(c['extremo_host_s'])} -> "
                      f"{sec.seg(c['extremo_residente_s'])}")
    out["memoria_final"] = memoria()
    os.makedirs(a.salida, exist_ok=True)
    ruta = os.path.join(a.salida, f"cugraph_{csr.nombre}_{datetime.now():%Y%m%d_%H%M%S}.json")
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print(f"Resultados en {ruta}")
    ok = all(m.get("iguales_a_referencia", False) and m.get("invariantes", False) for m in modos.values())
    raise SystemExit(0 if ok and modos else 1)


if __name__ == "__main__":
    main()
