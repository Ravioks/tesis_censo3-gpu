import argparse
import time

import numpy as np
from numba import get_num_threads, njit, prange, set_num_threads

import censo_3n as c3
import censo_3n2a as sec
import censo_3n2a_gpu as gpu
import censo_3n3a_gpu as tri

LIMITE_DENSO = 1 << 24
LIMITE_PARCIALES = 1 << 30
ESTRATEGIAS_CAMINOS = ["enumeracion", "combinatorio"]


@njit
def _id_camino(lc, b1, b2, B):
    if b1 > b2:
        b1, b2 = b2, b1
    return (np.int64(lc) * B + b1) * B + b2


@njit
def _id_triangulo(t0, t1, t2, B):
    if t0 > t1:
        t0, t1 = t1, t0
    if t1 > t2:
        t1, t2 = t2, t1
    if t0 > t1:
        t0, t1 = t1, t0
    return (np.int64(t0) * B + t1) * B + t2


@njit
def _enumeracion_rango(offsets, indices, et_nodo, et_arista, inicio, fin, L, B, hist):
    for c in range(inicio, fin):
        ini = offsets[c]
        fin_c = offsets[c + 1]
        lc = et_nodo[c]
        for i in range(ini, fin_c):
            bi = np.int64(et_arista[i]) * L + et_nodo[indices[i]]
            for j in range(i + 1, fin_c):
                hist[_id_camino(lc, bi, np.int64(et_arista[j]) * L + et_nodo[indices[j]], B)] += 1


@njit
def _combinatorio_rango(offsets, indices, et_nodo, et_arista, inicio, fin, L, B, hist):
    cuenta = np.zeros(B, np.int64)
    presentes = np.zeros(B, np.int64)
    for c in range(inicio, fin):
        k = 0
        for p in range(offsets[c], offsets[c + 1]):
            b = np.int64(et_arista[p]) * L + et_nodo[indices[p]]
            if cuenta[b] == 0:
                presentes[k] = b
                k += 1
            cuenta[b] += 1
        lc = et_nodo[c]
        for x in range(k):
            b1 = presentes[x]
            n1 = cuenta[b1]
            if n1 > 1:
                hist[_id_camino(lc, b1, b1, B)] += n1 * (n1 - 1) // 2
            for y in range(x + 1, k):
                b2 = presentes[y]
                hist[_id_camino(lc, b1, b2, B)] += n1 * cuenta[b2]
        for x in range(k):
            cuenta[presentes[x]] = 0


@njit
def caminos_serial(offsets, indices, et_nodo, et_arista, L, B, P, combinatorio):
    hist = np.zeros(P, np.int64)
    n = offsets.shape[0] - 1
    if combinatorio:
        _combinatorio_rango(offsets, indices, et_nodo, et_arista, 0, n, L, B, hist)
    else:
        _enumeracion_rango(offsets, indices, et_nodo, et_arista, 0, n, L, B, hist)
    return hist


@njit(parallel=True)
def caminos_paralelo(offsets, indices, et_nodo, et_arista, cortes, L, B, P, combinatorio):
    nb = cortes.shape[0] - 1
    parcial = np.zeros((nb, P), np.int64)
    for t in prange(nb):
        if combinatorio:
            _combinatorio_rango(offsets, indices, et_nodo, et_arista, cortes[t], cortes[t + 1], L, B, parcial[t])
        else:
            _enumeracion_rango(offsets, indices, et_nodo, et_arista, cortes[t], cortes[t + 1], L, B, parcial[t])
    hist = np.zeros(P, np.int64)
    for t in range(nb):
        hist += parcial[t]
    return hist


@njit
def _triangulos_rango(off_o, ind_o, eta_o, fila_o, et_nodo, inicio, fin, L, B, hist_t):
    for p in range(inicio, fin):
        u = fila_o[p]
        v = ind_o[p]
        e_uv = np.int64(eta_o[p])
        lu = et_nodo[u]
        lv = et_nodo[v]
        i = off_o[u]
        fin_i = off_o[u + 1]
        j = off_o[v]
        fin_j = off_o[v + 1]
        while i < fin_i and j < fin_j:
            a = ind_o[i]
            b = ind_o[j]
            if a < b:
                i += 1
            elif a > b:
                j += 1
            else:
                hist_t[_id_triangulo(np.int64(eta_o[j]) * L + lu, np.int64(eta_o[i]) * L + lv,
                                     e_uv * L + et_nodo[a], B)] += 1
                i += 1
                j += 1


@njit
def triangulos_serial(off_o, ind_o, eta_o, fila_o, et_nodo, L, B, PT):
    hist_t = np.zeros(PT, np.int64)
    _triangulos_rango(off_o, ind_o, eta_o, fila_o, et_nodo, 0, ind_o.shape[0], L, B, hist_t)
    return hist_t


@njit(parallel=True)
def triangulos_paralelo(off_o, ind_o, eta_o, fila_o, et_nodo, cortes, L, B, PT):
    nb = cortes.shape[0] - 1
    parcial = np.zeros((nb, PT), np.int64)
    for t in prange(nb):
        _triangulos_rango(off_o, ind_o, eta_o, fila_o, et_nodo, cortes[t], cortes[t + 1], L, B, parcial[t])
    hist_t = np.zeros(PT, np.int64)
    for t in range(nb):
        hist_t += parcial[t]
    return hist_t


def cortes_balanceados(acum, n, bloques):
    W = int(acum[-1])
    if bloques <= 1 or W == 0:
        return np.array([0, n], dtype=np.int64)
    objetivos = (np.arange(1, bloques, dtype=np.int64) * W) // bloques
    internos = np.clip(np.searchsorted(acum, objetivos), 0, n)
    return np.unique(np.concatenate(([0], internos, [n]))).astype(np.int64)


def numero_bloques(hilos, bloques_por_hilo, P):
    bloques = max(1, bloques_por_hilo * hilos)
    if bloques * P * 8 > LIMITE_PARCIALES:
        bloques = max(1, min(hilos, LIMITE_PARCIALES // (8 * P)))
    return bloques


def _nonzero(hist):
    pids = np.nonzero(hist)[0]
    return pids, hist[pids]


def censo3_njit(csr, pre=None, paralelo=True, estrategia_caminos="enumeracion", bloques_por_hilo=8):
    if estrategia_caminos not in ESTRATEGIAS_CAMINOS:
        raise ValueError(f"Estrategia de caminos desconocida para njit: {estrategia_caminos}")
    preproceso_incluido = pre is None
    pre = pre or c3.preparar(csr)
    if pre["orient"] is None or pre["orient"].get("en_gpu"):
        orient = tri.orientar(csr)
        pre = dict(pre, orient=orient, t_orientacion_s=orient["t_s"])
    prep, orient = pre["prep"], pre["orient"]
    L, B, P = prep["L"], prep["B"], prep["P"]
    PT = tri.espacio_triangulos(prep)
    if P > LIMITE_DENSO or PT > LIMITE_DENSO:
        raise ValueError("Demasiadas etiquetas para el histograma denso de la CPU")
    combinatorio = estrategia_caminos == "combinatorio"
    base = (csr.offsets, csr.indices, csr.et_nodo, csr.et_arista)
    base_t = (orient["off_o"], orient["ind_o"], orient["eta_o"], orient["fila_o"], csr.et_nodo)
    hilos = get_num_threads()

    t0 = time.perf_counter()
    if paralelo:
        acum = csr.offsets if combinatorio else prep["acum"]
        cortes = cortes_balanceados(acum, csr.n, numero_bloques(hilos, bloques_por_hilo, P))
        hist_c = caminos_paralelo(*base, cortes, L, B, P, combinatorio)
    else:
        hist_c = caminos_serial(*base, L, B, P, combinatorio)
    t_caminos = time.perf_counter() - t0

    t0 = time.perf_counter()
    if paralelo:
        cortes_t = cortes_balanceados(orient["acum_costo"], orient["aristas"], numero_bloques(hilos, bloques_por_hilo, PT))
        hist_t = triangulos_paralelo(*base_t, cortes_t, L, B, PT)
    else:
        hist_t = triangulos_serial(*base_t, L, B, PT)
    t_triangulos = time.perf_counter() - t0

    t0 = time.perf_counter()
    no_inducido = gpu.decodificar(*_nonzero(hist_c), csr, prep)
    triangulos = tri.decodificar_triangulos(*_nonzero(hist_t), csr, prep)
    inducido = sec.censo_inducido(no_inducido, sec.resta_desde_triangulos(triangulos))
    t_post = time.perf_counter() - t0

    t_pre = pre["t_prefijo_s"] + pre["t_orientacion_s"]
    t_censo = t_caminos + t_triangulos + t_post
    return {
        "caminos_no_inducidos": no_inducido,
        "caminos_inducidos": inducido,
        "triangulos": triangulos,
        "W": int(hist_c.sum()),
        "T": int(hist_t.sum()),
        "hist": (hist_c, hist_t),
        "config": {"paralelo": paralelo, "hilos": hilos if paralelo else 1, "estrategia_caminos": estrategia_caminos},
        "tiempos": {"caminos_s": t_caminos, "triangulos_s": t_triangulos, "post_s": t_post, "censo_s": t_censo,
                    "preproceso_s": t_pre, "preproceso_incluido": preproceso_incluido, "extremo_s": t_pre + t_censo},
    }


def calentar():
    csr = gpu.grafo_a_csr(sec.generar_ejemplo())
    tri.calentar()
    pre = c3.preparar(csr)
    for paralelo in (False, True):
        for estrategia in ESTRATEGIAS_CAMINOS:
            censo3_njit(csr, pre, paralelo=paralelo, estrategia_caminos=estrategia)


def medir(fn, reps):
    corridas = [fn() for _ in range(reps)]
    est = {k: sec.estadisticas(c["tiempos"][k] for c in corridas)
           for k in ("caminos_s", "triangulos_s", "post_s", "censo_s", "preproceso_s", "extremo_s")}
    return corridas[-1], est, corridas


def crear_parser():
    p = argparse.ArgumentParser(description="Censo completo de 3 nodos en CPU compilada (Numba njit): 1 hilo y multihilo")
    c3.agregar_args_grafo(p)
    s = p.add_argument_group("CPU")
    s.add_argument("--modo", choices=["serial", "paralelo", "ambos"], default="ambos")
    s.add_argument("--estrategia-caminos-cpu", choices=ESTRATEGIAS_CAMINOS + ["ambas"], default="ambas",
                   help="Caminos por enumeración (misma lógica que la GPU) o combinatoria (sin enumerar)")
    s.add_argument("--hilos", type=int, default=None, help="Hilos de Numba (por defecto todos)")
    s.add_argument("--bloques-por-hilo", type=int, default=8)
    s.add_argument("--reps", type=int, default=3, help="Repeticiones medidas (tras compilar)")
    s.add_argument("--verificar", action="store_true", help="Compara con la referencia Python (oráculo)")
    s.add_argument("--forzar-python", action="store_true", help=f"Permite --verificar con más de {c3.LIMITE_PYTHON:,} caminos")
    s.add_argument("--gpu", action="store_true", help="Mide también el censo completo en GPU en la misma sesión")
    s.add_argument("--json", action="store_true")
    s.add_argument("--salida", default="resultados")
    c3.agregar_args_config(p, " (con --gpu)")
    return p


def main():
    a = crear_parser().parse_args()
    a.reps = max(1, a.reps)
    if a.hilos:
        set_num_threads(a.hilos)
    hilos = get_num_threads()
    cfg_gpu = c3.config_desde_args(a)

    csr, g, info, t_carga = c3.cargar_grafo(a)
    t0 = time.perf_counter()
    calentar()
    t_jit = time.perf_counter() - t0
    pre = c3.preparar(csr)
    prep = pre["prep"]
    W = prep["W"]

    print("\n=== GRAFO ===")
    print(f"{csr.nombre} | nodos {csr.n:,} | aristas {csr.nnz // 2:,} | W = {W:,} caminos | carga {sec.seg(t_carga)}")
    print(f"Hilos Numba: {hilos} | compilación JIT: {sec.seg(t_jit)} | tiempos = media ± desviación de {a.reps}")
    print(f"Preproceso (suma prefija + orientación, en CPU, compartido con la GPU): "
          f"{sec.seg(pre['t_prefijo_s'] + pre['t_orientacion_s'])}")

    modos = {"serial": [False], "paralelo": [True], "ambos": [False, True]}[a.modo]
    estrategias = ESTRATEGIAS_CAMINOS if a.estrategia_caminos_cpu == "ambas" else [a.estrategia_caminos_cpu]
    filas = {}
    resultados = {}
    for paralelo in modos:
        for estrategia in estrategias:
            nombre = f"njit {hilos if paralelo else 1} hilo{'s' if paralelo and hilos > 1 else ''} · {estrategia}"
            r, est, _ = medir(lambda: censo3_njit(csr, None, paralelo=paralelo, estrategia_caminos=estrategia,
                                                  bloques_por_hilo=a.bloques_por_hilo), a.reps)
            resultados[nombre] = r
            filas[nombre] = {"tipo": "njit", "paralelo": paralelo, "estrategia": estrategia,
                             "caminos": est["caminos_s"], "triangulos": est["triangulos_s"], "censo": est["censo_s"],
                             "extremo": est["extremo_s"]}
    ref_nombre = next(iter(resultados))
    ref = resultados[ref_nombre]

    verif = c3.invariantes(ref)
    verif["njit_configuraciones_iguales"] = all(c3.censos_iguales(r, ref) for r in resultados.values())
    if info and info.get("triangulos"):
        verif["T_igual_SNAP"] = ref["T"] == info["triangulos"]
    if a.verificar:
        if W > c3.LIMITE_PYTHON and not a.forzar_python:
            verif["igual_CPU_python"] = f"omitida (W > {c3.LIMITE_PYTHON:,}; usa --forzar-python)"
        else:
            if g is None:
                g = gpu.csr_a_grafo(csr)
            t0 = time.perf_counter()
            no_ind, _ = sec.censo_centro(g)
            t_pc = time.perf_counter() - t0
            t0 = time.perf_counter()
            triangulos, resta = sec.censo_triangulos(g)
            inducido = sec.censo_inducido(no_ind, resta)
            t_pt = time.perf_counter() - t0
            ref_py = {"caminos_no_inducidos": no_ind, "caminos_inducidos": inducido, "triangulos": triangulos}
            uno = lambda t: sec.estadisticas([t])
            filas["Python (referencia)"] = {"tipo": "python", "caminos": uno(t_pc), "triangulos": uno(t_pt),
                                            "censo": uno(t_pc + t_pt), "extremo": uno(t_pc + t_pt)}
            verif["igual_CPU_python"] = c3.censos_iguales(ref_py, ref)

    cfg_usada = None
    if a.gpu:
        if not gpu.cuda.is_available():
            verif["gpu"] = "omitida (sin GPU CUDA)"
        else:
            cfg_usada = c3.completar_memorias(cfg_gpu, prep)
            gpu.cuda.to_device(np.zeros(1))
            gpu.cuda.synchronize()
            kw = dict(max_bloques=a.max_bloques, trabajo_max=a.trabajo_por_lanzamiento, preproceso=a.preproceso)
            c3.censo3(csr, pre, cfg_usada, **kw)
            corridas = [c3.censo3(csr, None, cfg_usada, **kw) for _ in range(a.reps)]
            est = c3.estadisticas_corridas(corridas)
            verif["gpu_igual_njit"] = all(c3.censos_iguales(c, ref) for c in corridas)
            txt_c, txt_t = c3.describir_config(cfg_usada)
            nombre = f"GPU · caminos {txt_c.split(' (')[0]} · tri. {txt_t.split(' (')[0]} · prep. {a.preproceso}"
            filas[nombre] = {"tipo": "gpu", "caminos": est["caminos_total"], "triangulos": est["triangulos_total"],
                             "censo": est["censo"], "extremo": est["extremo"]}
            filas["  (solo cálculo en GPU)"] = {"tipo": "gpu_calculo", "caminos": est["caminos_kernel"],
                                                "triangulos": est["triangulos_kernel"], "censo": est["calculo"],
                                                "extremo": None}

    njit_1 = [f["censo"]["media"] for f in filas.values() if f["tipo"] == "njit" and not f["paralelo"]]
    njit_n = [f["censo"]["media"] for f in filas.values() if f["tipo"] == "njit" and f["paralelo"]]
    ref_1 = min(njit_1) if njit_1 else None
    ref_n = min(njit_n) if njit_n else None
    print("\n=== TIEMPOS ===")
    print("Censo = caminos + triángulos + resta e inducido. Extremo a extremo = censo + preparación (preproceso en CPU; "
          "en GPU, además, carga del grafo y preproceso en GPU).")
    print("Aceleraciones sobre la media del censo, contra el njit más rápido de cada tipo.")
    ancho = max(len(k) for k in filas) + 2
    print(f"{'Método':<{ancho}}{'Caminos':>24}{'Triángulos':>24}{'Censo':>24}{'Extremo a extremo':>24}"
          f"{'vs njit 1 hilo':>16}{f'vs njit {hilos} hilos':>18}")
    for nombre, f in filas.items():
        t = f["censo"]["media"]
        v1 = f"{ref_1 / t:,.1f}x" if ref_1 and t > 0 else "-"
        vn = f"{ref_n / t:,.1f}x" if ref_n and t > 0 else "-"
        ext = sec.seg_est(f["extremo"]) if f["extremo"] else "-"
        print(f"{nombre:<{ancho}}{sec.seg_est(f['caminos']):>24}{sec.seg_est(f['triangulos']):>24}"
              f"{sec.seg_est(f['censo']):>24}{ext:>24}{v1:>16}{vn:>18}")

    print("\n=== TOTALES ===")
    print(f"Caminos no inducidos: {ref['W']:,} | triángulos: {ref['T']:,} | caminos inducidos: {ref['W'] - 3 * ref['T']:,}")

    print("\n=== VERIFICACIÓN ===")
    for k, v in verif.items():
        print(f"{k}: {v}")

    if a.json:
        prefijo = sec.preparar_salida(a.salida, csr.nombre, "njit_")
        sec.guardar_json({
            "grafo": {"nombre": csr.nombre, "nodos": csr.n, "aristas": csr.nnz // 2, "W": ref["W"], "T": ref["T"],
                      "L": prep["L"], "E": prep["E"], "P_caminos": prep["P"],
                      "P_triangulos": tri.espacio_triangulos(prep), "t_carga_s": t_carga},
            "hilos": hilos,
            "reps": a.reps,
            "t_jit_s": t_jit,
            "config_gpu": cfg_usada,
            "preproceso_gpu": a.preproceso if cfg_usada else None,
            "tiempos": {k: {campo: (sec.aplanar("t", e) if e else None) for campo, e in f.items()
                            if campo in ("caminos", "triangulos", "censo", "extremo")} for k, f in filas.items()},
            "verificacion": verif,
        }, prefijo + ".json")
        print(f"\nArchivo generado: {prefijo}.json")


if __name__ == "__main__":
    main()
