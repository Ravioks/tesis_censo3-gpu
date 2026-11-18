import gc
import os
import time

import numpy as np

import censo_3n as c3
import censo_3n2a_gpu as gpu
import censo_3n3a_gpu as tri

cuda = gpu.cuda
TAM_TROZO = 1 << 25


def modulo_arreglos():
    if os.environ.get("NUMBA_ENABLE_CUDASIM") == "1" or os.environ.get("CENSO_XP") == "numpy":
        return np
    import cupy
    return cupy


def usar_rmm_en_cupy():
    try:
        import cupy
        from rmm.allocators.cupy import rmm_cupy_allocator
        cupy.cuda.set_allocator(rmm_cupy_allocator)
        return True
    except Exception:
        return False


def liberar():
    gc.collect()
    gpu.liberar_gpu()
    xp = modulo_arreglos()
    if xp is not np:
        try:
            xp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass


def sincronizar():
    cuda.synchronize()
    xp = modulo_arreglos()
    if xp is not np:
        xp.cuda.Device().synchronize()


def columna(xp, s):
    return xp.asarray(getattr(s, "values", s))


def a_host(x):
    return x.get() if hasattr(x, "get") and not isinstance(x, np.ndarray) else np.asarray(x)


def a_numba(x):
    return cuda.as_cuda_array(x) if hasattr(x, "__cuda_array_interface__") else cuda.to_device(np.ascontiguousarray(x))


def puntero(x):
    try:
        return int(x.__cuda_array_interface__["data"][0])
    except Exception:
        try:
            return int(x.__array_interface__["data"][0])
        except Exception:
            return None


def bits(k):
    return max(1, int(k - 1).bit_length()) if k > 1 else 1


def a_entero(xp, x, dtype):
    if x.dtype.kind == "f":
        x = xp.rint(x)
    return x.astype(dtype, copy=False)


def desempaquetar(xp, clave, corrimiento, mascara, dtype=np.int32):
    salida = xp.empty(clave.shape[0], dtype=dtype)
    for i in range(0, clave.shape[0], TAM_TROZO):
        salida[i:i + TAM_TROZO] = ((clave[i:i + TAM_TROZO] >> corrimiento) & mascara).astype(dtype)
    return salida


def ordenar_filas(xp, fila, ind, eta, n, E):
    bn, be = bits(n), bits(E)
    if 2 * bn + be <= 63:
        clave = (fila.astype(np.int64) << (bn + be)) | (ind.astype(np.int64) << be) | eta.astype(np.int64)
        clave.sort()
        ind = desempaquetar(xp, clave, be, (1 << bn) - 1)
        eta = desempaquetar(xp, clave, 0, (1 << be) - 1)
        del clave
        return ind, eta
    orden = xp.lexsort(xp.stack([eta, ind, fila]))
    return ind[orden], eta[orden]


def revisar(xp, fila, ind):
    if ind.shape[0] == 0:
        return {"desordenadas": 0, "repetidas": 0, "lazos": 0, "simetrico": True}
    misma = fila[1:] == fila[:-1]
    return {
        "desordenadas": int(xp.count_nonzero(misma & (ind[1:] < ind[:-1]))),
        "repetidas": int(xp.count_nonzero(misma & (ind[1:] == ind[:-1]))),
        "lazos": int(xp.count_nonzero(ind == fila)),
        "simetrico": int(xp.count_nonzero(ind > fila)) == int(xp.count_nonzero(ind < fila)),
    }


def desde_arreglos(offsets, indices, et_nodo, et_arista, nombres_nodo, nombres_arista, nombre="residente",
                   max_bloques=4096):
    xp = modulo_arreglos()
    sincronizar()
    t0 = time.perf_counter()
    off_in, ind_in = columna(xp, offsets), columna(xp, indices)
    etn_in = columna(xp, et_nodo)
    n = int(etn_in.shape[0])
    off = off_in.astype(np.int64, copy=False)
    nnz = int(a_host(off[-1:])[0])
    if off.shape[0] - 1 < n:
        off = xp.concatenate([off, xp.full(n + 1 - off.shape[0], nnz, dtype=np.int64)])
    elif off.shape[0] - 1 > n:
        raise ValueError(f"El grafo tiene {off.shape[0] - 1:,} vértices y hay {n:,} etiquetas de nodo")
    ind = ind_in.astype(np.int32, copy=False)
    etn = a_entero(xp, etn_in, np.int32)
    if et_arista is None:
        eta_in = None
        eta = xp.zeros(nnz, dtype=np.int32)
    else:
        eta_in = columna(xp, et_arista)
        eta = a_entero(xp, eta_in, np.int32)
    L, E = len(nombres_nodo), len(nombres_arista)
    if nnz:
        if int(eta.min()) < 0 or int(eta.max()) >= E:
            raise ValueError(f"Etiquetas de arista fuera de [0, {E})")
    if n and (int(etn.min()) < 0 or int(etn.max()) >= L):
        raise ValueError(f"Etiquetas de nodo fuera de [0, {L})")

    d_off = a_numba(off)
    d_fila = gpu.fila_en_gpu({"offsets": d_off}, n, nnz, max_bloques=max_bloques)
    fila = xp.asarray(d_fila) if xp is not np else d_fila.copy_to_host()
    rev = revisar(xp, fila, ind)
    if rev["lazos"]:
        raise ValueError(f"El grafo tiene {rev['lazos']:,} lazos; el censo supone un grafo simple")
    if not rev["simetrico"]:
        raise ValueError("La lista de adyacencia no es simétrica (grafo no dirigido con ambas direcciones)")
    ordenado = False
    if rev["desordenadas"]:
        ind, eta = ordenar_filas(xp, fila, ind, eta, n, E)
        ordenado = True
        rev_final = revisar(xp, fila, ind)
        rev["repetidas"] = rev_final["repetidas"]
    if rev["repetidas"]:
        raise ValueError(f"El grafo tiene {rev['repetidas']:,} aristas repetidas; el censo supone un grafo simple")

    off_host = a_host(off)
    sin_copia = {"offsets": puntero(off) == puntero(off_in), "indices": puntero(ind) == puntero(ind_in),
                 "et_nodo": puntero(etn) == puntero(etn_in),
                 "et_arista": eta_in is not None and puntero(eta) == puntero(eta_in)}
    disp = {"offsets": d_off, "indices": a_numba(ind), "et_nodo": a_numba(etn), "et_arista": a_numba(eta)}
    sincronizar()
    t = time.perf_counter() - t0
    csr = gpu.CSR(off_host, None, None, None, list(nombres_nodo), list(nombres_arista), nombre)
    return {"csr": csr, "disp": disp, "fila": d_fila, "vivos": [off, ind, etn, eta, fila],
            "info": {"n": n, "nnz": nnz, "reordenado": ordenado, "sin_copia": sin_copia, **rev},
            "t_adaptador_s": t}


def csr_simetrico(xp, a, b, e, n, E):
    k = int(a.shape[0])
    if k and (int(e.min()) < 0 or int(e.max()) >= E):
        raise ValueError(f"Etiquetas de arista fuera de [0, {E})")
    if k and (min(int(a.min()), int(b.min())) < 0 or max(int(a.max()), int(b.max())) >= n):
        raise ValueError(f"IDs de vértice fuera de [0, {n})")
    bn, be = bits(n), bits(E)
    if 2 * bn + be > 63:
        raise ValueError("Demasiados vértices para empaquetar la clave de orden en 64 bits")
    s, mn, me = bn + be, (1 << bn) - 1, (1 << be) - 1
    centinela = np.iinfo(np.int64).max
    clave = xp.empty(k, dtype=np.int64)
    for i in range(0, k, TAM_TROZO):
        j = min(i + TAM_TROZO, k)
        aa, bb = a[i:j].astype(np.int64), b[i:j].astype(np.int64)
        c = (xp.minimum(aa, bb) << s) | (xp.maximum(aa, bb) << be) | e[i:j].astype(np.int64)
        clave[i:j] = xp.where(aa == bb, centinela, c)
    clave.sort()
    validas = k - int(xp.count_nonzero(clave == centinela))
    lazos = k - validas
    clave = clave[:validas]
    repetidas = 0
    if validas > 1:
        nueva = xp.empty(validas, dtype=bool)
        nueva[0] = True
        for i in range(1, validas, TAM_TROZO):
            j = min(i + TAM_TROZO, validas)
            nueva[i:j] = (clave[i:j] >> be) != (clave[i - 1:j - 1] >> be)
        repetidas = validas - int(xp.count_nonzero(nueva))
        if repetidas:
            clave = clave[nueva]
        del nueva
    m = int(clave.shape[0])
    sim = xp.empty(2 * m, dtype=np.int64)
    for i in range(0, m, TAM_TROZO):
        j = min(i + TAM_TROZO, m)
        c = clave[i:j]
        uu, vv, ee = c >> s, (c >> be) & mn, c & me
        sim[i:j] = (uu << s) | (vv << be) | ee
        sim[m + i:m + j] = (vv << s) | (uu << be) | ee
    del clave
    sim.sort()
    off = xp.searchsorted(sim, xp.arange(n + 1, dtype=np.int64) << s)
    ind = desempaquetar(xp, sim, be, mn)
    eta = desempaquetar(xp, sim, 0, me)
    del sim
    return off, ind, eta, {"lazos_quitados": lazos, "repetidas_quitadas": repetidas, "aristas": m}


def desde_cudf(df, et_nodo, nombres_nodo, nombres_arista, origen="src", destino="dst", etiqueta=None,
               nombre="cudf", max_bloques=4096):
    xp = modulo_arreglos()
    sincronizar()
    t0 = time.perf_counter()
    a = columna(xp, df[origen])
    b = columna(xp, df[destino])
    e = a_entero(xp, columna(xp, df[etiqueta]), np.int32) if etiqueta else xp.zeros(a.shape[0], dtype=np.int32)
    n = int(columna(xp, et_nodo).shape[0])
    off, ind, eta, info = csr_simetrico(xp, a, b, e, n, len(nombres_arista))
    del a, b, e
    sincronizar()
    t_construccion = time.perf_counter() - t0
    entrada = desde_arreglos(off, ind, et_nodo, eta, nombres_nodo, nombres_arista, nombre, max_bloques)
    entrada["info"].update({**info, "t_construccion_csr_s": t_construccion})
    entrada["t_adaptador_s"] += t_construccion
    return entrada


def vista_adyacencia(G):
    try:
        return G.view_adj_list()
    except AttributeError:
        return G._Impl.view_adj_list()


def renumerado(G):
    for nombre in ("is_renumbered", "renumbered"):
        valor = getattr(G, nombre, None)
        if callable(valor):
            try:
                return bool(valor())
            except Exception:
                continue
        if valor is not None:
            return bool(valor)
    return False


def desde_cugraph(G, et_nodo, nombres_nodo, nombres_arista, nombre="cugraph", max_bloques=4096):
    if renumerado(G):
        raise ValueError("El grafo de cuGraph está renumerado: constrúyelo con renumber=False e IDs en [0, n)")
    sincronizar()
    t0 = time.perf_counter()
    off, ind, val = vista_adyacencia(G)
    sincronizar()
    t_vista = time.perf_counter() - t0
    xp = modulo_arreglos()
    t0 = time.perf_counter()
    n = int(columna(xp, et_nodo).shape[0])
    o = columna(xp, off).astype(np.int64, copy=False)
    nnz = int(a_host(o[-1:])[0])
    if o.shape[0] - 1 < n:
        o = xp.concatenate([o, xp.full(n + 1 - o.shape[0], nnz, dtype=np.int64)])
    i = columna(xp, ind)
    d_fila = gpu.fila_en_gpu({"offsets": a_numba(o)}, n, nnz, max_bloques=max_bloques)
    fila = xp.asarray(d_fila) if xp is not np else d_fila.copy_to_host()
    simetrica = int(xp.count_nonzero(i > fila)) == int(xp.count_nonzero(i < fila))
    if not simetrica:
        e = a_entero(xp, columna(xp, val), np.int32) if val is not None else xp.zeros(nnz, dtype=np.int32)
        off, ind, val, info = csr_simetrico(xp, fila, i, e, n, len(nombres_arista))
        del e
    del fila, d_fila, o, i
    sincronizar()
    t_simetrizar = time.perf_counter() - t0
    entrada = desde_arreglos(off, ind, et_nodo, val, nombres_nodo, nombres_arista, nombre, max_bloques)
    entrada["info"].update({"t_vista_s": t_vista, "simetrizado": not simetrica, "t_simetrizar_s": t_simetrizar})
    if not simetrica:
        entrada["info"].update(info)
    entrada["t_adaptador_s"] += t_vista + t_simetrizar
    return entrada


def preparar_residente(entrada, estrategias_caminos, estrategias_tri, max_bloques=4096):
    csr = entrada["csr"]
    pre = c3.preparar(csr, orientar=False)
    disp = dict(entrada["disp"])
    t_carga = 0.0
    if "balanceado" in estrategias_caminos:
        sincronizar()
        t0 = time.perf_counter()
        disp["acum"] = cuda.to_device(pre["prep"]["acum"])
        sincronizar()
        t_carga = time.perf_counter() - t0
    t0 = time.perf_counter()
    if "entrada" in estrategias_caminos:
        disp["fila"] = entrada["fila"]
    orient = tri.orientar_gpu(csr, disp, binaria="binaria" in estrategias_tri, max_bloques=max_bloques)
    sincronizar()
    t_gpu = time.perf_counter() - t0
    dispositivo = {"disp": disp, "orient": orient, "preproceso": "gpu",
                   "tiempos": {"preproceso_cpu_s": pre["t_prefijo_s"], "carga_s": t_carga, "preproceso_gpu_s": t_gpu}}
    return pre, dispositivo


def censo3_residente(entrada, config=None, max_bloques=4096, trabajo_max=gpu.TRABAJO_POR_LANZAMIENTO, **cambios):
    cfg = c3.resolver_config(config, **cambios)
    pre, dispositivo = preparar_residente(entrada, [cfg["estrategia_caminos"]], [cfg["estrategia_tri"]], max_bloques)
    res = c3.censo3(entrada["csr"], pre=pre, config=cfg, max_bloques=max_bloques, trabajo_max=trabajo_max,
                    dispositivo=dispositivo)
    t = res["tiempos"]
    td = dispositivo["tiempos"]
    t["adaptador_s"] = entrada["t_adaptador_s"]
    t["preproceso_cpu_s"] = td["preproceso_cpu_s"]
    t["carga_s"] = td["carga_s"]
    t["preproceso_gpu_s"] = td["preproceso_gpu_s"]
    t["preparacion_s"] = t["adaptador_s"] + t["preproceso_cpu_s"] + t["carga_s"] + t["preproceso_gpu_s"]
    t["extremo_s"] = t["preparacion_s"] + t["censo_s"]
    t["preproceso_incluido"] = True
    res["preproceso"] = "residente"
    res["adaptador"] = dict(entrada["info"])
    del dispositivo
    liberar()
    return res
