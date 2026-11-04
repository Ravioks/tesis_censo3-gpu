import math
import time

import numpy as np
from numba import cuda, int64, njit

import censo_3n_ref as sec
import censo_3n2a_gpu as gpu

BINS_SHARED = gpu.BINS_SHARED
MEMORIAS = gpu.MEMORIAS
ESTRATEGIAS = ["arista", "binaria", "balanceado"]
TRABAJO_POR_LANZAMIENTO = 1_000_000_000
CHUNK_TRI = 256


@njit
def _orientar(offsets, indices, et_arista):
    n = offsets.shape[0] - 1
    cuenta = np.zeros(n + 1, np.int64)
    for u in range(n):
        du = offsets[u + 1] - offsets[u]
        for p in range(offsets[u], offsets[u + 1]):
            v = indices[p]
            dv = offsets[v + 1] - offsets[v]
            if du < dv or (du == dv and u < v):
                cuenta[u + 1] += 1
    off_o = np.cumsum(cuenta)
    m = off_o[n]
    ind_o = np.empty(m, np.int32)
    eta_o = np.empty(m, np.int32)
    fila_o = np.empty(m, np.int32)
    k = 0
    for u in range(n):
        du = offsets[u + 1] - offsets[u]
        for p in range(offsets[u], offsets[u + 1]):
            v = indices[p]
            dv = offsets[v + 1] - offsets[v]
            if du < dv or (du == dv and u < v):
                ind_o[k] = v
                eta_o[k] = et_arista[p]
                fila_o[k] = u
                k += 1
    return off_o, ind_o, eta_o, fila_o


@njit
def _costos(off_o, ind_o, fila_o):
    m = ind_o.shape[0]
    acum = np.zeros(m + 1, np.int64)
    acum_bin = np.zeros(m + 1, np.int64)
    cota = 0
    for p in range(m):
        du = off_o[fila_o[p] + 1] - off_o[fila_o[p]]
        dv = off_o[ind_o[p] + 1] - off_o[ind_o[p]]
        corto = min(du, dv)
        largo = max(du, dv)
        pasos = 1
        while (1 << pasos) <= largo:
            pasos += 1
        acum[p + 1] = acum[p] + du + dv
        acum_bin[p + 1] = acum_bin[p] + corto * pasos + 1
        cota += corto
    return acum, acum_bin, cota


def orientar(csr):
    t0 = time.perf_counter()
    off_o, ind_o, eta_o, fila_o = _orientar(csr.offsets, csr.indices, csr.et_arista)
    if not gpu.filas_ordenadas(off_o, ind_o):
        orden = np.lexsort((ind_o, fila_o))
        ind_o, eta_o = ind_o[orden], eta_o[orden]
    acum, acum_bin, cota = _costos(off_o, ind_o, fila_o)
    grado_salida = np.diff(off_o)
    return {
        "off_o": off_o,
        "ind_o": ind_o,
        "eta_o": eta_o,
        "fila_o": fila_o,
        "acum_costo": acum,
        "acum_binaria": acum_bin,
        "total_costo": int(acum[-1]),
        "total_binaria": int(acum_bin[-1]),
        "cota_triangulos": int(cota),
        "aristas": int(len(ind_o)),
        "grado_salida_max": int(grado_salida.max()) if len(grado_salida) else 0,
        "en_gpu": False,
        "t_s": time.perf_counter() - t0,
    }


@cuda.jit
def _k_contar_salida(offsets, indices, n, cuenta, maximo):
    local_max = int64(0)
    for u in range(cuda.grid(1), n, cuda.gridsize(1)):
        du = offsets[u + 1] - offsets[u]
        c = int64(0)
        for p in range(offsets[u], offsets[u + 1]):
            v = indices[p]
            dv = offsets[v + 1] - offsets[v]
            if du < dv or (du == dv and u < v):
                c += 1
        cuenta[u] = c
        if c > local_max:
            local_max = c
    cuda.atomic.max(maximo, 0, local_max)


@cuda.jit
def _k_llenar_orientado(offsets, indices, et_arista, off_o, n, ind_o, eta_o, fila_o):
    for u in range(cuda.grid(1), n, cuda.gridsize(1)):
        du = offsets[u + 1] - offsets[u]
        k = off_o[u]
        for p in range(offsets[u], offsets[u + 1]):
            v = indices[p]
            dv = offsets[v + 1] - offsets[v]
            if du < dv or (du == dv and u < v):
                ind_o[k] = v
                eta_o[k] = et_arista[p]
                fila_o[k] = u
                k += 1


@cuda.jit
def _k_costos(off_o, ind_o, fila_o, m, costo, cota):
    local = int64(0)
    for p in range(cuda.grid(1), m, cuda.gridsize(1)):
        u = fila_o[p]
        v = ind_o[p]
        du = off_o[u + 1] - off_o[u]
        dv = off_o[v + 1] - off_o[v]
        costo[p] = du + dv
        local += min(du, dv)
    cuda.atomic.add(cota, 0, local)


@cuda.jit
def _k_costos_binaria(off_o, ind_o, fila_o, m, costo):
    for p in range(cuda.grid(1), m, cuda.gridsize(1)):
        u = fila_o[p]
        v = ind_o[p]
        du = off_o[u + 1] - off_o[u]
        dv = off_o[v + 1] - off_o[v]
        corto = min(du, dv)
        largo = max(du, dv)
        pasos = 1
        while (1 << pasos) <= largo:
            pasos += 1
        costo[p] = corto * pasos + 1


def _acumulado_gpu(kernel, orient, m, bloque, max_bloques, *extra):
    d_acum = cuda.device_array(m + 1, np.int64)
    gpu.escribir(d_acum, 0, 0)
    if m:
        bloques, hilos = gpu.configuracion_lanzamiento(m, bloque, max_bloques)
        kernel[bloques, hilos](orient["off_o"], orient["ind_o"], orient["fila_o"], m, d_acum[1:], *extra)
        gpu.scan_inclusivo(d_acum[1:], d_acum[1:], m)
    return d_acum


def orientar_gpu(csr, disp, binaria=False, bloque=256, max_bloques=4096):
    cuda.synchronize()
    t0 = time.perf_counter()
    n = csr.n
    bloques, hilos = gpu.configuracion_lanzamiento(n, bloque, max_bloques)
    d_cuenta = cuda.device_array(max(1, n), np.int64)
    d_max = cuda.to_device(np.zeros(1, np.int64))
    _k_contar_salida[bloques, hilos](disp["offsets"], disp["indices"], n, d_cuenta, d_max)
    d_off_o = gpu.prefijo_exclusivo(d_cuenta, n)
    del d_cuenta
    m = gpu.leer_escalar(d_off_o, n)
    orient = {"off_o": d_off_o, "ind_o": cuda.device_array(max(1, m), np.int32),
              "eta_o": cuda.device_array(max(1, m), np.int32), "fila_o": cuda.device_array(max(1, m), np.int32),
              "aristas": m, "en_gpu": True, "acum_binaria": None, "total_binaria": None}
    _k_llenar_orientado[bloques, hilos](disp["offsets"], disp["indices"], disp["et_arista"], d_off_o, n,
                                        orient["ind_o"], orient["eta_o"], orient["fila_o"])
    d_cota = cuda.to_device(np.zeros(1, np.int64))
    orient["acum_costo"] = _acumulado_gpu(_k_costos, orient, m, bloque, max_bloques, d_cota)
    orient["total_costo"] = gpu.leer_escalar(orient["acum_costo"], m)
    orient["cota_triangulos"] = int(d_cota.copy_to_host()[0])
    orient["grado_salida_max"] = int(d_max.copy_to_host()[0])
    if binaria:
        asegurar_binaria(orient, bloque, max_bloques)
    cuda.synchronize()
    orient["t_s"] = time.perf_counter() - t0
    return orient


def asegurar_binaria(orient, bloque=256, max_bloques=4096):
    if orient.get("acum_binaria") is None:
        m = orient["aristas"]
        orient["acum_binaria"] = _acumulado_gpu(_k_costos_binaria, orient, m, bloque, max_bloques)
        orient["total_binaria"] = gpu.leer_escalar(orient["acum_binaria"], m)
    return orient


def subir_orientacion(orient, acum_en_gpu=False):
    subida = dict(orient)
    for clave in ("off_o", "ind_o", "eta_o", "fila_o"):
        subida[clave] = gpu.a_dispositivo(orient[clave])
    if acum_en_gpu:
        subida["acum_costo"] = gpu.a_dispositivo(orient["acum_costo"])
    subida["en_gpu"] = True
    return subida


def calentar():
    orientar(gpu.grafo_a_csr(sec.generar_ejemplo()))


def espacio_triangulos(prep):
    PT = prep["B"] ** 3
    if PT >= 1 << 62:
        raise ValueError("Demasiadas etiquetas para codificar el triángulo en 64 bits")
    return PT


def motivo_invalida(estrategia, memoria, prep):
    return gpu.motivo_memoria(memoria, espacio_triangulos(prep))


def config_valida(estrategia, memoria, prep):
    return motivo_invalida(estrategia, memoria, prep) is None


def memoria_por_defecto(prep):
    return gpu.memoria_por_defecto(espacio_triangulos(prep))


@cuda.jit(device=True, inline=True)
def id_triangulo(t0, t1, t2, B):
    if t0 > t1:
        t0, t1 = t1, t0
    if t1 > t2:
        t1, t2 = t2, t1
    if t0 > t1:
        t0, t1 = t1, t0
    return (int64(t0) * B + t1) * B + t2


@cuda.jit(device=True)
def acumular(pid, ultimo, cant, modo, hist, sh, claves, valores):
    if pid == ultimo:
        return ultimo, cant + 1
    gpu.volcar(ultimo, cant, modo, hist, sh, claves, valores)
    return pid, int64(1)


@cuda.jit(device=True)
def mezclar(p, off_o, ind_o, eta_o, fila_o, et_nodo, L, B, modo, hist, sh, claves, valores, ultimo, cant):
    u = fila_o[p]
    v = ind_o[p]
    e_uv = eta_o[p]
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
            pid = id_triangulo(eta_o[j] * L + lu, eta_o[i] * L + lv, e_uv * L + et_nodo[a], B)
            ultimo, cant = acumular(pid, ultimo, cant, modo, hist, sh, claves, valores)
            i += 1
            j += 1
    return ultimo, cant


@cuda.jit(device=True)
def buscar(p, off_o, ind_o, eta_o, fila_o, et_nodo, L, B, modo, hist, sh, claves, valores, ultimo, cant):
    u = fila_o[p]
    v = ind_o[p]
    e_uv = eta_o[p]
    lu = et_nodo[u]
    lv = et_nodo[v]
    iu = off_o[u]
    fu = off_o[u + 1]
    iv = off_o[v]
    fv = off_o[v + 1]
    corto_es_u = fu - iu <= fv - iv
    if corto_es_u:
        ic, fc, il, fl = iu, fu, iv, fv
    else:
        ic, fc, il, fl = iv, fv, iu, fu
    for s in range(ic, fc):
        w = ind_o[s]
        lo = il
        hi = fl
        while lo < hi:
            mid = (lo + hi) // 2
            if ind_o[mid] < w:
                lo = mid + 1
            else:
                hi = mid
        il = lo
        if lo < fl and ind_o[lo] == w:
            if corto_es_u:
                e_uw = eta_o[s]
                e_vw = eta_o[lo]
            else:
                e_vw = eta_o[s]
                e_uw = eta_o[lo]
            pid = id_triangulo(e_vw * L + lu, e_uw * L + lv, e_uv * L + et_nodo[w], B)
            ultimo, cant = acumular(pid, ultimo, cant, modo, hist, sh, claves, valores)
            il = lo + 1
    return ultimo, cant


def _fabricar_kernels(modo):
    tam_sh = BINS_SHARED if modo == 1 else 1

    @cuda.jit
    def k_arista(off_o, ind_o, eta_o, fila_o, et_nodo, inicio, fin, L, B, PT, hist, claves, valores):
        sh = cuda.shared.array(tam_sh, int64)
        gpu.iniciar_shared(sh, PT, modo)
        ultimo = int64(-1)
        cant = int64(0)
        for p in range(inicio + cuda.grid(1), fin, cuda.gridsize(1)):
            ultimo, cant = mezclar(p, off_o, ind_o, eta_o, fila_o, et_nodo, L, B, modo, hist, sh, claves, valores,
                                   ultimo, cant)
        gpu.volcar(ultimo, cant, modo, hist, sh, claves, valores)
        gpu.vaciar_shared(sh, PT, modo, hist)

    @cuda.jit
    def k_binaria(off_o, ind_o, eta_o, fila_o, et_nodo, inicio, fin, L, B, PT, hist, claves, valores):
        sh = cuda.shared.array(tam_sh, int64)
        gpu.iniciar_shared(sh, PT, modo)
        ultimo = int64(-1)
        cant = int64(0)
        for p in range(inicio + cuda.grid(1), fin, cuda.gridsize(1)):
            ultimo, cant = buscar(p, off_o, ind_o, eta_o, fila_o, et_nodo, L, B, modo, hist, sh, claves, valores,
                                  ultimo, cant)
        gpu.volcar(ultimo, cant, modo, hist, sh, claves, valores)
        gpu.vaciar_shared(sh, PT, modo, hist)

    @cuda.jit
    def k_balanceado(off_o, ind_o, eta_o, fila_o, et_nodo, acum, m, chunk, it_inicio, it_fin, L, B, PT,
                     hist, claves, valores):
        sh = cuda.shared.array(tam_sh, int64)
        gpu.iniciar_shared(sh, PT, modo)
        ultimo = int64(-1)
        cant = int64(0)
        for it in range(it_inicio + cuda.grid(1), it_fin, cuda.gridsize(1)):
            c0 = it * chunk
            c1 = c0 + chunk
            lo = 0
            hi = m
            while lo < hi:
                mid = (lo + hi) // 2
                if acum[mid] < c0:
                    lo = mid + 1
                else:
                    hi = mid
            p = lo
            while p < m and acum[p] < c1:
                ultimo, cant = mezclar(p, off_o, ind_o, eta_o, fila_o, et_nodo, L, B, modo, hist, sh, claves, valores,
                                       ultimo, cant)
                p += 1
        gpu.volcar(ultimo, cant, modo, hist, sh, claves, valores)
        gpu.vaciar_shared(sh, PT, modo, hist)

    return {"arista": k_arista, "binaria": k_binaria, "balanceado": k_balanceado}


_KERNELS = {}


def kernels(memoria):
    if memoria not in _KERNELS:
        _KERNELS[memoria] = _fabricar_kernels(MEMORIAS[memoria])
    return _KERNELS[memoria]


def lotes(estrategia, orient, chunk, trabajo_max):
    m = orient["aristas"]
    if estrategia == "balanceado":
        nit = (orient["total_costo"] + chunk - 1) // chunk
        paso = max(1, trabajo_max // chunk)
        return [(i, min(i + paso, nit)) for i in range(0, nit, paso)]
    if estrategia == "binaria":
        asegurar_binaria(orient)
        acum, total = orient["acum_binaria"], orient["total_binaria"]
    else:
        acum, total = orient["acum_costo"], orient["total_costo"]
    if total <= trabajo_max:
        cortes = np.array([0, m], dtype=np.int64)
    else:
        objetivos = np.arange(trabajo_max, total, trabajo_max, dtype=np.int64)
        cortes = np.unique(np.concatenate([[0], gpu.buscar_cortes(acum, objetivos), [m]]))
    return [(int(a), int(b)) for a, b in zip(cortes[:-1], cortes[1:]) if b > a] or [(0, 0)]


def capacidad(prep, orient):
    return gpu.capacidad_hash(orient["cota_triangulos"], espacio_triangulos(prep))


def bytes_necesarios(csr, prep, orient, estrategia, memoria, et_nodo_en_gpu=False):
    def en_host(a):
        return 0 if a is None or gpu.en_gpu(a) else a.nbytes

    total = sum(en_host(orient[k]) for k in ("off_o", "ind_o", "eta_o", "fila_o"))
    total += 0 if et_nodo_en_gpu else csr.et_nodo.nbytes
    if estrategia == "balanceado":
        total += en_host(orient["acum_costo"])
    if MEMORIAS[memoria] < 2:
        total += 8 * espacio_triangulos(prep)
    else:
        total += 16 * capacidad(prep, orient)
    return total


def hilos_teoricos(orient, estrategia, bloque, max_bloques, chunk):
    items = orient["aristas"]
    if estrategia == "balanceado":
        items = (orient["total_costo"] + chunk - 1) // chunk
    return max(1, min(math.ceil(items / bloque), max_bloques)) * bloque


def decodificar_triangulos(pids, cuentas, csr, prep):
    L, B = prep["L"], prep["B"]
    nn, na = csr.nombres_nodo, csr.nombres_arista
    censo = {}
    for pid, c in zip(pids.tolist(), cuentas.tolist()):
        t0, resto = divmod(pid, B * B)
        t1, t2 = divmod(resto, B)
        censo[tuple(sorted((na[t // L], nn[t % L]) for t in (t0, t1, t2)))] = c
    return censo


def ejecutar_triangulos_gpu(csr, prep, orient, estrategia, memoria, bloque=256, max_bloques=4096, chunk=CHUNK_TRI,
                            trabajo_max=TRABAJO_POR_LANZAMIENTO, et_nodo=None):
    motivo = motivo_invalida(estrategia, memoria, prep)
    if motivo:
        raise ValueError(f"Configuración {estrategia} + {memoria} no válida para triángulos: {motivo}")
    t0 = time.perf_counter()
    rangos = lotes(estrategia, orient, chunk, trabajo_max)
    t_preparacion = time.perf_counter() - t0
    k = kernels(memoria)[estrategia]
    modo = MEMORIAS[memoria]
    L, B = prep["L"], prep["B"]
    PT = espacio_triangulos(prep)
    m = orient["aristas"]

    t0 = time.perf_counter()
    d_off = gpu.a_dispositivo(orient["off_o"])
    d_ind = gpu.a_dispositivo(orient["ind_o"])
    d_eta = gpu.a_dispositivo(orient["eta_o"])
    d_fila = gpu.a_dispositivo(orient["fila_o"])
    d_etn = et_nodo if et_nodo is not None else gpu.a_dispositivo(csr.et_nodo)
    if estrategia == "balanceado":
        d_acum = gpu.a_dispositivo(orient["acum_costo"])
    d_hist = cuda.to_device(np.zeros(PT if modo < 2 else 1, dtype=np.int64))
    cap = capacidad(prep, orient) if modo == 2 else 1
    d_claves = cuda.to_device(np.full(cap, gpu.VACIO, dtype=np.int64))
    d_valores = cuda.to_device(np.zeros(cap, dtype=np.int64))
    cuda.synchronize()
    t_h2d = time.perf_counter() - t0

    hilos = 0
    t0 = time.perf_counter()
    for inicio, fin in rangos:
        bloques = max(1, min(math.ceil((fin - inicio) / bloque), max_bloques))
        hilos = max(hilos, bloques * bloque)
        if estrategia == "balanceado":
            k[bloques, bloque](d_off, d_ind, d_eta, d_fila, d_etn, d_acum, m, chunk, inicio, fin, L, B, PT,
                               d_hist, d_claves, d_valores)
        else:
            k[bloques, bloque](d_off, d_ind, d_eta, d_fila, d_etn, inicio, fin, L, B, PT, d_hist, d_claves, d_valores)
    cuda.synchronize()
    t_kernel = time.perf_counter() - t0

    t0 = time.perf_counter()
    if modo < 2:
        hist = d_hist.copy_to_host()
        pids = np.nonzero(hist)[0]
        cuentas = hist[pids]
    else:
        claves = d_claves.copy_to_host()
        valores = d_valores.copy_to_host()
        ocupadas = claves != gpu.VACIO
        pids, cuentas = claves[ocupadas], valores[ocupadas]
    triangulos = decodificar_triangulos(pids, cuentas, csr, prep)
    t_d2h = time.perf_counter() - t0

    return {
        "estrategia": estrategia,
        "memoria": memoria,
        "triangulos": triangulos,
        "T": int(cuentas.sum()),
        "hilos": max(hilos, 1),
        "lanzamientos": len(rangos),
        "cap_hash": cap if modo == 2 else None,
        "t_preparacion": t_preparacion,
        "t_h2d": t_h2d,
        "t_kernel": t_kernel,
        "t_d2h": t_d2h,
    }
