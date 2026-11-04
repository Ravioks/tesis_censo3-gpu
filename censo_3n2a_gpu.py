import argparse
import gc
import math
import os
import time

import numpy as np
from numba import cuda, int32, int64, njit

import censo_3n2a as sec

VACIO = -1
BINS_SHARED = 2048
LIMITE_DENSO = 1 << 24
LIMITE_VERIFICAR_IDS = 5_000_000
LIMITE_TRABAJO_HILO = 50_000_000
MAX_TIPOS_COMBINATORIO = 128
TRABAJO_POR_LANZAMIENTO = 1_000_000_000
MEMORIAS = {"global": 0, "shared": 1, "hash": 2}
ESTRATEGIAS = ["nodo", "entrada", "balanceado", "combinatorio"]


class CSR:
    def __init__(self, offsets, indices, et_nodo, et_arista, nombres_nodo, nombres_arista, nombre="csr", id_arista=None):
        self.offsets = offsets
        self.indices = indices
        self.et_nodo = et_nodo
        self.et_arista = et_arista
        self.nombres_nodo = nombres_nodo
        self.nombres_arista = nombres_arista
        self.nombre = nombre
        self._id_arista = id_arista

    @property
    def n(self):
        return len(self.offsets) - 1

    @property
    def nnz(self):
        return int(self.offsets[-1])

    @property
    def id_arista(self):
        if self._id_arista is None:
            self._id_arista = calcular_id_arista(self.offsets, self.indices)
        return self._id_arista


def calcular_id_arista(offsets, indices):
    n = len(offsets) - 1
    fila = np.repeat(np.arange(n, dtype=np.int64), np.diff(offsets))
    col = indices.astype(np.int64)
    if np.any(fila == col):
        raise ValueError("El CSR contiene lazos (u == v)")
    clave = fila * n + col
    orden = np.argsort(clave, kind="stable")
    ordenada = clave[orden]
    if np.any(np.diff(ordenada) == 0):
        raise ValueError("El CSR contiene aristas repetidas")
    inversa = col * n + fila
    pos = np.searchsorted(ordenada, inversa)
    if np.any(pos >= len(ordenada)) or np.any(ordenada[np.minimum(pos, len(ordenada) - 1)] != inversa):
        raise ValueError("El CSR no es simétrico: falta la copia inversa de alguna arista")
    pos = orden[pos]
    ids = np.full(len(indices), -1, dtype=np.int64)
    superior = fila < col
    ids[superior] = np.arange(int(superior.sum()), dtype=np.int64)
    ids[~superior] = ids[pos[~superior]]
    return ids


def grafo_a_csr(g):
    nodos = list(g.etiqueta_nodo)
    pos = {nid: i for i, nid in enumerate(nodos)}
    nombres_nodo = sorted({str(e) for e in g.etiqueta_nodo.values()})
    nombres_arista = sorted({str(e[2]) for e in g.aristas}) or ["-"]
    idx_n = {e: i for i, e in enumerate(nombres_nodo)}
    idx_a = {e: i for i, e in enumerate(nombres_arista)}
    grados = np.array([len(g.adyacencia[v]) for v in nodos], dtype=np.int64)
    offsets = np.zeros(len(nodos) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(grados)
    indices = np.empty(int(offsets[-1]), dtype=np.int32)
    et_arista = np.empty(int(offsets[-1]), dtype=np.int32)
    id_arista = np.empty(int(offsets[-1]), dtype=np.int64)
    for i, v in enumerate(nodos):
        o = int(offsets[i])
        for k, (w, eid) in enumerate(sorted((pos[w], eid) for w, eid in g.adyacencia[v])):
            indices[o + k] = w
            et_arista[o + k] = idx_a[str(g.aristas[eid][2])]
            id_arista[o + k] = eid
    et_nodo = np.array([idx_n[str(g.etiqueta_nodo[v])] for v in nodos], dtype=np.int32)
    return CSR(offsets, indices, et_nodo, et_arista, nombres_nodo, nombres_arista, g.nombre, id_arista)


def csr_a_grafo(csr):
    g = sec.Grafo(csr.nombre)
    for u in range(csr.n):
        g.agregar_nodo(u, csr.nombres_nodo[csr.et_nodo[u]])
    for u in range(csr.n):
        for p in range(csr.offsets[u], csr.offsets[u + 1]):
            v = int(csr.indices[p])
            if u < v:
                g.agregar_arista(u, v, csr.nombres_arista[csr.et_arista[p]])
    return g


@njit
def filas_ordenadas(offsets, indices):
    for c in range(len(offsets) - 1):
        for p in range(offsets[c] + 1, offsets[c + 1]):
            if indices[p] <= indices[p - 1]:
                return False
    return True


@njit
def _llenar_csr(a, b, lab, offsets):
    cursor = offsets[:-1].copy()
    indices = np.empty(offsets[-1], np.int32)
    et = np.empty(offsets[-1], np.int32)
    for k in range(a.shape[0]):
        u = a[k]
        v = b[k]
        e = lab[k]
        indices[cursor[u]] = v
        et[cursor[u]] = e
        cursor[u] += 1
        indices[cursor[v]] = u
        et[cursor[v]] = e
        cursor[v] += 1
    return indices, et


def csr_desde_aristas(a, b, n, et_nodo, et_arista, nombres_nodo, nombres_arista, nombre="csr"):
    grados = np.bincount(a, minlength=n) + np.bincount(b, minlength=n)
    offsets = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(grados, out=offsets[1:])
    indices, et = _llenar_csr(a, b, et_arista.astype(np.int32), offsets)
    return CSR(offsets, indices, et_nodo.astype(np.int32), et, nombres_nodo, nombres_arista, nombre)


def cargar_npz(ruta, validar=True):
    d = np.load(ruta, allow_pickle=False)
    offsets = d["offsets"].astype(np.int64, copy=False)
    indices = d["indices"].astype(np.int32, copy=False)
    et_nodo = d["etiqueta_nodo"].astype(np.int32, copy=False)
    et_arista = d["etiqueta_arista"].astype(np.int32, copy=False) if "etiqueta_arista" in d.files else np.zeros(len(indices), np.int32)
    n = len(offsets) - 1
    if len(et_nodo) != n or len(et_arista) != len(indices) or offsets[-1] != len(indices):
        raise ValueError("Tamaños inconsistentes entre offsets, indices y etiquetas")
    if validar:
        if np.any(np.diff(offsets) < 0) or (len(indices) and (indices.min() < 0 or indices.max() >= n)):
            raise ValueError("offsets no monótonos o indices fuera de rango")
        if not filas_ordenadas(offsets, indices):
            orden = np.lexsort((indices, np.repeat(np.arange(n), np.diff(offsets))))
            indices, et_arista = indices[orden], et_arista[orden]
    nn = int(et_nodo.max()) + 1 if len(et_nodo) else 1
    na = int(et_arista.max()) + 1 if len(et_arista) else 1
    nombres_nodo = [str(x) for x in d["nombres_nodo"]] if "nombres_nodo" in d.files else [str(i) for i in range(nn)]
    nombres_arista = [str(x) for x in d["nombres_arista"]] if "nombres_arista" in d.files else [str(i) for i in range(na)]
    csr = CSR(offsets, indices, et_nodo, et_arista, nombres_nodo, nombres_arista, os.path.basename(ruta))
    if validar:
        csr.id_arista
    return csr


def guardar_npz(csr, ruta, comprimir=True):
    guardar = np.savez_compressed if comprimir else np.savez
    guardar(
        ruta,
        offsets=csr.offsets,
        indices=csr.indices,
        etiqueta_nodo=csr.et_nodo,
        etiqueta_arista=csr.et_arista,
        nombres_nodo=np.array(csr.nombres_nodo),
        nombres_arista=np.array(csr.nombres_arista),
    )


@njit
def etiquetar_componentes(offsets, indices, n):
    comp = np.full(n, -1, np.int64)
    pila = np.empty(max(1, n), np.int64)
    ncomp = 0
    for s in range(n):
        if comp[s] >= 0:
            continue
        comp[s] = ncomp
        pila[0] = s
        tope = 1
        while tope > 0:
            tope -= 1
            u = pila[tope]
            for p in range(offsets[u], offsets[u + 1]):
                v = indices[p]
                if comp[v] < 0:
                    comp[v] = ncomp
                    pila[tope] = v
                    tope += 1
        ncomp += 1
    return comp, ncomp


def conexidad_csr(csr):
    comp, ncomp = etiquetar_componentes(csr.offsets, csr.indices, csr.n)
    mayor = int(np.bincount(comp).max()) if csr.n else 0
    return {
        "conexo": ncomp <= 1,
        "componentes": int(ncomp),
        "componente_mayor": mayor,
        "pct_componente_mayor": 100 * mayor / csr.n if csr.n else 0.0,
        "transacciones_conexas": 1 if ncomp <= 1 else 0,
    }


@cuda.jit(device=True, inline=True)
def id_patron(lc, b1, b2, B):
    if b1 > b2:
        b1, b2 = b2, b1
    return (int64(lc) * B + b1) * B + b2


@cuda.jit(device=True)
def insertar_hash(claves, valores, pid, cant):
    mascara = claves.shape[0] - 1
    h = int64(pid) * int64(2654435761)
    h = (h ^ (h >> 17)) & mascara
    while True:
        actual = claves[h]
        if actual == VACIO:
            actual = cuda.atomic.cas(claves, h, VACIO, pid)
            if actual == VACIO:
                actual = pid
        if actual == pid:
            cuda.atomic.add(valores, h, cant)
            return
        h = (h + 1) & mascara


@cuda.jit(device=True)
def volcar(pid, cant, modo, hist, sh, claves, valores):
    if cant > 0:
        if modo == 0:
            cuda.atomic.add(hist, pid, cant)
        elif modo == 1:
            cuda.atomic.add(sh, pid, cant)
        else:
            insertar_hash(claves, valores, pid, cant)


@cuda.jit(device=True)
def iniciar_shared(sh, P, modo):
    if modo == 1:
        for t in range(cuda.threadIdx.x, P, cuda.blockDim.x):
            sh[t] = 0
    cuda.syncthreads()


@cuda.jit(device=True)
def vaciar_shared(sh, P, modo, hist):
    cuda.syncthreads()
    if modo == 1:
        for t in range(cuda.threadIdx.x, P, cuda.blockDim.x):
            if sh[t] > 0:
                cuda.atomic.add(hist, t, sh[t])


@cuda.jit(device=True)
def desrankear(k, d):
    t = 2 * d - 1
    i = int((t - math.sqrt(t * t - 8.0 * k)) / 2.0)
    if i < 0:
        i = 0
    while i > 0 and i * (t - i) // 2 > k:
        i -= 1
    while (i + 1) * (t - i - 1) // 2 <= k:
        i += 1
    return i, i + 1 + k - i * (t - i) // 2


def _fabricar_kernels(modo):
    tam_sh = BINS_SHARED if modo == 1 else 1

    @cuda.jit
    def k_nodo(offsets, indices, et_nodo, et_arista, inicio, fin, L, B, P, hist, claves, valores):
        sh = cuda.shared.array(tam_sh, int64)
        iniciar_shared(sh, P, modo)
        ultimo = -1
        cant = 0
        for c in range(inicio + cuda.grid(1), fin, cuda.gridsize(1)):
            ini = offsets[c]
            fin_c = offsets[c + 1]
            lc = et_nodo[c]
            for i in range(ini, fin_c):
                bi = et_arista[i] * L + et_nodo[indices[i]]
                for j in range(i + 1, fin_c):
                    pid = id_patron(lc, bi, et_arista[j] * L + et_nodo[indices[j]], B)
                    if pid == ultimo:
                        cant += 1
                    else:
                        volcar(ultimo, cant, modo, hist, sh, claves, valores)
                        ultimo = pid
                        cant = 1
        volcar(ultimo, cant, modo, hist, sh, claves, valores)
        vaciar_shared(sh, P, modo, hist)

    @cuda.jit
    def k_entrada(offsets, indices, et_nodo, et_arista, fila, inicio, fin, L, B, P, hist, claves, valores):
        sh = cuda.shared.array(tam_sh, int64)
        iniciar_shared(sh, P, modo)
        ultimo = -1
        cant = 0
        for p in range(inicio + cuda.grid(1), fin, cuda.gridsize(1)):
            c = fila[p]
            fin_c = offsets[c + 1]
            lc = et_nodo[c]
            bi = et_arista[p] * L + et_nodo[indices[p]]
            for q in range(p + 1, fin_c):
                pid = id_patron(lc, bi, et_arista[q] * L + et_nodo[indices[q]], B)
                if pid == ultimo:
                    cant += 1
                else:
                    volcar(ultimo, cant, modo, hist, sh, claves, valores)
                    ultimo = pid
                    cant = 1
        volcar(ultimo, cant, modo, hist, sh, claves, valores)
        vaciar_shared(sh, P, modo, hist)

    @cuda.jit
    def k_balanceado(offsets, indices, et_nodo, et_arista, id_arista, acum, n, W, chunk, ch_inicio, ch_fin,
                     L, B, P, hist, claves, valores, emitir, out1, out2):
        sh = cuda.shared.array(tam_sh, int64)
        iniciar_shared(sh, P, modo)
        ultimo = -1
        cant = 0
        for ch in range(ch_inicio + cuda.grid(1), ch_fin, cuda.gridsize(1)):
            w = ch * chunk
            fin_w = min(w + chunk, W)
            lo = 0
            hi = n
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if acum[mid] <= w:
                    lo = mid
                else:
                    hi = mid - 1
            c = lo
            ini = offsets[c]
            d = offsets[c + 1] - ini
            lc = et_nodo[c]
            i, j = desrankear(w - acum[c], d)
            bi = et_arista[ini + i] * L + et_nodo[indices[ini + i]]
            while w < fin_w:
                pid = id_patron(lc, bi, et_arista[ini + j] * L + et_nodo[indices[ini + j]], B)
                if emitir:
                    a = id_arista[ini + i]
                    b = id_arista[ini + j]
                    if a > b:
                        a, b = b, a
                    out1[w] = a
                    out2[w] = b
                if pid == ultimo:
                    cant += 1
                else:
                    volcar(ultimo, cant, modo, hist, sh, claves, valores)
                    ultimo = pid
                    cant = 1
                w += 1
                if w < fin_w:
                    j += 1
                    if j == d:
                        i += 1
                        if i >= d - 1:
                            c += 1
                            while acum[c + 1] == acum[c]:
                                c += 1
                            ini = offsets[c]
                            d = offsets[c + 1] - ini
                            lc = et_nodo[c]
                            i = 0
                        j = i + 1
                        bi = et_arista[ini + i] * L + et_nodo[indices[ini + i]]
        volcar(ultimo, cant, modo, hist, sh, claves, valores)
        vaciar_shared(sh, P, modo, hist)

    @cuda.jit
    def k_combinatorio(offsets, indices, et_nodo, et_arista, inicio, fin, L, B, P, hist, claves, valores):
        sh = cuda.shared.array(tam_sh, int64)
        iniciar_shared(sh, P, modo)
        cuenta = cuda.local.array(MAX_TIPOS_COMBINATORIO, int32)
        presentes = cuda.local.array(MAX_TIPOS_COMBINATORIO, int32)
        for t in range(B):
            cuenta[t] = 0
        for c in range(inicio + cuda.grid(1), fin, cuda.gridsize(1)):
            k = 0
            for p in range(offsets[c], offsets[c + 1]):
                b = et_arista[p] * L + et_nodo[indices[p]]
                if cuenta[b] == 0:
                    presentes[k] = b
                    k += 1
                cuenta[b] += 1
            lc = et_nodo[c]
            for x in range(k):
                b1 = presentes[x]
                n1 = int64(cuenta[b1])
                volcar(id_patron(lc, b1, b1, B), n1 * (n1 - 1) // 2, modo, hist, sh, claves, valores)
                for y in range(x + 1, k):
                    b2 = presentes[y]
                    volcar(id_patron(lc, b1, b2, B), n1 * cuenta[b2], modo, hist, sh, claves, valores)
            for x in range(k):
                cuenta[presentes[x]] = 0
        vaciar_shared(sh, P, modo, hist)

    return {"nodo": k_nodo, "entrada": k_entrada, "balanceado": k_balanceado, "combinatorio": k_combinatorio}


_KERNELS = {}


def kernels(memoria):
    if memoria not in _KERNELS:
        _KERNELS[memoria] = _fabricar_kernels(MEMORIAS[memoria])
    return _KERNELS[memoria]


SCAN_HILOS = 256
SCAN_POR_HILO = 4
SCAN_BLOQUE = SCAN_HILOS * SCAN_POR_HILO


@cuda.jit
def _k_scan_bloque(entrada, salida, sumas, n):
    sh = cuda.shared.array(SCAN_HILOS, int64)
    local = cuda.local.array(SCAN_POR_HILO, int64)
    t = cuda.threadIdx.x
    base = cuda.blockIdx.x * SCAN_BLOQUE + t * SCAN_POR_HILO
    acc = int64(0)
    for k in range(SCAN_POR_HILO):
        i = base + k
        if i < n:
            acc += entrada[i]
        local[k] = acc
    sh[t] = acc
    cuda.syncthreads()
    d = 1
    while d < SCAN_HILOS:
        v = int64(0)
        if t >= d:
            v = sh[t - d]
        cuda.syncthreads()
        sh[t] += v
        cuda.syncthreads()
        d *= 2
    previo = int64(0)
    if t > 0:
        previo = sh[t - 1]
    for k in range(SCAN_POR_HILO):
        i = base + k
        if i < n:
            salida[i] = local[k] + previo
    if t == SCAN_HILOS - 1:
        sumas[cuda.blockIdx.x] = sh[t]


@cuda.jit
def _k_sumar_bloques(salida, sumas, n):
    b = cuda.blockIdx.x
    if b > 0:
        extra = sumas[b - 1]
        base = b * SCAN_BLOQUE
        for k in range(cuda.threadIdx.x, SCAN_BLOQUE, cuda.blockDim.x):
            i = base + k
            if i < n:
                salida[i] += extra


@cuda.jit
def _k_escribir(arr, i, valor):
    if cuda.grid(1) == 0:
        arr[i] = valor


@cuda.jit
def _k_fila(offsets, n, fila):
    for c in range(cuda.grid(1), n, cuda.gridsize(1)):
        for p in range(offsets[c], offsets[c + 1]):
            fila[p] = c


@cuda.jit
def _k_buscar(acum, m, objetivos, salida):
    i = cuda.grid(1)
    if i < objetivos.shape[0]:
        x = objetivos[i]
        lo = 0
        hi = m
        while lo < hi:
            mid = (lo + hi) // 2
            if acum[mid] < x:
                lo = mid + 1
            else:
                hi = mid
        salida[i] = lo


def en_gpu(a):
    return hasattr(a, "__cuda_array_interface__")


def configuracion_lanzamiento(items, bloque=256, max_bloques=4096):
    return max(1, min(math.ceil(items / bloque), max_bloques)), bloque


def scan_inclusivo(d_entrada, d_salida, n):
    if n <= 0:
        return
    nb = (n + SCAN_BLOQUE - 1) // SCAN_BLOQUE
    d_sumas = cuda.device_array(nb, np.int64)
    _k_scan_bloque[nb, SCAN_HILOS](d_entrada, d_salida, d_sumas, n)
    if nb > 1:
        scan_inclusivo(d_sumas, d_sumas, nb)
        _k_sumar_bloques[nb, SCAN_HILOS](d_salida, d_sumas, n)


def prefijo_exclusivo(d_valores, n):
    d_salida = cuda.device_array(n + 1, np.int64)
    _k_escribir[1, 1](d_salida, 0, 0)
    scan_inclusivo(d_valores, d_salida[1:], n)
    return d_salida


def escribir(d_arr, i, valor):
    _k_escribir[1, 1](d_arr, i, valor)


def leer_escalar(d_arr, i):
    return int(d_arr[i:i + 1].copy_to_host()[0])


def buscar_cortes(acum, objetivos):
    objetivos = np.ascontiguousarray(objetivos, dtype=np.int64)
    if not en_gpu(acum):
        return np.searchsorted(acum, objetivos)
    if len(objetivos) == 0:
        return np.empty(0, np.int64)
    d_salida = cuda.device_array(len(objetivos), np.int64)
    _k_buscar[(len(objetivos) + 255) // 256, 256](acum, acum.shape[0], cuda.to_device(objetivos), d_salida)
    return d_salida.copy_to_host()


def subir_grafo(csr):
    return {"offsets": cuda.to_device(csr.offsets), "indices": cuda.to_device(csr.indices),
            "et_nodo": cuda.to_device(csr.et_nodo), "et_arista": cuda.to_device(csr.et_arista)}


def fila_en_gpu(disp, n, nnz, bloque=256, max_bloques=4096):
    d_fila = cuda.device_array(max(1, nnz), np.int32)
    bloques, hilos = configuracion_lanzamiento(n, bloque, max_bloques)
    _k_fila[bloques, hilos](disp["offsets"], n, d_fila)
    return d_fila


def _tomar(disp, clave, host):
    return disp[clave] if clave in disp else a_dispositivo(host)


def preparar(csr):
    grados = np.diff(csr.offsets)
    cunas = grados * (grados - 1) // 2
    acum = np.zeros(csr.n + 1, dtype=np.int64)
    np.cumsum(cunas, out=acum[1:])
    L = len(csr.nombres_nodo)
    E = len(csr.nombres_arista)
    B = E * L
    P = L * B * B
    if P >= 1 << 62:
        raise ValueError("Demasiadas etiquetas para codificar el patrón en 64 bits")
    return {"grados": grados, "cunas": cunas, "acum": acum, "fila": None,
            "W": int(acum[-1]), "L": L, "E": E, "B": B, "P": P,
            "grado_max": int(grados.max()) if csr.n else 0, "max_cunas": int(cunas.max()) if csr.n else 0}


def obtener_fila(csr, prep):
    if prep["fila"] is None:
        prep["fila"] = np.repeat(np.arange(csr.n, dtype=np.int32), prep["grados"])
    return prep["fila"]


def a_dispositivo(a):
    return a if hasattr(a, "__cuda_array_interface__") else cuda.to_device(a)


def capacidad_hash(W, P):
    return 1 << max(10, math.ceil(math.log2(max(2, 2 * min(W, P)))))


def decodificar(pids, cuentas, csr, prep):
    L, B = prep["L"], prep["B"]
    nn, na = csr.nombres_nodo, csr.nombres_arista
    censo = {}
    for pid, c in zip(pids.tolist(), cuentas.tolist()):
        lc, resto = divmod(pid, B * B)
        b1, b2 = divmod(resto, B)
        r1 = (na[b1 // L], nn[b1 % L])
        r2 = (na[b2 // L], nn[b2 % L])
        censo[(nn[lc],) + tuple(sorted((r1, r2)))] = c
    return censo


def motivo_memoria(memoria, espacio):
    if memoria == "shared" and espacio > BINS_SHARED:
        return f"{espacio:,} patrones posibles superan las {BINS_SHARED:,} casillas de memoria compartida"
    if memoria == "global" and espacio > LIMITE_DENSO:
        return f"{espacio:,} patrones posibles superan el histograma denso ({LIMITE_DENSO:,} casillas)"
    return None


def memoria_por_defecto(espacio):
    for memoria in ("shared", "global", "hash"):
        if motivo_memoria(memoria, espacio) is None:
            return memoria


def motivo_invalida(estrategia, memoria, prep):
    if estrategia == "combinatorio" and prep["B"] > MAX_TIPOS_COMBINATORIO:
        return (f"B = {prep['B']} tipos de rama superan el máximo de la estrategia combinatoria "
                f"({MAX_TIPOS_COMBINATORIO})")
    if estrategia == "nodo" and prep["max_cunas"] > LIMITE_TRABAJO_HILO:
        return (f"un solo hilo procesaría C({prep['grado_max']:,}, 2) = {prep['max_cunas']:,} caminos "
                f"(límite {LIMITE_TRABAJO_HILO:,} por riesgo de watchdog)")
    return motivo_memoria(memoria, prep["P"])


def config_valida(estrategia, memoria, prep):
    return motivo_invalida(estrategia, memoria, prep) is None


def memoria_gpu():
    try:
        libre, total = cuda.current_context().get_memory_info()
        return int(libre), int(total)
    except Exception:
        return None, None


def liberar_gpu():
    gc.collect()
    try:
        cuda.current_context().deallocations.clear()
    except Exception:
        pass


def bytes_necesarios(csr, prep, estrategia, memoria, emitir=False, disp=None):
    disp = disp or {}
    total = sum(getattr(csr, campo).nbytes for clave, campo in (("offsets", "offsets"), ("indices", "indices"),
                                                                ("et_nodo", "et_nodo"), ("et_arista", "et_arista"))
                if clave not in disp)
    if estrategia == "entrada" and "fila" not in disp:
        total += 4 * csr.nnz
    if estrategia == "balanceado":
        total += 0 if "acum" in disp else prep["acum"].nbytes
        if emitir:
            total += 8 * csr.nnz + 16 * prep["W"]
    total += 8 * prep["P"] if MEMORIAS[memoria] < 2 else 16 * capacidad_hash(prep["W"], prep["P"])
    return total


def lotes(estrategia, csr, prep, chunk, trabajo_max):
    W = prep["W"]
    if estrategia == "balanceado":
        nch = (W + chunk - 1) // chunk
        paso = max(1, trabajo_max // chunk)
        return [(i, min(i + paso, nch)) for i in range(0, nch, paso)]
    n = csr.n
    acum, total = (csr.offsets, csr.nnz) if estrategia == "combinatorio" else (prep["acum"], W)
    if total <= trabajo_max:
        cortes = np.array([0, n], dtype=np.int64)
    else:
        objetivos = np.arange(trabajo_max, total, trabajo_max, dtype=np.int64)
        cortes = np.unique(np.concatenate([[0], np.searchsorted(acum, objetivos), [n]]))
    if estrategia == "entrada":
        cortes = csr.offsets[cortes]
    return [(int(a), int(b)) for a, b in zip(cortes[:-1], cortes[1:]) if b > a] or [(0, 0)]


def ejecutar_gpu(csr, prep, estrategia, memoria, bloque, max_bloques, chunk, emitir=False,
                 trabajo_max=TRABAJO_POR_LANZAMIENTO, disp=None):
    motivo = motivo_invalida(estrategia, memoria, prep)
    if motivo:
        raise ValueError(f"Configuración {estrategia} + {memoria} no válida: {motivo}")
    if emitir and estrategia != "balanceado":
        raise ValueError("Solo la estrategia 'balanceado' emite los IDs de cada camino")
    k = kernels(memoria)[estrategia]
    modo = MEMORIAS[memoria]
    W, P, n = prep["W"], prep["P"], csr.n
    L, B = prep["L"], prep["B"]
    disp = disp or {}

    t0 = time.perf_counter()
    fila = obtener_fila(csr, prep) if estrategia == "entrada" and "fila" not in disp else None
    t_preparacion = time.perf_counter() - t0

    t0 = time.perf_counter()
    d_off = _tomar(disp, "offsets", csr.offsets)
    d_ind = _tomar(disp, "indices", csr.indices)
    d_etn = _tomar(disp, "et_nodo", csr.et_nodo)
    d_eta = _tomar(disp, "et_arista", csr.et_arista)
    d_hist = cuda.to_device(np.zeros(P if modo < 2 else 1, dtype=np.int64))
    cap = capacidad_hash(W, P) if modo == 2 else 1
    d_claves = cuda.to_device(np.full(cap, VACIO, dtype=np.int64))
    d_valores = cuda.to_device(np.zeros(cap, dtype=np.int64))
    if estrategia == "entrada":
        d_fila = disp["fila"] if "fila" in disp else a_dispositivo(fila)
    if estrategia == "balanceado":
        d_acum = _tomar(disp, "acum", prep["acum"])
        if emitir:
            d_ida = a_dispositivo(csr.id_arista)
            d_out1 = cuda.device_array(max(1, W), dtype=np.int64)
            d_out2 = cuda.device_array(max(1, W), dtype=np.int64)
        else:
            d_ida = cuda.to_device(np.zeros(1, dtype=np.int64))
            d_out1 = cuda.device_array(1, dtype=np.int64)
            d_out2 = cuda.device_array(1, dtype=np.int64)
    cuda.synchronize()
    t_h2d = time.perf_counter() - t0

    rangos = lotes(estrategia, csr, prep, chunk, trabajo_max)
    hilos = 0
    t0 = time.perf_counter()
    for inicio, fin in rangos:
        bloques = max(1, min(math.ceil((fin - inicio) / bloque), max_bloques))
        hilos = max(hilos, bloques * bloque)
        if estrategia in ("nodo", "combinatorio"):
            k[bloques, bloque](d_off, d_ind, d_etn, d_eta, inicio, fin, L, B, P, d_hist, d_claves, d_valores)
        elif estrategia == "entrada":
            k[bloques, bloque](d_off, d_ind, d_etn, d_eta, d_fila, inicio, fin, L, B, P, d_hist, d_claves, d_valores)
        else:
            k[bloques, bloque](d_off, d_ind, d_etn, d_eta, d_ida, d_acum, n, W, chunk, inicio, fin,
                               L, B, P, d_hist, d_claves, d_valores, 1 if emitir else 0, d_out1, d_out2)
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
        ocupadas = claves != VACIO
        pids, cuentas = claves[ocupadas], valores[ocupadas]
    censo = decodificar(pids, cuentas, csr, prep)
    embeddings = (d_out1.copy_to_host(), d_out2.copy_to_host()) if emitir and W else None
    t_d2h = time.perf_counter() - t0

    return {
        "estrategia": estrategia,
        "memoria": memoria,
        "censo": censo,
        "embeddings": embeddings,
        "hilos": max(hilos, 1),
        "lanzamientos": len(rangos),
        "cap_hash": cap if modo == 2 else None,
        "t_preparacion": t_preparacion,
        "t_h2d": t_h2d,
        "t_kernel": t_kernel,
        "t_d2h": t_d2h,
    }


@njit
def _carga_max_nodo(cunas, hilos):
    carga = np.zeros(hilos, np.int64)
    for c in range(cunas.shape[0]):
        carga[c % hilos] += cunas[c]
    return carga.max()


@njit
def _carga_max_entrada(offsets, hilos):
    carga = np.zeros(hilos, np.int64)
    for c in range(offsets.shape[0] - 1):
        fin = offsets[c + 1]
        for p in range(offsets[c], fin):
            carga[p % hilos] += fin - p - 1
    return carga.max()


def desbalance(csr, prep, estrategia, hilos, chunk):
    W = prep["W"]
    if estrategia == "combinatorio":
        return None
    if W == 0 or hilos <= 0:
        return 1.0
    prom = W / hilos
    if estrategia == "balanceado":
        nchunks = (W + chunk - 1) // chunk
        q, r = divmod(nchunks, hilos)
        ultimo = W - (nchunks - 1) * chunk
        if r == 0:
            maximo = q * chunk if hilos > 1 else W
        elif r == 1:
            maximo = q * chunk + ultimo
        else:
            maximo = (q + 1) * chunk
        return maximo / prom
    if estrategia == "nodo":
        return float(_carga_max_nodo(prep["cunas"], hilos)) / prom
    return float(_carga_max_entrada(csr.offsets, hilos)) / prom


def hilos_teoricos(csr, prep, estrategia, bloque, max_bloques, chunk):
    items = {"nodo": csr.n, "combinatorio": csr.n, "entrada": csr.nnz,
             "balanceado": (prep["W"] + chunk - 1) // chunk}[estrategia]
    return max(1, min(math.ceil(items / bloque), max_bloques)) * bloque


def medir_gpu(csr, prep, estrategias, memorias, bloque, max_bloques, chunk, reps, censo_ref=None, t_cpu=None,
              trabajo_max=TRABAJO_POR_LANZAMIENTO):
    filas = []
    censo_gpu = None
    for estrategia in estrategias:
        for memoria in memorias:
            base = {"estrategia": estrategia, "memoria": memoria, "bloque": bloque}
            motivo = motivo_invalida(estrategia, memoria, prep)
            if motivo:
                filas.append({**base, "omitida": motivo})
                continue
            libre, _ = memoria_gpu()
            necesario = bytes_necesarios(csr, prep, estrategia, memoria)
            if libre is not None and necesario > 0.9 * libre:
                filas.append({**base, "omitida": f"requiere {necesario / 2**30:.2f} GB y hay {libre / 2**30:.2f} GB libres"})
                continue
            primera = ejecutar_gpu(csr, prep, estrategia, memoria, bloque, max_bloques, chunk, trabajo_max=trabajo_max)
            liberar_gpu()
            corridas = []
            for _ in range(reps):
                corridas.append(ejecutar_gpu(csr, prep, estrategia, memoria, bloque, max_bloques, chunk,
                                             trabajo_max=trabajo_max))
                liberar_gpu()
            est = {k: sec.estadisticas(c[f"t_{k}"] for c in corridas) for k in ("h2d", "kernel", "d2h")}
            est["total"] = sec.estadisticas(c["t_h2d"] + c["t_kernel"] + c["t_d2h"] for c in corridas)
            censo = corridas[-1]["censo"]
            if censo_ref is None:
                censo_ref = censo
            correcto = all(c["censo"] == censo_ref for c in corridas) and sum(censo.values()) == prep["W"]
            censo_gpu = censo_gpu or censo
            total, calculo = est["total"]["media"], est["kernel"]["media"]
            fila = {
                **base,
                "hilos": corridas[-1]["hilos"],
                "lanzamientos": corridas[-1]["lanzamientos"],
                "cap_hash": corridas[-1]["cap_hash"],
                "reps": reps,
                "t_jit_s": max(0.0, primera["t_kernel"] - calculo),
                **sec.aplanar("t_h2d", est["h2d"]),
                **sec.aplanar("t_kernel", est["kernel"]),
                **sec.aplanar("t_d2h", est["d2h"]),
                **sec.aplanar("t_total", est["total"]),
                "caminos_por_s_kernel": prep["W"] / calculo if calculo > 0 else None,
                "speedup_exploracion": None,
                "speedup_centro": None,
                "speedup_kernel_centro": None,
                "desbalance": desbalance(csr, prep, estrategia, corridas[-1]["hilos"], chunk),
                "correcto": bool(correcto),
            }
            if t_cpu:
                if t_cpu.get("exploracion") and total > 0:
                    fila["speedup_exploracion"] = t_cpu["exploracion"] / total
                if t_cpu.get("centro") and total > 0:
                    fila["speedup_centro"] = t_cpu["centro"] / total
                    fila["speedup_kernel_centro"] = t_cpu["centro"] / calculo if calculo > 0 else None
            filas.append(fila)
    return filas, censo_gpu


def embeddings_secuenciales(g):
    ids = set()
    for vecinos in g.adyacencia.values():
        for i in range(len(vecinos)):
            for j in range(i + 1, len(vecinos)):
                a, b = vecinos[i][1], vecinos[j][1]
                ids.add((a, b) if a < b else (b, a))
    return ids


def medir_secuencial(g, reps):
    salida = {}
    for nombre, metodo in sec.METODOS.items():
        tiempos = []
        censo = None
        for _ in range(reps):
            gc.collect()
            gc.disable()
            t0 = time.perf_counter()
            censo, _ = metodo(g)
            tiempos.append(time.perf_counter() - t0)
            gc.enable()
        salida[nombre] = {"censo": censo, "est": sec.estadisticas(tiempos)}
    return salida


def verificar_ids(csr, prep, g, bloque=256, max_bloques=4096, chunk=64, trabajo_max=TRABAJO_POR_LANZAMIENTO):
    W = prep["W"]
    if W > LIMITE_VERIFICAR_IDS:
        return {"estado": f"omitida (W = {W} > {LIMITE_VERIFICAR_IDS})"}
    r = ejecutar_gpu(csr, prep, "balanceado", "hash", bloque, max_bloques, chunk, emitir=True, trabajo_max=trabajo_max)
    liberar_gpu()
    if r["embeddings"] is None:
        return {"estado": "OK", "emitidos_gpu": 0, "unicos_gpu": 0, "duplicados_gpu": 0,
                "unicos_secuencial": 0, "iguales_a_secuencial": True}
    e1, e2 = r["embeddings"]
    m = max(1, int(csr.id_arista.max()) + 1) if csr.nnz else 1
    claves_gpu = e1 * m + e2
    unicos = len(np.unique(claves_gpu))
    sec_ids = np.array(sorted(a * m + b for a, b in embeddings_secuenciales(g)), dtype=np.int64)
    iguales = unicos == W and np.array_equal(np.sort(claves_gpu), sec_ids)
    return {
        "estado": "OK" if iguales else "ERROR",
        "emitidos_gpu": int(W),
        "unicos_gpu": int(unicos),
        "duplicados_gpu": int(W - unicos),
        "unicos_secuencial": int(len(sec_ids)),
        "iguales_a_secuencial": bool(iguales),
    }


def crear_parser():
    p = argparse.ArgumentParser(description="Censo paralelo (CUDA/Numba) de caminos de 3 nodos (no inducidos) sobre CSR")
    p.add_argument("--csr", help="Archivo .npz con el grafo en CSR (offsets, indices, etiqueta_nodo, etiqueta_arista)")
    p.add_argument("--guardar-csr", action="store_true", help="Guarda el CSR usado en un .npz")
    p.add_argument("--grafo", choices=["ejemplo", "triangulo", "estrella", "completo", "aleatorio", "ba"])
    p.add_argument("--archivo", help="Archivo gSpan (se convierte a CSR)")
    p.add_argument("--ids", nargs="+")
    p.add_argument("--max-grafos", type=int, default=None)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--m", type=int, default=5000)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--etiquetas-nodo", type=int, default=3)
    p.add_argument("--etiquetas-arista", type=int, default=2)
    p.add_argument("--semilla", type=int, default=42)
    p.add_argument("--estrategia", choices=ESTRATEGIAS + ["todos"], default="todos", help="Reparto del trabajo")
    p.add_argument("--memoria", choices=list(MEMORIAS) + ["todos"], default="todos", help="Dónde se acumula el histograma")
    p.add_argument("--bloque", type=int, default=256, help="Hilos por bloque")
    p.add_argument("--max-bloques", type=int, default=4096)
    p.add_argument("--chunk", type=int, default=64, help="Caminos por hilo en la estrategia balanceado")
    p.add_argument("--trabajo-por-lanzamiento", type=int, default=TRABAJO_POR_LANZAMIENTO,
                   help="Máximo de caminos por lanzamiento de kernel")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--sin-secuencial", action="store_true", help="No ejecuta la versión secuencial")
    p.add_argument("--verificar-ids", action="store_true",
                   help="Emite los IDs de cada subgrafo en GPU y los compara con los secuenciales")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--json", action="store_true", help="Exporta el resultado completo a JSON")
    p.add_argument("--dibujar", action="store_true", help="Guarda un dibujo del grafo (.png)")
    p.add_argument("--salida", default="resultados", help="Carpeta para los archivos generados")
    p.add_argument("--max-dibujo", type=int, default=300, help="Máximo de nodos para dibujar")
    return p


def imprimir_tabla_gpu(filas):
    print(f"{'Estrategia':<13}{'Memoria':<8}{'Hilos':>9}{'Lanz':>6}{'JIT':>12}{'H2D':>12}{'Cálculo':>24}{'D2H+dec':>12}"
          f"{'Total':>24}{'xExpl':>9}{'xCentro':>10}{'Desbal':>9}  OK")
    for f in filas:
        if f.get("omitida"):
            print(f"{f['estrategia']:<13}{f['memoria']:<8}  omitida: {f['omitida']}")
            continue
        xe = f"{f['speedup_exploracion']:.2f}" if f["speedup_exploracion"] else "-"
        xc = f"{f['speedup_centro']:.2f}" if f["speedup_centro"] else "-"
        des = f"{f['desbalance']:.2f}" if f["desbalance"] is not None else "-"
        calc = f"{sec.seg(f['t_kernel_s'])} ± {sec.seg(f['t_kernel_desv_s'])}"
        total = f"{sec.seg(f['t_total_s'])} ± {sec.seg(f['t_total_desv_s'])}"
        print(f"{f['estrategia']:<13}{f['memoria']:<8}{f['hilos']:>9}{f['lanzamientos']:>6}{sec.seg(f['t_jit_s']):>12}"
              f"{sec.seg(f['t_h2d_s']):>12}{calc:>24}{sec.seg(f['t_d2h_s']):>12}"
              f"{total:>24}{xe:>9}{xc:>10}{des:>9}  {'sí' if f['correcto'] else 'NO'}")
    print("Tiempos: media ± desviación de las repeticiones, tras 1 ejecución de calentamiento")
    print("xExpl / xCentro: aceleración del tiempo total medio (H2D + cálculo + D2H) sobre la media de cada método secuencial")
    print("Desbal: trabajo del hilo más cargado / trabajo promedio por hilo (1.00 = perfecto; '-' = no modelado)")


def imprimir_reporte(r, top):
    gr = r["grafo"]
    print("\n=== GRAFO (CSR) ===")
    print(f"Nombre: {gr['nombre']} | nodos: {gr['nodos']} | aristas: {gr['aristas']} | entradas CSR: {gr['nnz']}")
    sec.imprimir_conexidad(gr)
    print(f"Grado máx: {gr['grado_max']} | caminos esperados W = Σ C(d,2): {gr['W']}")
    print(f"Etiquetas nodo: {gr['L']} | etiquetas arista: {gr['E']} | espacio de patrones P: {gr['P']}")
    print(f"Construcción: {sec.seg(gr['t_construccion_s'])} | preproceso CSR (scan): {sec.seg(gr['t_preproceso_s'])}")

    s = r["secuencial"]
    if s:
        print("\n=== SECUENCIAL (CPU, Python) ===")
        for nombre, d in s.items():
            print(f"{nombre:>12}: media {sec.seg(d['t_s'])} ± {sec.seg(d['t_desv_s'])} | mín {sec.seg(d['t_min_s'])}")

    print("\n=== PARALELO (GPU) ===")
    imprimir_tabla_gpu(r["gpu"])

    v = r["verificacion_ids"]
    if v:
        print("\n=== VERIFICACIÓN DE IDs DE SUBGRAFO ===")
        for k, val in v.items():
            print(f"{k}: {val}")

    filas = r["censo"][:top] if top else r["censo"]
    if filas:
        print(f"\n=== CENSO GPU (top {len(filas)} de {len(r['censo'])} patrones) ===")
        ancho = max(len(f["patron"]) for f in filas)
        for i, f in enumerate(filas, 1):
            print(f"{i:>4}  {f['patron']:<{ancho}}  {f['frecuencia']:>12}")


def main():
    args = crear_parser().parse_args()
    if not cuda.is_available():
        raise SystemExit("No se detectó GPU CUDA. Para depurar sin GPU: NUMBA_ENABLE_CUDASIM=1 con grafos muy pequeños.")
    if not (args.csr or args.grafo or args.archivo):
        args.grafo = "ejemplo"
    args.reps = max(1, args.reps)

    t0 = time.perf_counter()
    if args.csr:
        csr = cargar_npz(args.csr)
        g = csr_a_grafo(csr) if (not args.sin_secuencial or args.verificar_ids) else None
    else:
        g = sec.construir_grafo(args)
        csr = grafo_a_csr(g)
    t_construccion = time.perf_counter() - t0

    t0 = time.perf_counter()
    prep = preparar(csr)
    t_preproceso = time.perf_counter() - t0
    W = prep["W"]

    cuda.to_device(np.zeros(1))
    cuda.synchronize()

    secuencial = medir_secuencial(g, args.reps) if not args.sin_secuencial else None
    censo_ref = secuencial["centro"]["censo"] if secuencial else None
    t_cpu = {k: v["est"]["media"] for k, v in secuencial.items()} if secuencial else None

    estrategias = ESTRATEGIAS if args.estrategia == "todos" else [args.estrategia]
    memorias = list(MEMORIAS) if args.memoria == "todos" else [args.memoria]
    filas_gpu, censo_gpu = medir_gpu(csr, prep, estrategias, memorias, args.bloque, args.max_bloques, args.chunk,
                                     args.reps, censo_ref, t_cpu, args.trabajo_por_lanzamiento)

    verificacion = None
    if args.verificar_ids:
        if g is None:
            g = csr_a_grafo(csr)
        verificacion = verificar_ids(csr, prep, g, args.bloque, args.max_bloques, args.chunk, args.trabajo_por_lanzamiento)

    censo_final = censo_gpu or {}
    total_censo = sum(censo_final.values()) or 1
    resultado = {
        "grafo": {
            "nombre": csr.nombre,
            "nodos": csr.n,
            "aristas": csr.nnz // 2,
            "nnz": csr.nnz,
            "grado_max": prep["grado_max"],
            "W": W,
            "L": prep["L"],
            "E": prep["E"],
            "P": prep["P"],
            "t_construccion_s": t_construccion,
            "t_preproceso_s": t_preproceso,
            "transacciones": g.num_transacciones if g is not None else 1,
            **(sec.analizar_conexidad(g) if g is not None else conexidad_csr(csr)),
        },
        "secuencial": {k: sec.aplanar("t", v["est"]) for k, v in secuencial.items()} if secuencial else None,
        "secuenciales_coinciden": secuencial["exploracion"]["censo"] == secuencial["centro"]["censo"] if secuencial else None,
        "gpu": filas_gpu,
        "verificacion_ids": verificacion,
        "censo": [
            {"patron": sec.formato_patron(k), "frecuencia": v, "porcentaje": 100 * v / total_censo}
            for k, v in sorted(censo_final.items(), key=lambda x: (-x[1], sec.formato_patron(x[0])))
        ],
    }
    imprimir_reporte(resultado, args.top)

    generados = []
    if args.json or args.dibujar or args.guardar_csr:
        prefijo = sec.preparar_salida(args.salida, csr.nombre, "gpu_")
        if args.json:
            sec.guardar_json(resultado, prefijo + ".json")
            generados.append(prefijo + ".json")
        if args.guardar_csr:
            guardar_npz(csr, prefijo + "_csr.npz")
            generados.append(prefijo + "_csr.npz")
        if args.dibujar and sec.dibujar_grafo(g if g is not None else csr_a_grafo(csr), prefijo + ".png",
                                              args.max_dibujo, args.semilla):
            generados.append(prefijo + ".png")
    if generados:
        print("\nArchivos generados:")
        for ruta in generados:
            print(f"  {ruta}")


if __name__ == "__main__":
    main()
