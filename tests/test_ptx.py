import os
import sys
import types as pytypes

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("NUMBA_ENABLE_CUDASIM") == "1",
                                reason="el simulador de Numba no compila a PTX")

CC = (12, 0)


def _firmas():
    from numba import types

    A64, A32, I = types.int64[::1], types.int32[::1], types.int64
    caminos = {
        "nodo": (A64, A32, A32, A32, I, I, I, I, I, A64, A64, A64),
        "combinatorio": (A64, A32, A32, A32, I, I, I, I, I, A64, A64, A64),
        "entrada": (A64, A32, A32, A32, A32, I, I, I, I, I, A64, A64, A64),
        "balanceado": (A64, A32, A32, A32, A64, A64, I, I, I, I, I, I, I, I, A64, A64, A64, I, A64, A64),
    }
    triangulos = {
        "arista": (A64, A32, A32, A32, A32, I, I, I, I, I, A64, A64, A64),
        "binaria": (A64, A32, A32, A32, A32, I, I, I, I, I, A64, A64, A64),
        "balanceado": (A64, A32, A32, A32, A32, A64, I, I, I, I, I, I, I, A64, A64, A64),
    }
    return caminos, triangulos


@pytest.fixture
def compilar(monkeypatch):
    import censo_3n2a_gpu as gpu
    from numba import cuda

    modulo = sys.modules[type(gpu.id_patron).__module__]
    if not hasattr(modulo, "get_current_device"):
        pytest.skip("esta versión de numba-cuda no permite fijar la arquitectura de las funciones de dispositivo")
    monkeypatch.setattr(modulo, "get_current_device", lambda: pytypes.SimpleNamespace(compute_capability=CC))

    def _compilar(kernel, firma):
        try:
            ptx, _ = cuda.compile_ptx(kernel.py_func, firma, cc=CC)
        except Exception as e:
            if "NVVM" in str(e) or "libnvvm" in str(e).lower():
                pytest.skip(f"NVVM no disponible: {e}")
            raise
        return ptx

    return _compilar


def _shared_bytes(ptx):
    total = 0
    for linea in ptx.splitlines():
        if ".shared" in linea and ".align" in linea and "[" in linea:
            total += int(linea.split("[")[1].split("]")[0])
    return total


@pytest.mark.parametrize("memoria", ["global", "shared", "hash"])
def test_kernels_caminos_compilan(compilar, memoria):
    import censo_3n2a_gpu as gpu

    firmas, _ = _firmas()
    assert set(firmas) == set(gpu.ESTRATEGIAS)
    for estrategia, firma in firmas.items():
        ptx = compilar(gpu.kernels(memoria)[estrategia], firma)
        assert f".target sm_{CC[0]}{CC[1]}" in ptx
        esperado = 8 * gpu.BINS_SHARED if memoria == "shared" else 8
        assert _shared_bytes(ptx) == esperado, (estrategia, memoria)


@pytest.mark.parametrize("memoria", ["global", "shared", "hash"])
def test_kernels_triangulos_compilan(compilar, memoria):
    import censo_3n3a_gpu as tri

    _, firmas = _firmas()
    assert set(firmas) == set(tri.ESTRATEGIAS)
    for estrategia, firma in firmas.items():
        ptx = compilar(tri.kernels(memoria)[estrategia], firma)
        esperado = 8 * tri.BINS_SHARED if memoria == "shared" else 8
        assert _shared_bytes(ptx) == esperado, (estrategia, memoria)


def test_kernels_de_preproceso_compilan(compilar):
    from numba import types

    import censo_3n2a_gpu as gpu
    import censo_3n3a_gpu as tri

    A64, A32, I = types.int64[::1], types.int32[::1], types.int64
    kernels = [
        (gpu._k_scan_bloque, (A64, A64, A64, I)),
        (gpu._k_sumar_bloques, (A64, A64, I)),
        (gpu._k_escribir, (A64, I, I)),
        (gpu._k_fila, (A64, I, A32)),
        (gpu._k_buscar, (A64, I, A64, A64)),
        (tri._k_contar_salida, (A64, A32, I, A64, A64)),
        (tri._k_llenar_orientado, (A64, A32, A32, A64, I, A32, A32, A32)),
        (tri._k_costos, (A64, A32, A32, I, A64, A64)),
        (tri._k_costos_binaria, (A64, A32, A32, I, A64)),
    ]
    for kernel, firma in kernels:
        ptx = compilar(kernel, firma)
        assert f".target sm_{CC[0]}{CC[1]}" in ptx
    assert _shared_bytes(compilar(gpu._k_scan_bloque, (A64, A64, A64, I))) == 8 * gpu.SCAN_HILOS
