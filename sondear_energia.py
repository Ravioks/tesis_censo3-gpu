import argparse
import json
import os
import shutil
import subprocess
import time

import numpy as np

import medidores_energia as me


def probar(nombre, fn):
    try:
        r = fn()
        print(f"[OK]    {nombre}: {r}")
        return {"ok": True, "valor": r}
    except Exception as e:
        print(f"[FALLA] {nombre}: {e!r}")
        return {"ok": False, "error": repr(e)}


def carga_cpu(segundos):
    from numba import njit, prange

    @njit(parallel=True)
    def trabajo(n):
        s = np.zeros(n)
        for i in prange(n):
            x = 0.0
            for k in range(20000):
                x += np.sin(i + k)
            s[i] = x
        return s.sum()
    trabajo(64)
    t0 = time.monotonic()
    while time.monotonic() - t0 < segundos:
        trabajo(4096)


def carga_gpu(segundos):
    from numba import cuda

    @cuda.jit
    def trabajo(x, it):
        i = cuda.grid(1)
        if i < x.shape[0]:
            v = x[i]
            for _ in range(it):
                v = v * 1.0000001 + 0.5
            x[i] = v
    x = cuda.to_device(np.ones(1 << 22, dtype=np.float32))
    trabajo[(x.shape[0] + 255) // 256, 256](x, 10)
    cuda.synchronize()
    t0 = time.monotonic()
    while time.monotonic() - t0 < segundos:
        trabajo[(x.shape[0] + 255) // 256, 256](x, 4000)
        cuda.synchronize()


def main():
    p = argparse.ArgumentParser(description="Qué fuentes de energía de CPU y GPU funcionan en este equipo (WSL)")
    p.add_argument("--url-lhm", default=None)
    p.add_argument("--salida", default="sondeo_energia.json")
    p.add_argument("--segundos", type=float, default=6.0)
    a = p.parse_args()
    r = {"fecha": time.strftime("%Y-%m-%d %H:%M:%S")}

    print("== CPU ==")
    r["rapl_wsl"] = probar("RAPL en WSL (/sys/class/powercap)", lambda: me.rapl_disponible() or
                           (_ for _ in ()).throw(FileNotFoundError(me.RAPL)))
    print("== GPU ==")
    r["nvml_potencia"] = probar("NVML potencia (W)", lambda: me.potencia_nvml()())

    def contador():
        leer = me.energia_nvml()
        e0, t0 = leer(), time.monotonic()
        time.sleep(1.0)
        e1, t1 = leer(), time.monotonic()
        return f"{(e1 - e0) / (t1 - t0):.1f} W promedio en 1 s (contador de energía)"
    r["nvml_energia"] = probar("NVML contador de energía", contador)
    if shutil.which("nvidia-smi"):
        for campo in ("power.draw", "power.draw.instant", "power.draw.average"):
            r[f"smi_{campo}"] = probar(f"nvidia-smi {campo}", lambda c=campo: me.leer_smi(c))
        try:
            r["smi_q_power"] = subprocess.run(["nvidia-smi", "-q", "-d", "POWER"], capture_output=True, text=True,
                                              timeout=20).stdout
        except Exception as e:
            r["smi_q_power"] = repr(e)
    print("== LibreHardwareMonitor (Windows) ==")
    urls = [a.url_lhm] if a.url_lhm else me.urls_lhm()
    print("URLs probadas:", ", ".join(urls))
    lhm = me.buscar_lhm(urls)
    if lhm:
        print(f"[OK]    responde en {lhm['url']}")
        pot = [s for s in lhm["sensores"] if me.es_potencia(s)]
        for s in pot:
            print(f"        {' / '.join(s['camino'])} = {s['valor']} (id {s.get('id')})")
        print(f"        sensor CPU elegido: {lhm['cpu'] and ' / '.join(lhm['cpu']['camino'])}")
        print(f"        sensor GPU elegido: {lhm['gpu'] and ' / '.join(lhm['gpu']['camino'])}")
        t0 = time.monotonic()
        me.descargar_json(lhm["url"])
        print(f"        latencia de una lectura: {1000 * (time.monotonic() - t0):.1f} ms")
        r["lhm"] = {"url": lhm["url"], "potencias": pot, "cpu": lhm["cpu"], "gpu": lhm["gpu"]}
    else:
        print("[FALLA] LibreHardwareMonitor no respondió (¿abierto como administrador, servidor web activo en "
              "0.0.0.0:8085 y regla de firewall?)")
        r["lhm"] = None

    print("== Prueba con carga (reposo vs carga) ==")
    try:
        m = me.Medidores(url_lhm=lhm["url"] if lhm else None)
        m.iniciar()
        res = {comp: {} for comp in m.componentes()}
        for nombre, fn in (("reposo", time.sleep), ("carga_cpu", carga_cpu), ("carga_gpu", carga_gpu)):
            ta = m.marcar(nombre + "_0")
            fn(a.segundos)
            tb = m.marcar(nombre + "_1")
            for comp in m.componentes():
                res[comp][nombre] = m.energia(comp, nombre + "_0", nombre + "_1", ta, tb)
        m.parar()
        for comp, v in res.items():
            linea = " | ".join(f"{k} {x['watts']:.1f} W" for k, x in v.items() if x.get("watts") is not None)
            print(f"        {comp}: {linea}")
        r["prueba_carga"] = {"resultado": res, "medidores": m.info()}
        print("        fuentes:", json.dumps(m.info()["descripcion"], ensure_ascii=False, default=str)[:400])
        print("        tasas:", {k: v["hz"] for k, v in m.info()["muestreadores"].items()})
    except Exception as e:
        print(f"[FALLA] prueba con carga: {e!r}")
        r["prueba_carga"] = {"error": repr(e)}

    with open(a.salida, "w", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=2, default=str)
    print(f"Sondeo guardado en {os.path.abspath(a.salida)}")
    ok_cpu = r["rapl_wsl"]["ok"] or bool(lhm and lhm.get("cpu"))
    ok_gpu = r["nvml_energia"]["ok"] or r["nvml_potencia"]["ok"] or any(
        r.get(f"smi_{c}", {}).get("ok") for c in ("power.draw", "power.draw.instant")) or bool(lhm and lhm.get("gpu"))
    print(f"RESUMEN: energía de CPU {'disponible' if ok_cpu else 'NO disponible'} | "
          f"energía de GPU {'disponible' if ok_gpu else 'NO disponible'}")
    raise SystemExit(0 if ok_cpu and ok_gpu else 1)


if __name__ == "__main__":
    main()
