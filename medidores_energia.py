import glob
import json
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.request

import numpy as np

NUMERO = re.compile(r"[-+]?\d+(?:[.,]\d+)?")
RAPL = "/sys/class/powercap/intel-rapl:0/energy_uj"


def ahora():
    return time.monotonic()


def integrar(muestras, t0, t1):
    if t1 <= t0:
        return {"julios": 0.0, "watts": 0.0, "muestras": 0}
    if not muestras:
        return {"julios": None, "watts": None, "muestras": 0}
    t = np.array([m[0] for m in muestras], dtype=float)
    p = np.array([m[1] for m in muestras], dtype=float)
    orden = np.argsort(t)
    t, p = t[orden], p[orden]
    dentro = (t > t0) & (t < t1)
    tt = np.concatenate([[t0], t[dentro], [t1]])
    pp = np.concatenate([[np.interp(t0, t, p)], p[dentro], [np.interp(t1, t, p)]])
    julios = float(np.sum((tt[1:] - tt[:-1]) * (pp[1:] + pp[:-1]) / 2))
    return {"julios": julios, "watts": julios / (t1 - t0), "muestras": int(dentro.sum())}


class Muestreador(threading.Thread):
    def __init__(self, nombre, leer, periodo):
        super().__init__(daemon=True)
        self.nombre = nombre
        self.leer = leer
        self.periodo = periodo
        self.muestras = []
        self.errores = 0
        self.ultimo_error = None
        self._parar = threading.Event()

    def run(self):
        while not self._parar.is_set():
            t = ahora()
            try:
                w = self.leer()
            except Exception as e:
                self.errores += 1
                self.ultimo_error = repr(e)
                w = None
            if w is not None:
                self.muestras.append((t, float(w)))
            espera = self.periodo - (ahora() - t)
            if espera > 0:
                self._parar.wait(espera)

    def parar(self):
        self._parar.set()
        self.join(timeout=5)

    def energia(self, t0, t1):
        return integrar(self.muestras, t0, t1)

    def info(self):
        n = len(self.muestras)
        dur = self.muestras[-1][0] - self.muestras[0][0] if n > 1 else 0.0
        return {"fuente": self.nombre, "muestras": n, "hz": (n - 1) / dur if dur else None,
                "errores": self.errores, "ultimo_error": self.ultimo_error}


class MuestreadorProceso(Muestreador):
    def __init__(self, nombre, comando):
        super().__init__(nombre, None, 0)
        self.comando = comando
        self.proc = None

    def run(self):
        self.proc = subprocess.Popen(self.comando, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                                     bufsize=1)
        for linea in self.proc.stdout:
            if self._parar.is_set():
                break
            m = NUMERO.search(linea)
            if m:
                self.muestras.append((ahora(), float(m.group(0).replace(",", "."))))
            else:
                self.errores += 1
                self.ultimo_error = linea.strip()[:80]

    def parar(self):
        self._parar.set()
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
        self.join(timeout=5)


class Contador:
    def __init__(self, nombre, leer_julios, rango=None):
        self.nombre = nombre
        self.leer_julios = leer_julios
        self.rango = rango
        self.marcas = {}

    def marcar(self, clave):
        self.marcas[clave] = (ahora(), self.leer_julios())

    def energia_entre(self, a, b):
        (t0, e0), (t1, e1) = self.marcas[a], self.marcas[b]
        d = e1 - e0
        if d < 0 and self.rango:
            d += self.rango
        return {"julios": d, "watts": d / (t1 - t0) if t1 > t0 else 0.0, "muestras": 2}


def nvml_dispositivo(indice=0):
    import pynvml
    pynvml.nvmlInit()
    return pynvml, pynvml.nvmlDeviceGetHandleByIndex(indice)


def potencia_nvml(indice=0):
    pynvml, h = nvml_dispositivo(indice)
    return lambda: pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0


def energia_nvml(indice=0):
    pynvml, h = nvml_dispositivo(indice)
    return lambda: pynvml.nvmlDeviceGetTotalEnergyConsumption(h) / 1000.0


def comando_smi(campo, periodo_ms):
    return ["nvidia-smi", f"--query-gpu={campo}", "--format=csv,noheader,nounits", f"-lms={periodo_ms}"]


def leer_smi(campo):
    salida = subprocess.run(["nvidia-smi", f"--query-gpu={campo}", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, timeout=10).stdout.strip()
    m = NUMERO.search(salida)
    if not m:
        raise ValueError(f"nvidia-smi {campo}: {salida!r}")
    return float(m.group(0).replace(",", "."))


def rapl_disponible():
    try:
        with open(RAPL) as f:
            int(f.read())
        return True
    except Exception:
        return False


def energia_rapl():
    with open(os.path.join(os.path.dirname(RAPL), "max_energy_range_uj")) as f:
        rango = int(f.read()) / 1e6

    def leer():
        with open(RAPL) as f:
            return int(f.read()) / 1e6
    return leer, rango


def ip_windows():
    ips = []
    try:
        salida = subprocess.run(["ip", "route", "show", "default"], capture_output=True, text=True).stdout
        m = re.search(r"default via (\S+)", salida)
        if m:
            ips.append(m.group(1))
    except Exception:
        pass
    try:
        with open("/etc/resolv.conf") as f:
            for linea in f:
                if linea.startswith("nameserver"):
                    ips.append(linea.split()[1])
    except Exception:
        pass
    return list(dict.fromkeys(ips))


def urls_lhm(puerto=8085):
    return [f"http://localhost:{puerto}/data.json"] + [f"http://{ip}:{puerto}/data.json" for ip in ip_windows()]


def descargar_json(url, timeout=2.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", errors="replace"))


def sensores_lhm(arbol):
    salida = []

    def recorrer(nodo, camino):
        texto = str(nodo.get("Text", ""))
        hijos = nodo.get("Children") or []
        valor = str(nodo.get("Value", ""))
        if not hijos and valor:
            salida.append({"camino": camino + [texto], "texto": texto, "valor": valor,
                           "id": nodo.get("SensorId"), "tipo": nodo.get("Type")})
        for h in hijos:
            recorrer(h, camino + [texto])
    recorrer(arbol, [])
    return salida


def es_potencia(s):
    return (s.get("tipo") == "Power") or ("/power/" in str(s.get("id") or "")) or s["valor"].strip().endswith("W")


def valor_watts(s):
    m = NUMERO.search(s["valor"])
    return float(m.group(0).replace(",", ".")) if m else None


def elegir_sensor(sensores, cual, filtro=None):
    pot = [s for s in sensores if es_potencia(s) and valor_watts(s) is not None]
    if filtro:
        cand = [s for s in pot if filtro.lower() in (" / ".join(s["camino"]) + " " + str(s.get("id"))).lower()]
        return cand[0] if cand else None
    if cual == "cpu":
        cand = [s for s in pot if ("cpu" in str(s.get("id") or "").lower() or
                                   any(k in " ".join(s["camino"]) for k in ("Intel", "AMD", "Core", "Ryzen")))
                and "package" in s["texto"].lower() and "gpu" not in str(s.get("id") or "").lower()]
    else:
        cand = [s for s in pot if ("gpu" in str(s.get("id") or "").lower() or "NVIDIA" in " ".join(s["camino"]))
                and ("package" in s["texto"].lower() or "power" in s["texto"].lower() or "board" in s["texto"].lower())]
    return cand[0] if cand else None


def lector_lhm(url, sensor):
    clave_id = sensor.get("id")
    camino = sensor["camino"]

    def leer():
        for s in sensores_lhm(descargar_json(url)):
            if (clave_id and s.get("id") == clave_id) or (not clave_id and s["camino"] == camino):
                return valor_watts(s)
        raise KeyError(f"sensor {clave_id or ' / '.join(camino)} no encontrado")
    return leer


def buscar_lhm(urls=None, filtro_cpu=None, filtro_gpu=None):
    for url in urls or urls_lhm():
        try:
            sensores = sensores_lhm(descargar_json(url))
        except Exception:
            continue
        return {"url": url, "sensores": sensores, "cpu": elegir_sensor(sensores, "cpu", filtro_cpu),
                "gpu": elegir_sensor(sensores, "gpu", filtro_gpu)}
    return None


class Medidores:
    def __init__(self, fuente_cpu="auto", fuente_gpu="auto", url_lhm=None, filtro_cpu=None, filtro_gpu=None,
                 periodo=0.05, periodo_lhm=0.1, potencia_falsa=None):
        self.muestreadores = {}
        self.contadores = {}
        self.descripcion = {}
        self.lhm = None
        if "lhm" in (fuente_cpu, fuente_gpu) or "auto" in (fuente_cpu, fuente_gpu):
            self.lhm = buscar_lhm([url_lhm] if url_lhm else None, filtro_cpu, filtro_gpu)
        self._preparar_cpu(fuente_cpu, periodo_lhm, potencia_falsa)
        self._preparar_gpu(fuente_gpu, periodo, periodo_lhm, potencia_falsa)
        if self.lhm and self.lhm.get("gpu") and fuente_gpu != "falsa" and \
                self.descripcion.get("gpu", {}).get("fuente") != "lhm":
            self.muestreadores["gpu_lhm"] = Muestreador("lhm_gpu", lector_lhm(self.lhm["url"], self.lhm["gpu"]),
                                                        periodo_lhm)
            self.descripcion["gpu_lhm"] = {"fuente": "lhm", "url": self.lhm["url"], "sensor": self.lhm["gpu"]}

    def _preparar_cpu(self, fuente, periodo_lhm, falsa):
        if fuente == "falsa":
            self.muestreadores["cpu"] = Muestreador("falsa", lambda: falsa["cpu"], 0.02)
        elif fuente in ("auto", "rapl") and rapl_disponible():
            leer, rango = energia_rapl()
            self.contadores["cpu"] = Contador("rapl", leer, rango)
        elif fuente in ("auto", "lhm") and self.lhm and self.lhm.get("cpu"):
            self.muestreadores["cpu"] = Muestreador("lhm_cpu", lector_lhm(self.lhm["url"], self.lhm["cpu"]),
                                                    periodo_lhm)
            self.descripcion["cpu"] = {"fuente": "lhm", "url": self.lhm["url"], "sensor": self.lhm["cpu"]}
        if "cpu" not in self.muestreadores and "cpu" not in self.contadores:
            raise RuntimeError("Sin fuente de energía de CPU: RAPL no existe en WSL y LibreHardwareMonitor no "
                               "respondió (revisa LEEME_energia.md)")
        self.descripcion.setdefault("cpu", {"fuente": (self.contadores.get("cpu") or
                                                       self.muestreadores.get("cpu")).nombre})

    def _preparar_gpu(self, fuente, periodo, periodo_lhm, falsa):
        if fuente == "falsa":
            self.muestreadores["gpu"] = Muestreador("falsa", lambda: falsa["gpu"], 0.02)
            self.descripcion["gpu"] = {"fuente": "falsa"}
            return
        if fuente in ("auto", "nvml_energia"):
            try:
                leer = energia_nvml()
                leer()
                self.contadores["gpu"] = Contador("nvml_energia", leer)
                self.descripcion["gpu"] = {"fuente": "nvml_energia"}
            except Exception as e:
                if fuente == "nvml_energia":
                    raise
                self.descripcion["gpu_nvml_energia_error"] = repr(e)
        if fuente in ("auto", "nvml") and "gpu" not in self.contadores:
            try:
                leer = potencia_nvml()
                leer()
                self.muestreadores["gpu"] = Muestreador("nvml_potencia", leer, periodo)
                self.descripcion["gpu"] = {"fuente": "nvml_potencia"}
            except Exception as e:
                if fuente == "nvml":
                    raise
                self.descripcion["gpu_nvml_error"] = repr(e)
        if fuente in ("auto", "smi") and "gpu" not in self.contadores and "gpu" not in self.muestreadores \
                and shutil.which("nvidia-smi"):
            for campo in ("power.draw.instant", "power.draw"):
                try:
                    leer_smi(campo)
                except Exception:
                    continue
                self.muestreadores["gpu"] = MuestreadorProceso(f"smi_{campo}",
                                                               comando_smi(campo, int(periodo * 1000)))
                self.descripcion["gpu"] = {"fuente": f"nvidia-smi {campo}"}
                break
        if fuente in ("auto", "lhm") and "gpu" not in self.contadores and "gpu" not in self.muestreadores \
                and self.lhm and self.lhm.get("gpu"):
            self.muestreadores["gpu"] = Muestreador("lhm_gpu", lector_lhm(self.lhm["url"], self.lhm["gpu"]),
                                                    periodo_lhm)
            self.descripcion["gpu"] = {"fuente": "lhm", "url": self.lhm["url"], "sensor": self.lhm["gpu"]}
        if "gpu" not in self.contadores and "gpu" not in self.muestreadores:
            raise RuntimeError("Sin fuente de energía de GPU (NVML, nvidia-smi ni LibreHardwareMonitor)")

    def iniciar(self):
        for m in self.muestreadores.values():
            m.start()
        time.sleep(0.5)

    def marcar(self, clave):
        t = ahora()
        for c in self.contadores.values():
            c.marcar(clave)
        return t

    def parar(self):
        for m in self.muestreadores.values():
            m.parar()

    def energia(self, componente, a, b, ta, tb):
        if componente in self.contadores:
            return self.contadores[componente].energia_entre(a, b)
        return self.muestreadores[componente].energia(ta, tb)

    def componentes(self):
        return sorted(set(self.contadores) | set(self.muestreadores))

    def info(self):
        return {"descripcion": self.descripcion,
                "muestreadores": {k: m.info() for k, m in self.muestreadores.items()},
                "contadores": {k: c.nombre for k, c in self.contadores.items()}}

    def muestras(self):
        return {k: np.array(m.muestras, dtype=float) for k, m in self.muestreadores.items()}
