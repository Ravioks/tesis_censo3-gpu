import argparse
import glob
import json
import os

from figuras_articulo import MACROS

NOMBRE_MODO = {"gpu_censo": "GpuCenso", "gpu_extremo": "GpuExtremo", "cpu6_censo": "CpuSixCenso",
               "cpu6_extremo": "CpuSixExtremo", "cpu1_censo": "CpuOneCenso"}


def julios(x):
    if x < 1:
        return f"{x * 1e3:.1f}\\,mJ" if x * 1e3 >= 100 else f"{x * 1e3:.2f}\\,mJ"
    return f"{x:.0f}\\,J" if x >= 100 else (f"{x:.1f}\\,J" if x >= 10 else f"{x:.2f}\\,J")


def watts(x):
    return f"{x:.1f}\\,W"


def veces(x):
    return f"{x:.0f}$\\times$" if x >= 100 else f"{x:.1f}$\\times$"


PARES = {"RazonCenso": ("cpu6_censo", "gpu_censo"), "RazonCensoOne": ("cpu1_censo", "gpu_censo"),
         "RazonExtremo": ("cpu6_extremo", "gpu_extremo")}


def ultimo_por_grafo(carpeta):
    res = {}
    for ruta in sorted(glob.glob(os.path.join(carpeta, "energia_*.json"))):
        with open(ruta, encoding="utf-8") as f:
            d = json.load(f)
        clave = "ligero" if d["parametros"].get("sin_gc") else "base"
        res.setdefault(d["grafo"], {})[clave] = d
    return res


def tiempo_interno(d, modo):
    t = [x["tiempo_por_corrida"]["media"] for x in d["mediciones"] if x.get("modo") == modo and "tiempo_por_corrida" in x]
    return sum(t) / len(t) if t else None


def segundos(x):
    return f"{x * 1e3:.0f}\\,ms" if x < 1 else f"{x:.2f}\\,s"


def macros_modo(m, q, d, modo):
    b = d["resumen"][modo]
    m[q + "J"] = julios(b["julios_por_corrida"]["media"])
    m[q + "JSd"] = f"{b['julios_por_corrida']['desv']:.2f}\\,J"
    m[q + "JDin"] = julios(b["julios_dinamicos_por_corrida"]["media"])
    m[q + "JCpu"] = julios(b["julios_cpu_por_corrida"]["media"])
    m[q + "JGpu"] = julios(b["julios_gpu_por_corrida"]["media"])
    m[q + "W"] = watts(b["watts_total_bucle"]["media"])
    m[q + "WCpu"] = watts(b["watts_cpu_bucle"]["media"])
    m[q + "WGpu"] = watts(b["watts_gpu_bucle"]["media"])
    m[q + "WReposo"] = watts(b["watts_total_reposo"]["media"])
    m[q + "T"] = segundos(b["tiempo_por_corrida_s"]["media"])
    ti = tiempo_interno(d, modo)
    if ti is not None:
        m[q + "TFase"] = segundos(ti)
    m[q + "Rondas"] = str(b["rondas"])


def macros_razones(m, p, cpu, gpu, suf=""):
    for nombre, (mc, mg) in PARES.items():
        if mc not in cpu["resumen"] or mg not in gpu["resumen"]:
            continue
        c, g = cpu["resumen"][mc], gpu["resumen"][mg]
        pares = (("", "julios_por_corrida", "julios_por_corrida"),
                 ("Din", "julios_dinamicos_por_corrida", "julios_dinamicos_por_corrida"),
                 ("Tiempo", "tiempo_por_corrida_s", "tiempo_por_corrida_s"),
                 ("CpuSolo", "julios_cpu_por_corrida", "julios_por_corrida"))
        for s2, kc, kg in pares:
            if g[kg]["media"] > 0:
                m[p + nombre + suf + s2] = veces(c[kc]["media"] / g[kg]["media"])


def globales(res, m):
    desv, dif, hz = [], [], []
    for ds in res.values():
        for clave, d in ds.items():
            for modo, b in d["resumen"].items():
                if clave == "base" and modo.startswith("gpu") and "ligero" in ds:
                    continue
                j = b["julios_por_corrida"]
                if j["media"] > 0:
                    desv.append(j["desv"] / j["media"])
            for x in d["mediciones"]:
                c = x.get("componentes", {})
                if x.get("modo", "").startswith("gpu") and "watts_bucle" in c.get("gpu", {}) \
                        and "watts_bucle" in c.get("gpu_lhm", {}):
                    dif.append(abs(c["gpu_lhm"]["watts_bucle"] / c["gpu"]["watts_bucle"] - 1))
            hz.append(d.get("medidores", {}).get("muestreadores", {}).get("cpu", {}).get("hz"))
    hz = [h for h in hz if h]
    if desv:
        m["EnDesvRelMax"] = f"{100 * max(desv):.1f}\\,\\%"
    if dif:
        m["EnDifGpuLhmMax"] = f"{100 * max(dif):.1f}\\,\\%"
    if hz:
        m["EnHzLhm"] = f"{min(hz):.1f}\\,Hz"


def cifras_perfil(ruta, m):
    if not ruta or not os.path.exists(ruta):
        return
    with open(ruta, encoding="utf-8") as f:
        p = json.load(f)
    pre = MACROS.get(p["grafo"], "") + "En"
    m[pre + "Gc"] = segundos(p["gc_collect_s"])
    m[pre + "PerfilPared"] = segundos(p["normal"]["pared_s"])
    m[pre + "PerfilParedSinGc"] = segundos(p["sin_gc"]["pared_s"])
    m[pre + "PerfilFase"] = segundos(p["sin_gc"]["censo_s"])
    m[pre + "PerfilFuera"] = f"{p['normal']['fuera_s'] * 1e3:.0f}\\,ms"
    m[pre + "PerfilFueraSinGc"] = f"{p['sin_gc']['fuera_s'] * 1e3:.1f}\\,ms"


def cifras(res):
    m = {}
    globales(res, m)
    for nombre, ds in res.items():
        p = MACROS.get(nombre)
        if p is None or "base" not in ds:
            continue
        p += "En"
        base = ds["base"]
        for modo in base["resumen"]:
            macros_modo(m, p + NOMBRE_MODO[modo], base, modo)
        macros_razones(m, p, base, base)
        if "ligero" in ds:
            lig = ds["ligero"]
            for modo in lig["resumen"]:
                if modo.startswith("gpu"):
                    macros_modo(m, p + NOMBRE_MODO[modo] + "Ligero", lig, modo)
            macros_razones(m, p, base, lig, "Ligero")
    return m


def main():
    p = argparse.ArgumentParser(description="cifras_energia.tex a partir de resultados/v3_energia")
    p.add_argument("--resultados", default=os.path.join("resultados", "v3_energia"))
    p.add_argument("--salida", default=os.path.join("resultados", "v3_energia", "cifras_energia.tex"))
    p.add_argument("--perfil", default=None, help="perfil_lj.json de perfil_llamada.py")
    a = p.parse_args()
    res = ultimo_por_grafo(a.resultados)
    m = cifras(res)
    cifras_perfil(a.perfil, m)
    todos = [d for ds in res.values() for d in ds.values()]
    commits = sorted({d.get("commit") for d in todos if d.get("commit")})
    fuentes = sorted({json.dumps(d["medidores"]["descripcion"].get(c, {}).get("fuente")) for d in todos
                      for c in ("cpu", "gpu")})
    with open(a.salida, "w", encoding="utf-8") as f:
        f.write("% Generado por cifras_energia.py a partir de resultados/v3_energia. No editar a mano.\n")
        f.write(f"% Commits: {', '.join(commits)} | fuentes: {', '.join(fuentes)}\n")
        for k in sorted(m):
            f.write(f"\\newcommand{{\\num{k}}}{{{m[k]}}}\n")
    print(f"Grafos: {', '.join(res)} | {len(m)} macros -> {a.salida}")


if __name__ == "__main__":
    main()
