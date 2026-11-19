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


def cifras(res):
    m = {}
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
    a = p.parse_args()
    res = ultimo_por_grafo(a.resultados)
    m = cifras(res)
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
