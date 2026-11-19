import argparse
import glob
import json
import os

from figuras_articulo import MACROS

NOMBRE_MODO = {"gpu_censo": "GpuCenso", "gpu_extremo": "GpuExtremo", "cpu6_censo": "CpuSixCenso",
               "cpu6_extremo": "CpuSixExtremo", "cpu1_censo": "CpuOneCenso"}
NOMBRE_RAZON = {"cpu6_censo/gpu_censo": "RazonCenso", "cpu1_censo/gpu_censo": "RazonCensoOne",
                "cpu6_extremo/gpu_extremo": "RazonExtremo"}


def julios(x):
    if x < 1:
        return f"{x * 1e3:.1f}\\,mJ" if x * 1e3 >= 100 else f"{x * 1e3:.2f}\\,mJ"
    return f"{x:.0f}\\,J" if x >= 100 else (f"{x:.1f}\\,J" if x >= 10 else f"{x:.2f}\\,J")


def watts(x):
    return f"{x:.1f}\\,W"


def veces(x):
    return f"{x:.0f}$\\times$" if x >= 100 else f"{x:.1f}$\\times$"


def ultimo_por_grafo(carpeta):
    res = {}
    for ruta in sorted(glob.glob(os.path.join(carpeta, "energia_*.json"))):
        with open(ruta, encoding="utf-8") as f:
            d = json.load(f)
        res[d["grafo"]] = d
    return res


def cifras(res):
    m = {}
    for nombre, d in res.items():
        p = MACROS.get(nombre)
        if p is None:
            continue
        p += "En"
        for modo, b in d["resumen"].items():
            q = p + NOMBRE_MODO[modo]
            m[q + "J"] = julios(b["julios_por_corrida"]["media"])
            m[q + "JSd"] = julios(b["julios_por_corrida"]["desv"])
            m[q + "JDin"] = julios(b["julios_dinamicos_por_corrida"]["media"])
            m[q + "W"] = watts(b["watts_total_bucle"]["media"])
            m[q + "WCpu"] = watts(b["watts_cpu_bucle"]["media"])
            m[q + "WGpu"] = watts(b["watts_gpu_bucle"]["media"])
            m[q + "WReposo"] = watts(b["watts_total_reposo"]["media"])
            m[q + "Rondas"] = str(b["rondas"])
        for k, v in d["razones"].items():
            for campo, suf in (("energia", ""), ("energia_dinamica", "Din"), ("tiempo", "Tiempo")):
                if v.get(campo) is not None:
                    m[p + NOMBRE_RAZON[k] + suf] = veces(v[campo])
    return m


def main():
    p = argparse.ArgumentParser(description="cifras_energia.tex a partir de resultados/v3_energia")
    p.add_argument("--resultados", default=os.path.join("resultados", "v3_energia"))
    p.add_argument("--salida", default=os.path.join("resultados", "v3_energia", "cifras_energia.tex"))
    a = p.parse_args()
    res = ultimo_por_grafo(a.resultados)
    m = cifras(res)
    commits = sorted({d.get("commit") for d in res.values() if d.get("commit")})
    fuentes = sorted({json.dumps(d["medidores"]["descripcion"].get(c, {}).get("fuente")) for d in res.values()
                      for c in ("cpu", "gpu")})
    with open(a.salida, "w", encoding="utf-8") as f:
        f.write("% Generado por cifras_energia.py a partir de resultados/v3_energia. No editar a mano.\n")
        f.write(f"% Commits: {', '.join(commits)} | fuentes: {', '.join(fuentes)}\n")
        for k in sorted(m):
            f.write(f"\\newcommand{{\\num{k}}}{{{m[k]}}}\n")
    print(f"Grafos: {', '.join(res)} | {len(m)} macros -> {a.salida}")


if __name__ == "__main__":
    main()
