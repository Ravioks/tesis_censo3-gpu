import argparse
import glob
import json
import os

from figuras_articulo import MACROS, ms


def ultimo_por_grafo(carpeta):
    res = {}
    for ruta in sorted(glob.glob(os.path.join(carpeta, "cugraph_*.json"))):
        with open(ruta, encoding="utf-8") as f:
            d = json.load(f)
        res[d["grafo"]] = d
    return res


def media(bloque, campo):
    return bloque["tiempos"][campo]["media"]


def cifras(res):
    m = {}
    for nombre, d in res.items():
        p = MACROS.get(nombre)
        if p is None:
            continue
        p += "Cug"
        modos = d["modos"]
        h = modos.get("host", {})
        g = modos.get("cugraph", {})
        c = modos.get("cudf", {})
        gd = modos.get("cugraph_dirigido", {})
        if "tiempos" in h:
            m[p + "HostCarga"] = ms(media(h, "carga"))
            m[p + "HostPreparacion"] = ms(media(h, "preparacion"))
            m[p + "HostCenso"] = ms(media(h, "censo"))
            m[p + "HostExtremo"] = ms(media(h, "extremo"))
        for clave, b in (("Grafo", g), ("Tabla", c), ("GrafoDir", gd)):
            if "tiempos" not in b:
                continue
            m[p + clave + "Adaptador"] = ms(media(b, "adaptador"))
            m[p + clave + "PreprocesoGpu"] = ms(media(b, "preproceso_gpu"))
            m[p + clave + "Preparacion"] = ms(media(b, "preparacion"))
            m[p + clave + "Censo"] = ms(media(b, "censo"))
            m[p + clave + "Extremo"] = ms(media(b, "extremo"))
            m[p + clave + "Iguales"] = "yes" if b.get("iguales_a_referencia") else "no"
            if "tiempos" in h:
                m[p + clave + "Ahorro"] = ms(media(h, "extremo") - media(b, "extremo"))
            a = b.get("adaptador") or {}
            m[p + clave + "Reordenado"] = "yes" if a.get("reordenado") else "no"
            m[p + clave + "Simetrizado"] = "yes" if a.get("simetrizado") else "no"
            for k, nombre_k in (("t_simetrizar_s", "Simetrizar"), ("t_construccion_csr_s", "ConstruccionCsr"),
                                ("t_conversion_s", "Conversion"), ("t_revision_s", "Revision")):
                if a.get(k) is not None:
                    m[p + clave + nombre_k] = ms(a[k])
        e = d.get("extra", {})
        if "construccion_cugraph_s" in e:
            m[p + "Construccion"] = ms(e["construccion_cugraph_s"])
        if "subida_tabla_aristas_s" in e:
            m[p + "SubidaTabla"] = ms(e["subida_tabla_aristas_s"])
        tc = e.get("triangle_count_cugraph", {})
        if "tiempos" in tc:
            m[p + "TriangleCount"] = ms(tc["tiempos"]["media"])
        se = e.get("censo_sin_etiquetas", {})
        if "tiempos" in se:
            m[p + "SinEtTriKernel"] = ms(se["tiempos"]["triangulos_kernel"]["media"])
            m[p + "SinEtCenso"] = ms(se["tiempos"]["censo"]["media"])
        if "tiempos" in g:
            m[p + "ConEtTriKernel"] = ms(g["tiempos"]["triangulos_kernel"]["media"])
    return m


def main():
    p = argparse.ArgumentParser(description="cifras_cugraph.tex a partir de resultados/v2_cugraph")
    p.add_argument("--resultados", default=os.path.join("resultados", "v2_cugraph"))
    p.add_argument("--salida", default=os.path.join("resultados", "v2_cugraph", "cifras_cugraph.tex"))
    a = p.parse_args()
    res = ultimo_por_grafo(a.resultados)
    m = cifras(res)
    commits = sorted({d.get("commit") for d in res.values() if d.get("commit")})
    with open(a.salida, "w", encoding="utf-8") as f:
        f.write("% Generado por cifras_cugraph.py a partir de resultados/v2_cugraph. No editar a mano.\n")
        f.write(f"% Commits de las corridas: {', '.join(commits)}\n")
        for k in sorted(m):
            f.write(f"\\newcommand{{\\num{k}}}{{{m[k]}}}\n")
    print(f"Grafos: {', '.join(res)} | {len(m)} macros -> {a.salida}")


if __name__ == "__main__":
    main()
