# Censo etiquetado de 3 nodos en GPU

Herramienta de tesis para contar, en un grafo grande no dirigido con etiquetas en nodos y aristas, **todos los subgrafos conexos de 3 nodos**, agrupados por patrón de etiquetas. Corre en GPU NVIDIA con kernels CUDA escritos en Python (Numba) sobre el grafo en **CSR**, el mismo formato de adyacencia que usa RAPIDS cuGraph.

Hay dos formas posibles de 3 nodos: **caminos** `a – c – b` (2 aristas) y **triángulos** (3 aristas). Por cada patrón se entregan tres censos:

| Censo | Qué cuenta |
|---|---|
| Caminos **no inducidos** | Todo camino `a – c – b`, aunque `a` y `b` también estén unidos |
| Caminos **inducidos** | Solo los tríos cuyo subgrafo inducido es exactamente un camino (sin la arista `a – b`) |
| **Triángulos** | Cada trío de nodos mutuamente unidos. Es igual en ambas convenciones |

Cada triángulo contiene 3 caminos no inducidos, así que el inducido se obtiene **restando, patrón por patrón**, los caminos que están dentro de triángulos. En total: caminos inducidos = W − 3T, con W = Σ C(grado, 2) y T = número de triángulos.

La documentación técnica (algoritmos, kernels y funciones) está en [`docs/FUNCIONAMIENTO.md`](docs/FUNCIONAMIENTO.md). Los cambios de cada versión están en [`docs/CAMBIOS.md`](docs/CAMBIOS.md).

## Contenido

| Archivo | Para qué sirve |
|---|---|
| `censo_3n.py` | **Punto de entrada principal.** Censo completo en GPU: API `censo3()` y línea de comandos |
| `barrido.py` | **Compara configuraciones GPU** (estrategia × memoria × tamaño de bloque), valida cada una y guarda la mejor en `mejor_config.json` |
| `censo_3n_cpu_njit.py` | Baseline de CPU compilada (Numba `njit`), 1 hilo y multihilo, con la misma codificación que la GPU |
| `demo.py` | Corre los experimentos (validación, barrido, CPU vs GPU, escalabilidad, grafo real) y genera el informe |
| `censo_3n2a_gpu.py` | Caminos en GPU: CSR, 4 estrategias × 3 modos de memoria, verificación camino por camino |
| `censo_3n3a_gpu.py` | Triángulos en GPU: orientación por grado y 3 estrategias de intersección × 3 modos de memoria |
| `censo_3n2a.py` | Referencia secuencial en Python (oráculo), generadores de grafos chicos, lector gSpan y utilidades |
| `generadores.py` | Generadores rápidos (NumPy/Numba) que arman el CSR directo, para grafos de millones de aristas |
| `datos_reales.py` | Descarga grafos de SNAP, los limpia, les asigna etiquetas y los guarda en caché |
| `tests/` | Tests con pytest: fuerza bruta → oráculo → njit → GPU, más compilación a PTX |
| `docs/` | Documentación técnica, historial de cambios y resultados guardados |

Los nombres siguen la forma del subgrafo: `3n2a` = 3 nodos y 2 aristas (caminos), `3n3a` = 3 nodos y 3 aristas (triángulos), `3n` = censo completo.

## Requisitos

- **GPU NVIDIA con CUDA** para la parte paralela. Sin GPU funcionan la referencia Python, el baseline njit, los tests de CPU y la compilación a PTX.
- **Driver NVIDIA reciente.** En WSL2 el driver se instala en Windows, no dentro de Linux.
- **Python 3.10 a 3.13.**
- **numpy < 2.5:** `numba-cuda` 0.30 usa `np.row_stack`, que numpy 2.5 eliminó. Ya está fijado en `requirements.txt`.
- **Memoria para LiveJournal:** funcionó con los 7,7 GB de RAM que WSL2 asigna por defecto. En GPU usa alrededor de 1 GB; en disco, ~1 GB entre descarga y caché.
- **Orkut** (opcional, 3,5 veces más grande): se recomiendan 16 GB de RAM y ~3 GB de memoria de GPU.

Entorno de desarrollo: WSL2 con Ubuntu, Python 3.12, NVIDIA RTX 5060 (8 GB, sm_120), driver 610.57 (CUDA 13.3).

## Instalación

```bash
git clone https://github.com/Ravioks/tesis_FSM_MartinSuarez.git
cd tesis_FSM_MartinSuarez
conda create -n censo-gpu python=3.12 -y
conda activate censo-gpu
pip install -r requirements.txt
pip install "numba-cuda[cu13]"
```

Usa `[cu12]` en vez de `[cu13]` si `nvidia-smi` muestra CUDA 12.x. Estos paquetes traen las librerías de CUDA que necesita Numba, así que no hace falta instalar el CUDA Toolkit.

Si ya tienes el entorno `censo-gpu` funcionando, no lo recrees: basta con `pip install -r requirements.txt`, que no actualiza `numba-cuda`.

Comprueba que Numba ve la GPU:

```bash
python -c "from numba import cuda; print(cuda.is_available()); cuda.detect()"
```

## Inicio rápido

```bash
python -m pytest -q tests
python censo_3n.py --grafo ejemplo --verificar
python barrido.py --grafo ba --generador numpy --n 1000000 --k 8
python censo_3n.py --grafo ba --generador numpy --n 1000000 --k 8 --config resultados/barrido_<grafo>_<fecha>/mejor_config.json --verificar-njit
python demo.py --rapido
```

1. Los tests validan todo el código: oráculo contra fuerza bruta, CPU, GPU y compilación de cada kernel.
2. El censo del grafo de ejemplo se compara contra la referencia Python.
3. El barrido mide todas las configuraciones en un grafo de un millón de nodos y guarda la mejor.
4. El censo completo usa esa configuración y se compara contra la CPU compilada.
5. La demo rápida corre todos los experimentos en grafos medianos.

## Qué se calcula

**Patrón de un camino:** la etiqueta del centro y las dos ramas (etiqueta de la arista, etiqueta del nodo extremo), con las ramas ordenadas, porque `A–C–B` y `B–C–A` son el mismo camino. En pantalla: `O -[d]- C -[s]- C`, con el centro al medio.

**Patrón de un triángulo:** el multiconjunto de pares (arista opuesta, nodo). Cada nodo se empareja con la arista que no lo toca. Ese multiconjunto no cambia al permutar los nodos y determina el triángulo completo. En pantalla: `C -[s]- C -[s]- N -[d]- C`.

**Inducido.** El patrón de cada uno de los 3 caminos de un triángulo queda determinado por el patrón del triángulo. Por eso la resta se calcula en CPU a partir del censo de triángulos (`resta_desde_triangulos`), y la GPU solo cuenta triángulos. Los tests comprueban que esa resta coincide con la que se obtiene camino por camino.

## Modos de ejecución

| Comando | Para qué |
|---|---|
| `censo_3n.py` | Censo completo de un grafo con una configuración GPU: tiempos, verificación, top de patrones, JSON y CSV |
| `barrido.py` | Elegir la mejor configuración GPU para un grafo; cada configuración se valida contra una referencia |
| `censo_3n_cpu_njit.py` | Medir la CPU compilada (1 hilo y multihilo, enumeración y combinatoria) y compararla con la GPU |
| `demo.py` | Correr los experimentos completos y generar `resumen.md`, `resultados.json` y figuras |
| `censo_3n2a_gpu.py` | Solo caminos: estrategias × memorias contra la referencia Python, y verificación camino por camino de los IDs |
| `censo_3n2a.py` | Referencia Python: exploración contra centro, memoria, dibujo del grafo, soporte por transacción (gSpan) |
| `python -m pytest -q tests` | Verificación automática |

### Grafo de entrada (común a todos)

| Opción | Grafo |
|---|---|
| `--grafo ejemplo` / `triangulo` | Grafos fijos de prueba |
| `--grafo estrella --n N` | Un centro y N − 1 hojas: el caso extremo de hub |
| `--grafo completo --n N` | K_N con etiquetas al azar |
| `--grafo aleatorio --n N --m M` | Erdős–Rényi G(N, M) |
| `--grafo ba --n N --k K` | Barabási–Albert: libre de escala, con hubs |
| `--generador numpy` | Con `ba`, `aleatorio` o `completo`: arma el CSR directo con NumPy/Numba. Necesario sobre ~10⁵ aristas; el generador `python` construye diccionarios y es lento a esa escala |
| `--archivo ruta.txt` | Archivo gSpan (`t` / `v` / `e`) |
| `--csr grafo.npz` | CSR guardado (`offsets`, `indices`, `etiqueta_nodo`, `etiqueta_arista`) |
| `--dataset livejournal` | Grafo de SNAP (`livejournal`, `orkut`, `twitter`), con caché en `--datos` |
| `--etiquetas-nodo L --etiquetas-arista E --semilla S` | Etiquetas al azar uniformes (por defecto 3 × 2 en sintéticos y 5 × 2 en SNAP) |

Los dos generadores producen grafos distintos para la misma semilla. El BA de `numpy` muestrea los destinos con reemplazo y descarta las aristas repetidas, así que cada nodo nuevo queda con *hasta* k aristas.

### `censo_3n.py`: censo completo en GPU

```bash
python censo_3n.py --grafo ba --generador numpy --n 1000000 --k 8 --verificar-njit --json --csv
python censo_3n.py --dataset livejournal --config mejor_config.json --verificar-njit
python censo_3n.py --grafo aleatorio --n 2000 --m 20000 --estrategia-caminos combinatorio --estrategia-tri binaria --verificar
```

| Opción | Efecto |
|---|---|
| `--estrategia-caminos` | `nodo`, `entrada` (por defecto), `balanceado`, `combinatorio` |
| `--memoria-caminos` | `global`, `shared`, `hash`. Por defecto, el primer modo válido en ese orden |
| `--estrategia-tri` | `arista` (por defecto), `binaria`, `balanceado` |
| `--memoria-tri` | Igual que para caminos |
| `--bloque-caminos`, `--bloque-tri` | Hilos por bloque (256 por defecto) |
| `--chunk`, `--chunk-tri` | Trabajo por ítem en las estrategias `balanceado` (64 caminos; 256 unidades de costo) |
| `--preproceso` | `gpu` (por defecto): el grafo se copia a la GPU y la orientación y el arreglo `fila` se construyen allá. `cpu`: se construyen en CPU y se copian |
| `--config archivo.json` | Configuración de `barrido.py`. Las opciones explícitas tienen prioridad sobre el archivo |
| `--reps N` | Repeticiones medidas, tras una de calentamiento |
| `--verificar` | Compara contra la referencia Python, hasta 20 M de caminos (`--forzar-python` para más) |
| `--verificar-njit` | Compara contra la CPU compilada |
| `--json`, `--csv` | Exporta configuración, tiempos, verificación y censos; el CSV tiene `tipo, patron, no_inducido, inducido` |

En LiveJournal y Orkut compara además T contra el número de triángulos que publica SNAP.

### `barrido.py`: comparar configuraciones

```bash
python barrido.py --grafo ba --generador numpy --n 1000000 --k 8 --bloques 128 256 512
python barrido.py --dataset livejournal --reps 3
python barrido.py --grafo completo --generador numpy --n 1000 --parte triangulos --estrategias-tri arista binaria
```

El grafo se copia a la GPU y se preprocesa **una sola vez** (`--preproceso gpu` o `cpu`), y todas las configuraciones lo comparten. Así, cada una mide solo lo propio: reservar su histograma o tabla, su kernel y la lectura del resultado. El tiempo de esa preparación compartida se informa aparte.

Mide cada combinación de estrategia × memoria × bloque: una ejecución de calentamiento (incluye la compilación JIT) y `--reps` medidas. Valida cada configuración contra la referencia (`--referencia njit`, por defecto; `python` en grafos chicos; `ninguna` compara las configuraciones entre sí). Omite y explica las configuraciones que no aplican: memoria insuficiente, demasiados patrones para el modo, o un hub que dejaría a un hilo con demasiado trabajo.

Elige la mejor configuración correcta por tiempo total medio (`--criterio total`) o solo de cálculo (`--criterio calculo`). Después corre el censo completo con ella y lo verifica (`--sin-completo` lo omite).

Guarda en `resultados/barrido_<grafo>_<fecha>/`:

| Archivo | Contenido |
|---|---|
| `barrido.csv` | Una fila por configuración: tiempos (media, desviación y mínimo por fase), hilos, lanzamientos, desbalance, si es correcta o por qué se omitió |
| `barrido.json` | Lo mismo, más grafo, opciones, referencia, tiempos de la preparación compartida y censo completo |
| `mejor_config.json` | La mejor configuración, lista para `censo_3n.py --config` y `censo_3n_cpu_njit.py --config` |

### `censo_3n_cpu_njit.py`: baseline de CPU compilada

```bash
python censo_3n_cpu_njit.py --grafo ba --generador numpy --n 1000000 --k 8 --gpu --config mejor_config.json
python censo_3n_cpu_njit.py --dataset livejournal --modo paralelo --gpu --json
```

Mide el censo completo con `njit` en 1 hilo (`--modo serial`), multihilo (`--modo paralelo`) o ambos. Los caminos se cuentan por enumeración, la misma lógica que la GPU, o por conteo combinatorio (`--estrategia-caminos-cpu`). Con `--gpu` mide también la GPU en la misma sesión, con las mismas opciones de configuración que `censo_3n.py` (incluido `--preproceso`). Las aceleraciones se calculan contra el njit más rápido de cada tipo (1 hilo y multihilo).

### `demo.py`: experimentos

```bash
python demo.py
python demo.py --rapido
python demo.py --escala prueba
python demo.py --solo barrido cpu_gpu --config mejor_config.json
python demo.py --dataset orkut
```

| Escala | Grafos | Uso |
|---|---|---|
| `completa` (por defecto) | Hasta millones de aristas, más LiveJournal | Resultados del artículo |
| `rapida` (`--rapido`) | Cientos de miles de aristas, sin grafo real | Revisión en minutos |
| `prueba` | Decenas de nodos | Comprobar que todo corre, incluso en el simulador |

| # | Experimento (`--solo`) | Qué mide |
|---|---|---|
| 1 | `validacion` | Por grafo: invariantes, exploración = centro, las 4 variantes njit, las 12 configuraciones de caminos y las 9 de triángulos en GPU, el censo completo y los IDs camino por camino. Incluye grafos con 13 y 260 etiquetas de nodo para pasar por los modos `global` y `hash` |
| 2 | `estrategias` | Python: exploración (hash de IDs) contra centro, en tiempo, memoria y trabajo descartado |
| 3 | `barrido` | Estrategia × memoria × bloque en BA, Erdős–Rényi y grafo completo. Guarda `mejor_config_<grafo>.json` |
| 4 | `cpu_gpu` | Censo completo: Python (si W ≤ 5 M), njit 1 hilo y multihilo (enumeración y combinatoria) y GPU con la mejor configuración del barrido, con preproceso en GPU y en CPU. Separa preparación y censo |
| 5 | `escalabilidad` | Tamaño (BA), densidad (Erdős–Rényi) y número de etiquetas (P de 1 a ~10⁶), con una configuración GPU fija |
| 6 | `real` | LiveJournal u otro: T contra SNAP, barrido en el grafo real (`--sin-barrido-real` lo omite) y CPU vs GPU |
| 7 | resumen | Tabla consolidada y figuras |

Cada ejecución crea `resultados/demo_<fecha>_<hora>/` con `resumen.md` (todas las tablas), `resultados.json` (todos los datos), `cpu_vs_gpu.png`, `escalabilidad_tamano.png`, `escalabilidad_densidad.png`, `escalabilidad_etiquetas.png` y los `mejor_config_*.json`. Si un experimento falla, se registra el error y la demo sigue.

### `censo_3n2a_gpu.py` y `censo_3n2a.py`

```bash
python censo_3n2a_gpu.py --grafo ba --n 50000 --k 3 --reps 3 --verificar-ids
python censo_3n2a.py --grafo ba --n 20000 --k 3 --comparar --memoria
python censo_3n2a.py --archivo ejemplos/moleculas.txt --dibujar
```

`censo_3n2a_gpu.py` mide solo caminos no inducidos: todas las estrategias × memorias (`--estrategia`, `--memoria`) contra los dos métodos de Python, y con `--verificar-ids` comprueba que la GPU cuenta exactamente los mismos caminos que la CPU, sin duplicados. `censo_3n2a.py` es la referencia secuencial; `--sin-triangulos` omite triángulos e inducido.

### Desde Python

```python
import censo_3n as c3
import censo_3n2a_gpu as gpu
import generadores

csr = generadores.csr_sintetico("ba", 1_000_000, k=8, L=3, E=2)
res = c3.censo3(csr)
res = c3.censo3(csr, estrategia_caminos="combinatorio", memoria_tri="hash")
res = c3.censo3(csr, config=c3.cargar_config("mejor_config.json"))
res = c3.censo3(csr, preproceso="cpu")

res["caminos_no_inducidos"], res["caminos_inducidos"], res["triangulos"]
res["tiempos"]
c3.invariantes(res)
df = c3.a_dataframe(res)
```

Cada censo es un diccionario `{patrón: frecuencia}`. `res["tiempos"]` trae cada fase por separado; `a_dataframe` arma la tabla de pandas. También se puede cargar un CSR guardado con `gpu.cargar_npz("grafo.npz")`.

## Estrategias y modos de memoria

La GPU separa dos decisiones: **cómo se reparte el trabajo** entre hilos (estrategia) y **dónde se acumula el histograma** de patrones (memoria). Cada combinación compila su propio kernel, así que la memoria compartida se reserva solo en el modo `shared`. Lo comprueba `tests/test_ptx.py` sobre el PTX generado.

### Caminos

| Estrategia | Cada hilo procesa | Ventaja | Costo o límite |
|---|---|---|---|
| `nodo` | Un nodo central completo: C(d, 2) pares | La más simple | Un hub deja a un hilo con casi todo. Se omite si un hilo tendría más de 50 M de caminos |
| `entrada` | Una entrada `p` del CSR (vecino `i` de `c`) y todos los `j > i` de esa fila | Reparto más fino | Necesita el arreglo `fila` (4 bytes por entrada) |
| `balanceado` | Un bloque fijo de `chunk` caminos (64 por defecto) | Balance casi perfecto | Una búsqueda binaria y un desrankeo por bloque |
| `combinatorio` | Un nodo: cuenta sus vecinos por tipo de rama y suma C(n_b, 2) o n_b₁ · n_b₂ por patrón | No enumera: O(m + Σ k²), con k tipos de rama presentes en el nodo | No da instancias. Exige B = E · L ≤ 128 |

Las tres primeras **enumeran** cada camino. Se mantienen por la verificación camino por camino (solo `balanceado` emite IDs), como base para extensiones que necesitan las instancias (soporte MNI, listado) y como estudio de balanceo de carga.

### Triángulos

El grafo se orienta de menor a mayor grado (desempate por id). Cada triángulo aparece una sola vez: la arista orientada u → v lo encuentra en out(u) ∩ out(v).

| Estrategia | Cada hilo procesa | Cuándo conviene |
|---|---|---|
| `arista` | Una arista orientada; intersección por mezcla de dos listas ordenadas: O(d_u + d_v) | Grados parecidos |
| `binaria` | Una arista orientada; recorre la lista corta y busca cada elemento en la larga: O(mín · log máx) | Grados muy distintos |
| `balanceado` | Un ítem de `chunk-tri` unidades de costo (aristas contiguas con costo total parecido) | Desbalance de costo entre aristas y localidad |

### Modos de memoria (caminos y triángulos)

| Modo | Estructura | Válido si |
|---|---|---|
| `global` | Histograma denso en memoria global, un atómico por actualización | Patrones posibles ≤ 2²⁴ |
| `shared` | Histograma por bloque en memoria compartida (2048 casillas, 16 KB); al final se vuelca a global | Patrones posibles ≤ 2048 |
| `hash` | Tabla hash en memoria global (sondeo lineal, `atomic.cas`), ocupación ≤ 50% | Siempre |

Patrones posibles: caminos P = L · B², triángulos PT = B³, con B = E · L. Ejemplos: 3 × 2 etiquetas dan P = 108 y PT = 216; 5 × 2 dan P = 500 y PT = 1000; 13 × 1 dan P = PT = 2197 (ya no cabe en `shared`). En las estrategias que enumeran, cada hilo junta en registros las repeticiones seguidas del mismo patrón y hace un solo atómico por racha; `combinatorio` suma directamente n_b₁ · n_b₂.

### CPU compilada

| Estrategia | Qué hace |
|---|---|
| `enumeracion` | La misma enumeración por centro que la GPU, con histogramas parciales por bloque de trabajo |
| `combinatorio` | El mismo conteo combinatorio que la estrategia GPU `combinatorio` |

Los triángulos usan la misma mezcla que la estrategia GPU `arista`. En paralelo, los bloques se cortan por trabajo (suma prefija de C(d, 2), de grados o de costos), no por número de nodos.

## Cómo se mide

La GPU trabaja en dos etapas: **preparación** (dejar el grafo y sus arreglos derivados en la GPU) y **censo** (contar con el grafo ya preparado).

| Fase | Qué incluye |
|---|---|
| Preproceso CPU | Suma prefija de C(d, 2) por nodo. Con `--preproceso cpu`, además, la orientación y el arreglo `fila` |
| Carga | Copiar el CSR a la GPU, una vez para caminos y triángulos. Con `--preproceso cpu`, además, los arreglos construidos en CPU |
| Preproceso GPU | Con `--preproceso gpu`: orientación (conteo, suma prefija, llenado y costos) y arreglo `fila`, con kernels propios |
| **Preparación** | Preproceso CPU + carga + preproceso GPU |
| H2D | Reservar el histograma o la tabla de cada parte |
| Cálculo | Todos los lanzamientos del kernel |
| D2H | Traer el histograma, quedarse con lo no nulo y decodificar |
| Resta e inducido | En CPU, a partir del censo de triángulos |
| **Censo** | Caminos (H2D + cálculo + D2H) + triángulos (ídem) + resta e inducido |
| **Extremo a extremo** | Preparación + censo |

En la CPU compilada, la preparación es la suma prefija y la orientación en CPU (`njit`). La separación tiene una lectura directa para la integración con RAPIDS: si el grafo ya estuviera en la GPU, la carga desaparecería.

- **Repeticiones:** una de calentamiento, que paga la compilación JIT y no se cuenta, y luego `--reps` medidas.
- **Estadístico:** media ± desviación estándar, igual en CPU y GPU. El mínimo queda en los JSON y CSV. Las aceleraciones dividen medias.
- **Preparación:** en `censo_3n.py`, `censo_3n_cpu_njit.py` y la demo se vuelve a medir en cada repetición. En el barrido se hace una vez y se informa aparte, porque se comparan solo las fases de cada configuración.
- **Lotes:** el trabajo se divide en lanzamientos de hasta `--trabajo-por-lanzamiento` unidades (10⁹ por defecto), porque Windows/WSL aborta los kernels demasiado largos cuando la GPU también maneja la pantalla.
- **Desbalance:** es un modelo analítico (trabajo del hilo más cargado / promedio), no una medición. No se modela para `combinatorio` ni para triángulos.

## Correctitud

```bash
python -m pytest -q tests
NUMBA_ENABLE_CUDASIM=1 python -m pytest -q tests
```

| Archivo | Qué comprueba |
|---|---|
| `test_oraculo.py` | El oráculo Python contra fuerza bruta sobre cada trío de nodos (los tres censos, 40 grafos aleatorios y moléculas); la resta derivada del censo de triángulos; casos borde e invariantes |
| `test_csr.py` | CSR ida y vuelta; `csr_desde_aristas` (la ruta de SNAP) igual a `grafo_a_csr`; `.npz` ida y vuelta, filas desordenadas y CSR asimétrico; orientación; que los lotes cubran todo sin solaparse |
| `test_config.py` | Configuración: valores por defecto, archivo, errores y modo de memoria según el número de etiquetas (`shared` → `global` → `hash`) |
| `test_cpu_njit.py` | njit (enumeración y combinatoria, 1 hilo y multihilo) contra el oráculo, también con muchas etiquetas |
| `test_generadores.py` | Los generadores rápidos dan CSR válidos, reproducibles y de tamaño esperado |
| `test_gpu.py` | Las 12 configuraciones de caminos y las 9 de triángulos (con lotes chicos que fuerzan varios lanzamientos); `censo3` con los dos preprocesos; IDs camino por camino; muchas etiquetas; el barrido, incluido que detecte una configuración incorrecta; scan en GPU contra `np.cumsum`; orientación y `fila` en GPU idénticas, arreglo por arreglo, a las de CPU; cortes de lotes en GPU; grafo residente compartido entre configuraciones |
| `test_ptx.py` | Que los 30 kernels (21 de conteo y 9 de preproceso) compilen para sm_120 y que los de conteo reserven memoria compartida solo en el modo `shared` |

- **Sin GPU:** corren todos menos `test_gpu.py`.
- **Con el simulador** (`NUMBA_ENABLE_CUDASIM=1`): corren todos menos `test_ptx.py`. El simulador ejecuta los kernels en Python, así que valida la lógica, pero no la compilación.
- **Con GPU:** corre todo.

En cada corrida, además, `censo_3n.py` y `barrido.py` comprueban las invariantes (Σ no inducido = W, Σ triángulos = T, Σ inducido = W − 3T), que todas las repeticiones den el mismo censo y, en SNAP, T contra el valor publicado.

## Datos reales

| Opción | Grafo | Nodos | Aristas | Triángulos (SNAP) | Fuente |
|---|---|---|---|---|---|
| `livejournal` | com-LiveJournal | 3.997.962 | 34.681.189 | 177.820.130 | [SNAP](https://snap.stanford.edu/data/com-LiveJournal.html) |
| `orkut` | com-Orkut | 3.072.441 | 117.185.083 | 627.584.181 | [SNAP](https://snap.stanford.edu/data/com-Orkut.html) |
| `twitter` | ego-Twitter (como no dirigido) | — | — | — | [SNAP](https://snap.stanford.edu/data/ego-Twitter.html) |

Los grafos de SNAP no traen etiquetas: se asignan al azar, uniformes, con semilla fija (5 de nodo y 2 de arista por defecto). Se eliminan lazos y aristas repetidas, y los nodos se renumeran de 0 a n − 1. Si la descarga falla, baja el `.gz` desde la página de SNAP y déjalo en `datasets/` con su nombre original (`com-lj.ungraph.txt.gz`, `com-orkut.ungraph.txt.gz` o `twitter_combined.txt.gz`).

Cita de los datos: Jure Leskovec y Andrej Krevl, *SNAP Datasets: Stanford Large Network Dataset Collection*, http://snap.stanford.edu/data, 2014.

## Solución de problemas

- **`cuda.is_available()` devuelve `False`:** revisa que `nvidia-smi` funcione y que el extra de `numba-cuda` (`cu12` o `cu13`) coincida con el driver.
- **`AttributeError: module 'numpy' has no attribute 'row_stack'`:** tienes numpy 2.5 o superior. Corre `pip install "numpy<2.5"`.
- **`CUDA_ERROR_UNSUPPORTED_PTX_VERSION`:** las librerías de CUDA instaladas por pip son más nuevas que el driver. Puede pasar al crear un entorno nuevo, porque pip instala la versión más reciente. Lo más simple es actualizar el driver; la alternativa es instalar las ruedas de CUDA de la versión del driver (revisa cuáles quedaron con `pip list | grep -i -E "cuda|nvvm"`).
- **Pantalla congelada o `CUDA_ERROR_LAUNCH_TIMEOUT`:** baja el trabajo por lanzamiento, por ejemplo `--trabajo-por-lanzamiento 200000000`.
- **Una configuración aparece como omitida en el barrido:** el motivo queda en la tabla y en el CSV (patrones posibles, memoria de GPU o hub).
- **`MemoryError` o proceso muerto al cargar Orkut:** falta RAM. En WSL2 se configura en `.wslconfig`.
- **Advertencias `NumbaPerformanceWarning`:** aparecen con grafos chicos y son inofensivas.

## Limitaciones conocidas

- **Entrada en memoria de CPU.** El CSR tiene el mismo formato que usa cuGraph (`offsets`, `indices`), pero hoy entra como arreglos de NumPy y se copia a la GPU una vez por censo (fase de carga). Falta el adaptador desde cuDF o `cugraph.Graph` sin pasar por la CPU.
- **Suma prefija en CPU.** La suma prefija de C(d, 2) por nodo se sigue calculando en CPU (es O(n)); los lotes y la estrategia `balanceado` la usan.
- **Memoria de la preparación.** Con el grafo residente, la GPU guarda a la vez el CSR, el arreglo `fila` (si la estrategia es `entrada`) y la orientación con sus sumas de costo. En LiveJournal ocupa poco; en Orkut puede acercarse al límite de 8 GB, sobre todo en el barrido, que prepara los arreglos de todas las estrategias.
- **Etiquetas sintéticas.** En SNAP se asignan al azar, así que el censo de esos grafos sale casi plano. Falta un grafo con etiquetas reales.
- **Límites por modo.** `shared` admite hasta 2048 patrones posibles, `combinatorio` hasta 128 tipos de rama, y el baseline njit usa histogramas densos (hasta 2²⁴ patrones). `hash` no tiene límite de patrones.
- **Soporte por transacción.** Para bases de varios grafos (gSpan), el soporte por transacción existe solo en la referencia Python.
- **Una GPU.** No hay versión multi-GPU.
