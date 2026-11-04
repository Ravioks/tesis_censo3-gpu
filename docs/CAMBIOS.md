# Historial de cambios

## Versión 3.1: preparación en GPU y grafo residente (4-nov-2026)

Primera corrida de v3 en la RTX 5060: los 225 tests pasaron y LiveJournal dio T = 177.820.130, igual a SNAP y a la CPU compilada. Esa corrida mostró dos problemas de medición, que esta versión corrige.

- **El arreglo `fila` se medía como H2D.** La estrategia `entrada` construía `fila` con `np.repeat` dentro del cronómetro de copia, así que la H2D de caminos aparecía inflada. Ahora `fila` se construye antes, y en GPU por defecto.
- **La orientación en CPU dominaba el extremo a extremo** (njit, 1 hilo). Ahora se hace en GPU con kernels propios: conteo de salida, suma prefija (scan en tres fases escrito en Numba), llenado que preserva el orden de cada fila, y costos con su suma prefija. El resultado es idéntico, arreglo por arreglo, al de CPU.
- **El grafo se copia a la GPU una sola vez** y lo comparten caminos y triángulos. Antes, cada parte copiaba el suyo y la etiqueta de nodo viajaba dos veces.
- **Fases de tiempo nuevas:** preproceso CPU, carga a GPU y preproceso GPU (juntas, la *preparación*), y luego el censo con el grafo ya preparado. Extremo a extremo = preparación + censo. La H2D de cada parte queda en reservar su histograma.
- **`--preproceso gpu|cpu`** en `censo_3n.py`, `censo_3n_cpu_njit.py --gpu`, `barrido.py` y `demo.py`, para comparar las dos preparaciones. La demo mide ambas en CPU vs GPU.
- **El barrido prepara el grafo una vez** y todas las configuraciones lo comparten: cada una mide solo su histograma, su kernel y su lectura.
- **Tests nuevos:** scan contra `np.cumsum`, orientación y `fila` en GPU iguales a las de CPU, cortes de lotes en GPU, `censo3` con los dos preprocesos y grafo residente compartido. La compilación PTX cubre ahora 30 kernels.

## Versión 3: estrategias, barrido y medición coherente (4-nov-2026)

### Correcciones que afectan las mediciones

- **Kernels especializados por modo de memoria.** Antes, todos los kernels reservaban el histograma de memoria compartida aunque el modo fuera `global` o `hash`: 16 KB por bloque en caminos y 32 KB en triángulos. Esa reserva baja la ocupancia y sesgaba la comparación de modos. Ahora cada modo compila su kernel y solo `shared` reserva memoria compartida. El PTX generado lo confirma (`tests/test_ptx.py`).
- **Un solo criterio de tiempo.** Todos los scripts informan media ± desviación de las repeticiones, igual para CPU y GPU, y guardan también el mínimo. Antes, la CPU usaba el mínimo de N repeticiones y la GPU el promedio, salvo `censo_3n_cpu_njit.py --gpu`, que usaba el mínimo también para la GPU.
- **Tiempo extremo a extremo.** El preproceso en CPU (suma prefija y orientación) ahora se mide en cada repetición y se suma en una columna "extremo a extremo". Antes la orientación quedaba fuera del total de GPU.
- **`--verificar-njit` ya no incluye la compilación JIT** en el tiempo que informa.

### Simplificación

- **La resta para el inducido se deriva del censo de triángulos** (`censo_3n2a.resta_desde_triangulos`). El patrón de los 3 caminos de un triángulo queda determinado por el patrón del triángulo, así que se elimina el segundo histograma de la GPU: se pasa de 4 a 1 operación atómica por triángulo y de 32 a 16 KB de memoria compartida en modo `shared`. Lo mismo en el baseline njit.

### Estrategias nuevas

- **Caminos `combinatorio`** (GPU y njit): cuenta los vecinos por tipo de rama y suma C(n_b, 2) o n_b₁ · n_b₂ por patrón, sin enumerar. Cuesta O(m + Σ k²) en vez de Θ(Σ d²). Exige B ≤ 128 tipos de rama.
- **Triángulos `binaria`:** recorre la lista corta y busca cada elemento en la larga.
- **Triángulos `balanceado`:** reparte ítems de costo fijo de mezcla en vez de una arista por ítem.
- **Guarda contra hubs:** la estrategia `nodo` se marca como no válida si un solo hilo tendría más de 50 M de caminos.

### Modos de ejecución nuevos

- **`barrido.py`:** recorre estrategia × memoria × tamaño de bloque, valida cada configuración contra la CPU compilada y guarda la mejor en `mejor_config.json`.
- **`--config`** en `censo_3n.py`, `censo_3n_cpu_njit.py` y `demo.py` para usar esa configuración.
- **Opciones separadas para caminos y triángulos:** `--estrategia-caminos`, `--memoria-caminos`, `--bloque-caminos`, `--estrategia-tri`, `--memoria-tri`, `--bloque-tri`, `--chunk-tri`.
- **`--generador numpy`** y `generadores.py`: grafos sintéticos de millones de aristas armados directo en CSR.
- **`censo_3n_cpu_njit.py --estrategia-caminos-cpu`:** enumeración, combinatoria o ambas.

### Nombres

- "Kernel" pasa a llamarse **estrategia** y "modo de conteo" pasa a llamarse **memoria**, en código, opciones y salidas: `KERNELS` → `ESTRATEGIAS`, `MODOS` → `MEMORIAS`, `--kernel` → `--estrategia` y `--conteo` → `--memoria` en `censo_3n2a_gpu.py`. Los JSON de versiones anteriores usan los nombres viejos.

### Demo reorganizada

Experimentos: validación, estrategias secuenciales, barrido, CPU vs GPU, escalabilidad (tamaño, densidad, etiquetas) y grafo real. Se pueden elegir con `--solo` y hay tres escalas (`completa`, `rapida`, `prueba`). Se eliminaron las estimaciones de tiempo de CPU: toda comparación usa tiempos medidos.

### Tests

De 90 a 225 tests (`tests/test_*.py`):

- la fuerza bruta valida ahora también el no inducido por patrón;
- la ruta de SNAP (`csr_desde_aristas`, `.npz`) se prueba contra `grafo_a_csr`;
- grafos con 13 y 260 etiquetas pasan por los modos `global` y `hash`;
- hay test para la verificación camino por camino, la configuración, los lotes, los generadores y el barrido (incluido que detecte una configuración incorrecta);
- todos los kernels se compilan a PTX para sm_120 sin necesidad de GPU.

## Versión 2: censo completo (28-oct-2026)

Triángulos en GPU, inducido por patrón, API `censo3`, baseline njit, demo sección 6 y tests con pytest. Se validó solo en el simulador.

## Versión 1: caminos (26-oct-2026)

Censo de caminos no inducidos en CPU y GPU: 3 kernels × 3 modos de conteo, verificación camino por camino y resultados en `docs/resultados/demo_20261026_194819/`.
