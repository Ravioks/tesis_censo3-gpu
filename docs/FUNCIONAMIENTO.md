# Funcionamiento técnico

Este documento explica qué calcula la herramienta, cómo lo calcula en CPU y en GPU, cómo se mide y cómo se verifica. Las referencias apuntan a funciones (`archivo.función`), no a números de línea.

---

## 0. Resumen

- **Entrada:** un grafo no dirigido, sin lazos ni aristas repetidas, con una etiqueta por nodo y una por arista, en formato CSR.
- **Salida:** tres censos por patrón de etiquetas: caminos no inducidos, caminos inducidos y triángulos.
- **Idea central para caminos:** cada camino `a – c – b` tiene un único nodo central. Si cada nodo forma los pares de sus vecinos con i < j, cada camino aparece exactamente una vez, sin duplicados ni estado compartido. El total es W = Σ C(d, 2).
- **Idea central para triángulos:** se orienta cada arista del nodo de menor grado al de mayor grado. Cada triángulo aparece una sola vez como intersección out(u) ∩ out(v) de una arista orientada u → v.
- **Inducido:** el patrón de los 3 caminos de un triángulo se deduce del patrón del triángulo. La resta se hace en CPU sobre el censo de triángulos, que es chico.
- **GPU:** kernels propios en Numba CUDA. Caminos: 4 estrategias de reparto; triángulos: 3. Ambos con 3 modos de memoria para el histograma. Cada combinación compila su propio kernel.
- **Preparación en GPU:** el grafo se copia una sola vez y la orientación y el arreglo `fila` se construyen en la GPU, con un scan (suma prefija) propio. La orientación resultante es idéntica, arreglo por arreglo, a la de CPU.

---

## 1. Qué se calcula

| Censo | Definición | Total |
|---|---|---|
| Caminos no inducidos | Pares de aristas que comparten un nodo | W = Σ C(d, 2) |
| Triángulos | Tríos de nodos mutuamente unidos | T |
| Caminos inducidos | Tríos con exactamente 2 aristas | W − 3T |

Cada triángulo contiene 3 caminos no inducidos (uno centrado en cada vértice) y ningún camino inducido. Por eso, patrón por patrón:

> inducido(p) = no inducido(p) − (caminos de patrón p que están dentro de un triángulo)

### 1.1 Patrones

**Camino:** `(etiqueta del centro, rama₁, rama₂)` con cada rama = `(etiqueta de la arista, etiqueta del nodo extremo)` y las ramas ordenadas (`censo_3n2a.clave_patron`). Ordenarlas hace que `A–C–B` y `B–C–A` den la misma clave.

**Triángulo:** multiconjunto ordenado de los pares `(arista opuesta, nodo)` (`censo_3n2a.clave_triangulo`). Cada nodo se empareja con la arista que no lo toca. Una permutación de los nodos mueve los pares completos, así que el multiconjunto no cambia. Y desde los tres pares se reconstruye el triángulo: la arista entre los nodos i y j es la opuesta del tercero. Entonces dos triángulos tienen la misma clave si y solo si son isomorfos con etiquetas.

### 1.2 Resta a partir del censo de triángulos

Sea un triángulo con pares (e₀, n₀), (e₁, n₁), (e₂, n₂). El camino centrado en el nodo 0 tiene dos ramas:

- hacia el nodo 1, por la arista 0–1, que es la opuesta del nodo 2: rama (e₂, n₁);
- hacia el nodo 2, por la arista 0–2, que es la opuesta del nodo 1: rama (e₁, n₂).

Su patrón es `(n₀; (e₂, n₁), (e₁, n₂))`. Lo mismo vale para los nodos 1 y 2. `censo_3n2a.resta_desde_triangulos` recorre los patrones de triángulo, suma su frecuencia a esos tres patrones de camino y devuelve la resta. Cuesta O(patrones de triángulo), no O(T).

Ejemplo: un triángulo con nodos `C`, `C`, `O`, arista `d` entre los dos `C`, y aristas `s` desde cada `C` al `O`.

| Nodo | Arista opuesta | Par |
|---|---|---|
| C₁ | C₂–O = s | (s, C) |
| C₂ | C₁–O = s | (s, C) |
| O | C₁–C₂ = d | (d, O) |

Clave: `((d, O), (s, C), (s, C))`. Resta: centrado en O, `(O; (s, C), (s, C))`; centrado en cada C, `(C; (d, C), (s, O))`, dos veces.

`tests/test_oraculo.py` comprueba en 40 grafos aleatorios que esta resta es idéntica a la calculada camino por camino (`censo_3n2a.censo_triangulos`).

---

## 2. Archivos y funciones clave

| Archivo | Funciones |
|---|---|
| `censo_3n2a.py` | `Grafo`; `censo_centro`, `censo_exploracion` (caminos); `censo_triangulos` (triángulos + resta directa); `resta_desde_triangulos`; `censo_inducido`; `censo_completo` (oráculo); generadores `generar_*`; `cargar_gspan`; `estadisticas`, `aplanar`, `seg_est` (medición) |
| `censo_3n2a_gpu.py` | `CSR`, `grafo_a_csr`, `csr_desde_aristas`, `cargar_npz`/`guardar_npz`; `preparar`; funciones de dispositivo `id_patron`, `insertar_hash`, `volcar`, `iniciar_shared`, `vaciar_shared`, `desrankear`; `_fabricar_kernels` y `kernels(memoria)`; scan en GPU (`_k_scan_bloque`, `_k_sumar_bloques`, `scan_inclusivo`, `prefijo_exclusivo`); `subir_grafo`, `fila_en_gpu`, `buscar_cortes`; `motivo_memoria`, `motivo_invalida`, `memoria_por_defecto`; `lotes`; `ejecutar_gpu`; `medir_gpu`; `verificar_ids` |
| `censo_3n3a_gpu.py` | `orientar` (`_orientar`, `_costos`) en CPU; `orientar_gpu` (`_k_contar_salida`, `_k_llenar_orientado`, `_k_costos`, `_k_costos_binaria`), `asegurar_binaria`, `subir_orientacion`; `id_triangulo`, `acumular`, `mezclar`, `buscar`; `_fabricar_kernels` y `kernels(memoria)`; `motivo_invalida`; `lotes`; `ejecutar_triangulos_gpu`; `decodificar_triangulos` |
| `censo_3n.py` | `preparar`; `resolver_config`, `completar_memorias`, `cargar_config`; `preparar_dispositivo`; `censo3`; `estadisticas_corridas`; `invariantes`; `a_dataframe`; argumentos comunes (`agregar_args_grafo`, `agregar_args_config`, `cargar_grafo`) |
| `censo_3n_cpu_njit.py` | `_enumeracion_rango`, `_combinatorio_rango`, `_triangulos_rango`; versiones `*_serial` y `*_paralelo`; `cortes_balanceados`; `censo3_njit` |
| `barrido.py` | `barrer_caminos`, `barrer_triangulos`, `barrer`, `mejor`, `config_desde_mejores`, `calcular_referencia` |
| `generadores.py` | `csr_sintetico` (`ba`, `aleatorio`, `completo`), `grafo_etiquetas` |
| `datos_reales.py` | `descargar`, `leer_lista_aristas`, `normalizar_aristas`, `cargar_dataset` |
| `demo.py` | `exp_validacion`, `exp_estrategias`, `exp_barrido`, `exp_cpu_gpu`, `exp_escalabilidad`, `exp_real`, figuras |

---

## 3. Representación

### 3.1 CSR

| Arreglo | dtype | Tamaño | Contenido |
|---|---|---|---|
| `offsets` | int64 | n + 1 | Los vecinos de `c` están en `indices[offsets[c] : offsets[c+1]]` |
| `indices` | int32 | 2m | Id del vecino; cada arista aparece en ambos sentidos |
| `et_nodo` | int32 | n | Etiqueta del nodo (0..L−1) |
| `et_arista` | int32 | 2m | Etiqueta de la arista de esa entrada (0..E−1) |

Las filas quedan ordenadas por id de vecino. Los triángulos lo necesitan para intersecar por mezcla o búsqueda binaria.

- `grafo_a_csr` ordena explícitamente.
- `csr_desde_aristas` (la ruta de SNAP y de los generadores rápidos) las deja ordenadas si las aristas llegan ordenadas por (mín, máx), como las entrega `normalizar_aristas`. Para un nodo x, primero se insertan las aristas (y, x) con y < x, en orden creciente de y, y después las (x, z), en orden creciente de z.
- `cargar_npz` ordena si hace falta.
- `orientar` vuelve a comprobarlo.

`tests/test_csr.py` verifica que ambas rutas den el mismo CSR.

Las etiquetas son índices sobre `nombres_nodo` y `nombres_arista`. `grafo_a_csr` ordena los nombres alfabéticamente; los generadores usan `alfabeto(L)`. La decodificación vuelve a ordenar por nombre, así que las claves de la GPU, la CPU compilada y Python coinciden siempre.

### 3.2 Codificación de patrones en enteros

```
rama = etiqueta_arista × L + etiqueta_nodo                     (0 .. B−1),  B = E·L
camino:    pid = (etiqueta_centro × B + mín(r₁, r₂)) × B + máx(r₁, r₂)      P  = L·B²
triángulo: pid = (t₀ × B + t₁) × B + t₂,  con t = arista_opuesta × L + nodo, ordenados   PT = B³
```

Como cada patrón es un entero distinto, el histograma puede ser un arreglo indexado por `pid` cuando P o PT es chico. `decodificar` y `decodificar_triangulos` invierten la fórmula con `divmod`.

Ejemplo con 3 etiquetas de nodo y 2 de arista: B = 6, P = 3 · 36 = 108 y PT = 216.

---

## 4. Algoritmos secuenciales (referencia y baseline)

### 4.1 Caminos por centro y por exploración (Python)

| | Exploración (`censo_exploracion`) | Centro (`censo_centro`) |
|---|---|---|
| Recorrido | Desde cada nodo, un paso por una arista y otro desde cada extremo | Cada nodo `c` forma los pares `(i, j)` de sus vecinos con i < j |
| Veces que ve cada camino | 4 | 1 |
| Cómo evita contar de más | `set` con el par de IDs de aristas | No hace falta |
| Memoria | Crece con W | Constante |

### 4.2 Caminos por conteo combinatorio

Un camino centrado en `c` queda determinado por sus dos ramas. Si `c` tiene n_b vecinos con rama de tipo b:

- dos ramas del mismo tipo b: C(n_b, 2) caminos;
- ramas de tipos b₁ ≠ b₂: n_b₁ · n_b₂ caminos.

Ejemplo: un centro `C` con vecinos por ramas `(s, O)`, `(s, O)`, `(s, O)` y `(d, N)`.

| Tipos | Caminos |
|---|---|
| (s, O) con (s, O) | C(3, 2) = 3 |
| (s, O) con (d, N) | 3 · 1 = 3 |
| Total | 6 = C(4, 2) |

El algoritmo recorre la fila una vez para contar los tipos presentes (k ≤ mín(d, B)) y luego suma los k(k+1)/2 pares de tipos. Cuesta O(d + k²) por nodo, en vez de Θ(d²). No produce las instancias, solo las cuentas.

### 4.3 Triángulos por orientación e intersección

1. `orientar`: deja solo las aristas u → v con (d_u, u) < (d_v, v) (`_orientar`). El grado de salida queda acotado por O(√m).
2. Para cada arista orientada u → v, los w comunes a out(u) y out(v) cierran un triángulo. Cada triángulo se encuentra una sola vez: desde la arista que une sus dos vértices de menor orden.
3. `_costos` calcula la suma prefija del costo de mezcla (d_u + d_v, siempre ≥ 1), la del costo de búsqueda binaria (mín · ⌈log₂(máx + 1)⌉ + 1) y la cota Σ mín(d_u, d_v) ≥ T, que dimensiona la tabla hash.

---

## 5. GPU

### 5.1 Principio común

```python
for x in range(inicio + cuda.grid(1), fin, cuda.gridsize(1)):
```

- Cada hilo empieza en su índice global y avanza de a "total de hilos" (bucle grid-stride).
- Por defecto: 256 hilos por bloque y hasta 4096 bloques.
- Los hilos no se comunican, salvo al sumar al histograma.
- El estado por hilo es O(1): no hay frontera BFS ni pila que crezca.

### 5.2 Especialización por modo de memoria

`_fabricar_kernels(modo)` define los kernels dentro de una función que recibe el modo. Numba trata `modo` y el tamaño del arreglo compartido como constantes de compilación, así que `kernels(memoria)` compila un kernel distinto por modo y lo guarda en caché.

| Modo | Memoria compartida por bloque |
|---|---|
| `shared` | 2048 × 8 B = 16 KB |
| `global`, `hash` | 8 B (un arreglo de tamaño 1) |

Antes, todos los modos reservaban el histograma compartido aunque no lo usaran; los triángulos reservaban 32 KB, por el segundo histograma de resta. Esa reserva limita cuántos bloques caben por SM y sesgaba la comparación en contra de `global` y `hash`. `tests/test_ptx.py` compila los 21 kernels para sm_120 y verifica en el PTX el tamaño de `.shared` de cada uno.

### 5.3 Estrategias de caminos (`censo_3n2a_gpu`)

| Estrategia | Ítem del bucle | Trabajo por ítem |
|---|---|---|
| `nodo` | Un nodo central | C(d, 2) pares |
| `entrada` | Una entrada `p` del CSR: el vecino `i` de `c` contra todos los `j > i` de su fila | d − 1 − posición |
| `balanceado` | Un bloque fijo de `chunk` caminos | `chunk` |
| `combinatorio` | Un nodo central | O(d + k²) |

**`balanceado`** sabe qué caminos le tocan así:

1. Los caminos se numeran de 0 a W − 1, centro por centro, con la suma prefija `acum` de C(d, 2).
2. Una búsqueda binaria en `acum` encuentra el centro `c` con `acum[c] ≤ w < acum[c+1]`.
3. `desrankear` convierte el índice local k = w − acum[c] en el par (i, j).
   - Antes de la fila i hay i · (2d − 1 − i) / 2 pares.
   - Con d = 4, los pares en orden son (0,1) (0,2) (0,3) (1,2) (1,3) (2,3). La fila 0 tiene 3 pares, así que k = 4 cae en la fila 1, posición 4 − 3 = 1: el par (1, 3).
4. Avanza secuencialmente hasta completar el bloque; si se acaba un centro, pasa al siguiente.

Con `emitir`, `balanceado` escribe el par de IDs de aristas de cada camino. `verificar_ids` lo usa para comparar el conjunto exacto de caminos con la CPU.

**`combinatorio`** guarda en memoria local por hilo dos arreglos de `MAX_TIPOS_COMBINATORIO` = 128 enteros: `cuenta` (por tipo) y `presentes` (los tipos vistos en el nodo actual). Al terminar cada nodo vuelve a cero solo los tipos presentes, así que no recorre los B tipos por nodo. Es válida si B ≤ 128.

**Hubs en `nodo`:** si el hilo más cargado procesaría más de `LIMITE_TRABAJO_HILO` = 50 M de caminos, `motivo_invalida` marca la configuración como no válida. Sirve para evitar el watchdog de Windows/WSL. En LiveJournal, por ejemplo, el hub de grado 14.815 daría C(14.815, 2) ≈ 110 M.

### 5.4 Estrategias de triángulos (`censo_3n3a_gpu`)

| Estrategia | Ítem | Intersección |
|---|---|---|
| `arista` | Una arista orientada | `mezclar`: dos punteros sobre listas ordenadas, O(d_u + d_v) |
| `binaria` | Una arista orientada | `buscar`: recorre la lista corta y busca cada elemento en la larga. El límite inferior de la búsqueda avanza, porque la lista corta está ordenada. O(mín · log máx) |
| `balanceado` | `chunk-tri` unidades de costo de mezcla | El ítem `it` procesa las aristas cuyo costo acumulado previo cae en [it · chunk, (it + 1) · chunk); usa `mezclar` |

En `balanceado`, cada arista pertenece a exactamente un ítem: los costos son ≥ 1, así que `acum_costo` es estrictamente creciente y los intervalos [it · chunk, (it + 1) · chunk) lo particionan. Una arista cara se procesa completa en el ítem donde empieza.

### 5.5 Modos de memoria

| Modo | Estructura | Válido si |
|---|---|---|
| `global` | Arreglo denso de P (o PT) enteros; `atomic.add(hist, pid, n)` | espacio ≤ 2²⁴ |
| `shared` | Histograma de 2048 casillas en memoria compartida por bloque. Se pone en cero (`iniciar_shared`), se suma ahí (solo compiten los hilos del bloque) y se vuelca a global con un atómico por patrón (`vaciar_shared`) | espacio ≤ 2048 |
| `hash` | Tabla en memoria global: hash multiplicativo `pid × 2654435761`, sondeo lineal y `atomic.cas` para reclamar casilla. Capacidad = potencia de 2 ≥ 2 · mín(cota, espacio), así que nunca se llena más de la mitad | siempre |

**Agregación local:** cada hilo guarda en registros `(último patrón, cuántas veces seguidas)` y solo hace la operación atómica cuando el patrón cambia. Diez caminos seguidos del mismo patrón cuestan un atómico de +10. En triángulos lo hace `acumular`; en caminos está escrito dentro de cada kernel. `combinatorio` suma directamente n_b₁ · n_b₂.

`memoria_por_defecto(espacio)` elige el primer modo válido en el orden `shared`, `global`, `hash`.

### 5.6 Lanzamientos por lotes

`lotes` divide el trabajo en lanzamientos de hasta `trabajo_max` unidades (10⁹ por defecto). Los cortes se calculan con `searchsorted` sobre la suma prefija correspondiente:

| Estrategia | Suma prefija |
|---|---|
| `nodo`, `entrada` | C(d, 2) por nodo (`entrada` traduce los cortes de nodos a entradas con `offsets`) |
| `combinatorio` | Grados (`offsets`) |
| `balanceado` (caminos) | Bloques de `chunk` caminos |
| `arista`, `balanceado` (triángulos) | Costo de mezcla |
| `binaria` | Costo de búsqueda binaria |

Los cortes de las sumas que viven en la GPU (orientación hecha en GPU) se buscan con un kernel (`buscar_cortes` → `_k_buscar`, búsqueda binaria por objetivo, misma semántica que `np.searchsorted`); solo vuelven a la CPU los índices de corte.

`tests/test_csr.py` verifica que los lotes cubran todo el rango sin solaparse, y `tests/test_gpu.py` corre todas las configuraciones con lotes chicos que fuerzan decenas de lanzamientos y compara los lotes calculados en GPU con los de CPU.

### 5.7 Preparación en GPU

**Scan (suma prefija).** `scan_inclusivo` en tres fases, con bloques de 256 hilos que procesan 4 elementos cada uno (1024 por bloque):

1. `_k_scan_bloque`: cada hilo suma sus 4 elementos en registros; el bloque hace un scan de Hillis–Steele de los 256 subtotales en memoria compartida; cada hilo escribe sus 4 prefijos más el subtotal de los hilos anteriores, y el último hilo guarda el total del bloque.
2. Los totales de bloque se escanean con la misma función, recursivamente (un nivel cada 1024×).
3. `_k_sumar_bloques` suma a cada bloque el total acumulado de los bloques anteriores.

Cada hilo lee sus elementos antes de escribirlos, así que el scan puede hacerse en el mismo arreglo. `prefijo_exclusivo` antepone un 0 (formato `offsets`). Ejemplo con 4 elementos `[3, 1, 4, 1]`: inclusivo `[3, 4, 8, 9]`, exclusivo `[0, 3, 4, 8, 9]`.

**Orientación** (`orientar_gpu`), con el CSR ya en la GPU:

| Paso | Kernel | Qué hace |
|---|---|---|
| 1 | `_k_contar_salida` | Un hilo por nodo cuenta sus vecinos con (grado, id) mayor; de paso calcula el grado de salida máximo con `atomic.max` |
| 2 | `prefijo_exclusivo` | Convierte las cuentas en `off_o` |
| 3 | `_k_llenar_orientado` | Un hilo por nodo recorre su fila en orden y escribe los vecinos que quedan; como la fila de entrada está ordenada, la orientada también |
| 4 | `_k_costos` + scan | Costo de mezcla d_u + d_v por arista orientada y su suma prefija (`acum_costo`), en el mismo arreglo; la cota Σ mín(d_u, d_v) se suma con un atómico por hilo |
| 5 | `_k_costos_binaria` + scan | Solo si se usa la estrategia `binaria` (`asegurar_binaria`) |

Como cada hilo procesa nodos completos y respeta el orden de la fila, el resultado es determinista e idéntico al de `orientar` en CPU. `tests/test_gpu.py` lo comprueba arreglo por arreglo, incluidos totales, cota y grado de salida máximo.

**Arreglo `fila`** (`fila_en_gpu` → `_k_fila`): un hilo por nodo escribe su id en las entradas de su fila. Reemplaza el `np.repeat` de CPU y su copia.

---

## 6. El censo completo (`censo_3n.censo3`)

1. `preparar(csr, orientar)`: suma prefija de C(d, 2) en CPU (`censo_3n2a_gpu.preparar`) y, solo con preproceso en CPU, la orientación (`censo_3n3a_gpu.orientar`). Si se pasa `pre`, se reutiliza.
2. Configuración: `resolver_config` mezcla, en orden de prioridad, los valores por defecto, el archivo de configuración y las opciones explícitas, y rechaza estrategias, memorias o claves desconocidas. `completar_memorias` reemplaza las memorias `None` por la primera válida.
3. `preparar_dispositivo`: copia el CSR a la GPU una vez (`subir_grafo`); copia también la suma prefija si la estrategia de caminos es `balanceado`. Luego construye lo que piden las estrategias elegidas:
   - con `preproceso="gpu"`: `fila` (si es `entrada`) y la orientación, con los kernels de la sección 5.7;
   - con `preproceso="cpu"`: los mismos arreglos en CPU, que se copian junto con el grafo (`subir_orientacion`).
4. Caminos no inducidos en GPU (`ejecutar_gpu` con `disp`: usa los arreglos ya residentes). Se liberan `fila` y la suma prefija.
5. Triángulos en GPU (`ejecutar_triangulos_gpu` con la orientación residente y la etiqueta de nodo ya copiada). Se libera todo.
6. En CPU: `resta_desde_triangulos` y `censo_inducido`. Esta última falla si un conteo sale negativo o si la resta tiene un patrón que no está en el no inducido; las dos situaciones indicarían un error.

Con `dispositivo=` se le pasa a `censo3` un grafo ya preparado (lo usa el barrido); en ese caso no se copia ni se libera nada y la preparación no se cuenta.

`res["tiempos"]` trae, para caminos y triángulos, `h2d_s` (solo reservar el histograma o la tabla), `kernel_s`, `d2h_s`, `total_s`, lanzamientos e hilos, y además:

| Campo | Contenido |
|---|---|
| `preproceso_cpu_s` | Paso 1 y, con preproceso en CPU, la orientación y `fila` |
| `carga_s` | Copia del grafo (y, con preproceso en CPU, de los arreglos construidos en CPU) |
| `preproceso_gpu_s` | Orientación y `fila` en GPU |
| `preparacion_s` | Suma de las tres anteriores (`preproceso_incluido` indica si `pre` se midió en esta llamada) |
| `inducido_s` | Paso 6 |
| `calculo_s` | Kernels de caminos + kernels de triángulos |
| `censo_s` | Total de caminos + total de triángulos + `inducido_s` |
| `extremo_s` | `preparacion_s` + `censo_s` |

Antes de esta versión, `ejecutar_gpu` construía `fila` con `np.repeat` dentro del cronómetro de H2D, y cada parte copiaba su propio grafo. Ahora `fila` se construye antes del cronómetro (`t_preparacion` en el resultado) cuando no viene en `disp`.

---

## 7. Barrido (`barrido.py`)

`barrer` recorre estrategia × memoria × bloque por separado para caminos y triángulos.

0. **Preparar una vez** (`preparar_dispositivo` con todas las estrategias del barrido): CSR, `fila`, suma prefija y orientación quedan en la GPU y las comparten todas las configuraciones. Su tiempo va en `preparacion`, aparte. Al terminar se libera.
1. **Validar** (`motivo_invalida` y `motivo_vram`): si la configuración no aplica, se registra el motivo y no se ejecuta.
2. **Medir** (`medir`): una ejecución de calentamiento, que incluye la compilación JIT, y `reps` medidas. Por fase: media, desviación y mínimo.
3. **Comparar** cada repetición con la referencia: censo de caminos (y Σ = W) o de triángulos (y T). Si no hay referencia, con la primera configuración medida.
4. **Elegir** (`mejor`) la configuración correcta con menor tiempo medio, total o de cálculo.
5. **Censo completo** con la mejor combinación, con su propia preparación (como en `censo_3n.py`): una ejecución de calentamiento y `reps` medidas. Se verifican invariantes, repeticiones idénticas e igualdad con la referencia.

La referencia por defecto es `censo3_njit` multihilo: código independiente de los kernels, con la misma codificación. En grafos chicos se puede usar el oráculo Python.

---

## 8. Baseline de CPU compilada (`censo_3n_cpu_njit.py`)

| Función | Paralelización |
|---|---|
| `caminos_serial` / `caminos_paralelo` | `prange` sobre bloques de nodos con histograma parcial por bloque, sin atómicos. `combinatorio` elige entre enumeración y conteo combinatorio |
| `triangulos_serial` / `triangulos_paralelo` | La misma mezcla que la estrategia GPU `arista`, con histograma parcial por bloque |
| `cortes_balanceados` | Corta los bloques por trabajo: C(d, 2) en enumeración, grados en combinatoria, costo de mezcla en triángulos |
| `numero_bloques` | 8 bloques por hilo por defecto; limita la memoria de los parciales a 1 GB |

`censo3_njit` devuelve lo mismo que `censo3`, con tiempos `caminos_s`, `triangulos_s`, `post_s` (decodificación, resta e inducido), `censo_s`, `preproceso_s` (suma prefija y orientación en CPU; es su preparación) y `extremo_s`. Si recibe un `pre` sin orientación en CPU, la calcula.

---

## 9. Medición

- **Reloj:** `time.perf_counter`, con `cuda.synchronize()` al final de cada fase de GPU.
- **Estadístico:** media ± desviación estándar en todos los scripts y para CPU y GPU por igual (`censo_3n2a.estadisticas`). El mínimo queda en los JSON y CSV (`aplanar` genera `*_s`, `*_desv_s`, `*_min_s`).
- **Calentamiento:** la primera ejecución de cada configuración se descarta. En los JSON, `t_jit_s` es el exceso de esa primera ejecución sobre la media.
- **Preparación:** se mide en cada repetición en `censo_3n.py`, `censo_3n_cpu_njit.py` y la demo. En el barrido se hace una vez y se informa aparte.
- **Aceleraciones:** cociente de medias del censo. En `censo_3n_cpu_njit.py` y en la demo, contra el njit más rápido de 1 hilo y el más rápido multihilo.

---

## 10. Verificación

1. **Fuerza bruta → oráculo** (`tests/test_oraculo.py`): para cada trío de nodos se cuentan sus aristas; con 2 es un camino (inducido y no inducido), con 3 es un triángulo más 3 caminos no inducidos. Los tres censos del oráculo deben ser idénticos en 40 grafos aleatorios y en las moléculas.
2. **Resta derivada** = resta camino por camino (`tests/test_oraculo.py`).
3. **Oráculo → njit** (`tests/test_cpu_njit.py`): enumeración y combinatoria, 1 hilo y multihilo, también con muchas etiquetas. Los histogramas serial y paralelo son idénticos.
4. **Oráculo → GPU** (`tests/test_gpu.py`): las 12 configuraciones de caminos y las 9 de triángulos, cada una con dos tamaños de lote. Además, `censo3` con varias configuraciones y con los dos preprocesos, grafos con 13 y 260 etiquetas (`global` y `hash`), IDs camino por camino sin duplicados y el grafo residente compartido entre configuraciones.
5. **Preparación en GPU = CPU** (`tests/test_gpu.py`): scan contra `np.cumsum` (tamaños que cruzan los límites de bloque, también en el mismo arreglo), orientación y `fila` idénticas arreglo por arreglo, y cortes de lotes iguales.
6. **Compilación** (`tests/test_ptx.py`): los 30 kernels (21 de conteo, 9 de preparación) compilan para sm_120, y los de conteo reservan memoria compartida solo en `shared`.
7. **En cada corrida:** invariantes (Σ no inducido = W, Σ triángulos = T, Σ inducido = W − 3T), repeticiones idénticas y, en SNAP, T contra el valor publicado (LiveJournal 177.820.130; Orkut 627.584.181).

El simulador de Numba (`NUMBA_ENABLE_CUDASIM=1`) ejecuta los kernels en Python, así que valida la lógica de los tests de GPU, pero no la compilación; para eso está `test_ptx.py`.

---

## 11. Limitaciones y trabajo futuro

| Tema | Estado |
|---|---|
| Entrada desde RAPIDS | El CSR tiene el formato de cuGraph, pero entra en memoria de CPU. Falta el adaptador desde cuDF y `cugraph.Graph` en GPU (verificar en la documentación de cuGraph cómo extraer el CSR y cómo maneja la renumeración) |
| Preparación | La orientación y `fila` se hacen en GPU; la suma prefija de C(d, 2) sigue en CPU (O(n)). Con el grafo residente, en Orkut la memoria puede acercarse a los 8 GB |
| Etiquetas | SNAP sin etiquetas: se asignan al azar y el censo sale casi plano. Falta un grafo con etiquetas reales |
| Límites | `shared` ≤ 2048 patrones; `combinatorio` B ≤ 128; njit con histograma denso ≤ 2²⁴ |
| Perfilado | Las diferencias entre estrategias no se han explicado con un profiler (Nsight Compute) |
| Numba vs CUDA C++ | No hay comparación contra los mismos kernels escritos en CUDA C++ |
| Alcance | k = 4, soporte MNI con umbral, multi-GPU y A100 (Patagon) quedan como trabajo futuro |
