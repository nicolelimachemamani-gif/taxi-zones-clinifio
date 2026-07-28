# Informe Técnico — NYC Demand Zones

**Aprendizaje de Máquina, IX ciclo grupo B — Universidad Nacional del Altiplano Puno**
**Integrantes:** _(completar)_
**Repositorio:** _(URL)_ · **Aplicación:** _(URL)_ · **Video:** _(URL)_

> Este documento está estructurado según la rúbrica de evaluación (8 puntos).
> Cada sección corresponde a un criterio calificado. Rellenen los marcadores
> `_(...)_` con capturas y valores reales antes de entregar.

---

## 1. Resumen

Aplicación web que identifica zonas de concentración de demanda de viajes en
Nueva York mediante clustering espacial no supervisado, desplegada en contenedor
y mantenida por flujos automatizados de monitoreo de deriva, reentrenamiento y
promoción condicional de modelos.

El destinatario de este informe es un equipo de TI que asumirá la operación del
sistema sin haber participado en su construcción.

---

## 2. Herramientas y plataformas *(1 punto)*

| Capa | Tecnología | Justificación |
|---|---|---|
| Modelado | scikit-learn 1.5 (MiniBatchKMeans) | Escala a millones de puntos; soporta `predict` sobre datos nuevos |
| Datos | pandas, PyArrow | Parquet para el perfil de demanda: menor tamaño y tipado estable |
| Serialización | joblib | Estándar de facto para artefactos de scikit-learn |
| Interfaz | Streamlit + PyDeck | Mapas 3D sin escribir front-end |
| Contenedor | Docker (`python:3.11-slim`) | Paridad entre entorno local y producción |
| Orquestación | GitHub Actions | Integrado al repositorio; sin infraestructura adicional |
| Alojamiento | _(Hugging Face Spaces / Render)_ | Plan gratuito suficiente: la imagen no lleva frameworks de aprendizaje profundo |
| Pruebas | pytest | Ocho pruebas ejecutadas en cada push |

**Costo total de operación: 0 USD** en los planes gratuitos utilizados.

---

## 3. Organización del código fuente *(1 punto)*

```
src/data.py      Carga, limpieza, particionado en lotes, perfil de demanda
src/train.py     Búsqueda de K, ajuste final, métricas → modelo candidato
src/monitor.py   Tres señales de deriva → código de salida 0/1
src/promote.py   Compuerta de calidad: promueve solo si mejora
app.py           Interfaz de usuario y registro de consultas
tests/           Suite de pruebas del pipeline
```

**Principio de separación aplicado:** entrenar y promover son pasos distintos.
`train.py` nunca escribe sobre el modelo en producción; solo genera un
candidato. La decisión de reemplazo vive exclusivamente en `promote.py`. Esto
permite que el flujo automatizado rechace un modelo peor sin intervención humana.

**Convenciones:** módulos importables como paquete (`python -m src.train`),
rutas resueltas relativas a la raíz del repositorio, semillas fijas
(`random_state=42`) para reproducibilidad.

_(Insertar aquí captura del árbol del repositorio.)_

---

## 4. Modelo y entrenamiento

### 4.1 Datos

`Uber Pickups in New York City` (Kaggle / FiveThirtyEight), seis archivos
mensuales de abril a septiembre de 2014.

| Partición | Meses | Registros tras limpieza | Rol |
|---|---|---|---|
| Entrenamiento | abr14, may14 | _(completar)_ | Modelo inicial en producción |
| Lotes de producción | jun14–sep14 | _(completar)_ | Ingesta mes a mes |

**Limpieza:** eliminación de nulos y recorte al bounding box de NYC
(lat 40.50–41.00, lon −74.30 a −73.70). Este recorte descarta registros en
coordenada (0, 0) que desplazarían los centroides fuera del continente.

### 4.2 Características

Únicamente `Lat` y `Lon`. La marca de tiempo se excluye deliberadamente del
modelo: combinar grados geográficos con horas en una misma métrica euclidiana
carece de sentido dimensional. La hora se aplica después del clustering para
construir el perfil `(zona, día, hora)`.

### 4.3 Hiperparámetros

| Parámetro | Rejilla | Valor elegido | Criterio |
|---|---|---|---|
| `n_clusters` (K) | 15 … 55, paso 5 | _(completar)_ | Máxima silueta |
| `batch_size` | — | 4096 | Compromiso memoria/convergencia |
| `n_init` | — | 10 | Reduce sensibilidad a la inicialización |

_(Insertar Figura 5 del notebook: silueta vs. inercia.)_

### 4.4 Evaluación

Al no existir etiquetas, la validación es **interna**:

| Métrica | Valor | Lectura |
|---|---|---|
| Coeficiente de silueta | _(completar)_ | Cohesión frente a separación |
| Calinski-Harabasz | _(completar)_ | Razón de dispersión inter/intra |
| Davies-Bouldin | _(completar)_ | Menor es mejor |

### 4.5 Alternativa descartada

DBSCAN produce zonas de forma irregular más fieles a la geografía urbana, pero
no implementa `predict`: asignar un punto nuevo exige reajustar el modelo
completo. Es incompatible con inferencia en línea y con reentrenamiento
automatizado, por lo que se descartó para producción pese a su mejor ajuste
descriptivo.

---

## 5. Consideraciones de despliegue inicial *(1 punto)*

### 5.1 Artefactos requeridos

El contenedor no puede arrancar sin los cuatro archivos siguientes, que deben
versionarse **conjuntamente**:

- `models/production.joblib`
- `models/production_metrics.json`
- `data/processed/demand_profile.parquet`
- `data/processed/centroids.csv`

Un modelo sin su perfil de demanda produce una aplicación que muestra zonas
inexistentes. Es el modo de fallo más probable de este sistema.

### 5.2 Construcción de la imagen

`requirements.txt` se copia antes que el código para aprovechar la caché de capas
de Docker: un cambio en `app.py` no reinstala dependencias. Se define un
`HEALTHCHECK` contra `/_stcore/health`, que es lo que la plataforma de
alojamiento consulta para decidir si el servicio está vivo.

### 5.3 Recursos

| Recurso | Requerido | Disponible en plan gratuito |
|---|---|---|
| RAM | ~400 MB | 512 MB (Render) / 16 GB (HF Spaces) |
| Imagen | ~450 MB | sin límite práctico |
| Arranque en frío | ~15 s | aceptable |

### 5.4 Procedimiento

1. Entrenar y promover localmente: `python -m src.train && python -m src.promote`
2. Confirmar artefactos al repositorio.
3. Conectar el repositorio a _(plataforma)_ con SDK Docker.
4. Verificar el health check y la carga del mapa.

_(Insertar captura de la aplicación en línea.)_

---

## 6. Flujos de mantenimiento e integración continua *(2 puntos)*

### 6.1 Integración continua — `ci.yml`

Disparadores: push a `main`, pull request, ejecución manual.

| Etapa | Acción | Criterio de fallo |
|---|---|---|
| Pruebas | `pytest tests/ -v` | Cualquier prueba falla |
| Pipeline | Entrenamiento y promoción en modo sintético | Excepción no controlada |
| Imagen | `docker build` | Error de construcción |
| Servicio | Health check en contenedor vivo | Sin respuesta en 60 s |

Las pruebas usan un generador de datos sintéticos porque GitHub Actions no
dispone de credenciales de Kaggle y versionar 2 GB de CSV sería incorrecto. El
generador reproduce la estructura del dataset real y permite inyectar deriva
controlada.

### 6.2 Mantenimiento — `retrain.yml`

Disparadores: cron semanal (lunes 06:00 UTC) y `workflow_dispatch` manual con
parámetros de mes y forzado.

```
Ingesta del lote mensual
      ↓
Monitoreo de deriva ─── sin deriva (código 0) ──→ fin, modelo intacto
      ↓ deriva (código 1)
Reentrenamiento → modelo candidato
      ↓
Compuerta de promoción ─── candidato peor ──→ rechazo, producción intacta
      ↓ candidato mejor
Archivo de la versión anterior → commit → redespliegue automático
```

### 6.3 Señales de deriva

| Señal | Umbral | Fundamento |
|---|---|---|
| Caída relativa de silueta | 15 % | Las zonas dejan de describir los datos actuales |
| Tasa de puntos lejanos (p99 de entrenamiento) | 10 % | Demanda en ubicaciones nunca observadas |
| Desplazamiento medio de centroides | informativo | Cuantifica la magnitud del cambio |

Los umbrales se calibraron con los reportes de las dos primeras ejecuciones y se
fijaron para evitar reentrenamientos innecesarios.

### 6.4 Política de versionado

Cada promoción incrementa la versión, archiva el modelo saliente en
`models/archive/production_vN.joblib` y registra en `production_metrics.json` la
métrica previa. La reversión consiste en restaurar el archivo anterior — no hay
migración de estado que deshacer.

---

## 7. Pruebas de funcionamiento *(2 puntos)*

### 7.1 Suite automatizada

| Prueba | Verifica |
|---|---|
| `test_cleaning_drops_out_of_bounds` | Se descartan coordenadas fuera de NYC |
| `test_features_are_only_coordinates` | Invariante de diseño: la hora no es característica |
| `test_model_trains_and_predicts` | Ajuste e inferencia consistentes |
| `test_silhouette_is_reasonable` | Umbral mínimo de calidad |
| `test_demand_profile_covers_all_zones` | Integridad del perfil |
| `test_model_roundtrip` | Serialización sin pérdida |
| `test_drift_not_flagged_on_similar_batch` | Ausencia de falsos positivos |
| `test_drift_flagged_on_shifted_batch` | Sensibilidad real a la deriva |

Las dos últimas son las relevantes: un detector que siempre alerta, o que nunca
alerta, es inútil. Se comprueban ambos extremos.

_(Insertar captura de la ejecución en GitHub Actions.)_

### 7.2 Prueba de extremo a extremo

1. Estado inicial: modelo v1 (abr–may) sirviendo en producción.
2. Ingesta de junio: el monitor devuelve código 1 y publica `drift_report.json`.
3. Reentrenamiento automático con los tres meses.
4. Compuerta: ganancia de silueta _(valor)_ sobre el umbral de +0.005 → promoción.
5. Aplicación redesplegada sirviendo la versión v2.

_(Insertar capturas de cada paso y la Figura 9 del notebook con la migración de
centroides.)_

### 7.3 Prueba negativa

Ejecución del flujo con un lote sin deriva: el monitor devuelve código 0 y el
proceso termina sin reentrenar. Confirma que el sistema no consume recursos ni
arriesga el modelo cuando no hay motivo.

---

## 8. Operación

| Situación | Acción |
|---|---|
| La app no arranca | Verificar que los cuatro artefactos estén presentes en la imagen |
| Deriva persistente tras reentrenar | Revisar los umbrales; puede requerir ampliar la rejilla de K |
| Candidato rechazado repetidamente | Normal: significa que el modelo actual sigue siendo el mejor |
| Reversión | Copiar `models/archive/production_vN.joblib` sobre `production.joblib` |

---

## 9. Limitaciones y trabajo futuro

- K-Means impone zonas convexas de tamaño similar, que no reflejan la geometría
  real de un barrio. HDBSCAN con `approximate_predict` sería la evolución natural.
- Los datos son de 2014 y no representan la demanda actual.
- El perfil de demanda es descriptivo, no predictivo: no anticipa demanda futura.
- El registro de consultas se guarda en el sistema de archivos del contenedor y
  se pierde al reiniciar. En producción real correspondería una base externa.

---

## 10. Anexos

- **A.** Notebook de entrenamiento con las nueve figuras.
- **B.** Reporte de deriva (`reports/drift_report.json`).
- **C.** Registro de ejecución de los flujos de GitHub Actions.
- **D.** Video de la exposición _(URL)_.
