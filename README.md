# NYC Demand Zones 🚕

Aplicación de aprendizaje **no supervisado** en producción: descubre zonas de
concentración de demanda de viajes en Nueva York mediante clustering espacial, y
se mantiene sola con flujos automatizados de monitoreo, reentrenamiento y
promoción condicional de modelos.

> Proyecto de la Segunda Unidad — Aprendizaje de Máquina, IX ciclo grupo B
> Universidad Nacional del Altiplano Puno, Ingeniería de Sistemas

---

## Puesta en marcha

```bash
git clone <URL-DEL-REPO> && cd uber-demand-zones
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Descargar el dataset en `data/raw/`:

```bash
kaggle datasets download -d fivethirtyeight/uber-pickups-in-new-york-city \
  -p data/raw --unzip
```

Entrenar, promover y levantar la app:

```bash
python -m src.train        # entrena con abr14 + may14, escribe candidate.joblib
python -m src.promote      # evalúa y promueve a production.joblib
streamlit run app.py       # http://localhost:8501
```

Sin el dataset, todo funciona igual en modo sintético: `python -m src.train --synthetic`.

## Estructura

```
uber-demand-zones/
├── src/
│   ├── data.py          Carga, limpieza, lotes mensuales, perfil de demanda
│   ├── train.py         Búsqueda de K, ajuste, métricas → modelo candidato
│   ├── monitor.py       Detección de deriva (3 señales) → código de salida
│   └── promote.py       Compuerta de calidad: promueve solo si mejora
├── notebooks/
│   └── 01_entrenamiento_zonas_demanda.ipynb
├── tests/test_pipeline.py
├── models/              production.joblib, métricas, archive/ de versiones
├── data/
│   ├── raw/             CSV de Kaggle (no versionado)
│   └── processed/       demand_profile.parquet, centroids.csv
├── .github/workflows/   ci.yml, retrain.yml
├── app.py               Interfaz Streamlit (en inglés)
└── Dockerfile
```

## El modelo

MiniBatchKMeans sobre `Lat` y `Lon` **exclusivamente**. K se elige por
coeficiente de silueta sobre una rejilla de 15 a 55.

La marca de tiempo nunca entra al modelo: mezclar grados con horas deforma la
distancia euclidiana. Se usa después del clustering para construir el perfil
`(zona, día, hora)` que la aplicación consulta.

Se evaluó DBSCAN como alternativa. Describe mejor zonas irregulares, pero carece
de `predict`, lo que impide asignar puntos nuevos sin reajustar el modelo
completo — incompatible con inferencia en línea y con reentrenamiento automático.

## Mantenimiento e integración continua

**`ci.yml`** — en cada push: pruebas unitarias, ejecución del pipeline completo
en modo sintético, construcción de la imagen Docker y verificación de que el
contenedor responde al health check.

**`retrain.yml`** — semanal por cron y manual por `workflow_dispatch`:

```
Ingesta del lote mensual
   ↓
Monitoreo de deriva  ──── sin deriva ───→ fin, modelo intacto
   ↓ deriva detectada
Reentrenamiento (candidato)
   ↓
Compuerta de promoción  ── candidato peor ──→ rechazo, producción intacta
   ↓ candidato mejor
Commit del modelo → redespliegue automático
```

### Señales de deriva

| Señal | Umbral | Qué detecta |
|---|---|---|
| Caída relativa de silueta | 15 % | Las zonas dejan de describir la realidad |
| Tasa de puntos lejanos | 10 % | Demanda en lugares nunca vistos |
| Desplazamiento de centroides | informativo | Magnitud del cambio geográfico |

## Guion de la demostración

```bash
# 1. Estado inicial: modelo v1 entrenado con abr–may
python -m src.train && python -m src.promote
streamlit run app.py

# 2. Llega junio: el monitor detecta deriva y devuelve código 1
python -m src.monitor --months jun14

# 3. Disparar el flujo desde GitHub → Actions → Scheduled retraining → Run workflow
#    Resultado: modelo v2 promovido y la app sirviendo las zonas nuevas
```

## Despliegue

Hugging Face Spaces (Docker SDK) o Render. La imagen base es `python:3.11-slim`
sin frameworks de aprendizaje profundo, por lo que entra sin problema en los
planes gratuitos.

## Pruebas

```bash
pytest tests/ -v
```

Ocho pruebas: limpieza de coordenadas fuera de rango, invariante de que la marca
de tiempo no es característica, entrenamiento e inferencia, umbral de calidad,
integridad del perfil de demanda, serialización del modelo, y los dos casos del
detector de deriva (lote equivalente → sin alerta; lote desplazado → alerta).

## Limitaciones

K-Means impone zonas convexas de tamaño similar, que no reflejan la geometría
real de un barrio. Los datos son de 2014 y no representan la demanda actual. El
perfil de demanda es descriptivo, no un pronóstico.
