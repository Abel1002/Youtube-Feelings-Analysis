# Youtube-Feelings-Analysis

## 🎬 Analizador Neuronal de Sentimiento en YouTube

Práctica completa de **PLN + Transfer Learning + PyTorch + Streamlit** para el
Máster en PLN y Análisis de Sentimiento.

El proyecto convierte comentarios reales de YouTube en un cuadro de mando
analítico: descarga los comentarios con **YouTube Data API v3**, los tokeniza,
los pasa por un **backbone Transformer preentrenado** y los clasifica con un
**cabezal neuronal construido a mano** en PyTorch. El resultado se visualiza con
Matplotlib y se puede exportar a CSV.

> **Advertencia pedagógica:** un cabezal sin entrenar produce salidas casi
> aleatorias (~33 % por clase). La app avisa explícitamente cuando no encuentra
> un checkpoint. Los gráficos de un modelo sin entrenar son una prueba de
> fontanería, **no** un análisis de sentimiento.

---

## 1. Qué incluye

| Módulo | Responsabilidad |
|---|---|
| `app.py` | Dashboard Streamlit: ingesta, inferencia, métricas, gráficos y CSV |
| `youtube_api.py` | Extracción y paginación de comentarios + errores traducidos |
| `model_architecture.py` | Backbone + cabezal `nn.Module` + checkpoints |
| `train.py` | Dataset, optimizadores, training loop, early stopping y reportes |
| `prepare_data.py` | Descarga y normalización del corpus etiquetado |
| `visualization.py` | Figuras Matplotlib reutilizables |
| `config.py` | Constantes, rutas y carga de la API Key desde `.env` |
| `smoke_test.py` | Verificación rápida de shapes, checkpoint, figuras y API |
| `data/` | CSV de entrenamiento y ficha metodológica |
| `checkpoints/` | Salida del entrenamiento: pesos, curvas y matrices (no se versiona) |

---

## 2. Arquitectura

```text
URL de YouTube
      │
      ▼
YouTube Data API v3  ──►  comentarios + likes + fecha  ──►  DataFrame
      │
      ▼
Tokenizador  ──►  input_ids [batch, 128]  +  attention_mask [batch, 128]
      │
      ▼
Backbone Transformer congelado  ──►  last_hidden_state [batch, 128, 768]
      │
      ▼
Pooling (mean / cls / max)  ──►  [batch, 768]
      │
      ▼
LayerNorm → Dropout → Linear(768→256) → GELU → Dropout → Linear(256→3)
      │
      ▼
Softmax  ──►  {Negativo, Neutro, Positivo} + confianza + polaridad neta
      │
      ▼
Dashboard Streamlit + Matplotlib + CSV
```

La salida del modelo son **logits crudos**. `CrossEntropyLoss` aplica
`log_softmax` internamente; la softmax solo se usa en inferencia. Añadir softmax
en el modelo y luego `CrossEntropyLoss` es el clásico error de **doble softmax**.

---

## 3. Instalación

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Verifica el entorno:

```bash
python -c "import torch, transformers, streamlit; print(torch.__version__, transformers.__version__, streamlit.__version__)"
python smoke_test.py --api
```

---

## 4. API Key de YouTube

1. Crea un proyecto en **Google Cloud Console**.
2. Habilita **YouTube Data API v3**.
3. Genera una **API Key** restringida a esa API.
4. Colócala en `.env`:

```env
YOUTUBE_API_KEY=tu_api_key_aqui
```

`.env` está en `.gitignore` y **nunca** debe subirse al repositorio. La app
también permite pegar la clave en el panel lateral; si existe `.env`, se
precarga automáticamente.

> Cuota gratuita: 10.000 unidades/día. `commentThreads.list` cuesta ~1 unidad
> por página, así que 500 comentarios ≈ 5 unidades. El límite práctico suele ser
> el cierre de comentarios del vídeo o el modo "solo aprobados".

---

## 5. Datos de entrenamiento

La app analiza comentarios de YouTube, pero **no puede entrenar con ellos**:
no tienen etiqueta. El entrenamiento usa un corpus público etiquetado en
español.

```bash
python prepare_data.py
```

Por defecto descarga:

- **`cardiffnlp/tweet_sentiment_multilingual`** (config `spanish`): 1.839
  ejemplos de train, 324 de validación y 870 de test, con 3 clases balanceadas.

Mapeo de etiquetas:

| Etiqueta original | ID del proyecto | Nombre |
|---|---|---|
| `negative` | 0 | Negativo |
| `neutral` | 1 | Neutro |
| `positive` | 2 | Positivo |

El script elimina URLs, menciones `@usuario`, textos vacíos y duplicados
exactos; conserva emojis, acentos y signos porque aportan señal. Genera:

- `data/sentiment_es.csv` con columnas `texto, etiqueta, split, fuente`.
- `data/dataset_card.json` con la ficha metodológica.

Alternativa TASS 2019:

```bash
python prepare_data.py --preset tass --output data/sentiment_es_tass.csv
```

> **Nota de honestidad metodológica:** entrenar con tuits y predecir sobre
> comentarios de YouTube es un **cambio de dominio** (*domain shift*). Parte del
> error en producción viene de ahí, no de la arquitectura. La jerga, la longitud
> y el contexto conversacional cambian.

---

## 6. Entrenamiento

Entrenamiento por defecto (feature extraction: backbone congelado):

```bash
python train.py
```

Los valores por defecto del CLI son `--epochs 3 --batch-size 16 --max-len 128
--hidden-dim 256 --seed 42`, pensados para que una primera prueba no se eternice.
La configuración con la que se generó el checkpoint de referencia del §9 fue esta:

```bash
python train.py \
  --data data/sentiment_es.csv \
  --model pysentimiento/robertuito-base-uncased \
  --pooling mean \
  --dropout 0.3 \
  --epochs 6 \
  --batch-size 32 \
  --optimizer AdamW \
  --lr-head 1e-3 \
  --lr-backbone 2e-5 \
  --patience 2 \
  --output checkpoints/best_sentiment_head.pt
```

El número de épocas y el tamaño de lote sí cambian el resultado: con 6 épocas hay
más oportunidades de mejorar el F1 macro (el early stopping corta igualmente) y un
lote de 32 reduce a la mitad los pasos de optimizador por época, con un gradiente
algo menos ruidoso. El resto de valores coincide con los defaults del CLI.

Salidas:

- `checkpoints/best_sentiment_head.pt` — pesos del cabezal y metadatos.
- `checkpoints/training_curves.png` — pérdida y métricas por época.
- `checkpoints/confusion_matrix_val.png` y `..._test.png`.
- `checkpoints/training_report.json` — informe completo con `classification_report`.

### 6.1 Por qué el checkpoint es pequeño

El backbone congelado no se guarda: se reconstruye desde Hugging Face usando
`model_name`. El checkpoint contiene el cabezal completo y, si se descongelaron
capas, solo los pesos de esas capas. Así pasa de ~435 MB a ~0,8 MB y sigue siendo
reproducible.

### 6.2 Experimentos de entrenamiento

```bash
# Comparar optimizadores
python train.py --optimizer SGD_Momentum --epochs 3 --output checkpoints/sgd.pt
python train.py --optimizer RMSprop --epochs 3 --output checkpoints/rmsprop.pt

# Fine-tuning parcial de las últimas 2 capas
python train.py --unfreeze-layers 2 --epochs 3 --output checkpoints/finetuned.pt

# Comparar pooling
python train.py --pooling cls --output checkpoints/cls.pt
python train.py --pooling max --output checkpoints/max.pt

# Prueba rápida con 300 ejemplos
python train.py --limit 300 --epochs 1 --output checkpoints/_smoke.pt
```

**Criterio de checkpoint:** F1 macro de validación, no accuracy. Con clases
desbalanceadas, la accuracy puede ser alta ignorando la clase neutra.

### 6.3 Coste de entrenamiento y equipos lentos

Con **Feature Extraction** el backbone está congelado: solo reciben gradientes los
~199.171 parámetros del cabezal (0,18 % del total), así que no hay *backward*
a través de los 109 M de parámetros del Transformer. Por eso el entrenamiento es
asequible sin GPU.

Medición real de este proyecto (CPU, `--batch-size 32`, 1.839 ejemplos de train):

| Dato | Valor |
|---|---|
| Tiempo total | **~202 s** (4 épocas ejecutadas de 6, con early stopping) |
| Coste por época | ~50 s |
| Backbone en memoria | ~435 MB solo en pesos (109 M × 4 bytes) |

Conclusión: entrenar el cabezal completo es **viable en un portátil normal**, y es
la razón por la que el diseño congela el backbone en lugar de hacer fine-tuning.

**Recomendación para portátiles lentos.** Si el equipo va justo de RAM o la
inferencia en Streamlit tarda demasiado, cambia el backbone por uno más ligero y
**reentrena el cabezal** con él:

```bash
# 12 capas de 384 dims: ~3× más rápida en inferencia
python train.py \
  --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
  --output checkpoints/minilm_head.pt

# 6 capas de 768 dims: ~40 % menos de RAM y ~2× más rápida
python train.py \
  --model dccuchile/distilbert-base-spanish-uncased \
  --output checkpoints/distilbert_es_head.pt
```

Matiz importante, porque es fácil equivocarse aquí. Medido en este equipo
(CPU, lote de 64 comentarios × 128 tokens, media de 3 pasadas):

| Modelo | Parámetros | Pesos fp32 | Inferencia por lote | RAM vs. actual |
|---|---|---|---|---|
| RoBERTuito (actual) | 109 M | 435 MB | 3,02 s | referencia |
| MiniLM-L12 multilingüe | 118 M | 471 MB | **0,94 s (3,2×)** | similar |
| DistilBERT español | 67 M | 270 MB | 1,53 s (2,0×) | **~40 % menos** |

MiniLM **no** baja la RAM: su vocabulario multilingüe de 250k compensa las capas
más estrechas. Lo que baja es el cómputo, porque 384 dims hacen del orden de 3–4×
menos operaciones por token que 768, y eso es lo que se nota al analizar 500
comentarios en Streamlit. DistilBERT español sí reduce ambas cosas (mitad de capas).
En los dos casos el precio es algo menos de precisión: MiniLM usa un tokenizador
multilingüe genérico, menos especializado en jerga y emojis españoles que RoBERTuito.

> **Ojo:** no basta con cambiar el desplegable "Modelo base" de la app. El cabezal
> entrenado tiene `fc1` de `[256, 768]`; con MiniLM (384 dims) la app lo detecta y
> falla con un error de dimensiones incompatibles. Hay que reentrenar y usar el
> checkpoint nuevo (`checkpoints/minilm_head.pt`). Con DistilBERT español las
> dimensiones coinciden, pero el backbone es otro: la app avisa y las predicciones
> dejan de ser fiables hasta reentrenar.

---

## 7. Lanzar el dashboard

```bash
streamlit run app.py

# Si Windows responde "streamlit no se reconoce", usa el módulo:
python -m streamlit run app.py
```

En el panel lateral hay 9 campos y un botón. Todos afectan a **qué datos entran**
o a **cómo se construye la red**, así que conviene saber qué toca cada uno antes
de moverlo:

| Control | Qué hace | Cómo elegirlo |
|---|---|---|
| **YouTube API Key v3** | Autentica las llamadas a `videos.list` y `commentThreads.list`. Se precarga desde `.env` (`YOUTUBE_API_KEY`) si existe; el campo va enmascarado. | No la escribas en el código ni la subas al repo. Si ves `403 quotaExceeded`, espera o usa otra clave. |
| **URL del vídeo** | Acepta `watch?v=…`, `youtu.be/…`, `/shorts/…`, `/live/…` o el ID de 11 caracteres. Se parsea a `video_id` con una regex; si no hay match, la app avisa. | Cualquier formato vale; el ID se muestra en la cabecera del análisis. |
| **Máx. comentarios** (20–500) | Cuántos comentarios de primer nivel descargar. Se pagina de 100 en 100 con `nextPageToken`. | 100–200 bastan para una demo. 500 ≈ 5 unidades de cuota de las 10.000 diarias. |
| **Orden de la muestra** | `Relevancia` = comentarios que YouTube considera destacados (sesga hacia los que tienen likes). `Recientes` = los últimos publicados (sesga hacia la conversación actual). | Cambia la muestra de forma drástica: repite el análisis con ambos y compara con el **Experimento 5** (volumen vs. impacto). |
| **Modelo base (Hugging Face)** | Backbone del que se extraen las representaciones latentes (768 dims en RoBERTuito). Se descarga la primera vez (~435 MB) y queda en caché. | Debe ser **el mismo con el que se entrenó el checkpoint**. Si lo cambias, la app avisa de que el cabezal no corresponde. |
| **Pooling** | Cómo se comprime `[batch, seq, 768]` → `[batch, 768]`: `mean` promedia los tokens reales, `cls` toma el token `[CLS]`, `max` toma el máximo por dimensión. | `mean` es el más robusto. El checkpoint se entrenó con un pooling concreto: cambiarlo degrada las predicciones (es el **Experimento 1**). |
| **Dropout del cabezal** | Tasa de neuronas que se apagan al azar. **En inferencia `model.eval()` lo desactiva**, así que no cambia las predicciones: solo define la arquitectura que se reconstruye. | Déjalo en el valor del entrenamiento (0,3). Sirve para el **Experimento 4** durante el entrenamiento, no aquí. |
| **Checkpoint entrenado (.pt)** | Ruta al archivo con los pesos del cabezal. Acepta ruta relativa al proyecto o absoluta. Debajo del formulario verás `Checkpoint detectado · 0.80 MB` o un aviso si no existe. | `checkpoints/best_sentiment_head.pt`. Sin checkpoint, las predicciones son de un cabezal aleatorio: la app lo advierte y no deben interpretarse. |
| **Batch de inferencia** (8–64) | Comentarios procesados por pasada en `predict_sentiment_batch`. Más grande = más rápido, pero más RAM. | 32 en CPU; baja a 16 si el portátil va justo de memoria. |
| **🚀 Extraer y analizar** | Ejecuta el pipeline completo: metadatos → comentarios → inferencia → dashboard. | — |

> El **dropout** y el **pooling** son hiperparámetros de arquitectura, no de
> inferencia: en producción no "mejoran" nada por sí solos. Su efecto se mide
> entrenando (`train.py`) y comparando el F1 macro, no moviendo el desplegable.

La app muestra:

- 5 métricas: volumen, % positivo, % negativo, confianza media y polaridad media.
- **Gráfico 1:** distribución por volumen (donut).
- **Gráfico 2:** impacto ponderado por `likes + 1` (barras).
- **Gráfico 3:** histograma de confianza softmax.
- **Gráfico 4:** polaridad neta vs. likes (dispersión, escala log).
- Tabla filtrable y exportación CSV en `utf-8-sig` (Excel con acentos).
- Pestaña de modelo con configuración, métricas e informe de carga.

Detalles técnicos relevantes:

| Detalle | Por qué |
|---|---|
| `@st.cache_resource` | Cargar un Transformer tarda ~10–20 s y ocupa cientos de MB |
| `@st.cache_data` | Evita repetir llamadas a YouTube al cambiar de pestaña |
| `model.eval()` | Desactiva Dropout: inferencia determinista |
| `torch.no_grad()` | Sin grafo de autograd: menos memoria y más velocidad |
| `plt.close(fig)` | Evita acumular figuras en memoria entre reruns |
| `weights_only=True` | No deserializa código arbitrario desde un `.pt` |
| `utf-8-sig` | BOM para que Excel abra bien los acentos |

---

## 8. Estructura del checkpoint

```python
{
  "format_version": 1,
  "state_dict": {...},          # cabezal + capas descongeladas
  "config": {
      "model_name": "pysentimiento/robertuito-base-uncased",
      "num_classes": 3,
      "pooling_type": "mean",
      "dropout_rate": 0.3,
      "hidden_dim": 256,
      "max_len": 128,
      "unfrozen_layers": 0
  },
  "label_map": {"0": "Negativo", "1": "Neutro", "2": "Positivo"},
  "metrics": {...},
  "history": [...],
  "training": {...},
  "parameter_scope": "head_only",
  "created_at": "..."
}
```

Si cambias el modelo base en la app, el checkpoint carga **solo el cabezal** y
avisa de que el backbone no coincide. Si cambias el pooling, la app avisa de que
el cabezal se entrenó con otra distribución de entrada: úsalo como experimento.

---

## 9. Resultados del checkpoint de referencia

> El checkpoint **no se versiona**: es un artefacto de entrenamiento y está en
> `.gitignore`. Se regenera con `python train.py` (~202 s en CPU).

Entrenamiento: 6 épocas solicitadas, early stopping en la época 4, mejor época 2.
Los valores exactos están en `checkpoints/training_report.json`.

| Métrica | Valor |
|---|---|
| Backbone | `pysentimiento/robertuito-base-uncased` |
| Pooling | `mean` |
| F1 macro validación | **0,650** |
| Accuracy validación | **0,651** |
| F1 macro test | **0,668** |
| Accuracy test | **0,667** |
| Parámetros entrenables | 199.171 / 108.986.371 (**0,18 %**) |
| Tiempo de entrenamiento | ~202 s en CPU |

**Resultados por clase en test:**

| Clase | Precision | Recall | F1 |
|---|---|---|---|
| Negativo | 0,750 | 0,631 | 0,685 |
| Neutro | 0,558 | 0,614 | 0,585 |
| Positivo | 0,713 | 0,755 | 0,734 |

**Lectura:** el neutro es la clase más difícil (F1 0,585), como era de esperar:
su frontera con positivo y negativo es difusa. La matriz de confusión muestra que
la mayor confusión es negativo→neutro (83 casos), seguida de neutro→positivo (64)
y positivo→neutro (58). El modelo es útil como demostración, pero **no** es un
clasificador de producción: se entrenó con tuits y se evalúa sobre YouTube
(domain shift).

**Cómo leer el informe:** mira el `classification_report` por clase antes que la
accuracy. Si el F1 de neutro es bajo pero la accuracy es alta, el modelo está
ignorando el neutro. El criterio de checkpoint es F1 macro precisamente para
evitar ese engaño.

---

## 10. Experimentos de laboratorio

Cada experimento exige **hipótesis → ejecución → evidencia → conclusión**.

### Experimento 1 · Pooling `cls` vs `mean` vs `max`

- **Hipótesis:** en comentarios largos con halago inicial y crítica final, `max`
  captura mejor el pico negativo que `mean`.
- **Procedimiento:** mismo vídeo, mismo checkpoint, tres valores de pooling.
- **Evidencia:** distribución y confianza media.
- **Cuidado:** el cabezal se entrenó con un pooling concreto; cambiarlo en
  inferencia es un experimento, no la configuración de producción.

### Experimento 2 · Optimizador `AdamW` vs `SGD + Momentum` vs `RMSprop`

- **Hipótesis:** AdamW baja rápido; SGD necesita más épocas.
- **Procedimiento:** 3 épocas, mismos datos, misma semilla, mismo `lr_head`.
- **Evidencia:** curvas de pérdida train/validación.
- **Pregunta:** ¿por qué un lr de `1e-3` que funciona en AdamW puede hacer
  divergir a SGD?

### Experimento 3 · Feature Extraction vs Fine-Tuning parcial

- **Hipótesis:** `unfreeze_last_layers(2)` mejora el F1 en jerga de YouTube a
  cambio de 3–5× más tiempo por época.
- **Procedimiento:** medir tiempo/época y F1 con y sin descongelación, usando
  Learning Rate Diferencial (`2e-5` backbone / `1e-3` cabezal).
- **Pregunta:** ¿qué evidencia en la curva de validación indicaría *catastrophic
  forgetting*?

### Experimento 4 · Dropout y sobreajuste

- **Hipótesis:** `dropout=0.1` sobreajusta antes (brecha train-val creciente);
  `dropout=0.5` estabiliza pero puede infraentrenar.
- **Procedimiento:** comparar `0.1 / 0.3 / 0.5` con más épocas.
- **Evidencia:** brecha train-val.

### Experimento 5 · Ponderación por likes

- **Hipótesis:** el ranking por volumen y el ranking por impacto ponderado
  difieren de forma sistemática.
- **Procedimiento:** comparar gráfico 1 (volumen) con gráfico 2 (likes) y
  cuantificar la discrepancia.
- **Pregunta:** ¿qué mide cada gráfico y cuál usaría un creador para decidir?

### Plantilla de informe

```markdown
## Experimento N — <título>
**Hipótesis:** ...
**Configuración:** modelo, pooling, dropout, optimizador, lr, épocas, semilla
**Resultados:** (tabla o figura)
**Conclusión:** ... (en términos del modelo, no de la librería)
**Qué haría distinto:** ...
```

---

## 11. Depuración rápida

| Síntoma | Causa probable | Solución |
|---|---|---|
| Predicciones ~33 % en cada clase | Cabezal sin entrenar o checkpoint no cargado | Verifica la ruta y el informe de carga |
| Todas las predicciones iguales | `pooling="cls"` con backbone no ajustado | Prueba `mean` |
| `RuntimeError: expected scalar type Long` | `attention_mask` en float | Mantenerlo en `torch.long` |
| `CUDA out of memory` | Batch o `max_len` altos | Baja a 16/64 o usa CPU |
| Loss `nan` | lr demasiado alto en SGD o gradientes explosivos | Baja lr; mantén `clip_grad_norm_` |
| Val loss sube y train baja | Sobreajuste | Más dropout, early stopping, menos épocas |
| F1 macro bajo con accuracy alta | Clases desbalanceadas | `stratify`, `--class-weights` |
| `HttpError 403 quotaExceeded` | Cuota diaria agotada | Espera o usa otra clave |
| `HttpError 403 commentsDisabled` | Comentarios cerrados | Prueba otro vídeo |
| App lenta en cada clic | Falta caché | `@st.cache_resource` / `@st.cache_data` |
| Memoria creciente | Figuras sin cerrar | `plt.close(fig)` |
| Acentos corruptos en Excel | CSV sin BOM | `encode("utf-8-sig")` |
| Texto ilegible en la barra lateral | CSS con fondo claro fijo sobre tema oscuro | El estilo usa colores del tema; no fijes `background` claros en `stSidebar` |

Si quieres forzar un tema concreto para toda la clase, crea
`.streamlit/config.toml`:

```toml
[theme]
base = "dark"          # o "light"
primaryColor = "#0f766e"
```

Protocolo de depuración de 5 minutos: ¿shape esperado? → ¿device correcto? →
¿`train()` vs `eval()`? → ¿checkpoint cargado? → ¿etiquetas correctas?

---

## 12. Nota metodológica (resumen)

- **Corpus:** tuits españoles etiquetados (`cardiffnlp/tweet_sentiment_multilingual`).
- **Mapeo:** `negative→0`, `neutral→1`, `positive→2`.
- **Limpieza:** URLs, menciones y duplicados fuera; emojis y acentos dentro.
- **Modelo:** RoBERTuito congelado + cabezal denso propio.
- **Criterio:** F1 macro de validación; test reservado para la evaluación final.
- **Límites:** cambio de dominio (tuits → YouTube), sesgo de muestra de YouTube,
  clase neutra difusa y confianza softmax que no equivale a verdad.
- **Uso responsable:** no presentar predicciones de un modelo sin entrenar como
  resultados; no usar el dashboard como único insumo de decisión.

---

## 13. Mapa pedagógico

| Fase | Entregable | Concepto |
|---|---|---|
| 0 | Entorno instalado | Reproducibilidad |
| 1 | DataFrame con comentarios y likes | Paginación, datos reales |
| 2 | Tensores `[batch, 128]` | PLN numérico, attention mask |
| 3 | `forward()` devuelve `[batch, 3]` | Composición de capas |
| 4 | Curva de pérdida + checkpoint | Backprop, optimizadores |
| 5 | Predicciones + `classification_report` | Softmax, métricas |
| 6 | Dashboard con 4 gráficos y CSV | Comunicación de resultados |
| 7 (opcional) | Fine-tuning parcial / LoRA | Transfer Learning avanzado |

---

## 14. Créditos y contexto

Proyecto didáctico del **Máster en PLN y Análisis de Sentimiento**. Construido
sobre ideas de `agents.md`, `ProyectoInicial.md` y los apuntes de Deep Learning y
Redes Neuronales.

Las librerías, APIs y *skills* son medios; el objetivo evaluable es **explicar,
experimentar y diagnosticar** el comportamiento del modelo.
