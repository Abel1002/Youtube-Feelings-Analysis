# 🧠 `agents.md` — Guía Pedagógica y Especificación Técnica

## Analizador Neuronal de Sentimiento en YouTube · PLN + Transfer Learning

Este documento es la **fuente de verdad** del proyecto: define el objetivo didáctico, el plan de construcción y el detalle técnico de cada módulo. Sirve tanto a **profesores** como a **alumnos** del Máster en PLN y Análisis de Sentimiento, y también como contexto para asistentes de IA de programación.

Integra y sustituye a:

| Documento | Aporte |
|-----------|--------|
| `guiapedagogica.md` | Objetivo pedagógico, competencias y los 5 pilares del proyecto |
| `agentsOriginal.md` | Especificación técnica, arquitectura de capas y código de referencia |
| `ProyectoInicial.md` | Base funcional de extracción de comentarios (YouTube Data API v3) |
| `Copia de Deep Learning i Xarxes Neuronals.md` | Fundamentos teóricos (secciones 2.1–2.8, 3.6–3.8) |

> **Aviso de enfoque.** Este documento enseña **redes neuronales, PLN y análisis de sentimiento**. El código, las librerías y las *skills* de desarrollo son **medios**, no objetivos de aprendizaje. Lo que se evalúa es la capacidad de **explicar, experimentar y diagnosticar** el comportamiento del modelo.

---

## Índice

1. [Objetivo pedagógico y competencias](#1-objetivo-pedagógico-y-competencias)
2. [Los 5 pilares de la práctica](#2-los-5-pilares-de-la-práctica)
3. [Visión general del pipeline](#3-visión-general-del-pipeline)
4. [Entorno y dependencias](#4-entorno-y-dependencias)
5. [Estructura del repositorio](#5-estructura-del-repositorio)
6. [Plan de construcción por fases](#6-plan-de-construcción-por-fases)
7. [Fase 1 · Ingesta de datos (YouTube Data API v3)](#7-fase-1--ingesta-de-datos-youtube-data-api-v3)
8. [Fase 2 · Tokenización y backbone](#8-fase-2--tokenización-y-backbone)
9. [Fase 3 · Capas personalizadas (`model_architecture.py`)](#9-fase-3--capas-personalizadas-model_architecturepy)
10. [Fase 4 · Entrenamiento (`train.py`)](#10-fase-4--entrenamiento-trainpy)
11. [Fase 5 · Inferencia y métricas](#11-fase-5--inferencia-y-métricas)
12. [Fase 6 · Dashboard (`app.py`)](#12-fase-6--dashboard-apppy)
13. [Experimentos de laboratorio](#13-experimentos-de-laboratorio)
14. [Skills como aceleradores de desarrollo](#14-skills-como-aceleradores-de-desarrollo)
15. [Errores comunes y depuración](#15-errores-comunes-y-depuración)
16. [Entregables y rúbrica](#16-entregables-y-rúbrica)
17. [Conclusión didáctica](#17-conclusión-didáctica)

---

## 1. Objetivo pedagógico y competencias

### 1.1 Objetivo principal

El propósito educativo es construir un **puente práctico** entre la teoría abstracta del Deep Learning y una aplicación real de ingeniería de datos, demostrando que las redes neuronales **no son "cajas negras mágicas"**, sino arquitecturas basadas en álgebra lineal y composición de capas.

Mediante un clasificador de sentimientos sobre comentarios reales de YouTube, el alumno aprende a:

- **Desacoplar la teoría del entrenamiento masivo:** no hace falta entrenar un modelo lingüístico desde cero para resolver un problema de negocio concreto.
- **Dominar el ciclo de vida de un proyecto de IA:** ingesta de datos crudos (API web) → vectorización → diseño de arquitectura → entrenamiento → visualización analítica.
- **Experimentar empíricamente:** modificar hiperparámetros (pooling, dropout, optimizador, learning rate) y observar el efecto en tiempo real, fomentando el razonamiento científico y la depuración de redes.

### 1.2 Competencias a desarrollar

| Tipo | Competencia |
|------|-------------|
| **Conceptual** | Explicar una red neuronal como composición de transformaciones afines + no linealidades |
| **Conceptual** | Justificar el *Inductive Transfer Learning* y el papel del *checkpoint* |
| **Conceptual** | Interpretar una curva de pérdida y decidir si el modelo converge o sobreajusta |
| **Técnica** | Convertir texto ruidoso en tensores (`input_ids`, `attention_mask`) correctamente |
| **Técnica** | Construir un cabezal `nn.Module` con pooling, `LayerNorm`, `Dropout`, activación y salida |
| **Técnica** | Implementar el bucle de entrenamiento de 5 pasos con `CrossEntropyLoss` y *gradient clipping* |
| **Técnica** | Comparar `AdamW` vs `SGD+Momentum` vs `RMSprop` con evidencia experimental |
| **Analítica** | Cruzar predicción neuronal con metadatos (likes) y comunicar el resultado en un dashboard |

### 1.3 Qué **no** es objetivo de este proyecto

- Aprender a usar asistentes de IA, *skills* o flujos multi-agente.
- Memorizar APIs de librerías.
- Desplegar en producción a escala industrial.

Estos elementos aparecen como **soporte de desarrollo** (sección 14) y quedan fuera de la evaluación.

---

## 2. Los 5 pilares de la práctica

### A. Redes neuronales y Transfer Learning — *el núcleo arquitectónico*

- **Foco:** comprender el concepto de *checkpoint* y la reutilización de conocimiento (*Inductive Transfer Learning*).
- **Aplicación:** descargar un Transformer ligero que ya modela la gramática del español y usarlo como **extractor de características latentes**, evitando el coste de entrenar desde cero.

### B. PLN en entornos reales

- **Foco:** transformar datos no estructurados y ruidosos (jerga, faltas de ortografía, emojis, longitud variable) en tensores numéricos estandarizados.
- **Aplicación:** tokenización + **pooling** para comprimir una secuencia de tokens en un vector de dimensión fija.

### C. Análisis de sentimiento con ponderación por impacto comunitario

- **Foco:** convertir una clasificación matemática en una métrica útil para la toma de decisiones.
- **Aplicación:** no solo etiquetar Positivo/Neutro/Negativo, sino **ponderar por likes** y visualizar el pulso real de la audiencia.

### D. Diseño de capas personalizadas (*custom neural heads*)

- **Foco:** la personalización real de una red vive en el **cabezal** acoplado al modelo base.
- **Aplicación:** construir explícitamente en PyTorch `nn.Linear`, `nn.GELU`, `nn.LayerNorm` y `nn.Dropout`.

### E. Optimización y dinámica del entrenador

- **Foco:** entender cómo el error se desplaza por el paisaje de pérdida y cómo el optimizador condiciona la convergencia.
- **Aplicación:** comparar `AdamW` (weight decay desacoplado), `RMSprop` y `SGD + Momentum`, aplicando **Learning Rate Diferencial**.

---

## 3. Visión general del pipeline

```text
[URL de YouTube en Streamlit]
       │
       ▼
[YouTube Data API v3] ──────► hasta 500 comentarios + likes  →  DataFrame (pandas)
       │
       ▼
[Preprocesamiento + Tokenización] ──► input_ids, attention_mask  (max_len = 128)
       │
       ▼
[Backbone: Transformer preentrenado] ──► last_hidden_state  [batch, seq, 768]
       │
       ▼
[Custom Neural Head · nn.Module]
   Pooling ─► LayerNorm ─► Dropout ─► Linear(768→256) ─► GELU ─► Dropout ─► Linear(256→3)
       │
       ▼
[Softmax] ──► clase {Negativo, Neutro, Positivo} + confianza (%) + polaridad neta
       │
       ▼
[Dashboard Streamlit + Matplotlib]
   métricas · donut de distribución · impacto por likes · histograma de confianza · CSV
```

**Dos modos de uso del mismo modelo:**

| Modo | Backbone | Para qué |
|------|----------|-----------|
| **Inferencia directa** | Congelado, sin entrenar | Validar el pipeline de extremo a extremo (Fase 5) |
| **Entrenado** | Congelado (feature extraction) o parcialmente descongelado | Producto final con `checkpoints/best_sentiment_head.pt` |

> ⚠️ **Advertencia pedagógica importante:** un cabezal **sin entrenar** produce salidas casi aleatorias (~33 % por clase). Si el alumno analiza comentarios sin haber entrenado, los gráficos son válidos como *prueba de fontanería*, no como análisis de sentimiento. El documento lo exige explícitamente en la rúbrica.

---

## 4. Entorno y dependencias

### 4.1 `requirements.txt`

```txt
streamlit>=1.32.0
google-api-python-client>=2.120.0
pandas>=2.2.0
numpy>=1.26.0
torch>=2.2.0
transformers>=4.38.0
datasets>=2.18.0
scikit-learn>=1.4.0
matplotlib>=3.8.0
peft>=0.9.0
accelerate>=0.27.0
```

### 4.2 Propósito de cada librería

| Librería | Rol en el proyecto |
|----------|--------------------|
| `streamlit` | Frontend interactivo: credenciales, selectores de hiperparámetros, tablas y `st.pyplot` |
| `google-api-python-client` | Endpoint `commentThreads.list` de YouTube Data API v3 |
| `torch` | `nn.Module`, capas, `CrossEntropyLoss`, `AdamW`/`SGD`/`RMSprop`, training loop |
| `transformers` | `AutoModel` / `AutoTokenizer` para Transfer Learning |
| `datasets` | Carga del corpus etiquetado en español para entrenar |
| `scikit-learn` | `train_test_split`, `classification_report`, matriz de confusión |
| `matplotlib` | Gráficos analíticos de alta calidad embebidos en Streamlit |
| `peft` | *(Avanzado)* adaptadores **LoRA** para fine-tuning eficiente |
| `accelerate` | *(Avanzado)* `mixed precision` y entrenamiento multi-GPU |

### 4.3 Instalación

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 4.4 Credenciales

1. Crear un proyecto en **Google Cloud Console**.
2. Habilitar **YouTube Data API v3**.
3. Generar una **API Key** (restringida a esa API).
4. Pegarla en la barra lateral de la app (nunca en el código ni en el repositorio).

> Cuota gratuita: 10 000 unidades/día. `commentThreads.list` cuesta ~1 unidad por página, así que 500 comentarios ≈ 5 unidades. El límite práctico es el **cierre de comentarios** del vídeo o el modo "solo aprobados".

---

## 5. Estructura del repositorio

```text
Analisis_Sentimiento_Youtube/
│
├── app.py                  # Frontend Streamlit: ingesta API + inferencia + dashboard
├── model_architecture.py   # Backbone + cabezal neuronal en PyTorch
├── train.py                # Dataset, optimizadores, training loop, checkpoints
│
├── checkpoints/            # Pesos entrenados (.pt)  ← incluir en .gitignore si son grandes
├── data/                   # CSV de entrenamiento etiquetado (texto, etiqueta)
│
├── requirements.txt
├── README.md               # Uso básico: instalar, entrenar, lanzar
└── AGENTS.md               # Este documento
```

**Extracciones opcionales** (recomendadas cuando el código crezca):

- `youtube_api.py` → `extract_video_id()` y `fetch_youtube_comments()` para poder testearlos sin Streamlit.
- `visualization.py` → `render_matplotlib_dashboard()` para reutilizar gráficos en `train.py`.

**Principio rector:** cada archivo debe poder explicarse en una frase. Si un archivo necesita tres, hay que partirlo.

---

## 6. Plan de construcción por fases

El proyecto se construye **incrementalmente**. Cada fase termina en algo **ejecutable y demostrable**; nunca se escribe código que no se vaya a ejecutar en esa misma sesión.

| Fase | Objetivo | Entregable verificable | Concepto evaluado |
|------|----------|------------------------|-------------------|
| **0** | Entorno listo | `pip install -r requirements.txt` sin errores; `python -c "import torch; print(torch.__version__)"` | Reproducibilidad |
| **1** | Ingesta | DataFrame con ≥100 comentarios y sus likes | Datos reales, paginación |
| **2** | Tokenización | Tensores con shapes `[1, 128]` y `[1, 128]` | PLN numérico |
| **3** | Cabezal | `forward()` devuelve `[batch, 3]` | Composición de capas |
| **4** | Entrenamiento | Curva de pérdida descendente + checkpoint guardado | Backprop, optimizadores |
| **5** | Inferencia | Predicciones + confianza + `classification_report` | Softmax, métricas |
| **6** | Dashboard | App completa con 3 gráficos y export CSV | Comunicación de resultados |

**Criterio de terminado (Definition of Done) por fase:** código que corre + una frase escrita por el alumno explicando **qué aprendió del modelo** en esa fase. Sin esa frase, la fase no está cerrada.

```text
Fase 0 ─► Fase 1 ─► Fase 2 ─► Fase 3 ─► Fase 4 ─► Fase 5 ─► Fase 6
entorno   datos    tensores   red       pesos     métricas   dashboard
                                   │
                                   └──► Fase 7 (opcional): fine-tuning parcial / LoRA
```

---

## 7. Fase 1 · Ingesta de datos (YouTube Data API v3)

### 7.1 Concepto

La API devuelve **JSON paginado**: cada `commentThread` contiene un `topLevelComment` con el texto, el autor, la fecha y el número de likes. El trabajo del alumno es **aplanar** esa estructura anidada en una tabla (DataFrame) y respetar el `nextPageToken` para pedir más páginas.

### 7.2 Código de referencia

```python
import re
import pandas as pd
from googleapiclient.discovery import build

def extract_video_id(url: str):
    """Extrae el ID de 11 caracteres de cualquier formato de URL de YouTube."""
    pattern = r"(?:v=|\/|youtu\.be\/|embed\/)([0-9A-Za-z_-]{11})"
    match = re.search(pattern, url)
    return match.group(1) if match else None


def fetch_youtube_comments(api_key: str, video_id: str, max_comments: int = 200) -> pd.DataFrame:
    """Extrae comentarios principales paginando con YouTube Data API v3."""
    youtube = build("youtube", "v3", developerKey=api_key)
    comments_data = []
    next_page_token = None

    while len(comments_data) < max_comments:
        batch_size = min(100, max_comments - len(comments_data))  # la API limita a 100
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=batch_size,
            order="relevance",
            textFormat="plainText",   # evita HTML en el texto
            pageToken=next_page_token,
        )
        response = request.execute()

        for item in response.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            comments_data.append({
                "autor": snippet.get("authorDisplayName", "Anónimo"),
                "comentario": snippet.get("textDisplay", ""),
                "likes": int(snippet.get("likeCount", 0)),
                "publicado": snippet.get("publishedAt", "")[:10],
            })

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    df = pd.DataFrame(comments_data)
    if not df.empty:
        df = df.sort_values(by="likes", ascending=False).reset_index(drop=True)
    return df
```

### 7.3 Puntos que el alumno debe saber explicar

- **`order="relevance"` vs `"time"`:** cambia radicalmente la muestra. Relevancia sesga hacia comentarios con más interacción.
- **Paginación:** sin `nextPageToken` solo se obtienen 100 comentarios.
- **Texto vacío:** comentarios solo-emoji o borrados llegan como `""`; hay que filtrarlos antes de tokenizar.
- **Sesgo de muestra:** los comentarios visibles son los que YouTube decide mostrar; el "sentimiento de la comunidad" es siempre una **estimación**, no un censo.

---

## 8. Fase 2 · Tokenización y backbone

### 8.1 Concepto: de texto a tensor

Un Transformer no lee letras: lee **enteros** que indexan un vocabulario. La tokenización produce dos tensores:

| Tensor | Shape | Significado |
|--------|-------|-------------|
| `input_ids` | `[batch, 128]` | IDs del vocabulario; `0`/`1`/`2` son tokens especiales (`<s>`, `<pad>`, `</s>`) |
| `attention_mask` | `[batch, 128]` | `1` = token real, `0` = relleno (*padding*) |

**El `attention_mask` no es decorativo:** sin él, el pooling promediaría cientos de tokens de relleno y diluiría la señal del comentario.

### 8.2 Código de referencia

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("pysentimiento/robertuito-base-uncased")

encoded = tokenizer(
    ["¡Buenísimo el vídeo! 🔥", "no me gustó nada, muy largo"],
    padding=True,          # rellena al máximo del batch
    truncation=True,       # corta si supera max_length
    max_length=128,
    return_tensors="pt",
)
print(encoded["input_ids"].shape)       # torch.Size([2, 128])
print(encoded["attention_mask"].sum(1)) # nº de tokens reales por comentario
```

### 8.3 Modelos base recomendados

| Checkpoint | Por qué |
|-----------|---------|
| `pysentimiento/robertuito-base-uncased` | RoBERTa entrenado en español de redes sociales: jerga, emojis, faltas. **Mejor punto de partida.** |
| `dccuchile/bert-base-spanish-wwm-cased` | BERT español robusto, referencia académica |
| `distilbert-base-multilingual-cased` | Ligero y multilingüe; útil con hardware modesto |

> **Ojo:** estos modelos NO son clasificadores de sentimiento por sí mismos. `pysentimiento/robertuito-base-uncased` es un **backbone**; el cabezal de clasificación lo construye el alumno. (La familia `pysentimiento/robertuito-sentiment-analysis` sí trae cabezal entrenado: sirve como *baseline* de comparación, no como sustituto del ejercicio.)

### 8.4 Dimensiones que hay que interiorizar

```text
comentario        →  "me encantó el vídeo"
input_ids         →  [batch, 128]              tokens
last_hidden_state →  [batch, 128, 768]         un vector de 768 dims por token
pooling           →  [batch, 768]              un vector por comentario
fc1               →  [batch, 256]              proyección intermedia
fc_out            →  [batch, 3]                logits por clase
softmax           →  [batch, 3]                probabilidades que suman 1
```

**Regla de oro:** si el alumno no puede dibujar esa cadena de shapes de memoria, no ha entendido el pipeline.

---

## 9. Fase 3 · Capas personalizadas (`model_architecture.py`)

### 9.1 Qué se construye y por qué

No se descarga un modelo cerrado: se descarga un **backbone** que entiende el idioma y se le acopla un **cabezal propio**. Ahí es donde el alumno experimenta.

| Componente | Capa | Fundamento teórico | Efecto observable |
|-----------|------|--------------------|-------------------|
| Pooling | `mean` / `max` / `cls` | Submuestreo secuencial (2.4.01) | Cómo se resume el comentario |
| Normalización | `nn.LayerNorm` | Estabilización de activaciones (2.4.03) | Entrenamiento más estable |
| Regularización | `nn.Dropout` | Prevención de overfitting (2.4.02) | Menos memorización de ruido |
| Capa oculta | `nn.Linear(768, 256)` | Transformación afín `Wx + b` (2.1, 2.3) | Capacidad representacional |
| Activación | `nn.GELU` | No linealidad (2.1) | Sin ella, la red es lineal |
| Salida | `nn.Linear(256, 3)` | Logits (2.2) | 3 clases de sentimiento |

### 9.2 Código de referencia

```python
import torch
import torch.nn as nn
from transformers import AutoModel


class YouTubeSentimentClassifier(nn.Module):
    """
    Red Neuronal para Análisis de Sentimiento basada en Transfer Learning.
    Conecta un Backbone Transformer preentrenado con un Cabezal Personalizado.
    """

    def __init__(
        self,
        model_name: str = "pysentimiento/robertuito-base-uncased",
        num_classes: int = 3,           # 0: Negativo, 1: Neutro, 2: Positivo
        pooling_type: str = "mean",     # 'cls', 'mean' o 'max'   (Sección 2.4.01)
        dropout_rate: float = 0.3,      # Prevención de overfitting (Sección 2.4.02)
        hidden_dim: int = 256,          # Neuronas de la capa oculta (Sección 2.3)
        freeze_backbone: bool = True,   # True = Feature Extraction | False = Fine-Tuning
    ):
        super().__init__()

        # 1. CARGA DEL MODELO BASE (Sección 3: Checkpoints)
        self.backbone = AutoModel.from_pretrained(model_name)
        embedding_dim = self.backbone.config.hidden_size   # 768 en BERT-base
        self.pooling_type = pooling_type
        self.embedding_dim = embedding_dim

        # 2. ESTRATEGIA DE TRANSFER LEARNING (Sección 3.7 vs 3.8)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # 3. CAPAS AUXILIARES Y CABEZAL DENSO (Secciones 2.1, 2.3, 2.4)
        self.layer_norm = nn.LayerNorm(embedding_dim)   # estabiliza la salida del backbone
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)  # 768 -> 256
        self.activation = nn.GELU()
        self.dropout2 = nn.Dropout(p=dropout_rate / 2)
        self.fc_out = nn.Linear(hidden_dim, num_classes)  # 256 -> 3 logits

    def unfreeze_last_layers(self, num_layers: int = 2):
        """
        Descongelación progresiva (Sección 3.8): desbloquea solo los últimos bloques
        del Transformer para refinar el modelo con jerga de YouTube.
        Debe combinarse con Learning Rate Diferencial.
        """
        # La ruta del encoder cambia según la arquitectura:
        #   BERT / RoBERTa -> backbone.encoder.layer
        #   DistilBERT     -> backbone.transformer.layer
        encoder = getattr(self.backbone, "encoder", None)
        if encoder is not None and hasattr(encoder, "layer"):
            layers = encoder.layer[-num_layers:]
        elif hasattr(self.backbone, "transformer"):
            layers = self.backbone.transformer.layer[-num_layers:]
        else:
            raise AttributeError("Arquitectura no soportada por unfreeze_last_layers()")

        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = True

    def apply_pooling(self, token_embeddings: torch.Tensor,
                       attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Submuestreo secuencial (Sección 2.4.01): comprime [batch, seq, 768] en [batch, 768].
        """
        if self.pooling_type == "cls":
            # Primer token especial [CLS]: resume la secuencia (estilo BERT)
            return token_embeddings[:, 0, :]

        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        if self.pooling_type == "max":
            # MaxPooling: máximo por dimensión, ignorando el padding
            # masked_fill es out-of-place: no corrompe el grafo de autograd
            masked = token_embeddings.masked_fill(mask_expanded == 0, -1e9)
            return torch.max(masked, dim=1)[0]

        # MeanPooling (por defecto): promedio de los tokens reales
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Paso 1: representaciones del modelo base
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state      # [batch, seq, 768]

        # Paso 2: submuestreo
        pooled = self.apply_pooling(token_embeddings, attention_mask)  # [batch, 768]

        # Paso 3: normalización + regularización
        x = self.layer_norm(pooled)
        x = self.dropout1(x)

        # Paso 4: capa densa oculta + activación no lineal
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout2(x)

        # Paso 5: logits crudos (la softmax la aplica CrossEntropyLoss / la inferencia)
        return self.fc_out(x)
```

### 9.3 Explicación pedagógica de cada decisión

- **¿Por qué congelar el backbone?** Entrenar 110 M de parámetros con ~2 000 comentarios destruiría el conocimiento preentrenado (*catastrophic forgetting*). Congelarlo convierte el problema en una regresión logística sobre features de 768 dims: barato, estable y suficiente como primera cota.
- **¿Por qué `LayerNorm` después del backbone?** Las activaciones del Transformer tienen escalas heterogéneas por dimensión; normalizarlas antes de una capa lineal mejora la condición del problema de optimización.
- **¿Por qué `GELU` y no `ReLU`?** GELU es suave y no anula gradientes en negativos pequeños; es la activación nativa de los Transformers, lo que reduce el desajuste entre backbone y cabezal.
- **¿Por qué dos dropouts con tasas distintas?** El primero protege la entrada del cabezal (la representación más informativa); el segundo, con tasa menor, evita apagar demasiado justo antes de la decisión final.
- **¿Por qué la salida no lleva softmax?** `nn.CrossEntropyLoss` espera **logits** y aplica `log_softmax` internamente, lo que es numéricamente más estable. Añadir `softmax` en el modelo y luego `CrossEntropyLoss` es un error clásico: **doble softmax**.
- **¿Por qué `masked_fill` en lugar de asignación in-place?** `token_embeddings[mask == 0] = -1e9` modifica un tensor que requiere gradiente y puede romper el grafo de autograd o corromper el `last_hidden_state` compartido.

### 9.4 Verificación de shapes (smoke test)

```python
if __name__ == "__main__":
    model = YouTubeSentimentClassifier(freeze_backbone=True)
    ids = torch.randint(0, 1000, (4, 128))
    mask = torch.ones(4, 128, dtype=torch.long)
    logits = model(ids, mask)
    print(logits.shape)                     # torch.Size([4, 3])

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Entrenables: {trainable:,} / {total:,} ({100 * trainable / total:.2f} %)")
    # Con backbone congelado ≈ 0.2 % de los parámetros
```

Ese porcentaje es una de las evidencias más potentes del *Transfer Learning*: se entrena **una fracción mínima** del modelo.

---

## 10. Fase 4 · Entrenamiento (`train.py`)

### 10.1 El bucle de 5 pasos (Sección 2.8)

| Paso | Llamada | Qué ocurre matemáticamente |
|------|---------|----------------------------|
| **0. Reinicio** | `optimizer.zero_grad()` | Se ponen los gradientes a cero para no acumularlos del batch anterior |
| **1. Inferencia** | `logits = model(...)` | Propagación hacia delante: `z = W₃·GELU(W₂·LN(pool(H)))` |
| **2. Evaluación** | `loss = criterion(logits, labels)` | Entropía cruzada: `-log(p_correcta)` |
| **3. Derivadas** | `loss.backward()` | Autograd aplica la regla de la cadena y llena `.grad` en cada parámetro |
| **4. Corrección** | `optimizer.step()` | Se actualizan `W` y `b` en la dirección opuesta al gradiente |

### 10.2 Código de referencia

```python
import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW, SGD, RMSprop
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from model_architecture import YouTubeSentimentClassifier


class CommentsDataset(Dataset):
    """Dataset de PyTorch: tokeniza comentarios crudos y devuelve tensores + etiqueta."""

    def __init__(self, texts, labels, tokenizer, max_len: int = 128):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(int(self.labels[idx]), dtype=torch.long),
        }


def build_optimizer(model, optimizer_name: str = "AdamW",
                    lr_head: float = 1e-3, lr_backbone: float = 2e-5):
    """
    Learning Rate Diferencial (Sección 3.8):
      - zancada diminuta en el backbone (2e-5): no destruir los pesos preentrenados
      - zancada normal en el cabezal (1e-3): aprender rápido la tarea nueva
    """
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = [p for n, p in model.named_parameters()
                   if not n.startswith("backbone.") and p.requires_grad]

    param_groups = [
        {"params": backbone_params, "lr": lr_backbone},
        {"params": head_params, "lr": lr_head},
    ]

    if optimizer_name == "AdamW":
        # Weight decay desacoplado: el "cable elástico" no se mezcla con el gradiente adaptativo
        return AdamW(param_groups, weight_decay=0.01)
    if optimizer_name == "RMSprop":
        # Frenos ABS: divide por la raíz de la media de gradientes al cuadrado
        return RMSprop(param_groups, weight_decay=1e-4)
    if optimizer_name == "SGD_Momentum":
        # Bola pesada: inercia 0.9 para atravesar mesetas
        return SGD(param_groups, momentum=0.9, weight_decay=1e-4)
    raise ValueError(f"Optimizador no soportado: {optimizer_name}")


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)          # Paso 0: REINICIO

        logits = model(input_ids=input_ids, attention_mask=attention_mask)  # Paso 1

        loss = criterion(logits, labels)               # Paso 2

        loss.backward()                                # Paso 3
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()                               # Paso 4

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(dataloader), correct / total


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Validación: model.eval() desactiva Dropout y usa estadísticas completas de LayerNorm."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    y_true, y_pred = [], []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    return total_loss / len(dataloader), correct / total, y_true, y_pred


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)

    model_name = "pysentimiento/robertuito-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # --- Datos: CSV con columnas (texto, etiqueta) donde etiqueta ∈ {0, 1, 2} ---
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv("data/sentiment_es.csv")
    df = df.dropna(subset=["texto", "etiqueta"])
    df = df[df["texto"].astype(str).str.strip() != ""]

    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["etiqueta"]
    )

    train_ds = CommentsDataset(train_df["texto"], train_df["etiqueta"], tokenizer)
    val_ds = CommentsDataset(val_df["texto"], val_df["etiqueta"], tokenizer)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    # --- Modelo (feature extraction) ---
    model = YouTubeSentimentClassifier(
        model_name=model_name, num_classes=3, pooling_type="mean",
        dropout_rate=0.3, freeze_backbone=True,
    ).to(device)

    criterion = nn.CrossEntropyLoss()   # aplica softmax internamente
    optimizer = build_optimizer(model, optimizer_name="AdamW")

    best_f1, patience, EPOCHS = -1.0, 0, 3
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc, y_true, y_pred = evaluate(model, val_loader, criterion, device)

        from sklearn.metrics import f1_score
        f1 = f1_score(y_true, y_pred, average="macro")

        print(f"Época {epoch}/{EPOCHS} | train loss {tr_loss:.4f} acc {tr_acc:.3f} "
              f"| val loss {va_loss:.4f} acc {va_acc:.3f} F1 {f1:.3f}")

        if f1 > best_f1:
            best_f1, patience = f1, 0
            torch.save(model.state_dict(), "checkpoints/best_sentiment_head.pt")
            print(f"  ↳ checkpoint guardado (F1 macro = {f1:.3f})")
        else:
            patience += 1
            if patience >= 2:
                print("Early stopping: la validación dejó de mejorar.")
                break


if __name__ == "__main__":
    main()
```

### 10.3 Decisiones que hay que justificar

| Decisión | Motivo |
|----------|--------|
| `CrossEntropyLoss` sin `softmax` en el modelo | Evita el doble softmax y mejora la estabilidad numérica (`log_softmax` + `NLLLoss`) |
| `clip_grad_norm_(..., 1.0)` | Evita la explosión de gradientes en las primeras iteraciones del cabezal |
| `stratify=df["etiqueta"]` | Comentarios neutros son minoría; sin estratificar, la clase neutra puede desaparecer de validación |
| `f1_score(average="macro")` como criterio de checkpoint | La accuracy engaña con clases desbalanceadas; el F1 macro penaliza ignorar la clase neutra |
| `torch.manual_seed(42)` | Reproducibilidad: sin semilla, los experimentos del alumno no son comparables |
| `dropna` + filtro de texto vacío | Comentarios vacíos producen solo tokens especiales: ruido puro |

### 10.4 De dónde salen los datos de entrenamiento

La app analiza comentarios de YouTube, pero **no puede entrenar con ellos**: no tienen etiqueta. El entrenamiento usa un corpus público etiquetado en español:

- `pysentimiento/es_sentiment` (Hugging Face) — reseñas y tuits en español.
- `mteb/spanish_sentiment` — benchmark multilingüe de sentimiento.
- Cualquier CSV propio con columnas `texto,etiqueta`.

**Tarea obligatoria del alumno:** escribir el *mapeo de etiquetas* del dataset elegido a `{0: Negativo, 1: Neutro, 2: Positivo}` y documentarlo. Si el dataset solo tiene 2 clases, hay que **decidir y justificar** qué hacer con el neutro (eliminarlo del diseño o buscarlo en otra fuente).

> **Nota de honestidad metodológica:** entrenar con reseñas y predecir sobre comentarios de YouTube es un **cambio de dominio** (*domain shift*). Parte del error final en producción viene de ahí, no del modelo. Saber identificarlo es parte de la competencia analítica.

---

## 11. Fase 5 · Inferencia y métricas

### 11.1 De logits a decisión

```python
import numpy as np
import torch
import torch.nn.functional as F

id2label = {0: "Negativo", 1: "Neutro", 2: "Positivo"}


@torch.no_grad()
def predict_sentiment_batch(df, tokenizer, model, device, batch_size: int = 32):
    """
    Inferencia por lotes + Softmax (Sección 2.2).
    Devuelve etiqueta, confianza (%) y polaridad neta en [-1, +1].
    """
    texts = df["comentario"].astype(str).tolist()
    preds_all, conf_all, neg_all, pos_all = [], [], [], []

    for i in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[i:i + batch_size],
            padding=True, truncation=True, max_length=128, return_tensors="pt",
        ).to(device)

        logits = model(encoded["input_ids"], encoded["attention_mask"])
        probs = F.softmax(logits, dim=1).cpu().numpy()   # [batch, 3]

        for prob_vec, pred, conf in zip(probs, probs.argmax(1), probs.max(1)):
            preds_all.append(id2label[int(pred)])
            conf_all.append(round(float(conf) * 100, 1))
            neg_all.append(float(prob_vec[0]))
            pos_all.append(float(prob_vec[2]))

    df = df.copy()
    df["sentimiento"] = preds_all
    df["confianza_%"] = conf_all
    df["polaridad_neta"] = np.round(np.array(pos_all) - np.array(neg_all), 3)
    return df
```

### 11.2 Las tres métricas que el alumno debe saber leer

| Métrica | Fórmula intuitiva | Qué revela |
|---------|-------------------|------------|
| **Confianza media** | media de `max(softmax)` | Si el modelo está decidiendo o dudando |
| **Polaridad neta** | `P(Positivo) − P(Negativo)` | Sentimiento continuo, mejor que la etiqueta dura para promediar |
| **Acuerdo con likes** | correlación polaridad ↔ likes | Si la comunidad "premia" el mismo tono que detecta la red |

### 11.3 Evaluación obligatoria del modelo entrenado

```python
from sklearn.metrics import classification_report, confusion_matrix

_, acc, y_true, y_pred = evaluate(model, val_loader, criterion, device)
print(classification_report(y_true, y_pred, target_names=["Negativo", "Neutro", "Positivo"], digits=3))
print(confusion_matrix(y_true, y_pred))
```

**Preguntas de laboratorio:** ¿en qué clase se confunde más? ¿Es hacia el neutro? ¿Por qué el neutro es estructuralmente difícil en PLN?

---

## 12. Fase 6 · Dashboard (`app.py`)

### 12.1 Arquitectura del frontal

```text
Sidebar                          Main
├── API Key (password)           ├── 4 métricas (volumen, % pos, % neg, confianza media)
├── URL del vídeo                ├── Matplotlib: 3 gráficos en una figura
├── Máx. comentarios (slider)    ├── Tabla de comentarios + predicciones
├── Modelo base (selectbox)      └── Botón de descarga CSV (utf-8-sig)
├── Pooling (selectbox)
├── Dropout (slider)
├── Ruta del checkpoint
└── Botón "Extraer y Analizar"
```

### 12.2 Código de referencia (integración completa)

```python
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from transformers import AutoTokenizer

from model_architecture import YouTubeSentimentClassifier

st.set_page_config(page_title="YouTube Sentiment DL Analyzer", page_icon="🧠", layout="wide")

# ==========================================
# 1. YOUTUBE API  (ver sección 7)
# ==========================================
def extract_video_id(url: str):
    pattern = r"(?:v=|\/|youtu\.be\/|embed\/)([0-9A-Za-z_-]{11})"
    match = re.search(pattern, url)
    return match.group(1) if match else None


def fetch_youtube_comments(api_key: str, video_id: str, max_comments: int = 200) -> pd.DataFrame:
    youtube = build("youtube", "v3", developerKey=api_key)
    comments_data, next_page_token = [], None

    while len(comments_data) < max_comments:
        batch_size = min(100, max_comments - len(comments_data))
        request = youtube.commentThreads().list(
            part="snippet", videoId=video_id, maxResults=batch_size,
            order="relevance", textFormat="plainText", pageToken=next_page_token,
        )
        response = request.execute()

        for item in response.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            comments_data.append({
                "autor": snippet.get("authorDisplayName", "Anónimo"),
                "comentario": snippet.get("textDisplay", ""),
                "likes": int(snippet.get("likeCount", 0)),
                "publicado": snippet.get("publishedAt", "")[:10],
            })

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    df = pd.DataFrame(comments_data)
    if not df.empty:
        df = df[df["comentario"].str.strip() != ""]
        df = df.sort_values("likes", ascending=False).reset_index(drop=True)
    return df


# ==========================================
# 2. MODELO E INFERENCIA
# ==========================================
@st.cache_resource
def load_neural_pipeline(model_name: str, pooling_type: str, dropout_rate: float,
                         checkpoint_path: str = None):
    """Carga tokenizador + red una sola vez (cache_resource) y aplica los pesos entrenados."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = YouTubeSentimentClassifier(
        model_name=model_name, num_classes=3, pooling_type=pooling_type,
        dropout_rate=dropout_rate, freeze_backbone=True,
    )

    if checkpoint_path and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state)

    model.to(device).eval()   # eval(): desactiva Dropout en producción
    return tokenizer, model, device


id2label = {0: "Negativo", 1: "Neutro", 2: "Positivo"}


@torch.no_grad()
def predict_sentiment_batch(df, tokenizer, model, device, batch_size: int = 32):
    texts = df["comentario"].astype(str).tolist()
    preds_all, conf_all, neg_all, pos_all = [], [], [], []

    for i in range(0, len(texts), batch_size):
        encoded = tokenizer(texts[i:i + batch_size], padding=True, truncation=True,
                            max_length=128, return_tensors="pt").to(device)
        logits = model(encoded["input_ids"], encoded["attention_mask"])
        probs = F.softmax(logits, dim=1).cpu().numpy()

        for prob_vec, pred, conf in zip(probs, probs.argmax(1), probs.max(1)):
            preds_all.append(id2label[int(pred)])
            conf_all.append(round(float(conf) * 100, 1))
            neg_all.append(float(prob_vec[0]))
            pos_all.append(float(prob_vec[2]))

    df = df.copy()
    df["sentimiento"] = preds_all
    df["confianza_%"] = conf_all
    df["polaridad_neta"] = np.round(np.array(pos_all) - np.array(neg_all), 3)
    return df


# ==========================================
# 3. VISUALIZACIÓN (MATPLOTLIB)
# ==========================================
def render_matplotlib_dashboard(df: pd.DataFrame):
    col_map = {"Positivo": "#2ecc71", "Neutro": "#95a5a6", "Negativo": "#e74c3c"}
    categorias = ["Positivo", "Neutro", "Negativo"]
    colores = [col_map[c] for c in categorias]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    fig.patch.set_facecolor("#ffffff")

    # --- 1. Distribución (donut) ---
    conteo = df["sentimiento"].value_counts().reindex(categorias, fill_value=0)
    _, _, autotexts = axes[0].pie(
        conteo, labels=categorias, autopct="%1.1f%%", startangle=140,
        colors=colores, pctdistance=0.78,
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2),
    )
    for t in autotexts:
        t.set_color("white")
        t.set_weight("bold")
    axes[0].set_title("1. Distribución de Sentimiento (Volumen)",
                      fontsize=12, fontweight="bold", pad=12)

    # --- 2. Impacto ponderado por likes ---
    df = df.copy()
    df["peso_interaccion"] = df["likes"] + 1     # +1: un comentario sin likes también cuenta
    impacto = df.groupby("sentimiento")["peso_interaccion"].sum().reindex(categorias, fill_value=0)

    bars = axes[1].bar(categorias, impacto, color=colores, edgecolor="#2c3e50", width=0.55)
    axes[1].set_title("2. Impacto Ponderado por Likes (Comunidad)",
                      fontsize=12, fontweight="bold", pad=12)
    axes[1].set_ylabel("Suma de (Likes + 1)")
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    for bar in bars:
        axes[1].annotate(f"{int(bar.get_height()):,}",
                         xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                         xytext=(0, 4), textcoords="offset points",
                         ha="center", va="bottom", fontweight="bold")

    # --- 3. Confianza de la red (softmax) ---
    for cat in categorias:
        subset = df.loc[df["sentimiento"] == cat, "confianza_%"]
        if not subset.empty:
            axes[2].hist(subset, bins=10, alpha=0.65, label=cat,
                         color=col_map[cat], edgecolor="black")
    axes[2].set_title("3. Confianza de la Red (Probabilidad Softmax)",
                      fontsize=12, fontweight="bold", pad=12)
    axes[2].set_xlabel("Confianza de Predicción (%)")
    axes[2].set_ylabel("Nº de Comentarios")
    axes[2].legend(loc="upper left")
    axes[2].grid(axis="y", linestyle="--", alpha=0.4)
    axes[2].spines["top"].set_visible(False)
    axes[2].spines["right"].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)          # libera memoria: Streamlit reejecuta el script en cada interacción


# ==========================================
# 4. INTERFAZ
# ==========================================
st.title("🎬 Analizador Neuronal de Sentimiento en YouTube")
st.markdown("Práctica de **Transfer Learning en PLN**, **capas personalizadas en PyTorch** "
            "y **YouTube Data API v3**.")

with st.sidebar:
    st.header("⚙️ 1. Credenciales y vídeo")
    api_key = st.text_input("YouTube API Key v3", type="password")
    video_url = st.text_input("URL del vídeo", placeholder="https://www.youtube.com/watch?v=...")
    max_comments = st.slider("Máximo de comentarios", 20, 500, 100, step=20)

    st.divider()
    st.header("🧠 2. Arquitectura de red (laboratorio)")
    model_choice = st.selectbox("Modelo base (checkpoint de Hugging Face)", [
        "pysentimiento/robertuito-base-uncased",
        "dccuchile/bert-base-spanish-wwm-cased",
        "distilbert-base-multilingual-cased",
    ])
    pooling_choice = st.selectbox(
        "Pooling", ["mean", "cls", "max"],
        help="Sección 2.4.01: cómo resumir las palabras del comentario en un solo vector.")
    dropout_choice = st.slider(
        "Dropout del cabezal", 0.1, 0.5, 0.3, 0.05,
        help="Sección 2.4.02: porcentaje de neuronas apagadas durante el entrenamiento.")
    checkpoint_file = st.text_input("Checkpoint entrenado (.pt)",
                                    value="checkpoints/best_sentiment_head.pt")

    run_btn = st.button("🚀 Extraer y analizar", use_container_width=True)

if run_btn:
    if not api_key or not video_url:
        st.error("Introduce la API Key de Google Cloud y una URL de YouTube válida.")
    else:
        video_id = extract_video_id(video_url)
        if not video_id:
            st.error("No se pudo extraer un ID de vídeo válido de la URL.")
        else:
            try:
                with st.spinner("1/2 Descargando comentarios (YouTube Data API v3)..."):
                    df_comments = fetch_youtube_comments(api_key, video_id, max_comments)

                if df_comments.empty:
                    st.warning("No se encontraron comentarios en este vídeo.")
                else:
                    with st.spinner("2/2 Ejecutando inferencia neuronal..."):
                        tokenizer, model, device = load_neural_pipeline(
                            model_choice, pooling_choice, dropout_choice, checkpoint_file)
                        df = predict_sentiment_batch(df_comments, tokenizer, model, device)

                    if not os.path.exists(checkpoint_file):
                        st.warning("⚠️ No se encontró checkpoint entrenado: las predicciones son "
                                   "de un cabezal sin entrenar y NO representan sentimiento real.")

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Comentarios analizados", len(df))
                    c2.metric("Sentimiento positivo", f'{(df["sentimiento"] == "Positivo").mean() * 100:.1f}%')
                    c3.metric("Sentimiento negativo", f'{(df["sentimiento"] == "Negativo").mean() * 100:.1f}%')
                    c4.metric("Confianza media", f'{df["confianza_%"].mean():.1f}%')

                    st.subheader("📊 Pulso de la audiencia (Matplotlib)")
                    render_matplotlib_dashboard(df)

                    st.subheader("💬 Detalle de comentarios y predicciones")
                    st.dataframe(df, use_container_width=True)

                    st.download_button(
                        "📥 Descargar resultados (CSV)",
                        data=df.to_csv(index=False).encode("utf-8-sig"),
                        file_name=f"sentimiento_youtube_{video_id}.csv",
                        mime="text/csv",
                    )
            except HttpError as e:
                st.error(f"Error de la API de YouTube (cuota agotada o comentarios desactivados): {e}")
```

### 12.3 Detalles técnicos que demuestran comprensión

| Detalle | Por qué importa |
|---------|-----------------|
| `@st.cache_resource` | Cargar un Transformer tarda ~10 s y ocupa cientos de MB; Streamlit reejecuta el script en cada interacción |
| `model.eval()` | Desactiva `Dropout`; sin esto las predicciones son **no deterministas** |
| `torch.no_grad()` | Evita construir el grafo de autograd: menos memoria y ~2× más rápido |
| `batch_size=32` | Equilibrio entre throughput y memoria; con secuencias de 128 tokens es seguro en CPU |
| `plt.close(fig)` | Sin esto, Matplotlib acumula figuras en memoria y la app se degrada |
| `utf-8-sig` | BOM para que Excel abra correctamente los acentos |
| `weights_only=True` en `torch.load` | Evita deserializar código arbitrario desde un `.pt` no confiable |
| Aviso si no hay checkpoint | Honestidad metodológica: no se presentan predicciones aleatorias como análisis |

---

## 13. Experimentos de laboratorio

Cada experimento exige **hipótesis → ejecución → evidencia gráfica → conclusión escrita**. Sin las cuatro partes no se considera entregado.

### Experimento 1 · Pooling: `cls` vs `mean` vs `max`

- **Hipótesis:** en comentarios largos donde se halaga al creador y luego se lanza una crítica, `max` capturará mejor el pico negativo que `mean`, que diluye la señal.
- **Procedimiento:** fijar semilla y checkpoint-base; ejecutar inferencia con los tres valores de `pooling_type` sobre el mismo vídeo; comparar distribución y confianza media.
- **Evidencia esperada:** `max` más extremo (más positivos *y* más negativos), `mean` más conservador, `cls` dependiente del preentrenamiento.
- **Pregunta de cierre:** ¿con qué pooling se obtiene mayor **entropía** de predicciones y por qué?

### Experimento 2 · Optimizador: `AdamW` vs `SGD + Momentum` vs `RMSprop`

- **Hipótesis:** `AdamW` desciende rápido en las primeras épocas; `SGD + Momentum` necesita más épocas pero puede alcanzar mejor generalización.
- **Procedimiento:** 3 épocas, mismos datos, misma semilla, mismo `lr_head`; graficar `loss` de train y validación por época.
- **Evidencia esperada:** curva de `AdamW` con caída inicial pronunciada; `SGD` con pendiente suave.
- **Pregunta de cierre:** ¿por qué un lr de `1e-3` que funciona en `AdamW` puede hacer divergir a `SGD`?

### Experimento 3 · Feature Extraction vs Fine-Tuning parcial

- **Hipótesis:** `unfreeze_last_layers(2)` mejora el F1 en jerga de YouTube a cambio de ~3–5× más tiempo por época.
- **Procedimiento:** medir tiempo/época y F1 final en ambos regímenes con **Learning Rate Diferencial** (`2e-5` backbone / `1e-3` cabezal).
- **Evidencia esperada:** mejora modesta de F1, coste computacional alto, riesgo de sobreajuste con pocos datos.
- **Pregunta de cierre:** ¿qué evidencia en la curva de validación indicaría *catastrophic forgetting*?

### Experimento 4 · Dropout y sobreajuste

- **Hipótesis:** `dropout=0.1` sobreajusta antes (brecha train-val creciente); `dropout=0.5` estabiliza pero puede infraentrenar.
- **Procedimiento:** comparar `0.1 / 0.3 / 0.5` con más épocas de las habituales.
- **Evidencia esperada:** brecha train-val como firma del sobreajuste.

### Experimento 5 · Ponderación por likes

- **Hipótesis:** el ranking por volumen y el ranking por impacto ponderado difieren de forma sistemática.
- **Procedimiento:** comparar el gráfico 1 (volumen) con el gráfico 2 (likes) y cuantificar la discrepancia.
- **Pregunta de cierre:** ¿qué mide realmente cada gráfico y cuál usaría un creador para decidir?

### Plantilla de informe (obligatoria)

```markdown
## Experimento N — <título>
**Hipótesis:** ...
**Configuración:** modelo, pooling, dropout, optimizador, lr, épocas, semilla
**Resultados:** (tabla o figura)
**Conclusión:** ... (en términos del modelo, no de la librería)
**Qué haría distinto:** ...
```

---

## 14. Skills como aceleradores de desarrollo

Las siguientes *skills* están instaladas en el entorno y **asisten la construcción** del proyecto. Se documentan aquí por dos razones: (a) reproducir el entorno de desarrollo, y (b) que el alumno sepa **a qué recurso acudir** cuando se atasque en un detalle de implementación.

> **No son objetivos de aprendizaje ni se evalúan.** El examen pregunta por capas, gradientes, pooling y optimizadores — no por nombres de herramientas.

### 14.1 Instalación

```bash
npx skills add https://github.com/streamlit/agent-skills --skill developing-with-streamlit
npx skills add https://github.com/pluginagentmarketplace/custom-plugin-python --skill 'Pandas Data Analysis'
npx skills add https://github.com/mindrally/skills --skill numpy-best-practices
npx skills add https://github.com/affaan-m/ecc --skill pytorch-patterns
npx skills add https://github.com/k-dense-ai/scientific-agent-skills --skill transformers
npx skills add https://github.com/k-dense-ai/scientific-agent-skills --skill scikit-learn
npx skills add https://github.com/k-dense-ai/scientific-agent-skills --skill matplotlib
npx skills add https://github.com/orchestra-research/ai-research-skills --skill peft-fine-tuning
npx skills add https://github.com/chalk-ai/chalk-ai-plugins --skill accelerate-python-resolvers
```

### 14.2 Mapa skill → fase del proyecto

| Fase | Skill de apoyo | Para qué exactamente |
|------|----------------|----------------------|
| 1 · Ingesta | `Pandas Data Analysis` | Aplanar JSON anidado, deduplicar, agrupar por sentimiento |
| 1 · Ingesta | `numpy-best-practices` | Vectores eficientes para `polaridad_neta` y agregaciones |
| 2 · Tokenización | `transformers` | `AutoTokenizer`, `AutoModel`, pooling, límites de longitud |
| 3 · Cabezal | `pytorch-patterns` | `nn.Module` idiomático, gestión de `device`, `train()`/`eval()` |
| 4 · Entrenamiento | `pytorch-patterns` | Bucle reproducible, `zero_grad(set_to_none=True)`, checkpoints |
| 4 · Entrenamiento | `scikit-learn` | `train_test_split` estratificado, `classification_report`, F1 macro |
| 4 · Avanzado | `peft-fine-tuning` | LoRA si el fine-tuning completo no cabe en el hardware |
| 4 · Avanzado | `accelerate-python-resolvers` | `mixed precision`, batches grandes, multi-GPU |
| 5 · Inferencia | `scikit-learn` | Matriz de confusión, análisis de errores por clase |
| 6 · Dashboard | `developing-with-streamlit` | `cache_resource`, `session_state`, layout, `st.pyplot` |
| 6 · Dashboard | `matplotlib` | Figuras limpias, anotaciones, paletas accesibles |

### 14.3 Uso responsable de la asistencia

```text
✅ Consultar la skill cuando:  no recuerdo la firma de una API o el patrón idiomático
✅ Consultar la skill cuando:  un warning/error técnico no sé interpretar
❌ Usar la skill para:         decidir la arquitectura, el pooling o el optimizador
❌ Usar la skill para:         escribir la conclusión de un experimento
```

**Regla del curso:** toda línea de código entregada debe poder **explicarse y modificarse** en vivo. Si el alumno no puede cambiar `hidden_dim` y predecir qué ocurrirá, no ha aprendido el concepto que esa línea representa.

### 14.4 Flujo de trabajo recomendado con asistencia

1. Escribir la **hipótesis** y el **cambio mínimo** (una variable a la vez).
2. Consultar la skill solo para el detalle sintáctico.
3. Ejecutar y guardar la evidencia (figura/tabla).
4. Escribir la conclusión **antes** de pedir la siguiente ayuda.
5. Registrar en el informe qué se consultó y qué se decidió por cuenta propia.

---

## 15. Errores comunes y depuración

| Síntoma | Causa probable | Solución |
|---------|----------------|----------|
| Predicciones ~33 % en cada clase | Cabezal sin entrenar o checkpoint no cargado | Verificar ruta y `load_state_dict`; comprobar `strict=True` |
| Todas las predicciones iguales | `pooling="cls"` con backbone no entrenado para esa tarea | Probar `mean` |
| `RuntimeError: expected scalar type Long` | `attention_mask` en `float` | Mantenerlo en `torch.long` |
| `CUDA out of memory` | `batch_size` o `max_length` altos | Reducir a 16/64 o usar CPU |
| Loss en `nan` | lr demasiado alto en `SGD`, o gradientes explosivos | Bajar lr, mantener `clip_grad_norm_` |
| Val loss sube mientras train baja | Sobreajuste | Más dropout, early stopping, menos épocas |
| F1 macro muy bajo con accuracy alta | Clases desbalanceadas (neutro minoritario) | `stratify`, `class_weight` en `CrossEntropyLoss` |
| `HttpError 403 quotaExceeded` | Cuota diaria agotada | Esperar o crear otra clave |
| `HttpError 403 commentsDisabled` | Comentarios cerrados | Probar otro vídeo |
| App lentísima en cada clic | Falta `@st.cache_resource` | Cachear tokenizador y modelo |
| Memoria creciente | Figuras de Matplotlib sin cerrar | `plt.close(fig)` |
| Acentos corruptos en Excel | CSV sin BOM | `encode("utf-8-sig")` |

**Protocolo de depuración (5 minutos):** ¿el shape es el esperado? → ¿el `device` coincide? → ¿`model.train()` vs `model.eval()`? → ¿se cargó el checkpoint? → ¿los datos tienen la etiqueta correcta?

---

## 16. Entregables y rúbrica

### 16.1 Entregables

1. **Repositorio funcional** con `app.py`, `model_architecture.py`, `train.py` y `requirements.txt`.
2. **Checkpoint entrenado** (`checkpoints/*.pt`) con su curva de pérdida.
3. **Informe de experimentos** (3 de los 5 propuestos, con la plantilla de la sección 13).
4. **Nota metodológica** (máx. 1 página): de qué dataset salieron las etiquetas, cómo se mapearon a 3 clases, y qué limitaciones tiene el modelo.
5. **Demo en vivo** de 5 minutos: analizar un vídeo nuevo y explicar un gráfico.

### 16.2 Rúbrica

| Criterio | Peso | Indicador de excelencia |
|----------|------|--------------------------|
| Comprensión conceptual | 30 % | Explica shapes y gradientes sin leer el código; distingue feature extraction de fine-tuning |
| Rigor experimental | 25 % | Una variable por experimento, semilla fijada, conclusión apoyada en evidencia |
| Calidad del modelo | 20 % | F1 macro reportado, análisis de errores por clase, baseline comparado |
| Producto funcional | 15 % | App estable, caché correcta, CSV exportable, aviso honesto si no hay checkpoint |
| Comunicación | 10 % | Gráficos legibles con conclusión escrita; distingue volumen de impacto |

**Penalizaciones explícitas:** presentar predicciones de un modelo sin entrenar como resultados (−20 %); reportar solo accuracy con clases desbalanceadas (−10 %); conclusiones que hablan de librerías en vez del modelo (−10 %).

---

## 17. Conclusión didáctica

Al finalizar la práctica, el alumno no solo habrá desplegado una herramienta web funcional, sino que habrá adquirido la capacidad crítica de **diagnosticar una red neuronal**:

- Sabrá **por qué** una capa de pooling altera la detección de matices en un comentario largo.
- Sabrá **cómo** el dropout previene la memorización de ruido en datasets pequeños.
- Sabrá **por qué** la elección del optimizador y del learning rate determina el éxito o el estancamiento del entrenamiento.
- Sabrá **qué no puede afirmar** con su modelo: sesgo de muestra, cambio de dominio y límites del neutro.

Ese es el objetivo: no replicar este caso concreto, sino **transferir** el razonamiento a cualquier problema propio de PLN — con otros datos, otro vocabulario y otras clases.
