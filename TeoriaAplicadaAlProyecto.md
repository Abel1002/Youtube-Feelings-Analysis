# Teoría aplicada al código · Cómo se entrena la red

> Documento de estudio del proyecto. Recorre el entrenamiento real línea a línea
> y lo conecta con la teoría de la presentación: Transfer Learning, capas,
> pooling, optimizadores y bucle de entrenamiento.

---

## Índice

1. [Qué se entrena de verdad](#1-qué-se-entrena-de-verdad)
2. [Transfer Learning en código: tres piezas](#2-transfer-learning-en-código-tres-piezas)
3. [El cabezal, capa por capa](#3-el-cabezal-capa-por-capa)
4. [Pooling: dónde se decide qué es "el comentario"](#4-pooling-dónde-se-decide-qué-es-el-comentario)
5. [Optimizador: el LR diferencial y cuándo está realmente activo](#5-optimizador-el-lr-diferencial-y-cuándo-está-realmente-activo)
6. [El bucle de 5 pasos](#6-el-bucle-de-5-pasos)
7. [La loss y el reparto de culpa](#7-la-loss-y-el-reparto-de-culpa)
8. [Cómo se decide cuál es "el mejor modelo"](#8-cómo-se-decide-cuál-es-el-mejor-modelo)
9. [Traza completa de un batch](#9-traza-completa-de-un-batch)
10. [La tabla teoría → código, verificada](#10-la-tabla-teoría--código-verificada)
11. [Los tres detalles que la presentación no cubre](#11-los-tres-detalles-que-la-presentación-no-cubre)
12. [Dónde se ve que no es una caja negra](#12-dónde-se-ve-que-no-es-una-caja-negra)

---

## 1. Qué se entrena de verdad

Antes de mirar capas, el dato que resume todo:

```text
Parámetros entrenables: 199.171 / 108.986.371 (0,18 %)
```

Ese 0,18 % **es** el Transfer Learning. Y se puede desglosar a mano, porque son
solo tres capas:

| Capa | Cálculo | Parámetros |
|---|---|---|
| `LayerNorm(768)` | 768 pesos + 768 sesgos | 1.536 |
| `Linear(768 → 256)` | 768×256 + 256 sesgos | 196.864 |
| `Linear(256 → 3)` | 256×3 + 3 sesgos | 771 |
| **Total** | | **199.171** |

Los 108.787.200 restantes (109 M − 199.171) son el Transformer, y están
congelados. Si preguntan "¿cuánto entrena tu red?", la respuesta no es "un
Transformer", es **"una regresión logística no lineal sobre features de 768
dimensiones"**.

---

## 2. Transfer Learning en código: tres piezas

### Pieza 1 · El checkpoint como punto de partida

`model_architecture.py:78`

```python
# 1. Checkpoint preentrenado: el conocimiento lingüístico ya está aquí.
self.backbone = AutoModel.from_pretrained(model_name)
self.embedding_dim = int(self.backbone.config.hidden_size)   # 768
```

`from_pretrained` descarga 435 MB de pesos ya entrenados con millones de textos.
No es un clasificador de sentimiento: es un **modelo de lenguaje** que sabe que
"no me gustó nada" y "me encantó" ocupan zonas distintas del espacio. Eso es el
*Inductive Transfer Learning*: reutilizas la representación, no la tarea.

### Pieza 2 · Congelar = decirle a autograd que no toque

`model_architecture.py:82-84`

```python
if freeze_backbone:
    for param in self.backbone.parameters():
        param.requires_grad = False
```

`requires_grad = False` tiene tres consecuencias encadenadas, y conviene saberlas
todas:

1. `loss.backward()` **no calcula** `.grad` para esos parámetros.
2. El optimizador no los ve (no están en ningún `param_group`), así que `step()`
   no los modifica.
3. No se reserva memoria para sus gradientes → por eso el entrenamiento cabe en
   CPU.

### Pieza 3 · Descongelar por bloques (fine-tuning parcial)

`model_architecture.py:108-125`

```python
encoder = getattr(self.backbone, "encoder", None)
if encoder is not None and hasattr(encoder, "layer"):
    layers = encoder.layer[-num_layers:]                    # BERT / RoBERTuito
elif hasattr(self.backbone, "transformer") and hasattr(self.backbone.transformer, "layer"):
    layers = self.backbone.transformer.layer[-num_layers:]   # DistilBERT

for layer in layers:
    for param in layer.parameters():
        param.requires_grad = True
```

Dos rutas porque BERT y DistilBERT nombran distinto su pila de bloques. Se
descongelan **las últimas** capas y no las primeras porque en un Transformer la
sintaxis vive abajo y la semántica arriba: lo que quieres adaptar a jerga de
YouTube es la parte alta.

### La pieza que la presentación no cuenta: `train()` no es lo mismo que "descongelado"

`model_architecture.py:180-190`

```python
def train(self, mode: bool = True) -> "YouTubeSentimentClassifier":
    """
    Si el backbone está completamente congelado, lo mantenemos en `eval`
    incluso durante el entrenamiento del cabezal.
    """
    super().train(mode)
    if mode and not any(p.requires_grad for p in self.backbone.parameters()):
        self.backbone.eval()
    return self
```

Esto es sutil y es un error clásico. **`requires_grad=False` congela los pesos,
pero no desactiva el Dropout interno del Transformer.** Sin este override, cada
`forward` apagaría al azar ~10 % de las neuronas del backbone: el "extractor de
características" dejaría de ser determinista y el cabezal estaría aprendiendo
sobre features que cambian en cada batch. El modelo acabaría funcionando, pero
peor y de forma inexplicable.

La regla mental: **congelado ≠ determinista**. Aquí se fuerza lo segundo.

---

## 3. El cabezal, capa por capa

`model_architecture.py:87-92`

```python
self.layer_norm = nn.LayerNorm(self.embedding_dim)        # 768
self.dropout1   = nn.Dropout(p=self.dropout_rate)         # 0.3
self.fc1        = nn.Linear(self.embedding_dim, self.hidden_dim)  # 768 -> 256
self.activation = nn.GELU()
self.dropout2   = nn.Dropout(p=self.dropout_rate / 2)     # 0.15
self.fc_out     = nn.Linear(self.hidden_dim, num_classes) # 256 -> 3
```

`forward` en `model_architecture.py:160-175`:

```python
outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
token_embeddings = outputs.last_hidden_state          # [batch, 128, 768]
pooled = self.apply_pooling(token_embeddings, attention_mask)   # [batch, 768]
x = self.layer_norm(pooled)      # estabiliza escalas
x = self.dropout1(x)             # protege la entrada del cabezal
x = self.fc1(x)                  # [batch, 256]  transformación afín
x = self.activation(x)           # GELU: no linealidad
x = self.dropout2(x)             # regularización fina
return self.fc_out(x)            # [batch, 3] LOGITS
```

Por qué cada decisión, en una frase cada una:

- **`LayerNorm`** → las 768 dimensiones del Transformer tienen escalas muy
  distintas; normalizarlas antes de una capa lineal mejora el condicionamiento
  del problema de optimización (es el "2.4.03" de la presentación).
- **`Dropout(0.3)` y luego `Dropout(0.15)`** → el primero es el más agresivo
  porque protege la representación más informativa; el segundo, más suave, para
  no apagar demasiado justo antes de decidir.
- **`GELU` en vez de `ReLU`** → GELU es suave y no anula el gradiente en
  negativos pequeños; es la activación nativa del Transformer, así que el cabezal
  no "desentona" con el backbone.
- **Sin `softmax` al final** → `nn.CrossEntropyLoss` espera logits y aplica
  `log_softmax` internamente. Poner softmax en el modelo y luego
  `CrossEntropyLoss` es el **doble softmax**: gradientes mal escalados y
  entrenamiento peor. El softmax vive solo en inferencia (`app.py:283`).

---

## 4. Pooling: dónde se decide qué es "el comentario"

`model_architecture.py:132-155`

```python
if self.pooling_type == "cls":
    return token_embeddings[:, 0, :]

mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

if self.pooling_type == "max":
    min_value = torch.finfo(token_embeddings.dtype).min
    masked = token_embeddings.masked_fill(mask_expanded == 0, min_value)
    return torch.max(masked, dim=1).values

sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
return sum_embeddings / sum_mask
```

La clave está en `mask_expanded`. Sin ella, el `mean` promediaría los `[PAD]`
hasta la posición 128 y diluiría la señal: un comentario de 5 palabras quedaría
dividido entre 128. Con ella, se divide solo entre los tokens reales
(`attention_mask.sum()`).

Y `masked_fill` es **out-of-place** a propósito: `token_embeddings[mask == 0] =
-1e9` modificaría un tensor que requiere gradiente y puede romper el grafo de
autograd o corromper el `last_hidden_state` que se comparte.

---

## 5. Optimizador: el LR diferencial y cuándo está realmente activo

`train.py:148-174`

```python
backbone_params = [p for n, p in model.named_parameters()
                   if n.startswith("backbone.") and p.requires_grad]
head_params = [p for n, p in model.named_parameters()
               if not n.startswith("backbone.") and p.requires_grad]

param_groups = []
if backbone_params:
    param_groups.append({"params": backbone_params, "lr": lr_backbone, "name": "backbone"})
param_groups.append({"params": head_params, "lr": lr_head, "name": "head"})
```

**Ojo con este detalle, porque la tabla de la presentación lo da por hecho y en el
modo por defecto no ocurre:** en Feature Extraction `backbone_params` está
**vacío** (todo `requires_grad=False`), así que el optimizador solo recibe un
grupo. El LR diferencial `2e-5 / 1e-3` está *latente*: se activa únicamente
cuando ejecutas `--unfreeze-layers 2`. Es una decisión correcta (no tiene sentido
crear un grupo vacío), pero conviene explicarlo así en la defensa.

Los tres optimizadores, con su intuición física:

| Optimizador | Regla mental | `weight_decay` por defecto |
|---|---|---|
| **AdamW** | Momento + escala por variable, con el *decay* **desacoplado** del gradiente adaptativo | 0.01 |
| **RMSprop** | Divide por la raíz de la media de gradientes² → "frenos ABS" | 1e-4 |
| **SGD_Momentum** | Bola pesada con inercia 0.9 → atraviesa mesetas, pero necesita más épocas | 1e-4 |

```python
if optimizer_name == "AdamW":
    return AdamW(param_groups, weight_decay=weight_decay)
if optimizer_name == "RMSprop":
    return RMSprop(param_groups, weight_decay=weight_decay)
if optimizer_name == "SGD_Momentum":
    return SGD(param_groups, momentum=0.9, weight_decay=weight_decay)
```

El "W" de AdamW es exactamente lo que se vio en 2.7: el *weight decay* se aplica
**después** del paso adaptativo, no sumado al gradiente. Con Adam normal, el decay
queda escalado por la varianza histórica y regulariza de forma despareja entre
parámetros.

---

## 6. El bucle de 5 pasos

`train.py:194-212`, literal:

```python
for batch in dataloader:
    input_ids      = batch["input_ids"].to(device, non_blocking=True)
    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
    labels         = batch["label"].to(device, non_blocking=True)

    optimizer.zero_grad(set_to_none=True)          # Paso 0: reinicio

    logits = model(input_ids=input_ids, attention_mask=attention_mask)  # Paso 1
    loss = criterion(logits, labels)               # Paso 2

    loss.backward()                                # Paso 3
    if grad_clip and grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
    optimizer.step()                               # Paso 4
```

| Paso | Llamada | Qué ocurre |
|---|---|---|
| 0 | `zero_grad(set_to_none=True)` | Libera los `.grad` del batch anterior. `set_to_none` es mejor que ponerlos a cero: ahorra una escritura y deja que autograd asigne memoria nueva |
| 1 | `model(...)` | `z = W₃ · GELU(W₂ · LN(pool(H)))` |
| 2 | `criterion(logits, labels)` | `−log(p_correcta)` |
| 3 | `loss.backward()` | Regla de la cadena; llena `.grad` **solo** en los 199.171 parámetros entrenables |
| 3.5 | `clip_grad_norm_(..., 1.0)` | Si la norma global supera 1, la reescala. Evita que un batch raro del cabezal recién inicializado dé un paso gigante |
| 4 | `optimizer.step()` | `W ← W − lr · (∇ + λW)` |

`set_to_none=True` no es cosmético: es el patrón recomendado por PyTorch desde 2.0
y reduce el pico de memoria, que en CPU importa.

---

## 7. La loss y el reparto de culpa

`train.py:580-591`

```python
class_weights = None
if args.class_weights:
    counts = train_df[args.label_col].value_counts().reindex([0, 1, 2], fill_value=0)
    class_weights = len(train_df) / (3.0 * counts.clip(lower=1))

weight_tensor = (torch.tensor(class_weights.values, dtype=torch.float32, device=device)
                 if class_weights is not None else None)
criterion = nn.CrossEntropyLoss(weight=weight_tensor)
```

`CrossEntropyLoss` = `log_softmax` + `NLLLoss`, en una sola pasada numéricamente
estable. El `weight` opcional es la respuesta al desbalanceo: multiplica la
pérdida de cada clase por su frecuencia inversa.

Aquí el corpus está balanceado por construcción (`prepare_data.py` usa
`cardiffnlp/tweet_sentiment_multilingual`, 613/613/613 en train), así que
`--class-weights` está apagado y **accuracy ≈ F1 macro** (0,651 vs 0,650). Ese
detalle es útil en la defensa: demuestra que el F1 no está maquillando nada, y que
se eligió la métrica correcta *antes* de saber que saldría bien.

---

## 8. Cómo se decide cuál es "el mejor modelo"

`train.py:664-686`

```python
if val_f1 > best_f1 + 1e-6:
    best_f1 = float(val_f1)
    best_epoch = epoch
    patience_counter = 0
    best_checkpoint = build_checkpoint(model=model, config=config,
                                       metrics={...}, history=list(history),
                                       training=training_meta)
    save_checkpoint(output_path, best_checkpoint)
else:
    patience_counter += 1
    if patience_counter >= args.patience:
        print("Early stopping: la validación no mejora...")
        break
```

Tres decisiones pedagógicamente relevantes:

1. **Criterio = F1 macro de validación**, no accuracy ni loss. Con clases
   desbalanceadas la accuracy premia ignorar la minoritaria; el F1 macro promedia
   por clase y castiga ese atajo.
2. **Early stopping con `patience=2`**: si dos épocas seguidas no mejoran, se
   corta. En el entrenamiento de referencia paró en la época 4 con mejor época 2 —
   es decir, **las épocas 3 y 4 ya estaban sobreajustando**.
3. **Se recarga el mejor checkpoint antes de evaluar** (`train.py:711`):

```python
load_checkpoint_into_model(model, best_checkpoint)
model.to(device)
```

Sin esto evaluarías los pesos de la *última* época, no los del mejor F1. Es un
error silencioso muy común: el informe diría "mejor F1 0,650" y la matriz de
confusión estaría calculada con otro modelo.

### El bug que se arregló en `build_checkpoint`

`model_architecture.py:238-242`

```python
state_dict = {
    key: value.detach().cpu().clone()
    for key, value in model.state_dict().items()
    if not key.startswith("backbone.") or key in trainable_names
}
```

`.detach().cpu()` **no copia** si el tensor ya está en CPU: comparte el mismo
almacenamiento. Sin `.clone()`, el checkpoint guardado en la época 2 seguiría
"viéndose" los pesos de las épocas 3 y 4, y guardarías como mejor modelo unos pesos
que nunca fueron los mejores. Es el tipo de bug que no da error, solo métricas
incoherentes.

Y el filtro `not key.startswith("backbone.") or key in trainable_names` es lo que
hace que el `.pt` pese **0,76 MB** en vez de 435 MB: se guarda el cabezal completo
+ solo las capas del backbone que estaban descongeladas. El resto se reconstruye
desde Hugging Face.

---

## 9. Traza completa de un batch

```text
32 comentarios
  └─ tokenizer → input_ids      [32, 128]      int64
                 attention_mask [32, 128]      int64 (1 real / 0 pad)
       └─ backbone (congelado, eval)
            └─ last_hidden_state [32, 128, 768] float32
                 └─ mean pooling con máscara
                      └─ pooled     [32, 768]
                           ├─ LayerNorm(768)
                           ├─ Dropout(0.3)   ← solo en train
                           ├─ Linear(768→256)  196.864 params
                           ├─ GELU
                           ├─ Dropout(0.15)
                           └─ Linear(256→3)      771 params
                                └─ logits  [32, 3]
                                     ├─ CrossEntropyLoss(logits, y) → escalar
                                     └─ softmax (solo inferencia) → [32, 3] probabilidades
```

Si no se puede reproducir esta cadena de shapes de memoria, es el primer punto a
repasar antes del examen.

---

## 10. La tabla teoría → código, verificada

| Fila de la presentación | Veredicto | Matiz real |
|---|---|---|
| Checkpoints / `from_pretrained` | ✅ | `model_architecture.py:78`. Correcto |
| `--freeze_backbone True` | ⚠️ **no existe ese flag** | Es argumento del constructor (`freeze_backbone=True`, fijado en `train.py:537`). El CLI expone `--unfreeze-layers N` (`train.py:461-466`) |
| "congela el 99 %" | ✅ aprox. | Exacto: **99,82 %** congelado, 0,18 % entrenable |
| LR diferencial `1e-3 / 2e-5` | ⚠️ **latente por defecto** | Solo hay grupo `backbone` si se usa `--unfreeze-layers`. En feature extraction el optimizador recibe un único grupo |
| Pooling `mean/cls/max` | ✅ | `apply_pooling`, con máscara y `masked_fill` out-of-place |
| `LayerNorm(768)` | ✅ | `model_architecture.py:87` |
| `Dropout(0.3)` | ✅ | Y un segundo `Dropout(0.15)` que la tabla omite |
| `Linear(768→256)+GELU+Linear(256→3)` | ✅ | `model_architecture.py:89-92` |
| `CrossEntropyLoss` sin softmax | ✅ | `train.py:591`; el softmax está en `app.py:283` |
| AdamW / RMSprop / SGD+Momentum | ✅ | `train.py:168-173`; SGD con `momentum=0.9` |
| Bucle de 5 pasos | ✅ | `train.py:199-207`, más `clip_grad_norm_` entre 3 y 4 |
| Domain shift | ✅ | Se imprime al final del entrenamiento (`train.py:809-813`) y está en `data/README.md` |

**Corrección de documentación ya aplicada al README:** decía que `python train.py`
"equivale a" `--epochs 6 --batch-size 32`, pero los defaults reales del CLI son
`--epochs 3 --batch-size 16` (`train.py:450-451`). Ahora distingue "valores por
defecto" de "configuración del checkpoint de referencia". También se unificaron
435 MB / 0,76 MB.

---

## 11. Los tres detalles que la presentación no cubre

Son los que más lucen en una defensa, porque demuestran que se entendió el *porqué*
y no solo el *qué*:

1. **`train()` con backbone en `eval`** → congelar pesos no desactiva el Dropout
   interno. Sin esto, el extractor no es determinista.
2. **`.clone()` en el checkpoint** → `.detach().cpu()` comparte memoria en CPU;
   sin clonar, el "mejor modelo" se contamina con épocas posteriores.
3. **Guardar solo parámetros entrenables** → el `.pt` pasa de 435 MB a 0,76 MB y
   sigue siendo reconstruible porque la config (`model_name`, `hidden_dim`,
   `pooling`) viaja dentro.

---

## 12. Dónde se ve que no es una caja negra

La parte que cierra el círculo pedagógico está en `train.py:217-249` (`evaluate`) y
en las figuras que genera:

```python
macro_f1 = f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2], zero_division=0)
```

El resultado medido — **F1 test 0,668 / accuracy 0,667**, con el neutro como clase
peor (F1 0,585) y la confusión dominante negativo→neutro (83 casos) — no se explica
por la librería, se explica por el modelo: la frontera entre "no me gustó" y "me da
igual" es difusa incluso para un humano. Eso es exactamente el tipo de conclusión
que pide la rúbrica ("en términos del modelo, no de la librería").

Y el `domain shift` es la otra mitad: se entrena con tuits y se predice sobre
YouTube. Parte del error en producción viene de ahí, no de la arquitectura — y el
propio script lo recuerda al terminar.
