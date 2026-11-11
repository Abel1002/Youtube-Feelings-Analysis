# `data/` — Corpus de entrenamiento

## Archivos

| Archivo | Contenido |
|---|---|
| `sentiment_es.csv` | Corpus normalizado con columnas `texto, etiqueta, split, fuente` |
| `dataset_card.json` | Ficha metodológica: origen, mapeo de etiquetas, limpieza y conteos |

## Contrato de columnas

| Columna | Valores | Significado |
|---|---|---|
| `texto` | string | Comentario/reseña limpia |
| `etiqueta` | `0`, `1`, `2` | `0=Negativo`, `1=Neutro`, `2=Positivo` |
| `split` | `train`, `validation`, `test` | Partición usada por `train.py` |
| `fuente` | string | Dataset y split de origen (trazabilidad) |

## Regenerar

```bash
python prepare_data.py
```

## Usar un CSV propio

Cualquier CSV con columnas de texto y etiqueta funciona:

```bash
python train.py --data mi_corpus.csv --text-col comentario --label-col sentimiento
```

Las etiquetas pueden ser numéricas (`0/1/2`) o textuales
(`negative/neutral/positive`, `negativo/neutro/positivo`).

> **Importante:** no uses los comentarios descargados de YouTube para entrenar
> sin etiquetarlos manualmente. La app los analiza, pero no los etiqueta.
