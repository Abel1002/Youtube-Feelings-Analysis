"""
Preparación del corpus etiquetado de entrenamiento.

La app analiza comentarios de YouTube, pero esos comentarios NO tienen etiqueta:
no sirven para entrenar. Este script descarga un corpus público de sentimiento
en español y lo normaliza a un CSV con el contrato del proyecto:

    texto, etiqueta, split, fuente

donde etiqueta ∈ {0: Negativo, 1: Neutro, 2: Positivo}.

Presets incluidos
-----------------
tweets : cardiffnlp/tweet_sentiment_multilingual (config "spanish")
         Tuits reales en español, 3 clases balanceadas.
tass   : mrm8488/tass-2019
         Tuits de la campaña TASS 2019; se descarta la clase NONE.

Uso
---
    python prepare_data.py
    python prepare_data.py --preset tass --output data/sentiment_es_tass.csv
    python prepare_data.py --max-per-class 300
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    from datasets import ClassLabel, load_dataset
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Falta la librería `datasets`. Instala las dependencias con:\n"
        "    pip install -r requirements.txt"
    ) from exc

from config import ID2LABEL, resolve_path

LABEL_ALIASES = {
    "negative": 0,
    "negativo": 0,
    "neg": 0,
    "n": 0,
    "neutral": 1,
    "neutro": 1,
    "neu": 1,
    "positive": 2,
    "positivo": 2,
    "pos": 2,
    "p": 2,
}

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MENTION_RE = re.compile(r"(?<!\w)@\w+")
_SPACES_RE = re.compile(r"\s+")


def clean_text(text: object) -> str:
    """
    Limpieza mínima y no destructiva:
      - elimina URLs y menciones @usuario (ruido específico de Twitter),
      - conserva emojis, signos y acentos (aportan señal de sentimiento),
      - colapsa espacios.
    """
    value = str(text or "")
    value = _URL_RE.sub(" ", value)
    value = _MENTION_RE.sub(" ", value)
    value = _SPACES_RE.sub(" ", value)
    return value.strip()


def _split_name(name: str) -> str:
    name = str(name).lower()
    if name in {"train", "training"}:
        return "train"
    if name in {"validation", "valid", "val", "dev"}:
        return "validation"
    if name in {"test", "testing"}:
        return "test"
    return name


def _label_mapping_from_features(feature: object) -> dict[int, int]:
    """Construye el mapeo id_crudo -> id_proyecto a partir de un ClassLabel."""
    if not isinstance(feature, ClassLabel):
        return {}
    mapping: dict[int, int] = {}
    for index, name in enumerate(feature.names):
        alias = LABEL_ALIASES.get(str(name).lower().strip())
        if alias is not None:
            mapping[index] = alias
    return mapping


def _dataset_to_frame(
    dataset,
    text_col: str,
    label_col: str,
    raw_to_project: dict,
    preset: str,
    split: str,
) -> pd.DataFrame:
    frame = dataset.to_pandas()
    if text_col not in frame.columns:
        raise ValueError(
            f"El dataset no contiene la columna de texto {text_col!r}. "
            f"Columnas disponibles: {list(frame.columns)}"
        )
    if label_col not in frame.columns:
        raise ValueError(
            f"El dataset no contiene la columna de etiqueta {label_col!r}. "
            f"Columnas disponibles: {list(frame.columns)}"
        )

    result = pd.DataFrame(
        {
            "texto": frame[text_col].map(clean_text),
            "etiqueta": frame[label_col].map(raw_to_project),
        }
    )
    result = result.dropna(subset=["etiqueta"])
    result["etiqueta"] = result["etiqueta"].astype(int)
    result = result[result["texto"].str.len() > 0]
    result["split"] = split
    result["fuente"] = f"{preset}:{split}"
    return result


def load_tweets() -> tuple[pd.DataFrame, dict]:
    """Corpus principal: tuits españoles con 3 clases balanceadas."""
    ds = load_dataset("cardiffnlp/tweet_sentiment_multilingual", "spanish")
    frames: list[pd.DataFrame] = []

    for raw_split, dataset in ds.items():
        label_feature = dataset.features.get("label")
        raw_to_project = _label_mapping_from_features(label_feature)
        if not raw_to_project:
            # Fallback documentado: 0=negative, 1=neutral, 2=positive.
            raw_to_project = {0: 0, 1: 1, 2: 2}

        frames.append(
            _dataset_to_frame(
                dataset=dataset,
                text_col="text",
                label_col="label",
                raw_to_project=raw_to_project,
                preset="tweets",
                split=_split_name(raw_split),
            )
        )

    metadata = {
        "dataset": "cardiffnlp/tweet_sentiment_multilingual",
        "config": "spanish",
        "text_column": "text",
        "label_column": "label",
        "label_mapping": "negative=0, neutral=1, positive=2",
        "license": "Consulta la ficha del dataset en Hugging Face.",
        "notes": (
            "Tuits reales en español con jerga y ruido. El cambio de dominio "
            "hacia comentarios de YouTube es una fuente de error en producción."
        ),
    }
    return pd.concat(frames, ignore_index=True), metadata


def load_tass() -> tuple[pd.DataFrame, dict]:
    """Alternativa: TASS 2019. Se descarta la clase NONE (sin sentimiento)."""
    ds = load_dataset("mrm8488/tass-2019")
    frames: list[pd.DataFrame] = []

    for raw_split, dataset in ds.items():
        frame = dataset.to_pandas()
        if "sentiments" in frame.columns:
            raw = frame["sentiments"].astype(str).str.upper().str.strip()
            project = raw.map({"N": 0, "NEU": 1, "P": 2})
        elif "labels" in frame.columns:
            # 0=N, 1=NEU, 2=P, 3=NONE
            project = frame["labels"].map({0: 0, 1: 1, 2: 2})
        else:
            raise ValueError("TASS no contiene 'sentiments' ni 'labels'.")

        text_col = "sentence" if "sentence" in frame.columns else "text"
        temp = pd.DataFrame({"texto": frame[text_col].map(clean_text), "etiqueta": project})
        temp = temp.dropna(subset=["etiqueta"])
        temp["etiqueta"] = temp["etiqueta"].astype(int)
        temp = temp[temp["texto"].str.len() > 0]
        temp["split"] = _split_name(raw_split)
        temp["fuente"] = f"tass:{_split_name(raw_split)}"
        frames.append(temp)

    combined = pd.concat(frames, ignore_index=True)

    # TASS solo trae train/test: separamos una validación estratificada del train.
    train = combined[combined["split"] == "train"]
    test = combined[combined["split"] == "test"]
    if len(train) > 20 and "validation" not in set(combined["split"]):
        from sklearn.model_selection import train_test_split

        train, validation = train_test_split(
            train, test_size=0.12, random_state=42, stratify=train["etiqueta"]
        )
        validation = validation.copy()
        validation["split"] = "validation"
        validation["fuente"] = "tass:validation"
        combined = pd.concat([train, validation, test], ignore_index=True)

    metadata = {
        "dataset": "mrm8488/tass-2019",
        "config": "default",
        "text_column": "sentence",
        "label_column": "sentiments",
        "label_mapping": "N=0 (Negativo), NEU=1 (Neutro), P=2 (Positivo); NONE descartado",
        "license": "Consulta la ficha del dataset en Hugging Face.",
        "notes": (
            "Tuits de la campaña TASS 2019. Algunos textos conservan caracteres "
            "corruptos del corpus original."
        ),
    }
    return combined, metadata


def _cap_per_class(df: pd.DataFrame, max_per_class: int, seed: int) -> pd.DataFrame:
    """Submuestreo estratificado por split y clase (útil para pruebas rápidas)."""
    parts: list[pd.DataFrame] = []
    for split, group in df.groupby("split", sort=False):
        for label, label_group in group.groupby("etiqueta", sort=True):
            if len(label_group) > max_per_class:
                label_group = label_group.sample(n=max_per_class, random_state=seed)
            parts.append(label_group)
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Descarga y normaliza un corpus español de sentimiento."
    )
    parser.add_argument(
        "--preset",
        choices=["tweets", "tass"],
        default="tweets",
        help="Corpus preconfigurado (por defecto: tweets).",
    )
    parser.add_argument(
        "--output",
        default="data/sentiment_es.csv",
        help="Ruta del CSV de salida.",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Máximo de ejemplos por clase y split (None = todos).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.preset == "tweets":
        df, metadata = load_tweets()
    else:
        df, metadata = load_tass()

    # Normalización final: sin duplicados exactos ni textos vacíos.
    df["texto"] = df["texto"].astype(str).str.strip()
    df = df[df["texto"] != ""]
    df = df.drop_duplicates(subset=["texto"]).reset_index(drop=True)

    if args.max_per_class:
        df = _cap_per_class(df, args.max_per_class, args.seed)

    df = df[["texto", "etiqueta", "split", "fuente"]]
    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")

    counts_raw = (
        df.groupby(["split", "etiqueta"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=[0, 1, 2], fill_value=0)
        .sort_index()
    )
    counts = counts_raw.rename(columns={0: "Negativo", 1: "Neutro", 2: "Positivo"})

    metadata.update(
        {
            "output": str(output_path),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "total_rows": int(len(df)),
            "counts": {
                split: {ID2LABEL[int(label)]: int(value) for label, value in row.items()}
                for split, row in counts_raw.iterrows()
            },
            "cleaning": (
                "Se eliminan URLs, menciones @usuario, espacios redundantes, "
                "textos vacíos y duplicados exactos. Se conservan emojis y acentos."
            ),
        }
    )
    card_path = output_path.with_name("dataset_card.json")
    card_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n✅ Corpus guardado en: {output_path}")
    print(f"   Ficha del dataset: {card_path}")
    print(f"   Total de ejemplos: {len(df):,}\n")
    print(counts.to_string())
    print("\nMapeo de etiquetas:")
    for label_id, label_name in ID2LABEL.items():
        print(f"   {label_id} -> {label_name}")


if __name__ == "__main__":
    main()
