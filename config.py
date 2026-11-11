"""
Configuración central del proyecto.

Aquí viven las constantes que comparten `app.py`, `train.py`, `prepare_data.py`
y el resto de módulos, además de la carga segura de la API Key desde `.env`.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Rutas del proyecto
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

# ---------------------------------------------------------------------------
# Credenciales
# ---------------------------------------------------------------------------
# `.env` nunca debe subirse al repositorio. Ver `.env.example`.
load_dotenv(PROJECT_ROOT / ".env")


def get_youtube_api_key() -> str:
    """Devuelve la API Key de YouTube desde el entorno o desde `.env`."""
    return os.getenv("YOUTUBE_API_KEY", "").strip()


# ---------------------------------------------------------------------------
# Etiquetas del problema
# ---------------------------------------------------------------------------
# 0: Negativo, 1: Neutro, 2: Positivo. Este orden es el contrato entre
# entrenamiento, checkpoint e inferencia: no cambiarlo sin reentrenar.
ID2LABEL: dict[int, str] = {0: "Negativo", 1: "Neutro", 2: "Positivo"}
LABEL2ID: dict[str, int] = {v: k for k, v in ID2LABEL.items()}
LABEL_NAMES: list[str] = [ID2LABEL[i] for i in range(3)]
SENTIMENT_COLORS: dict[str, str] = {
    "Positivo": "#2ecc71",
    "Neutro": "#95a5a6",
    "Negativo": "#e74c3c",
}

# ---------------------------------------------------------------------------
# Arquitectura y valores por defecto
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "pysentimiento/robertuito-base-uncased"
MODEL_OPTIONS: list[str] = [
    "pysentimiento/robertuito-base-uncased",
    "dccuchile/bert-base-spanish-wwm-cased",
    "distilbert-base-multilingual-cased",
]
DEFAULT_POOLING = "mean"
DEFAULT_DROPOUT = 0.3
DEFAULT_HIDDEN_DIM = 256
DEFAULT_MAX_LEN = 128
DEFAULT_BATCH_SIZE = 32
DEFAULT_CHECKPOINT = "checkpoints/best_sentiment_head.pt"
DEFAULT_DATA = "data/sentiment_es.csv"

# ---------------------------------------------------------------------------
# YouTube
# ---------------------------------------------------------------------------
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"
MAX_COMMENTS_LIMIT = 500


def resolve_path(path: str | Path) -> Path:
    """Resuelve una ruta relativa contra la raíz del proyecto."""
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()
