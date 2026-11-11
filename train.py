"""
Ciclo de entrenamiento del cabezal neuronal.

Implementa el bucle de 5 pasos de la teoría:
    0. optimizer.zero_grad()  -> reinicio
    1. forward                -> inferencia
    2. criterion(logits, y)   -> evaluación (CrossEntropy)
    3. loss.backward()        -> derivadas (autograd)
    4. optimizer.step()       -> corrección de W y b

Incluye:
  - Dataset de PyTorch con tokenización,
  - Learning Rate Diferencial (backbone / cabezal),
  - AdamW, RMSprop y SGD + Momentum,
  - gradient clipping, early stopping y F1 macro como criterio de checkpoint,
  - curvas de pérdida y matriz de confusión con Matplotlib.

Uso rápido:
    python train.py
    python train.py --optimizer SGD_Momentum --epochs 3
    python train.py --unfreeze-layers 2 --lr-backbone 2e-5
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.optim import SGD, AdamW, RMSprop
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from config import (
    DEFAULT_DATA,
    DEFAULT_DROPOUT,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_MAX_LEN,
    DEFAULT_MODEL,
    DEFAULT_POOLING,
    ID2LABEL,
    LABEL_NAMES,
    resolve_path,
)
from model_architecture import (
    YouTubeSentimentClassifier,
    build_checkpoint,
    load_checkpoint_into_model,
    save_checkpoint,
)
from visualization import (
    confusion_matrix_figure,
    save_figure,
    training_curves_figure,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Reproducibilidad
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class CommentsDataset(Dataset):
    """Tokeniza cada comentario y devuelve input_ids, attention_mask y label."""

    def __init__(
        self,
        texts: list[str] | pd.Series,
        labels: list[int] | pd.Series,
        tokenizer,
        max_len: int = DEFAULT_MAX_LEN,
    ) -> None:
        self.texts = [str(t) for t in texts]
        self.labels = [int(label) for label in labels]
        self.tokenizer = tokenizer
        self.max_len = int(max_len)
        if len(self.texts) != len(self.labels):
            raise ValueError("texts y labels deben tener la misma longitud.")

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Optimizador con Learning Rate Diferencial
# ---------------------------------------------------------------------------
def build_optimizer(
    model: YouTubeSentimentClassifier,
    optimizer_name: str = "AdamW",
    lr_head: float = 1e-3,
    lr_backbone: float = 2e-5,
    weight_decay: float | None = None,
):
    """
    Dos grupos de parámetros:
      - backbone: zancada diminuta (2e-5) para no destruir el preentrenamiento.
      - cabezal:  zancada normal (1e-3) para aprender rápido la tarea nueva.
    """
    backbone_params = [
        p for n, p in model.named_parameters()
        if n.startswith("backbone.") and p.requires_grad
    ]
    head_params = [
        p for n, p in model.named_parameters()
        if not n.startswith("backbone.") and p.requires_grad
    ]

    if not head_params:
        raise ValueError("El modelo no tiene parámetros entrenables en el cabezal.")

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": lr_backbone, "name": "backbone"})
    param_groups.append({"params": head_params, "lr": lr_head, "name": "head"})

    if weight_decay is None:
        weight_decay = 0.01 if optimizer_name == "AdamW" else 1e-4

    if optimizer_name == "AdamW":
        return AdamW(param_groups, weight_decay=weight_decay)
    if optimizer_name == "RMSprop":
        return RMSprop(param_groups, weight_decay=weight_decay)
    if optimizer_name == "SGD_Momentum":
        return SGD(param_groups, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Optimizador no soportado: {optimizer_name}")


# ---------------------------------------------------------------------------
# Bucle de entrenamiento y evaluación
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: YouTubeSentimentClassifier,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
) -> tuple[float, float]:
    if len(dataloader) == 0:
        raise ValueError("El DataLoader de entrenamiento está vacío.")

    model.train()  # El override mantiene el backbone congelado en eval.
    total_loss, correct, total = 0.0, 0, 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)          # Paso 0: reinicio

        logits = model(input_ids=input_ids, attention_mask=attention_mask)  # Paso 1
        loss = criterion(logits, labels)               # Paso 2

        loss.backward()                                # Paso 3
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()                               # Paso 4

        total_loss += float(loss.item())
        preds = torch.argmax(logits, dim=1)
        correct += int((preds == labels).sum().item())
        total += int(labels.size(0))

    return total_loss / len(dataloader), correct / max(total, 1)


@torch.no_grad()
def evaluate(
    model: YouTubeSentimentClassifier,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float, list[int], list[int]]:
    if len(dataloader) == 0:
        return 0.0, 0.0, 0.0, [], []

    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    y_true: list[int] = []
    y_pred: list[int] = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)

        total_loss += float(loss.item())
        preds = torch.argmax(logits, dim=1)
        correct += int((preds == labels).sum().item())
        total += int(labels.size(0))
        y_true.extend(int(v) for v in labels.cpu().tolist())
        y_pred.extend(int(v) for v in preds.cpu().tolist())

    accuracy = correct / max(total, 1)
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2], zero_division=0)
    return total_loss / len(dataloader), accuracy, float(macro_f1), y_true, y_pred


# ---------------------------------------------------------------------------
# Carga y partición de datos
# ---------------------------------------------------------------------------
def _read_csv_robusto(path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as error:
            last_error = error
    raise ValueError(f"No se pudo leer {path} con ninguna codificación habitual: {last_error}")


def _coerce_labels(series: pd.Series) -> pd.Series:
    """Acepta 0/1/2 o etiquetas textuales (negative/neutral/positive)."""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return numeric.astype(int)

    aliases = {
        "negative": 0, "negativo": 0, "neg": 0, "n": 0,
        "neutral": 1, "neutro": 1, "neu": 1,
        "positive": 2, "positivo": 2, "pos": 2, "p": 2,
    }
    mapped = series.astype(str).str.lower().str.strip().map(aliases)
    # Permite CSVs mixtos (algunas filas numéricas y otras textuales).
    return mapped.fillna(numeric)


def load_dataframe(path: str | Path, text_col: str, label_col: str) -> pd.DataFrame:
    path = resolve_path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No existe el CSV de datos: {path}\n"
            "Genera uno con `python prepare_data.py` o pasa --data."
        )

    frame = _read_csv_robusto(path)
    if text_col not in frame.columns:
        for alternative in ("texto", "text", "comentario"):
            if alternative in frame.columns:
                text_col = alternative
                break
    if label_col not in frame.columns:
        for alternative in ("etiqueta", "label", "sentiment"):
            if alternative in frame.columns:
                label_col = alternative
                break

    if text_col not in frame.columns or label_col not in frame.columns:
        raise ValueError(
            f"El CSV debe tener columnas de texto y etiqueta. "
            f"Encontradas: {list(frame.columns)}. "
            f"Esperadas: {text_col!r} y {label_col!r}."
        )

    frame = frame.dropna(subset=[text_col, label_col]).copy()
    frame[text_col] = frame[text_col].astype(str).str.strip()
    frame = frame[frame[text_col] != ""]
    frame[label_col] = _coerce_labels(frame[label_col])
    frame = frame.dropna(subset=[label_col])
    frame[label_col] = frame[label_col].astype(int)

    invalid = sorted(set(frame[label_col].unique()) - {0, 1, 2})
    if invalid:
        raise ValueError(
            f"Etiquetas inválidas: {invalid}. El contrato del proyecto es "
            "0=Negativo, 1=Neutro, 2=Positivo."
        )

    frame = frame.drop_duplicates(subset=[text_col]).reset_index(drop=True)
    return frame


def _safe_train_test_split(
    frame: pd.DataFrame, label_col: str, test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = frame[label_col].value_counts()
    stratify = frame[label_col] if len(counts) > 1 and counts.min() >= 2 else None
    if stratify is None:
        print("⚠️  No se puede estratificar (alguna clase tiene <2 ejemplos).")
    return train_test_split(
        frame, test_size=test_size, random_state=seed, stratify=stratify
    )


def split_dataframe(
    frame: pd.DataFrame,
    label_col: str,
    seed: int,
    val_ratio: float = 0.2,
    val_frame: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Devuelve (train, validation, test). Respeta la columna `split` si existe."""
    empty = frame.iloc[0:0].copy()

    if val_frame is not None:
        train_df = frame
        if "split" in frame.columns:
            split_values = frame["split"].astype(str).str.lower().str.strip()
            train_df = frame[split_values == "train"]
            test_df = frame[split_values == "test"]
        else:
            test_df = empty
        return train_df.reset_index(drop=True), val_frame.reset_index(drop=True), test_df.reset_index(drop=True)

    if "split" not in frame.columns:
        train_df, val_df = _safe_train_test_split(frame, label_col, val_ratio, seed)
        return (
            train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            empty.reset_index(drop=True),
        )

    split_values = frame["split"].astype(str).str.lower().str.strip()
    train_df = frame[split_values.isin({"train", "training"})]
    val_df = frame[split_values.isin({"validation", "valid", "val", "dev"})]
    test_df = frame[split_values.isin({"test", "testing"})]

    if train_df.empty:
        raise ValueError("El split de entrenamiento está vacío. Revisa la columna 'split'.")

    if val_df.empty:
        if len(train_df) > 20:
            print("ℹ️  No hay split de validación: se separa un 20 % del train.")
            train_df, val_df = _safe_train_test_split(train_df, label_col, val_ratio, seed)
        else:
            val_df = train_df.copy()
            print("⚠️  Dataset muy pequeño: se usa train como validación.")

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def limit_training_data(frame: pd.DataFrame, label_col: str, limit: int | None, seed: int) -> pd.DataFrame:
    if not limit or limit >= len(frame):
        return frame
    if limit < 2:
        raise ValueError("--limit debe ser al menos 2.")

    counts = frame[label_col].value_counts()
    stratify = frame[label_col] if len(counts) > 1 and counts.min() >= 2 else None
    try:
        subset, _ = train_test_split(
            frame, train_size=limit, random_state=seed, stratify=stratify
        )
    except ValueError:
        subset = frame.sample(n=limit, random_state=seed)
    return subset.reset_index(drop=True)


def _distribution(frame: pd.DataFrame, label_col: str) -> dict[str, int]:
    counts = frame[label_col].value_counts().reindex([0, 1, 2], fill_value=0)
    return {ID2LABEL[int(label)]: int(value) for label, value in counts.items()}


# ---------------------------------------------------------------------------
# JSON y reportes
# ---------------------------------------------------------------------------
def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entrena el cabezal de sentimiento sobre un corpus español."
    )
    parser.add_argument("--data", default=DEFAULT_DATA, help="CSV con texto y etiqueta.")
    parser.add_argument("--val-data", default=None, help="CSV opcional solo para validación.")
    parser.add_argument("--text-col", default="texto")
    parser.add_argument("--label-col", default="etiqueta")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Checkpoint del backbone.")
    parser.add_argument("--pooling", default=DEFAULT_POOLING, choices=["cls", "mean", "max"])
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-backbone", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument(
        "--optimizer",
        default="AdamW",
        choices=["AdamW", "RMSprop", "SGD_Momentum"],
    )
    parser.add_argument(
        "--unfreeze-layers",
        type=int,
        default=0,
        help="Descongela las últimas N capas del Transformer (0 = solo cabezal).",
    )
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Submuestra el train (pruebas rápidas).")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--class-weights", action="store_true", help="Pondera la loss por frecuencia inversa.")
    parser.add_argument("--output", default="checkpoints/best_sentiment_head.pt")
    parser.add_argument("--figures-dir", default="checkpoints")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs debe ser >= 1")
    if args.batch_size < 1:
        parser.error("--batch-size debe ser >= 1")
    if args.max_len < 8:
        parser.error("--max-len debe ser >= 8")
    if args.patience < 0:
        parser.error("--patience debe ser >= 0")
    if not 0.0 <= args.dropout < 1.0:
        parser.error("--dropout debe estar en [0, 1)")
    if args.unfreeze_layers < 0:
        parser.error("--unfreeze-layers debe ser >= 0")

    set_seed(args.seed)
    device = pick_device(args.device)
    output_path = resolve_path(args.output)
    figures_dir = resolve_path(args.figures_dir)

    print("=" * 78)
    print("🎓 ANALIZADOR NEURONAL DE SENTIMIENTO · ENTRENAMIENTO")
    print("=" * 78)
    print(f"Dispositivo: {device}")
    print(f"Backbone:    {args.model}")
    print(f"Pooling:     {args.pooling} | Dropout: {args.dropout} | hidden_dim: {args.hidden_dim}")
    print(f"Optimizador: {args.optimizer} | lr_head: {args.lr_head} | lr_backbone: {args.lr_backbone}")
    print(f"Semilla:     {args.seed}")

    # --- Datos ---------------------------------------------------------
    frame = load_dataframe(args.data, args.text_col, args.label_col)
    val_frame = (
        load_dataframe(args.val_data, args.text_col, args.label_col)
        if args.val_data
        else None
    )
    train_df, val_df, test_df = split_dataframe(
        frame, args.label_col, args.seed, val_frame=val_frame
    )
    train_df = limit_training_data(train_df, args.label_col, args.limit, args.seed)

    print("\n📊 Distribución de datos")
    print(f"   Train:      {len(train_df):>5}  {_distribution(train_df, args.label_col)}")
    print(f"   Validación: {len(val_df):>5}  {_distribution(val_df, args.label_col)}")
    if len(test_df):
        print(f"   Test:       {len(test_df):>5}  {_distribution(test_df, args.label_col)}")
    else:
        print("   Test:       (sin split de test)")

    if len(train_df) < 10:
        raise ValueError("Muy pocos ejemplos de entrenamiento (mínimo 10).")

    # --- Tokenizador y modelo -----------------------------------------
    print(f"\n⏳ Cargando tokenizador y backbone ({args.model})...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = YouTubeSentimentClassifier(
        model_name=args.model,
        num_classes=3,
        pooling_type=args.pooling,
        dropout_rate=args.dropout,
        hidden_dim=args.hidden_dim,
        freeze_backbone=True,
    )

    if args.unfreeze_layers > 0:
        unfrozen = model.unfreeze_last_layers(args.unfreeze_layers)
        print(f"🔓 Descongeladas las últimas {args.unfreeze_layers} capas ({unfrozen:,} parámetros).")

    model.to(device)
    trainable, total = model.count_parameters()
    print(f"🧠 Parámetros entrenables: {trainable:,} / {total:,} ({100 * trainable / total:.2f} %)")

    # --- DataLoaders ---------------------------------------------------
    train_dataset = CommentsDataset(
        train_df[args.text_col].tolist(), train_df[args.label_col].tolist(), tokenizer, args.max_len
    )
    val_dataset = CommentsDataset(
        val_df[args.text_col].tolist(), val_df[args.label_col].tolist(), tokenizer, args.max_len
    )
    test_dataset = (
        CommentsDataset(
            test_df[args.text_col].tolist(), test_df[args.label_col].tolist(), tokenizer, args.max_len
        )
        if len(test_df)
        else None
    )

    loader_kwargs = {
        "num_workers": max(0, args.num_workers),
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, **loader_kwargs
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.eval_batch_size, shuffle=False, **loader_kwargs
    )
    test_loader = (
        DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False, **loader_kwargs)
        if test_dataset is not None
        else None
    )

    # --- Loss, optimizador y checkpoint --------------------------------
    class_weights = None
    if args.class_weights:
        counts = train_df[args.label_col].value_counts().reindex([0, 1, 2], fill_value=0)
        class_weights = len(train_df) / (3.0 * counts.clip(lower=1))
        print(f"⚖️  Pesos de clase: {class_weights.round(3).to_dict()}")

    weight_tensor = (
        torch.tensor(class_weights.values, dtype=torch.float32, device=device)
        if class_weights is not None
        else None
    )
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = build_optimizer(
        model,
        optimizer_name=args.optimizer,
        lr_head=args.lr_head,
        lr_backbone=args.lr_backbone,
        weight_decay=args.weight_decay,
    )

    config = model.config_dict(
        max_len=args.max_len,
        unfrozen_layers=args.unfreeze_layers,
    )
    training_meta = {
        "dataset": str(resolve_path(args.data)),
        "val_dataset": str(resolve_path(args.val_data)) if args.val_data else None,
        "n_train": int(len(train_df)),
        "n_validation": int(len(val_df)),
        "n_test": int(len(test_df)),
        "epochs_requested": int(args.epochs),
        "optimizer": args.optimizer,
        "lr_head": float(args.lr_head),
        "lr_backbone": float(args.lr_backbone),
        "weight_decay": float(optimizer.defaults.get("weight_decay", 0.0)),
        "batch_size": int(args.batch_size),
        "grad_clip": float(args.grad_clip),
        "seed": int(args.seed),
        "device": str(device),
        "class_weights": (
            {str(label): float(value) for label, value in class_weights.round(4).items()}
            if class_weights is not None
            else None
        ),
        "train_distribution": _distribution(train_df, args.label_col),
        "validation_distribution": _distribution(val_df, args.label_col),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    # --- Entrenamiento ------------------------------------------------
    print("\n🚀 Iniciando entrenamiento\n" + "-" * 78)
    history: list[dict[str, Any]] = []
    best_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    best_checkpoint: dict[str, Any] | None = None
    training_start = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, grad_clip=args.grad_clip
        )
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - epoch_start

        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 6),
            "val_f1": round(val_f1, 6),
            "seconds": round(elapsed, 2),
        }
        history.append(record)

        print(
            f"Época {epoch:02d}/{args.epochs:02d} | "
            f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.3f} F1 {val_f1:.3f} | "
            f"{elapsed:.1f}s"
        )

        if val_f1 > best_f1 + 1e-6:
            best_f1 = float(val_f1)
            best_epoch = epoch
            patience_counter = 0
            best_checkpoint = build_checkpoint(
                model=model,
                config=config,
                metrics={
                    "best_val_f1_macro": float(val_f1),
                    "best_val_accuracy": float(val_acc),
                    "best_val_loss": float(val_loss),
                    "best_epoch": int(epoch),
                },
                history=list(history),
                training=training_meta,
            )
            save_checkpoint(output_path, best_checkpoint)
            print(f"   ↳ 💾 Checkpoint guardado (F1 macro = {val_f1:.3f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"⏹️  Early stopping: la validación no mejora desde hace {args.patience} épocas.")
                break

    if best_checkpoint is None:
        best_checkpoint = build_checkpoint(
            model=model,
            config=config,
            metrics={
                "best_val_f1_macro": float(best_f1),
                "best_val_accuracy": 0.0,
                "best_val_loss": 0.0,
                "best_epoch": best_epoch,
            },
            history=list(history),
            training=training_meta,
        )
        save_checkpoint(output_path, best_checkpoint)

    training_meta["epochs_run"] = len(history)
    training_meta["stopped_early"] = len(history) < args.epochs
    total_time = time.time() - training_start
    print("-" * 78)
    print(f"✅ Entrenamiento finalizado en {total_time:.1f}s | mejor época: {best_epoch} | F1 val: {best_f1:.4f}")

    # --- Evaluación final con los mejores pesos -----------------------
    print("\n🔎 Cargando el mejor checkpoint para la evaluación final...")
    load_checkpoint_into_model(model, best_checkpoint)
    model.to(device)

    val_loss, val_acc, val_f1, y_val_true, y_val_pred = evaluate(
        model, val_loader, criterion, device
    )
    val_report = classification_report(
        y_val_true, y_val_pred, labels=[0, 1, 2], target_names=LABEL_NAMES, digits=3, zero_division=0
    )
    cm_val = confusion_matrix(y_val_true, y_val_pred, labels=[0, 1, 2])

    print("\n📋 Validación")
    print(val_report)

    test_metrics: dict[str, Any] = {}
    test_report = ""
    cm_test = None
    if test_loader is not None:
        test_loss, test_acc, test_f1, y_test_true, y_test_pred = evaluate(
            model, test_loader, criterion, device
        )
        test_report = classification_report(
            y_test_true, y_test_pred, labels=[0, 1, 2], target_names=LABEL_NAMES, digits=3, zero_division=0
        )
        cm_test = confusion_matrix(y_test_true, y_test_pred, labels=[0, 1, 2])
        test_metrics = {
            "test_loss": float(test_loss),
            "test_accuracy": float(test_acc),
            "test_f1_macro": float(test_f1),
        }
        print("\n📋 Test")
        print(test_report)

    # --- Guardar métricas, historial y figuras -----------------------
    best_checkpoint["metrics"].update(test_metrics)
    best_checkpoint["metrics"].update(
        {
            "final_val_f1_macro": float(val_f1),
            "final_val_accuracy": float(val_acc),
            "final_val_loss": float(val_loss),
            "total_training_seconds": round(total_time, 2),
        }
    )
    best_checkpoint["history"] = history
    best_checkpoint["training"] = training_meta
    save_checkpoint(output_path, best_checkpoint)

    figures_dir.mkdir(parents=True, exist_ok=True)
    curves_path = figures_dir / "training_curves.png"
    fig = training_curves_figure(history)
    save_figure(fig, curves_path)
    import matplotlib.pyplot as plt
    plt.close(fig)

    cm_val_path = figures_dir / "confusion_matrix_val.png"
    fig = confusion_matrix_figure(cm_val, labels=LABEL_NAMES)
    save_figure(fig, cm_val_path)
    plt.close(fig)

    cm_test_path = None
    if cm_test is not None:
        cm_test_path = figures_dir / "confusion_matrix_test.png"
        fig = confusion_matrix_figure(cm_test, labels=LABEL_NAMES)
        save_figure(fig, cm_test_path)
        plt.close(fig)

    # El informe se guarda junto al checkpoint para que la app lo encuentre
    # aunque las figuras se redirijan a otra carpeta con --figures-dir.
    report_path = output_path.parent / "training_report.json"
    _save_json(
        report_path,
        {
            "config": config,
            "training": training_meta,
            "history": history,
            "metrics": best_checkpoint["metrics"],
            "validation_report": val_report,
            "test_report": test_report,
            "confusion_matrix_validation": cm_val.tolist(),
            "confusion_matrix_test": cm_test.tolist() if cm_test is not None else None,
            "artifacts": {
                "checkpoint": str(output_path),
                "training_curves": str(curves_path),
                "confusion_matrix_validation": str(cm_val_path),
                "confusion_matrix_test": str(cm_test_path) if cm_test_path else None,
            },
        },
    )

    print("\n" + "=" * 78)
    print("📦 Artefactos generados")
    print(f"   Checkpoint:          {output_path}")
    print(f"   Curvas:              {curves_path}")
    print(f"   Matriz validación:   {cm_val_path}")
    if cm_test_path:
        print(f"   Matriz test:         {cm_test_path}")
    print(f"   Informe JSON:        {report_path}")
    print("=" * 78)
    print(
        "Siguiente paso: `streamlit run app.py` y usa el checkpoint en el panel lateral.\n"
        "Recuerda: el modelo se entrenó con reseñas/tuits; los comentarios de YouTube "
        "son un dominio distinto (domain shift)."
    )


if __name__ == "__main__":
    main()
