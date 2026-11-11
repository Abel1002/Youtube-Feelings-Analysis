"""
Arquitectura de la red neuronal y utilidades de checkpoint.

La idea pedagógica central: no entrenamos un Transformer desde cero. Descargamos
un *backbone* preentrenado (Transfer Learning) y le acoplamos un cabezal propio
construido explícitamente con capas de PyTorch.

Cadena de shapes:
    input_ids          [batch, 128]
    last_hidden_state  [batch, 128, 768]
    pooling            [batch, 768]
    fc1                [batch, 256]
    fc_out             [batch, 3]   <- logits (sin softmax)
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModel

CHECKPOINT_FORMAT_VERSION = 1


class YouTubeSentimentClassifier(nn.Module):
    """
    Backbone Transformer congelable + cabezal denso personalizado.

    Parámetros
    ----------
    model_name : str
        Checkpoint de Hugging Face que actúa como extractor de características.
    num_classes : int
        Número de clases de salida (3: Negativo, Neutro, Positivo).
    pooling_type : str
        'cls', 'mean' o 'max'. Cómo se resume la secuencia en un vector.
    dropout_rate : float
        Tasa del primer Dropout; el segundo usa la mitad.
    hidden_dim : int
        Neuronas de la capa oculta densa.
    freeze_backbone : bool
        True = Feature Extraction (congela el backbone).
        False = Fine-Tuning completo (no recomendado con pocos datos).
    """

    def __init__(
        self,
        model_name: str = "pysentimiento/robertuito-base-uncased",
        num_classes: int = 3,
        pooling_type: str = "mean",
        dropout_rate: float = 0.3,
        hidden_dim: int = 256,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()

        if pooling_type not in {"cls", "mean", "max"}:
            raise ValueError(
                f"pooling_type inválido: {pooling_type!r}. Usa 'cls', 'mean' o 'max'."
            )
        if num_classes < 2:
            raise ValueError("num_classes debe ser al menos 2.")
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError("dropout_rate debe estar en [0, 1).")

        self.model_name = model_name
        self.num_classes = num_classes
        self.pooling_type = pooling_type
        self.dropout_rate = float(dropout_rate)
        self.hidden_dim = int(hidden_dim)
        self.freeze_backbone = bool(freeze_backbone)

        # 1. Checkpoint preentrenado: el conocimiento lingüístico ya está aquí.
        self.backbone = AutoModel.from_pretrained(model_name)
        self.embedding_dim = int(self.backbone.config.hidden_size)

        # 2. Estrategia de Transfer Learning.
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # 3. Cabezal personalizado.
        self.layer_norm = nn.LayerNorm(self.embedding_dim)
        self.dropout1 = nn.Dropout(p=self.dropout_rate)
        self.fc1 = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.activation = nn.GELU()
        self.dropout2 = nn.Dropout(p=self.dropout_rate / 2)
        self.fc_out = nn.Linear(self.hidden_dim, num_classes)

    # ------------------------------------------------------------------
    # Transfer Learning progresivo
    # ------------------------------------------------------------------
    def unfreeze_last_layers(self, num_layers: int = 2) -> int:
        """
        Descongela solo los últimos `num_layers` bloques del Transformer.

        Devuelve el número de parámetros que pasaron a ser entrenables.
        Debe combinarse con Learning Rate Diferencial (lr_backbone pequeño).
        """
        num_layers = int(num_layers)
        if num_layers <= 0:
            return 0

        encoder = getattr(self.backbone, "encoder", None)
        if encoder is not None and hasattr(encoder, "layer"):
            layers = encoder.layer[-num_layers:]
        elif hasattr(self.backbone, "transformer") and hasattr(
            self.backbone.transformer, "layer"
        ):
            layers = self.backbone.transformer.layer[-num_layers:]
        else:
            raise AttributeError(
                "Arquitectura no soportada por unfreeze_last_layers(): "
                f"{type(self.backbone).__name__}"
            )

        unfrozen = 0
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = True
                unfrozen += param.numel()
        self.freeze_backbone = False
        return unfrozen

    # ------------------------------------------------------------------
    # Pooling
    # ------------------------------------------------------------------
    def apply_pooling(
        self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Comprime [batch, seq, 768] en [batch, 768] ignorando el padding.

        `masked_fill` es out-of-place: no corrompe el grafo de autograd ni el
        `last_hidden_state` compartido.
        """
        if self.pooling_type == "cls":
            return token_embeddings[:, 0, :]

        mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )

        if self.pooling_type == "max":
            min_value = torch.finfo(token_embeddings.dtype).min
            masked = token_embeddings.masked_fill(mask_expanded == 0, min_value)
            return torch.max(masked, dim=1).values

        sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state

        pooled = self.apply_pooling(token_embeddings, attention_mask)

        x = self.layer_norm(pooled)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout2(x)

        # Logits crudos: CrossEntropyLoss aplica log_softmax internamente.
        return self.fc_out(x)

    # ------------------------------------------------------------------
    # Modo train/eval coherente con el Transfer Learning
    # ------------------------------------------------------------------
    def train(self, mode: bool = True) -> "YouTubeSentimentClassifier":
        """
        Si el backbone está completamente congelado, lo mantenemos en `eval`
        incluso durante el entrenamiento del cabezal. Así desactivamos su
        Dropout interno y las estadísticas de LayerNorm no se contaminan:
        el extractor de características debe ser determinista.
        """
        super().train(mode)
        if mode and not any(p.requires_grad for p in self.backbone.parameters()):
            self.backbone.eval()
        return self

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------
    def count_parameters(self) -> tuple[int, int]:
        """Devuelve (entrenables, totales)."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total

    def config_dict(self, **extra: Any) -> dict[str, Any]:
        """Configuración serializable necesaria para reconstruir el modelo."""
        config = {
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "pooling_type": self.pooling_type,
            "dropout_rate": self.dropout_rate,
            "hidden_dim": self.hidden_dim,
            "freeze_backbone": self.freeze_backbone,
        }
        config.update(extra)
        return config


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
def build_checkpoint(
    model: YouTubeSentimentClassifier,
    config: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    history: list[dict[str, Any]] | None = None,
    training: dict[str, Any] | None = None,
    label_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Construye el diccionario del checkpoint.

    Optimización de tamaño: guardamos el cabezal completo y solo los parámetros
    del backbone que eran entrenables (p. ej. las últimas capas descongeladas).
    El backbone congelado se reconstruye desde Hugging Face usando `model_name`.
    """
    trainable_names = {name for name, p in model.named_parameters() if p.requires_grad}
    # `.clone()` es imprescindible: `detach().cpu()` comparte almacenamiento con
    # el parámetro vivo en CPU. Sin el clon, el checkpoint "en memoria" seguiría
    # cambiando mientras el entrenamiento continúa y guardaríamos pesos que no
    # son los de la mejor época.
    state_dict = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if not key.startswith("backbone.") or key in trainable_names
    }

    backbone_saved = {k for k in state_dict if k.startswith("backbone.")}
    backbone_param_names = {f"backbone.{name}" for name, _ in model.backbone.named_parameters()}
    if not backbone_saved:
        scope = "head_only"
    elif backbone_saved >= backbone_param_names:
        scope = "full_model"
    else:
        scope = "head_plus_selected_backbone"

    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "state_dict": state_dict,
        "config": config or model.config_dict(),
        "label_map": label_map or {"0": "Negativo", "1": "Neutro", "2": "Positivo"},
        "metrics": metrics or {},
        "history": history or [],
        "training": training or {},
        "parameter_scope": scope,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def save_checkpoint(path: str | Path, checkpoint: dict[str, Any]) -> Path:
    """Guarda un checkpoint creando el directorio si no existe."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    return path


def read_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """
    Lee un checkpoint con `weights_only=True` (no deserializa código arbitrario).

    Soporta también checkpoints "legacy" que son un `state_dict` plano.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No existe el checkpoint: {path}")

    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint

    if isinstance(checkpoint, dict) and all(
        isinstance(value, torch.Tensor) for value in checkpoint.values()
    ):
        return {
            "format_version": 0,
            "state_dict": checkpoint,
            "config": {},
            "label_map": {"0": "Negativo", "1": "Neutro", "2": "Positivo"},
            "metrics": {},
            "history": [],
            "training": {},
            "parameter_scope": "legacy_state_dict",
        }

    raise ValueError(
        "El archivo no parece un checkpoint válido: falta la clave 'state_dict'."
    )


def load_checkpoint_into_model(
    model: YouTubeSentimentClassifier,
    checkpoint: dict[str, Any],
    strict_backbone: bool = False,
) -> dict[str, Any]:
    """
    Carga los pesos del checkpoint en el modelo.

    - Siempre exige que estén todas las claves del cabezal.
    - Si el checkpoint contiene pesos del backbone, intenta cargarlos.
    - Si el backbone del checkpoint es de otra arquitectura, carga solo el
      cabezal y lo reporta (no falla silenciosamente).

    Devuelve un informe con lo cargado y lo ignorado.
    """
    state_dict = checkpoint.get("state_dict", {})
    model_state = model.state_dict()

    head_keys = [k for k in model_state if not k.startswith("backbone.")]
    missing_head = [k for k in head_keys if k not in state_dict]
    if missing_head:
        raise RuntimeError(
            "El checkpoint no contiene todas las capas del cabezal. "
            f"Faltan: {missing_head[:5]}{'...' if len(missing_head) > 5 else ''}"
        )

    # Detectar incompatibilidades de shape antes de llamar a PyTorch.
    shape_mismatch = [
        k
        for k, v in state_dict.items()
        if k in model_state and tuple(v.shape) != tuple(model_state[k].shape)
    ]
    if shape_mismatch:
        raise RuntimeError(
            "El checkpoint no encaja con la arquitectura seleccionada. "
            f"Dimensiones incompatibles en: {shape_mismatch[:5]}"
        )

    backbone_keys = [k for k in state_dict if k.startswith("backbone.")]
    compatible_backbone = [k for k in backbone_keys if k in model_state]
    incompatible_backbone = [k for k in backbone_keys if k not in model_state]

    if strict_backbone and incompatible_backbone:
        raise RuntimeError(
            "El checkpoint contiene pesos de un backbone incompatible. "
            "Selecciona el mismo modelo base con el que se entrenó."
        )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    head_missing = [k for k in missing if not k.startswith("backbone.")]
    if head_missing:
        raise RuntimeError(f"No se pudieron cargar capas del cabezal: {head_missing[:5]}")

    if compatible_backbone:
        mode = "full" if not missing and not unexpected else "partial"
    else:
        mode = "head_only"

    return {
        "mode": mode,
        "loaded_head": True,
        "loaded_backbone_keys": len(compatible_backbone),
        "ignored_backbone_keys": len(incompatible_backbone),
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "message": (
            "Se cargó el cabezal entrenado; el backbone seleccionado no coincide "
            "con el del checkpoint, así que se usan sus pesos preentrenados."
            if incompatible_backbone
            else "Checkpoint cargado correctamente."
        ),
    }


def checkpoint_summary(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Resumen legible para mostrar en la interfaz."""
    config = checkpoint.get("config", {}) or {}
    metrics = checkpoint.get("metrics", {}) or {}
    training = checkpoint.get("training", {}) or {}
    return {
        "model_name": config.get("model_name", "(desconocido)"),
        "pooling_type": config.get("pooling_type", "(desconocido)"),
        "dropout_rate": config.get("dropout_rate"),
        "hidden_dim": config.get("hidden_dim"),
        "max_len": config.get("max_len"),
        "parameter_scope": checkpoint.get("parameter_scope", "(desconocido)"),
        "created_at": checkpoint.get("created_at", ""),
        "metrics": metrics,
        "training": training,
        "label_map": checkpoint.get("label_map", {}),
        "format_version": checkpoint.get("format_version", 0),
    }


# ---------------------------------------------------------------------------
# Smoke test de shapes (Fase 3)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = YouTubeSentimentClassifier(freeze_backbone=True)
    ids = torch.randint(0, 1000, (4, 128))
    mask = torch.ones(4, 128, dtype=torch.long)
    logits = model(ids, mask)
    print("logits:", tuple(logits.shape))  # torch.Size([4, 3])

    trainable, total = model.count_parameters()
    print(f"Entrenables: {trainable:,} / {total:,} ({100 * trainable / total:.2f} %)")
