"""
Smoke test del pipeline completo (sin Streamlit).

Comprueba las piezas que suelen romperse:
  - extracción del ID de YouTube,
  - shapes de tokenización y del modelo,
  - grupos de parámetros del optimizador,
  - guardado/carga de checkpoint,
  - figuras de Matplotlib,
  - opcionalmente, una llamada real a YouTube Data API v3.

Uso:
    python smoke_test.py
    python smoke_test.py --api
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import DEFAULT_MODEL, ID2LABEL, get_youtube_api_key
from model_architecture import (
    YouTubeSentimentClassifier,
    build_checkpoint,
    load_checkpoint_into_model,
    read_checkpoint,
    save_checkpoint,
)
from train import CommentsDataset, build_optimizer
from visualization import (
    confusion_matrix_figure,
    polarity_likes_figure,
    sentiment_dashboard_figure,
    training_curves_figure,
)
from youtube_api import extract_video_id, fetch_youtube_comments

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "✅" if condition else "❌"
    print(f"{status} {name}" + (f" · {detail}" if detail else ""))
    if not condition:
        raise AssertionError(name)


def test_video_id() -> None:
    cases = {
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ": "dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ?t=10": "dQw4w9WgXcQ",
        "https://www.youtube.com/shorts/dQw4w9WgXcQ": "dQw4w9WgXcQ",
        "https://www.youtube.com/embed/dQw4w9WgXcQ": "dQw4w9WgXcQ",
        "https://www.youtube-nocookie.com/embed/dQw4w9WgXcQ": "dQw4w9WgXcQ",
        "https://music.youtube.com/watch?v=dQw4w9WgXcQ": "dQw4w9WgXcQ",
        "https://www.youtube.com/live/dQw4w9WgXcQ": "dQw4w9WgXcQ",
        "dQw4w9WgXcQ": "dQw4w9WgXcQ",
        "https://example.com/video": None,
        "": None,
    }
    for url, expected in cases.items():
        actual = extract_video_id(url)
        check(f"extract_video_id({url!r})", actual == expected, f"-> {actual}")


def test_model_and_checkpoint() -> None:
    print(f"\n⏳ Cargando {DEFAULT_MODEL} (puede tardar la primera vez)...")
    model = YouTubeSentimentClassifier(
        model_name=DEFAULT_MODEL,
        pooling_type="mean",
        dropout_rate=0.3,
        hidden_dim=64,
        freeze_backbone=True,
    )

    ids = torch.randint(0, 1000, (4, 128))
    mask = torch.ones(4, 128, dtype=torch.long)
    logits = model(ids, mask)
    check("forward devuelve [batch, 3]", tuple(logits.shape) == (4, 3), str(tuple(logits.shape)))

    trainable, total = model.count_parameters()
    check(
        "backbone congelado (<1 % entrenable)",
        trainable < total * 0.01,
        f"{trainable:,}/{total:,} = {100 * trainable / total:.3f} %",
    )

    # Dataset y tokenización
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    dataset = CommentsDataset(
        ["¡Buenísimo el vídeo! 🔥", "no me gustó nada, muy largo"],
        [2, 0],
        tokenizer,
        max_len=128,
    )
    item = dataset[0]
    check("input_ids [128]", tuple(item["input_ids"].shape) == (128,), str(tuple(item["input_ids"].shape)))
    check(
        "attention_mask [128]",
        tuple(item["attention_mask"].shape) == (128,),
        str(tuple(item["attention_mask"].shape)),
    )
    check("label entero", item["label"].dtype == torch.long, str(item["label"].dtype))

    # Optimizador con dos grupos (backbone congelado -> solo cabezal)
    optimizer = build_optimizer(model, optimizer_name="AdamW", lr_head=1e-3)
    check("optimizador creado", optimizer is not None, f"{len(optimizer.param_groups)} grupo(s)")

    # Checkpoint de ida y vuelta
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "checkpoint.pt"
        checkpoint = build_checkpoint(
            model,
            config=model.config_dict(max_len=128, unfrozen_layers=0),
            metrics={"best_val_f1_macro": 0.5},
            training={"seed": 42},
        )
        save_checkpoint(path, checkpoint)
        loaded = read_checkpoint(path)
        check("checkpoint contiene state_dict", "state_dict" in loaded)
        check(
            "checkpoint guarda solo cabezal",
            all(not key.startswith("backbone.") for key in loaded["state_dict"]),
            f"{len(loaded['state_dict'])} tensores",
        )

        fresh = YouTubeSentimentClassifier(
            model_name=DEFAULT_MODEL,
            pooling_type="mean",
            dropout_rate=0.3,
            hidden_dim=64,
            freeze_backbone=True,
        )
        report = load_checkpoint_into_model(fresh, loaded)
        check("carga de checkpoint", report["loaded_head"] is True, report["mode"])

        # Verificar que los pesos del cabezal son idénticos.
        original = model.state_dict()["fc_out.weight"].clone()
        restored = fresh.state_dict()["fc_out.weight"]
        check("pesos del cabezal idénticos", torch.allclose(original, restored))

        # El checkpoint debe ser una foto inmutable, no una vista del modelo vivo.
        snapshot = checkpoint["state_dict"]["fc_out.weight"].clone()
        with torch.no_grad():
            model.fc_out.weight.add_(1.0)
        check(
            "checkpoint no comparte almacenamiento con el modelo",
            torch.allclose(checkpoint["state_dict"]["fc_out.weight"], snapshot),
        )

    # Descongelación progresiva
    unfrozen = model.unfreeze_last_layers(1)
    check("unfreeze_last_layers devuelve parámetros", unfrozen > 0, f"{unfrozen:,}")
    model.train()
    check("backbone en train tras descongelar", model.backbone.training is True)


def test_visualizations() -> None:
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "sentimiento": rng.choice(list(ID2LABEL.values()), size=40),
            "likes": rng.integers(0, 500, size=40),
            "confianza_%": rng.uniform(35, 99, size=40).round(1),
            "polaridad_neta": rng.uniform(-1, 1, size=40).round(3),
        }
    )
    fig = sentiment_dashboard_figure(df)
    check("figura dashboard", fig is not None)
    fig2 = polarity_likes_figure(df)
    check("figura polaridad/likes", fig2 is not None)

    history = [
        {"epoch": 1, "train_loss": 1.0, "val_loss": 1.1, "train_acc": 0.4, "val_acc": 0.35, "val_f1": 0.3},
        {"epoch": 2, "train_loss": 0.7, "val_loss": 0.8, "train_acc": 0.6, "val_acc": 0.55, "val_f1": 0.5},
    ]
    fig3 = training_curves_figure(history)
    check("figura curvas", fig3 is not None)
    fig4 = confusion_matrix_figure(np.array([[5, 1, 0], [1, 4, 1], [0, 1, 5]]))
    check("figura matriz de confusión", fig4 is not None)

    import matplotlib.pyplot as plt

    for figure in (fig, fig2, fig3, fig4):
        plt.close(figure)


def test_youtube_api() -> None:
    api_key = get_youtube_api_key()
    if not api_key:
        print("\n⚠️  Sin YOUTUBE_API_KEY: se omite la prueba real de la API.")
        return

    print("\n⏳ Probando YouTube Data API v3 (2 comentarios)...")
    df = fetch_youtube_comments(api_key, "dQw4w9WgXcQ", max_comments=2)
    check("API devuelve comentarios", not df.empty, f"{len(df)} filas")
    check(
        "columnas esperadas",
        {"autor", "comentario", "likes", "publicado"}.issubset(set(df.columns)),
        str(list(df.columns)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", action="store_true", help="Probar también la API de YouTube.")
    args = parser.parse_args()

    print("=" * 70)
    print("🧪 SMOKE TEST · Analizador Neuronal de Sentimiento")
    print("=" * 70)

    test_video_id()
    test_model_and_checkpoint()
    test_visualizations()
    if args.api:
        test_youtube_api()

    print("\n🎉 Todas las comprobaciones pasaron.")


if __name__ == "__main__":
    main()
