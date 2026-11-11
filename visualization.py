"""
Visualizaciones con Matplotlib.

Se separan del frontend para poder reutilizarlas en `train.py` (curvas de
pérdida, matriz de confusión) y en `app.py` (dashboard de la audiencia).

Todas las funciones devuelven una `Figure`; quien las usa decide si la muestra
con `st.pyplot` o la guarda en disco. Así evitamos figuras huérfanas en memoria.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")  # Backend sin GUI: funciona en Streamlit y en servidores.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import ID2LABEL, SENTIMENT_COLORS

PLOT_ORDER = [ID2LABEL[2], ID2LABEL[1], ID2LABEL[0]]  # Positivo, Neutro, Negativo
COLORS = [SENTIMENT_COLORS[c] for c in PLOT_ORDER]


def _require_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas para graficar: {missing}")


def sentiment_dashboard_figure(
    df: pd.DataFrame, figsize: tuple[float, float] = (18, 5.2)
) -> plt.Figure:
    """
    Tres gráficos del pulso de la audiencia:

    1. Distribución por volumen (donut).
    2. Impacto ponderado por likes (barras).
    3. Confianza de la red (histograma de max(softmax)).
    """
    _require_columns(df, ["sentimiento", "likes", "confianza_%"])

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.patch.set_facecolor("#ffffff")

    # ------------------------------------------------------------------
    # 1. Distribución de sentimiento (volumen)
    # ------------------------------------------------------------------
    conteo = df["sentimiento"].value_counts().reindex(PLOT_ORDER, fill_value=0)
    total = int(conteo.sum())

    if total > 0:
        def _autopct(pct: float) -> str:
            return f"{pct:.1f}%" if pct > 0.05 else ""

        _, _, autotexts = axes[0].pie(
            conteo,
            labels=PLOT_ORDER,
            autopct=_autopct,
            startangle=140,
            colors=COLORS,
            pctdistance=0.78,
            wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2),
        )
        for text in autotexts:
            text.set_color("white")
            text.set_weight("bold")
            text.set_fontsize(10)
        axes[0].text(
            0,
            0,
            f"{total}\ncomentarios",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="#2c3e50",
        )
    else:
        axes[0].text(0.5, 0.5, "Sin datos", ha="center", va="center")

    axes[0].set_title(
        "1. Distribución de Sentimiento (Volumen)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )

    # ------------------------------------------------------------------
    # 2. Impacto ponderado por likes
    # ------------------------------------------------------------------
    work = df.copy()
    work["peso_interaccion"] = pd.to_numeric(work["likes"], errors="coerce").fillna(0) + 1
    impacto = (
        work.groupby("sentimiento")["peso_interaccion"]
        .sum()
        .reindex(PLOT_ORDER, fill_value=0)
    )

    bars = axes[1].bar(
        PLOT_ORDER, impacto.values, color=COLORS, edgecolor="#2c3e50", width=0.55
    )
    axes[1].set_title(
        "2. Impacto Ponderado por Likes (Comunidad)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    axes[1].set_ylabel("Suma de (Likes + 1)")
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    axes[1].set_ylim(0, max(float(impacto.max()) * 1.18, 1))

    for bar in bars:
        height = float(bar.get_height())
        axes[1].annotate(
            f"{int(height):,}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9,
        )

    # ------------------------------------------------------------------
    # 3. Confianza de la red (softmax)
    # ------------------------------------------------------------------
    bins = np.linspace(0, 100, 11)
    for category, color in zip(PLOT_ORDER, COLORS):
        subset = pd.to_numeric(
            df.loc[df["sentimiento"] == category, "confianza_%"], errors="coerce"
        ).dropna()
        if not subset.empty:
            axes[2].hist(
                subset,
                bins=bins,
                alpha=0.65,
                label=category,
                color=color,
                edgecolor="black",
                linewidth=0.5,
            )

    axes[2].set_title(
        "3. Confianza de la Red (Probabilidad Softmax)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    axes[2].set_xlabel("Confianza de Predicción (%)")
    axes[2].set_ylabel("Nº de Comentarios")
    axes[2].set_xlim(0, 100)
    axes[2].legend(loc="upper left", frameon=False)
    axes[2].grid(axis="y", linestyle="--", alpha=0.4)
    axes[2].spines["top"].set_visible(False)
    axes[2].spines["right"].set_visible(False)

    fig.tight_layout()
    return fig


def polarity_likes_figure(
    df: pd.DataFrame, figsize: tuple[float, float] = (10, 5)
) -> plt.Figure:
    """
    Dispersión de polaridad neta frente a likes (escala logarítmica).

    Responde a la pregunta del Experimento 5: ¿la comunidad premia el mismo
    tono que detecta la red? La correlación de Spearman es orientativa: los
    likes no son una etiqueta de sentimiento.
    """
    _require_columns(df, ["sentimiento", "likes", "polaridad_neta"])

    fig, ax = plt.subplots(figsize=figsize)
    work = df.copy()
    work["likes_plot"] = pd.to_numeric(work["likes"], errors="coerce").fillna(0) + 1
    work["polaridad_neta"] = pd.to_numeric(work["polaridad_neta"], errors="coerce")

    for category, color in zip(PLOT_ORDER, COLORS):
        subset = work[work["sentimiento"] == category]
        if not subset.empty:
            ax.scatter(
                subset["likes_plot"],
                subset["polaridad_neta"],
                s=42,
                alpha=0.75,
                color=color,
                edgecolor="white",
                linewidth=0.6,
                label=category,
            )

    if float(work["likes_plot"].max()) > 1:
        ax.set_xscale("log")

    ax.axhline(0, color="#2c3e50", linewidth=1, linestyle="--", alpha=0.8)
    ax.set_xlabel("Likes + 1 (escala logarítmica)")
    ax.set_ylabel("Polaridad neta  P(Positivo) − P(Negativo)")
    ax.set_title(
        "Polaridad neuronal vs. impacto comunitario",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", frameon=False)

    try:
        rho = work[["likes", "polaridad_neta"]].corr(method="spearman").iloc[0, 1]
        if pd.notna(rho):
            ax.text(
                0.02,
                0.98,
                f"Spearman ρ = {rho:.3f}  (n = {len(work)})",
                transform=ax.transAxes,
                va="top",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )
    except Exception:
        # scipy puede no estar disponible; el gráfico sigue siendo válido.
        pass

    fig.tight_layout()
    return fig


def training_curves_figure(
    history: Sequence[dict[str, Any]], figsize: tuple[float, float] = (13, 4.6)
) -> plt.Figure:
    """Curvas de pérdida y métricas por época para diagnosticar convergencia."""
    if not history:
        raise ValueError("El historial de entrenamiento está vacío.")

    epochs = [int(h.get("epoch", i + 1)) for i, h in enumerate(history)]
    train_loss = [float(h.get("train_loss", np.nan)) for h in history]
    val_loss = [float(h.get("val_loss", np.nan)) for h in history]
    train_acc = [float(h.get("train_acc", np.nan)) for h in history]
    val_acc = [float(h.get("val_acc", np.nan)) for h in history]
    val_f1 = [float(h.get("val_f1", np.nan)) for h in history]

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.patch.set_facecolor("#ffffff")

    axes[0].plot(epochs, train_loss, marker="o", linewidth=2, label="Train loss", color="#3498db")
    axes[0].plot(epochs, val_loss, marker="s", linewidth=2, label="Validación loss", color="#e74c3c")
    axes[0].set_title("Curva de pérdida (CrossEntropy)", fontweight="bold")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend(frameon=False)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    axes[1].plot(epochs, train_acc, marker="o", linewidth=2, label="Train accuracy", color="#3498db")
    axes[1].plot(epochs, val_acc, marker="s", linewidth=2, label="Validación accuracy", color="#e74c3c")
    axes[1].plot(epochs, val_f1, marker="^", linewidth=2, label="Validación F1 macro", color="#2ecc71")
    axes[1].set_title("Métricas por época", fontweight="bold")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Valor")
    axes[1].set_ylim(0, 1.02)
    axes[1].grid(True, linestyle="--", alpha=0.35)
    axes[1].legend(frameon=False)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    if len(epochs) <= 12:
        axes[0].set_xticks(epochs)
        axes[1].set_xticks(epochs)

    valid_f1 = [(e, f) for e, f in zip(epochs, val_f1) if np.isfinite(f)]
    if valid_f1:
        best_epoch, best_value = max(valid_f1, key=lambda pair: pair[1])
        axes[1].axvline(best_epoch, color="#7f8c8d", linestyle=":", alpha=0.8)
        axes[1].annotate(
            f"mejor F1 = {best_value:.3f}",
            xy=(best_epoch, best_value),
            xytext=(6, -18),
            textcoords="offset points",
            fontsize=8,
            color="#2c3e50",
        )

    fig.tight_layout()
    return fig


def confusion_matrix_figure(
    cm: np.ndarray,
    labels: Sequence[str] = tuple(ID2LABEL[i] for i in range(3)),
    figsize: tuple[float, float] = (6.2, 5.2),
) -> plt.Figure:
    """Matriz de confusión con recuentos y porcentaje por fila (recall)."""
    cm = np.asarray(cm, dtype=float)
    fig, ax = plt.subplots(figsize=figsize)
    image = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="Etiqueta real",
        xlabel="Predicción de la red",
    )
    ax.set_title("Matriz de confusión (validación)", fontweight="bold", pad=12)

    row_totals = cm.sum(axis=1, keepdims=True)
    row_totals[row_totals == 0] = 1
    percentages = cm / row_totals * 100

    threshold = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{int(cm[i, j])}\n({percentages[i, j]:.0f}%)",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if cm[i, j] > threshold else "#2c3e50",
            )

    fig.tight_layout()
    return fig


def save_figure(fig: plt.Figure, path: str | Path, dpi: int = 150) -> Path:
    """Guarda una figura creando el directorio destino."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    return path
