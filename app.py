"""
Dashboard Streamlit del Analizador Neuronal de Sentimiento.

Flujo:
    URL de YouTube -> YouTube Data API v3 -> DataFrame
    -> tokenización -> backbone Transformer -> cabezal PyTorch
    -> softmax -> dashboard Matplotlib + tabla + CSV

Ejecutar:
    streamlit run app.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHECKPOINT,
    DEFAULT_DROPOUT,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_MAX_LEN,
    DEFAULT_MODEL,
    DEFAULT_POOLING,
    ID2LABEL,
    LABEL_NAMES,
    MODEL_OPTIONS,
    get_youtube_api_key,
    resolve_path,
)
from model_architecture import (
    YouTubeSentimentClassifier,
    checkpoint_summary,
    load_checkpoint_into_model,
    read_checkpoint,
)
from visualization import (
    polarity_likes_figure,
    sentiment_dashboard_figure,
)
from youtube_api import (
    YouTubeAPIError,
    extract_video_id,
    fetch_video_info,
    fetch_youtube_comments,
)

# ===========================================================================
# Configuración de página y estilo
# ===========================================================================
st.set_page_config(
    page_title="Analizador Neuronal de Sentimiento · YouTube",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.6rem; padding-bottom: 3rem; }

    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 55%, #0f766e 100%);
        border-radius: 18px;
        padding: 1.5rem 1.8rem;
        color: #f8fafc;
        margin-bottom: 1.4rem;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.18);
    }
    .hero h1 { color: #ffffff; font-size: 2rem; margin: 0.2rem 0 0.4rem 0; }
    .hero p { color: #dbeafe; margin: 0; font-size: 1rem; }
    .hero-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.14);
        border: 1px solid rgba(255, 255, 255, 0.25);
        color: #e0f2fe;
        padding: 0.18rem 0.6rem;
        border-radius: 999px;
        font-size: 0.72rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        font-weight: 600;
    }

    /* Tarjetas translúcidas: heredan el color de texto del tema (claro u oscuro).
       Un fondo fijo blanco dejaría el texto ilegible en tema oscuro. */
    div[data-testid="stMetric"] {
        background: rgba(148, 163, 184, 0.10);
        border: 1px solid rgba(148, 163, 184, 0.30);
        border-radius: 14px;
        padding: 0.8rem 0.95rem;
    }
    div[data-testid="stMetric"] label { opacity: 0.75; }
    div[data-testid="stMetricValue"] { color: inherit; }

    /* La barra lateral usa el fondo del tema; solo añadimos un borde de separación */
    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(148, 163, 184, 0.28);
    }

    .small-note { color: inherit; opacity: 0.72; font-size: 0.84rem; }
    .insight {
        background: rgba(15, 118, 110, 0.12);
        border-left: 4px solid #0f766e;
        border-radius: 8px;
        padding: 0.8rem 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ===========================================================================
# Utilidades de UI
# ===========================================================================
def _fmt_int(value: object) -> str:
    try:
        return f"{int(value):,}".replace(",", ".")
    except (TypeError, ValueError):
        return "—"


def _fmt_pct(value: object, digits: int = 1) -> str:
    try:
        return f"{float(value):.{digits}f}%"
    except (TypeError, ValueError):
        return "—"


def _checkpoint_signature(path: str | Path) -> tuple[str, int, int]:
    if not str(path).strip():
        return ("", 0, 0)
    resolved = resolve_path(path)
    if resolved.is_file():
        stat = resolved.stat()
        return (str(resolved), int(stat.st_mtime_ns), int(stat.st_size))
    return (str(resolved), 0, 0)


# ===========================================================================
# Caché de datos y modelo
# ===========================================================================
@st.cache_data(ttl=900, show_spinner=False)
def cached_fetch_video_info(api_key: str, video_id: str) -> dict:
    return fetch_video_info(api_key, video_id)


@st.cache_data(ttl=900, show_spinner=False)
def cached_fetch_comments(
    api_key: str, video_id: str, max_comments: int, order: str
) -> pd.DataFrame:
    return fetch_youtube_comments(api_key, video_id, max_comments, order)


@st.cache_data(show_spinner=False)
def cached_checkpoint_summary(path: str, signature: tuple[str, int, int]) -> dict | None:
    try:
        checkpoint = read_checkpoint(path, map_location="cpu")
        return checkpoint_summary(checkpoint)
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def load_neural_pipeline(
    model_name: str,
    pooling_type: str,
    dropout_rate: float,
    checkpoint_path: str,
    checkpoint_signature: tuple[str, int, int],
):
    """
    Carga tokenizador + red una sola vez por combinación de configuración.

    Si hay checkpoint:
      - reconstruye el cabezal con el `hidden_dim` con el que fue entrenado,
      - carga los pesos del cabezal (y las capas del backbone que se hubieran
        descongelado durante el fine-tuning),
      - informa si el backbone seleccionado no coincide con el del checkpoint.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    checkpoint = None
    checkpoint_path_obj = Path(checkpoint_path) if checkpoint_path else None
    if checkpoint_path_obj is not None and checkpoint_path_obj.is_file():
        checkpoint = read_checkpoint(checkpoint_path_obj, map_location="cpu")

    hidden_dim = DEFAULT_HIDDEN_DIM
    if checkpoint is not None:
        config = checkpoint.get("config", {}) or {}
        hidden_dim = int(config.get("hidden_dim", DEFAULT_HIDDEN_DIM))

    model = YouTubeSentimentClassifier(
        model_name=model_name,
        num_classes=3,
        pooling_type=pooling_type,
        dropout_rate=dropout_rate,
        hidden_dim=hidden_dim,
        freeze_backbone=True,
    )

    metadata: dict | None = None
    load_report: dict = {
        "mode": "untrained",
        "loaded_head": False,
        "loaded_backbone_keys": 0,
        "ignored_backbone_keys": 0,
        "message": (
            "No se encontró un checkpoint entrenado. El cabezal tiene pesos "
            "aleatorios: las predicciones NO representan sentimiento real."
        ),
    }

    if checkpoint is not None:
        metadata = checkpoint_summary(checkpoint)
        load_report = load_checkpoint_into_model(model, checkpoint)
        if metadata.get("model_name") and metadata["model_name"] != model_name:
            load_report["message"] = (
                "El checkpoint fue entrenado con otro backbone "
                f"({metadata['model_name']}). Se cargó el cabezal y se usó el "
                "backbone seleccionado sin ajustar."
            )

    model.to(device)
    model.eval()
    return tokenizer, model, device, metadata, load_report


# ===========================================================================
# Inferencia
# ===========================================================================
@torch.no_grad()
def predict_sentiment_batch(
    df: pd.DataFrame,
    tokenizer,
    model: YouTubeSentimentClassifier,
    device: torch.device,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_len: int = DEFAULT_MAX_LEN,
) -> pd.DataFrame:
    """
    Pasa los comentarios por la red y aplica softmax.

    Devuelve etiqueta, confianza (%), polaridad neta y las tres probabilidades
    para que el resultado sea auditable.
    """
    if df.empty:
        return df.copy()

    texts = df["comentario"].astype(str).tolist()
    batch_size = max(1, int(batch_size))

    preds_all: list[str] = []
    conf_all: list[float] = []
    neg_all: list[float] = []
    neu_all: list[float] = []
    pos_all: list[float] = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        ).to(device)

        logits = model(encoded["input_ids"], encoded["attention_mask"])
        probs = F.softmax(logits, dim=1).cpu().numpy()

        for prob_vec in probs:
            pred_id = int(np.argmax(prob_vec))
            preds_all.append(ID2LABEL[pred_id])
            conf_all.append(round(float(np.max(prob_vec)) * 100, 1))
            neg_all.append(float(prob_vec[0]))
            neu_all.append(float(prob_vec[1]))
            pos_all.append(float(prob_vec[2]))

    result = df.copy()
    result["sentimiento"] = preds_all
    result["confianza_%"] = conf_all
    result["polaridad_neta"] = np.round(np.array(pos_all) - np.array(neg_all), 3)
    result["p_negativo"] = np.round(neg_all, 4)
    result["p_neutro"] = np.round(neu_all, 4)
    result["p_positivo"] = np.round(pos_all, 4)
    return result


# ===========================================================================
# Renderizado de resultados
# ===========================================================================
def _render_video_header(video_info: dict) -> None:
    st.markdown(f"### 🎬 {video_info.get('titulo', 'Vídeo de YouTube')}")
    st.caption(
        f"Canal: **{video_info.get('canal', '—')}** · "
        f"Publicado: {video_info.get('publicado', '—')} · "
        f"{_fmt_int(video_info.get('vistas', 0))} vistas · "
        f"{_fmt_int(video_info.get('comentarios_totales', 0))} comentarios totales"
    )


def _render_metrics(df: pd.DataFrame) -> None:
    total = len(df)
    pct_pos = (df["sentimiento"] == "Positivo").mean() * 100 if total else 0
    pct_neg = (df["sentimiento"] == "Negativo").mean() * 100 if total else 0
    mean_conf = df["confianza_%"].mean() if total else 0
    mean_pol = df["polaridad_neta"].mean() if total else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Comentarios analizados", _fmt_int(total))
    c2.metric("Sentimiento positivo", _fmt_pct(pct_pos))
    c3.metric("Sentimiento negativo", _fmt_pct(pct_neg))
    c4.metric("Confianza media", _fmt_pct(mean_conf))
    c5.metric("Polaridad media", f"{mean_pol:+.3f}")


def _render_insights(df: pd.DataFrame, order: str = "relevance") -> None:
    if df.empty:
        return

    volume = df["sentimiento"].value_counts()
    impact = (
        df.assign(peso_interaccion=df["likes"].fillna(0) + 1)
        .groupby("sentimiento")["peso_interaccion"]
        .sum()
    )
    volume_winner = volume.idxmax() if not volume.empty else "—"
    impact_winner = impact.idxmax() if not impact.empty else "—"
    mean_pol = df["polaridad_neta"].mean()

    try:
        rho = df[["likes", "polaridad_neta"]].corr(method="spearman").iloc[0, 1]
        rho_text = f" Correlación de Spearman likes↔polaridad: **ρ = {rho:.3f}**." if pd.notna(rho) else ""
    except Exception:
        rho_text = ""

    st.markdown(
        f"""
        <div class="insight">
        <strong>Lectura rápida.</strong> Por <em>volumen</em> de comentarios domina
        <strong>{volume_winner}</strong>; al ponderar por <em>likes + 1</em> domina
        <strong>{impact_winner}</strong>. La polaridad media es
        <strong>{mean_pol:+.3f}</strong> en una escala de −1 a +1.{rho_text}
        Recuerda que la muestra está sesgada por el orden de YouTube
        (<code>{order}</code>) y no representa a toda la audiencia.
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_dashboard_tab(df: pd.DataFrame, order: str = "relevance") -> None:
    st.markdown("#### 📊 Tres vistas del pulso de la audiencia")
    fig = sentiment_dashboard_figure(df)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("#### 🔗 Polaridad neuronal vs. impacto comunitario")
    fig2 = polarity_likes_figure(df)
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

    _render_insights(df, order=order)

    st.markdown("#### 🏆 Comentarios con más impacto")
    top = (
        df.nlargest(5, "likes")[
            ["autor", "comentario", "likes", "sentimiento", "confianza_%", "polaridad_neta"]
        ]
        if not df.empty
        else df
    )
    st.dataframe(
        top,
        use_container_width=True,
        hide_index=True,
        column_config={
            "likes": st.column_config.NumberColumn("Likes", format="%d"),
            "confianza_%": st.column_config.ProgressColumn(
                "Confianza", min_value=0, max_value=100, format="%.1f%%"
            ),
            "polaridad_neta": st.column_config.NumberColumn("Polaridad", format="%+.3f"),
        },
    )


def _render_comments_tab(df: pd.DataFrame, video_id: str) -> None:
    st.markdown("#### 💬 Detalle y exportación")

    filter_cols = st.columns([2, 1, 2])
    with filter_cols[0]:
        sentiments = st.multiselect(
            "Filtrar por sentimiento",
            options=LABEL_NAMES,
            default=LABEL_NAMES,
            key="filter_sentiments",
        )
    with filter_cols[1]:
        min_likes = st.number_input(
            "Likes mínimos", min_value=0, value=0, step=1, key="filter_min_likes"
        )
    with filter_cols[2]:
        query = st.text_input(
            "Buscar texto", placeholder="palabra o frase...", key="filter_query"
        )

    filtered = df[df["sentimiento"].isin(sentiments) & (df["likes"] >= min_likes)]
    if query.strip():
        filtered = filtered[
            filtered["comentario"].str.contains(query.strip(), case=False, na=False)
        ]
    filtered = filtered.sort_values("likes", ascending=False).reset_index(drop=True)

    st.caption(f"Mostrando **{len(filtered):,}** de **{len(df):,}** comentarios.")

    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True,
        column_config={
            "comment_id": None,
            "autor": st.column_config.TextColumn("Autor"),
            "comentario": st.column_config.TextColumn("Comentario", width="large"),
            "likes": st.column_config.NumberColumn("Likes", format="%d"),
            "publicado": st.column_config.TextColumn("Publicado"),
            "respuestas": st.column_config.NumberColumn("Respuestas", format="%d"),
            "sentimiento": st.column_config.TextColumn("Sentimiento"),
            "confianza_%": st.column_config.ProgressColumn(
                "Confianza", min_value=0, max_value=100, format="%.1f%%"
            ),
            "polaridad_neta": st.column_config.NumberColumn("Polaridad", format="%+.3f"),
            "p_negativo": st.column_config.NumberColumn("P(Neg)", format="%.3f"),
            "p_neutro": st.column_config.NumberColumn("P(Neu)", format="%.3f"),
            "p_positivo": st.column_config.NumberColumn("P(Pos)", format="%.3f"),
        },
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    csv_all = df.to_csv(index=False).encode("utf-8-sig")
    csv_filtered = filtered.to_csv(index=False).encode("utf-8-sig")

    download_cols = st.columns(2)
    with download_cols[0]:
        st.download_button(
            "📥 Descargar todos los resultados (CSV)",
            data=csv_all,
            file_name=f"sentimiento_youtube_{video_id}_{timestamp}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with download_cols[1]:
        st.download_button(
            "📥 Descargar vista filtrada (CSV)",
            data=csv_filtered,
            file_name=f"sentimiento_youtube_{video_id}_filtrado_{timestamp}.csv",
            mime="text/csv",
            use_container_width=True,
            disabled=filtered.empty,
        )

    st.caption(
        "El CSV se codifica en UTF-8 con BOM (`utf-8-sig`) para que Excel "
        "muestre correctamente los acentos."
    )


def _render_checkpoint_card(
    metadata: dict | None, load_report: dict, checkpoint_path: str
) -> None:
    st.markdown("#### 🧠 Checkpoint y métricas")

    if metadata is None:
        st.error(
            "No se encontró un checkpoint entrenado. El modelo está usando un "
            "cabezal con pesos aleatorios: las predicciones **no** representan "
            "sentimiento real. Entrena con `python train.py` y vuelve a cargar."
        )
        return

    metrics = metadata.get("metrics", {}) or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "F1 macro (validación)",
        f"{metrics['best_val_f1_macro']:.3f}"
        if metrics.get("best_val_f1_macro") is not None
        else "—",
    )
    c2.metric(
        "Accuracy (validación)",
        f"{metrics['best_val_accuracy']:.3f}"
        if metrics.get("best_val_accuracy") is not None
        else "—",
    )
    c3.metric(
        "F1 macro (test)",
        f"{metrics['test_f1_macro']:.3f}" if metrics.get("test_f1_macro") is not None else "—",
    )
    c4.metric(
        "Mejor época",
        str(metrics.get("best_epoch", "—")),
    )

    st.caption(
        f"Entrenado: {metadata.get('created_at', '—')} · "
        f"Backbone: `{metadata.get('model_name', '—')}` · "
        f"Pooling: `{metadata.get('pooling_type', '—')}` · "
        f"Alcance de pesos: `{metadata.get('parameter_scope', '—')}`"
    )

    with st.expander("⚙️ Configuración del checkpoint", expanded=False):
        config_frame = pd.DataFrame(
            [
                ("Modelo base", metadata.get("model_name", "—")),
                ("Pooling", metadata.get("pooling_type", "—")),
                ("Dropout", metadata.get("dropout_rate", "—")),
                ("hidden_dim", metadata.get("hidden_dim", "—")),
                ("max_len", metadata.get("max_len", "—")),
                ("Clases", ", ".join(f"{k}={v}" for k, v in metadata.get("label_map", {}).items())),
                ("Formato", metadata.get("format_version", "—")),
                ("Checkpoint", checkpoint_path),
            ],
            columns=["Campo", "Valor"],
        ).astype(str)
        st.dataframe(config_frame, hide_index=True, use_container_width=True)

    st.markdown("##### Informe de carga de pesos")
    mode = load_report.get("mode")
    if mode == "head_only" and load_report.get("ignored_backbone_keys", 0) > 0:
        mode_label = "⚠️ Solo cabezal: el backbone seleccionado no coincide con el del checkpoint"
    elif mode == "head_only":
        mode_label = "✅ Cabezal entrenado + backbone preentrenado congelado"
    elif mode == "full":
        mode_label = "✅ Pesos completos (cabezal + backbone del checkpoint)"
    elif mode == "partial":
        mode_label = "✅ Pesos parciales (cabezal + capas descongeladas)"
    else:
        mode_label = "❌ Sin checkpoint"
    st.info(f"{mode_label}. {load_report.get('message', '')}")
    if load_report.get("ignored_backbone_keys"):
        st.caption(
            f"Claves de backbone ignoradas: {load_report['ignored_backbone_keys']}. "
            "Esto es normal si cambiaste el modelo base en el panel lateral."
        )

    # Informe de entrenamiento (si existe junto al checkpoint).
    report_path = resolve_path(checkpoint_path).parent / "training_report.json"
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            with st.expander("📋 Informe de validación y test", expanded=False):
                st.markdown("**Validación**")
                st.code(report.get("validation_report", "(sin informe)"), language="text")
                if report.get("test_report"):
                    st.markdown("**Test**")
                    st.code(report["test_report"], language="text")
                st.caption(
                    "El F1 macro penaliza ignorar la clase neutra. La accuracy "
                    "puede ser engañosa con clases desbalanceadas."
                )
        except Exception:
            pass


def _render_model_tab(
    metadata: dict | None, load_report: dict, checkpoint_path: str, config_used: dict
) -> None:
    _render_checkpoint_card(metadata, load_report, checkpoint_path)

    st.markdown("---")
    st.markdown("#### 🧩 Cómo funciona la red")
    st.markdown(
        """
        El modelo **no** es un clasificador descargado: es un *backbone* Transformer
        preentrenado al que se le acopla un cabezal construido en PyTorch.

        | Etapa | Operación | Shape |
        |---|---|---|
        | Texto | tokenización | `[batch, 128]` |
        | Backbone | Transformer congelado | `[batch, 128, 768]` |
        | Pooling | `mean` / `cls` / `max` | `[batch, 768]` |
        | Normalización | `LayerNorm` | `[batch, 768]` |
        | Regularización | `Dropout` | `[batch, 768]` |
        | Oculta | `Linear(768 → 256)` + `GELU` | `[batch, 256]` |
        | Salida | `Linear(256 → 3)` | `[batch, 3]` logits |
        | Decisión | `softmax` | `[batch, 3]` probabilidades |
        """
    )

    st.markdown("#### 🔬 Configuración usada en esta inferencia")
    used_frame = pd.DataFrame(
        [
            ("Modelo base", config_used.get("model_name")),
            ("Pooling", config_used.get("pooling_type")),
            ("Dropout (inactivo en inferencia)", config_used.get("dropout_rate")),
            ("hidden_dim", config_used.get("hidden_dim")),
            ("max_len", config_used.get("max_len")),
            ("Dispositivo", config_used.get("device")),
        ],
        columns=["Campo", "Valor"],
    ).astype(str)
    st.dataframe(used_frame, hide_index=True, use_container_width=True)

    st.markdown("#### 📚 Metodología y límites")
    st.markdown(
        """
        - **Datos de entrenamiento:** corpus público de sentimiento en español
          (tuíts), no comentarios de YouTube. Las etiquetas se mapean a
          `0=Negativo, 1=Neutro, 2=Positivo`.
        - **Cambio de dominio (*domain shift*):** entrenar con tuíts y predecir
          sobre YouTube introduce error. La jerga, la longitud y el contexto
          conversacional cambian.
        - **Sesgo de muestra:** `order="relevance"` prioriza comentarios con
          interacción. El resultado es una **estimación**, no un censo de la
          audiencia.
        - **Neutro es difícil:** suele ser la clase minoritaria y la más
          confundida; por eso el criterio de checkpoint es **F1 macro**, no
          accuracy.
        - **Confianza ≠ verdad:** `softmax` puede estar muy seguro y equivocarse.
          Contrasta siempre con una muestra manual de comentarios.
        """
    )

    if metadata and metadata.get("pooling_type") != config_used.get("pooling_type"):
        st.warning(
            f"El checkpoint se entrenó con pooling `{metadata.get('pooling_type')}` "
            f"y ahora estás usando `{config_used.get('pooling_type')}`. Cambiar el "
            "pooling altera la distribución que recibe el cabezal: úsalo como "
            "experimento, no como configuración de producción."
        )


def render_results(payload: dict) -> None:
    df: pd.DataFrame = payload["df"]
    video_info: dict = payload["video_info"]

    _render_video_header(video_info)

    if payload.get("checkpoint_meta") is None:
        st.warning(
            "⚠️ **Sin checkpoint entrenado.** Las etiquetas provienen de un cabezal "
            "con pesos aleatorios (~33 % por clase). Los gráficos son una prueba "
            "de fontanería, no un análisis de sentimiento. Ejecuta `python train.py`."
        )

    _render_metrics(df)

    tabs = st.tabs(
        ["📊 Pulso de la audiencia", "💬 Comentarios y exportación", "🧠 Modelo y metodología"]
    )
    with tabs[0]:
        _render_dashboard_tab(df, order=payload.get("order", "relevance"))
    with tabs[1]:
        _render_comments_tab(df, payload["video_id"])
    with tabs[2]:
        _render_model_tab(
            payload.get("checkpoint_meta"),
            payload.get("load_report", {}),
            payload.get("checkpoint_path", ""),
            payload.get("config_used", {}),
        )

    st.markdown("---")
    st.caption(
        "Herramienta didáctica de PLN. No uses sus predicciones como verdad "
        "absoluta ni como único insumo para decisiones de negocio."
    )


# ===========================================================================
# Pantalla de bienvenida
# ===========================================================================
def render_welcome() -> None:
    st.markdown(
        """
        <div class="hero">
            <span class="hero-badge">PLN · Transfer Learning · PyTorch</span>
            <h1>🎬 Analizador Neuronal de Sentimiento en YouTube</h1>
            <p>De la URL a un dashboard: ingesta con YouTube Data API v3,
            tokenización, backbone Transformer, cabezal propio y softmax.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 1 · Ingesta")
        st.markdown(
            "Se descargan hasta 500 comentarios con sus likes mediante "
            "`commentThreads.list`, respetando la paginación y filtrando vacíos."
        )
    with col2:
        st.markdown("### 2 · Red neuronal")
        st.markdown(
            "Un backbone preentrenado extrae representaciones de 768 dimensiones. "
            "Un cabezal propio las resume y clasifica en 3 clases."
        )
    with col3:
        st.markdown("### 3 · Lectura analítica")
        st.markdown(
            "Volumen, impacto ponderado por likes, confianza de la red y "
            "polaridad neta. Exportación a CSV para seguir analizando."
        )

    st.markdown("---")
    st.subheader("Estado del modelo")
    checkpoint_path = resolve_path(DEFAULT_CHECKPOINT)
    signature = _checkpoint_signature(checkpoint_path)
    metadata = cached_checkpoint_summary(str(checkpoint_path), signature)

    if metadata:
        metrics = metadata.get("metrics", {}) or {}
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "F1 macro (val)",
            f"{metrics['best_val_f1_macro']:.3f}" if metrics.get("best_val_f1_macro") is not None else "—",
        )
        c2.metric(
            "Accuracy (val)",
            f"{metrics['best_val_accuracy']:.3f}" if metrics.get("best_val_accuracy") is not None else "—",
        )
        c3.metric(
            "F1 macro (test)",
            f"{metrics['test_f1_macro']:.3f}" if metrics.get("test_f1_macro") is not None else "—",
        )
        c4.metric("Mejor época", str(metrics.get("best_epoch", "—")))
        st.success(
            f"Checkpoint listo: `{DEFAULT_CHECKPOINT}` · "
            f"{metadata.get('model_name', '—')} · pooling `{metadata.get('pooling_type', '—')}`"
        )
    else:
        st.warning(
            "Todavía no hay checkpoint entrenado en `checkpoints/`. "
            "Ejecuta `python prepare_data.py` y `python train.py` para generarlo."
        )

    with st.expander("🚀 Cómo empezar", expanded=True):
        st.markdown(
            """
            1. Escribe tu **API Key** de YouTube Data API v3 en el panel lateral
               (o configúrala en `.env` como `YOUTUBE_API_KEY`).
            2. Pega la **URL** de un vídeo con comentarios abiertos.
            3. Ajusta el máximo de comentarios y el modelo.
            4. Pulsa **Extraer y analizar**.

            La API Key nunca se escribe en el código ni se muestra en pantalla.
            """
        )

    st.markdown("#### 🔎 Qué mirar en los resultados")
    st.markdown(
        """
        - **Gráfico 1 (volumen):** cuántos comentarios hay de cada clase.
        - **Gráfico 2 (impacto):** cuánta interacción acumula cada clase. Un solo
          comentario con 1.000 likes pesa más que cien con 0.
        - **Gráfico 3 (confianza):** si la red decide o duda. Predicciones cerca
          del 33 % son casi aleatorias.
        - **Polaridad neta:** `P(Positivo) − P(Negativo)` en `[-1, +1]`; es más
          informativa que la etiqueta dura para promediar.
        """
    )


# ===========================================================================
# Sidebar
# ===========================================================================
default_api_key = get_youtube_api_key()

with st.sidebar:
    st.markdown("## ⚙️ Configuración")
    with st.form("analysis_form"):
        st.markdown("### 1 · Credenciales y vídeo")
        api_key = st.text_input(
            "YouTube API Key v3",
            value=default_api_key,
            type="password",
            help="Se lee de `.env` si existe. Nunca se guarda en el repositorio.",
        )
        video_url = st.text_input(
            "URL del vídeo",
            placeholder="https://www.youtube.com/watch?v=...",
        )
        col_a, col_b = st.columns(2)
        with col_a:
            max_comments = st.slider(
                "Máx. comentarios", min_value=20, max_value=500, value=100, step=20
            )
        with col_b:
            order = st.selectbox(
                "Orden de la muestra",
                options=["relevance", "time"],
                format_func=lambda x: "Relevancia" if x == "relevance" else "Recientes",
                help=(
                    "Relevancia sesga hacia comentarios con más interacción; "
                    "tiempo cambia radicalmente la muestra."
                ),
            )

        st.markdown("### 2 · Arquitectura de red")
        model_choice = st.selectbox(
            "Modelo base (Hugging Face)",
            options=MODEL_OPTIONS,
            index=MODEL_OPTIONS.index(DEFAULT_MODEL),
        )
        pooling_choice = st.selectbox(
            "Pooling",
            options=["mean", "cls", "max"],
            index=["mean", "cls", "max"].index(DEFAULT_POOLING),
            help="Cómo se resume la secuencia de tokens en un vector de 768 dimensiones.",
        )
        dropout_choice = st.slider(
            "Dropout del cabezal",
            min_value=0.1,
            max_value=0.5,
            value=DEFAULT_DROPOUT,
            step=0.05,
            help="Inactivo en inferencia; importa durante el entrenamiento.",
        )
        checkpoint_file = st.text_input(
            "Checkpoint entrenado (.pt)",
            value=DEFAULT_CHECKPOINT,
            help="Ruta relativa a la raíz del proyecto o absoluta.",
        )
        batch_size = st.slider(
            "Batch de inferencia",
            min_value=8,
            max_value=64,
            value=DEFAULT_BATCH_SIZE,
            step=8,
        )

        submitted = st.form_submit_button(
            "🚀 Extraer y analizar", use_container_width=True, type="primary"
        )

    checkpoint_resolved = (
        resolve_path(checkpoint_file) if str(checkpoint_file).strip() else None
    )
    if checkpoint_resolved is not None and checkpoint_resolved.is_file():
        st.success(f"Checkpoint detectado · {checkpoint_resolved.stat().st_size / 1e6:.2f} MB")
    elif checkpoint_resolved is not None and checkpoint_resolved.exists():
        st.warning("La ruta del checkpoint existe, pero no es un archivo `.pt`.")
    else:
        st.warning("Sin checkpoint: el cabezal no está entrenado.")

    st.caption(
        "La API Key se lee de `.env` (`YOUTUBE_API_KEY`) o del campo superior. "
        "No se escribe en el código."
    )


# ===========================================================================
# Flujo principal
# ===========================================================================
if submitted:
    st.session_state.pop("analysis", None)

    if not api_key.strip() or not video_url.strip():
        st.error("Introduce la API Key de YouTube y una URL de vídeo válida.")
    else:
        video_id = extract_video_id(video_url)
        if not video_id:
            st.error(
                "No se pudo extraer un ID de 11 caracteres de la URL. "
                "Formatos válidos: youtube.com/watch?v=..., youtu.be/..., /shorts/..."
            )
        else:
            try:
                with st.spinner("1/3 Leyendo metadatos del vídeo..."):
                    video_info = cached_fetch_video_info(api_key, video_id)

                with st.spinner(f"2/3 Descargando hasta {max_comments} comentarios..."):
                    df_comments = cached_fetch_comments(
                        api_key, video_id, int(max_comments), order
                    )

                if df_comments.empty:
                    st.warning(
                        "No se encontraron comentarios visibles en este vídeo. "
                        "Puede tener los comentarios desactivados o no tener ninguno."
                    )
                else:
                    with st.spinner("3/3 Ejecutando inferencia neuronal..."):
                        tokenizer, model, device, metadata, load_report = load_neural_pipeline(
                            model_choice,
                            pooling_choice,
                            float(dropout_choice),
                            str(checkpoint_resolved) if checkpoint_resolved else "",
                            _checkpoint_signature(checkpoint_file),
                        )
                        max_len = int((metadata or {}).get("max_len") or DEFAULT_MAX_LEN)
                        df_analyzed = predict_sentiment_batch(
                            df_comments,
                            tokenizer,
                            model,
                            device,
                            batch_size=int(batch_size),
                            max_len=max_len,
                        )

                    config_used = {
                        "model_name": model_choice,
                        "pooling_type": pooling_choice,
                        "dropout_rate": float(dropout_choice),
                        "hidden_dim": int((metadata or {}).get("hidden_dim") or DEFAULT_HIDDEN_DIM),
                        "max_len": max_len,
                        "device": str(device),
                    }
                    st.session_state["analysis"] = {
                        "df": df_analyzed,
                        "video_info": video_info,
                        "video_id": video_id,
                        "checkpoint_meta": metadata,
                        "load_report": load_report,
                        "checkpoint_path": str(checkpoint_resolved) if checkpoint_resolved else "",
                        "config_used": config_used,
                        "order": order,
                        "created_at": datetime.now().isoformat(timespec="seconds"),
                    }
                    st.rerun()

            except YouTubeAPIError as error:
                st.error(str(error))
            except Exception as error:  # pragma: no cover - defensivo
                st.error(f"Error inesperado: {error}")
                with st.expander("Detalles técnicos"):
                    st.exception(error)

if "analysis" in st.session_state:
    render_results(st.session_state["analysis"])
else:
    render_welcome()
