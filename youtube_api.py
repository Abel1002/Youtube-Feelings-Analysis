"""
Ingesta de datos desde YouTube Data API v3.

Responsabilidades:
  1. Extraer el ID de 11 caracteres de cualquier formato de URL.
  2. Paginar `commentThreads.list` hasta reunir N comentarios.
  3. Aplanar el JSON anidado a un DataFrame de pandas.
  4. Traducir los errores de Google a mensajes accionables en español.

Este módulo no depende de Streamlit: se puede testear y reutilizar desde
cualquier script o notebook.
"""

from __future__ import annotations

import re
from typing import Callable
from urllib.parse import parse_qs, urlparse

import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from config import (
    MAX_COMMENTS_LIMIT,
    YOUTUBE_API_SERVICE_NAME,
    YOUTUBE_API_VERSION,
)

_VIDEO_ID_RE = re.compile(r"^[0-9A-Za-z_-]{11}$")


class YouTubeAPIError(RuntimeError):
    """Error de YouTube con un mensaje pensado para mostrarse al usuario."""


def extract_video_id(url: str) -> str | None:
    """
    Extrae el ID de 11 caracteres de una URL de YouTube.

    Formatos soportados:
      - https://www.youtube.com/watch?v=XXXXXXXXXXX
      - https://youtu.be/XXXXXXXXXXX
      - https://www.youtube.com/embed/XXXXXXXXXXX
      - https://www.youtube.com/shorts/XXXXXXXXXXX
      - https://www.youtube.com/live/XXXXXXXXXXX
      - https://music.youtube.com/watch?v=XXXXXXXXXXX
      - el propio ID suelto: "XXXXXXXXXXX"
    """
    if not url:
        return None

    candidate = str(url).strip()
    if _VIDEO_ID_RE.match(candidate):
        return candidate

    try:
        parsed = urlparse(candidate if "://" in candidate else f"https://{candidate}")
    except ValueError:
        return None

    host = (parsed.netloc or "").lower().split(":")[0]
    if host.startswith("www."):
        host = host[4:]
    if host.startswith("m."):
        host = host[2:]

    if host in {"youtu.be", "youtube-nocookie.com"}:
        parts = [p for p in parsed.path.split("/") if p]
        if not parts:
            return None
        candidate_id = parts[1] if parts[0] == "embed" and len(parts) > 1 else parts[0]
        return candidate_id if _VIDEO_ID_RE.match(candidate_id) else None

    if host.endswith("youtube.com"):
        query = parse_qs(parsed.query)
        if "v" in query and query["v"]:
            candidate_id = query["v"][0]
            if _VIDEO_ID_RE.match(candidate_id):
                return candidate_id

        parts = [p for p in parsed.path.split("/") if p]
        if parts and parts[0] in {"embed", "shorts", "live", "v"} and len(parts) > 1:
            candidate_id = parts[1]
            if _VIDEO_ID_RE.match(candidate_id):
                return candidate_id

    # Último recurso: buscar un patrón de 11 caracteres en la cadena.
    match = re.search(r"(?:v=|/)([0-9A-Za-z_-]{11})(?:[?&/]|$)", candidate)
    return match.group(1) if match else None


def _friendly_http_error(error: HttpError) -> str:
    """Traduce los errores más frecuentes de YouTube Data API v3."""
    try:
        raw = error.content.decode("utf-8", errors="replace")
    except Exception:  # pragma: no cover - defensivo
        raw = str(error)

    lowered = raw.lower()
    if "quotaexceeded" in lowered or "quota exceeded" in lowered:
        return (
            "Se agotó la cuota diaria de YouTube Data API v3 (10.000 unidades). "
            "Espera al reinicio de la cuota o usa otra API Key."
        )
    if "commentsdisabled" in lowered or "comments disabled" in lowered:
        return (
            "El vídeo tiene los comentarios desactivados. Prueba con otro vídeo "
            "o con un vídeo cuyo creador permita comentarios."
        )
    if "videonotfound" in lowered or "video not found" in lowered:
        return "El vídeo no existe, es privado o no está disponible en tu región."
    if "keyinvalid" in lowered or "api key not valid" in lowered:
        return "La API Key no es válida. Revisa que esté bien copiada y que tenga habilitada YouTube Data API v3."
    if "accessnotconfigured" in lowered or "has not been used" in lowered:
        return (
            "La YouTube Data API v3 no está habilitada en el proyecto de Google Cloud "
            "asociado a esta API Key."
        )
    if "iprefererblocked" in lowered or "referer" in lowered:
        return (
            "La API Key tiene restricciones de referer/IP que bloquean esta petición. "
            "Revisa las restricciones en Google Cloud Console."
        )
    if error.resp.status == 403:
        return (
            "YouTube rechazó la petición (403). Causas habituales: cuota agotada, "
            "comentarios desactivados o API Key restringida."
        )
    return f"YouTube Data API devolvió un error: {raw[:400]}"


def _build_service(api_key: str):
    if not api_key or not str(api_key).strip():
        raise YouTubeAPIError(
            "Falta la API Key de YouTube. Configúrala en el panel lateral o en el archivo `.env`."
        )
    return build(
        YOUTUBE_API_SERVICE_NAME,
        YOUTUBE_API_VERSION,
        developerKey=str(api_key).strip(),
        cache_discovery=False,
    )


def fetch_video_info(api_key: str, video_id: str) -> dict:
    """Devuelve metadatos básicos del vídeo (título, canal, vistas, likes)."""
    youtube = _build_service(api_key)
    try:
        response = (
            youtube.videos()
            .list(part="snippet,statistics", id=video_id, maxResults=1)
            .execute()
        )
    except HttpError as error:
        raise YouTubeAPIError(_friendly_http_error(error)) from error

    items = response.get("items", [])
    if not items:
        raise YouTubeAPIError(
            "No se encontró el vídeo. Comprueba la URL o si el vídeo es privado."
        )

    item = items[0]
    snippet = item.get("snippet", {})
    stats = item.get("statistics", {})
    return {
        "video_id": video_id,
        "titulo": snippet.get("title", "(sin título)"),
        "canal": snippet.get("channelTitle", "(canal desconocido)"),
        "publicado": snippet.get("publishedAt", "")[:10],
        "vistas": int(stats.get("viewCount", 0) or 0),
        "likes": int(stats.get("likeCount", 0) or 0),
        "comentarios_totales": int(stats.get("commentCount", 0) or 0),
    }


def fetch_youtube_comments(
    api_key: str,
    video_id: str,
    max_comments: int = 200,
    order: str = "relevance",
    progress_callback: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """
    Extrae comentarios principales paginando `commentThreads.list`.

    Parámetros
    ----------
    api_key : str
        Clave de YouTube Data API v3.
    video_id : str
        ID de 11 caracteres del vídeo.
    max_comments : int
        Número máximo de comentarios (la API devuelve como máximo 100 por página).
    order : str
        "relevance" (sesga hacia comentarios con más interacción) o "time".
    progress_callback : callable, opcional
        Se llama con (descargados, objetivo) tras cada página.

    Devuelve
    --------
    pandas.DataFrame con columnas:
        comment_id, autor, comentario, likes, publicado, respuestas
    """
    if not _VIDEO_ID_RE.match(str(video_id or "")):
        raise YouTubeAPIError("El ID del vídeo no tiene el formato esperado (11 caracteres).")

    max_comments = max(1, min(int(max_comments), MAX_COMMENTS_LIMIT))
    if order not in {"relevance", "time"}:
        raise YouTubeAPIError("El orden debe ser 'relevance' o 'time'.")

    youtube = _build_service(api_key)
    rows: list[dict] = []
    seen_ids: set[str] = set()
    next_page_token: str | None = None

    try:
        while len(rows) < max_comments:
            batch_size = min(100, max_comments - len(rows))
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=batch_size,
                order=order,
                textFormat="plainText",
                pageToken=next_page_token,
            )
            response = request.execute()

            for item in response.get("items", []):
                thread_snippet = item.get("snippet", {})
                snippet = thread_snippet.get("topLevelComment", {}).get("snippet", {})
                comment_id = item.get("id", "")

                if comment_id and comment_id in seen_ids:
                    continue
                if comment_id:
                    seen_ids.add(comment_id)

                text = str(snippet.get("textDisplay", "") or "").strip()
                if not text:
                    # Comentarios borrados o solo con espacios: ruido puro para el tokenizador.
                    continue

                rows.append(
                    {
                        "comment_id": comment_id,
                        "autor": snippet.get("authorDisplayName", "Anónimo"),
                        "comentario": text,
                        "likes": int(snippet.get("likeCount", 0) or 0),
                        "publicado": str(snippet.get("publishedAt", ""))[:10],
                        "respuestas": int(thread_snippet.get("totalReplyCount", 0) or 0),
                    }
                )
                if len(rows) >= max_comments:
                    break

            if progress_callback is not None:
                progress_callback(len(rows), max_comments)

            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break
    except HttpError as error:
        raise YouTubeAPIError(_friendly_http_error(error)) from error

    df = pd.DataFrame(
        rows,
        columns=["comment_id", "autor", "comentario", "likes", "publicado", "respuestas"],
    )
    if not df.empty:
        df = (
            df.drop_duplicates(subset=["comment_id"])
            .sort_values("likes", ascending=False)
            .reset_index(drop=True)
        )
    return df
