from __future__ import annotations

from pathlib import Path
import time

from ai_mastering_lab.core.audio_io import persist_uploaded_bytes, supported_upload_extensions
from ai_mastering_lab.core.youtube_video import (
    DEFAULT_YOUTUBE_OUTPUT_DIR,
    DEFAULT_YOUTUBE_TEMPLATE,
    list_visual_templates,
    recommend_visual_template,
)
from ai_mastering_lab.ui.common import as_file_url, load_genre_options, render_project_context


def _sanitize_project_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in name.strip())
    return safe.strip("_") or "batch_track"


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    total_seconds = max(0, int(round(float(seconds))))
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}h {minutes}m"
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _collect_audio_files_from_folder(folder_path: str, recursive: bool) -> list[Path]:
    root = Path(folder_path).expanduser()
    if not root.exists() or not root.is_dir():
        return []
    patterns = [f"*{suffix}" for suffix in supported_upload_extensions()]
    collected: list[Path] = []
    for pattern in patterns:
        iterator = root.rglob(pattern) if recursive else root.glob(pattern)
        collected.extend(path for path in iterator if path.is_file())
    return sorted({path.resolve() for path in collected}, key=lambda path: path.name.lower())


def _build_preview_entries(uploads, folder_files: list[Path]) -> list[dict]:
    preview_entries = []
    for item in uploads or []:
        preview_entries.append(
            {
                "source_type": "upload",
                "display_name": item.name,
                "dedupe_key": f"upload::{item.name.lower()}::{len(item.getvalue())}",
            }
        )
    for path in folder_files:
        preview_entries.append(
            {
                "source_type": "folder",
                "display_name": path.name,
                "dedupe_key": f"folder::{str(path.resolve()).lower()}",
            }
        )

    unique_entries = []
    seen = set()
    for entry in preview_entries:
        if entry["dedupe_key"] in seen:
            continue
        seen.add(entry["dedupe_key"])
        unique_entries.append(entry)
    return unique_entries


def _detected_formats(preview_entries: list[dict]) -> list[str]:
    formats = sorted({Path(entry["display_name"]).suffix.lower().lstrip(".") for entry in preview_entries if Path(entry["display_name"]).suffix})
    return formats


def _render_scan_summary(st, preview_entries: list[dict]) -> None:
    formats = _detected_formats(preview_entries)
    cols = st.columns(3)
    cols[0].metric("Detected Tracks", len(preview_entries))
    cols[1].metric("Formats", ", ".join(formats).upper() if formats else "-")
    cols[2].metric("Input Mode", "Folder First")
    with st.expander("Final Processing Order", expanded=False):
        for index, entry in enumerate(preview_entries, start=1):
            st.write(f"{index:02d}. [{entry['source_type']}] {entry['display_name']}")


def _render_batch_dashboard(st, result: dict) -> None:
    album = result.get("album", {})
    album_report = result.get("album_report", {})
    video_rendering = result.get("video_rendering", {})

    st.subheader("Batch Dashboard")
    metric_cols = st.columns(5)
    metric_cols[0].metric("Total Queued", len(album_report.get("tracks", [])))
    metric_cols[1].metric("Mastered", result.get("mastered_count", album_report.get("mastered_count", 0)))
    metric_cols[2].metric("MP4 Rendered", result.get("video_rendered_count", album_report.get("video_rendered_count", 0)))
    metric_cols[3].metric("Failed", result.get("failed_count", album_report.get("failed_count", 0)))
    metric_cols[4].metric("Current Stage", "Complete")

    button_cols = st.columns(2)
    if album.get("album_root"):
        button_cols[0].link_button("Open Batch Folder", as_file_url(album["album_root"]), use_container_width=True)
    if video_rendering.get("video_output_dir"):
        button_cols[1].link_button("Open Mastered Videos Folder", as_file_url(video_rendering["video_output_dir"]), use_container_width=True)

    for track in album_report.get("tracks", []):
        with st.container(border=True):
            st.markdown(f"**{int(track['order']):02d}. {track['source_name']}**")
            status_cols = st.columns(2)
            status_cols[0].caption(f"Mastering: `{track.get('master_status', 'unknown')}`")
            status_cols[1].caption(f"MP4: `{track.get('video_status', 'disabled')}`")
            if track.get("audio_restore_used"):
                st.caption("AudioRestore applied before mastering")
            if track.get("song_stt_status") == "success":
                st.caption("Song STT completed for this track")
            if track.get("master_error"):
                st.warning(f"Mastering error: {track['master_error']}")
            if track.get("video_error"):
                st.warning(f"Video render error: {track['video_error']}")

            link_cols = st.columns(6)
            if track.get("wav"):
                link_cols[0].link_button("Open WAV", as_file_url(track["wav"]), use_container_width=True)
            if track.get("mp3_320"):
                link_cols[1].link_button("Open MP3", as_file_url(track["mp3_320"]), use_container_width=True)
            if track.get("youtube_mp4"):
                link_cols[2].link_button("Open MP4", as_file_url(track["youtube_mp4"]), use_container_width=True)
            if track.get("lyrics_lrc"):
                link_cols[3].link_button("Open LRC", as_file_url(track["lyrics_lrc"]), use_container_width=True)
            if track.get("aligned_srt"):
                link_cols[4].link_button("Open SRT", as_file_url(track["aligned_srt"]), use_container_width=True)
            if track.get("project_root"):
                link_cols[5].link_button("Open Project Folder", as_file_url(track["project_root"]), use_container_width=True)

            with st.expander("Paths", expanded=False):
                st.code(
                    "\n".join(
                        [
                            f"WAV: {track.get('wav', '')}",
                            f"MP3: {track.get('mp3_320', '')}",
                            f"MP4: {track.get('youtube_mp4', '')}",
                            f"LRC: {track.get('lyrics_lrc', '')}",
                            f"SRT: {track.get('aligned_srt', '')}",
                            f"Project: {track.get('project_root', '')}",
                        ]
                    ),
                    language="text",
                )


def _build_visual_asset_options(genre: str) -> tuple[list[Path], int]:
    assets = list_visual_templates()
    if not assets:
        return [], 0
    recommended = recommend_visual_template(genre)
    default_index = 0
    for index, asset in enumerate(assets):
        if asset == recommended:
            default_index = index
            break
    return assets, default_index


def _resolve_batch_inputs(engine, uploads, folder_files: list[Path]) -> tuple[list[Path], list[str]]:
    resolved_inputs: list[tuple[Path, str, str]] = []
    if uploads:
        upload_dir = engine.workspace.temp_dir / "batch_uploads"
        for item in uploads:
            saved = persist_uploaded_bytes(item.name, item.getvalue(), upload_dir)
            resolved_inputs.append((saved, Path(item.name).stem, f"upload::{item.name.lower()}::{len(item.getvalue())}"))
    for path in folder_files:
        resolved_inputs.append((Path(path), Path(path).stem, f"folder::{str(Path(path).resolve()).lower()}"))

    unique_inputs: list[Path] = []
    unique_project_names: list[str] = []
    seen = set()
    for path, display_name, dedupe_key in resolved_inputs:
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        unique_inputs.append(Path(path))
        unique_project_names.append(f"{len(unique_inputs):02d}_{_sanitize_project_name(display_name)}")
    return unique_inputs, unique_project_names


def render_batch_page(engine):
    import streamlit as st

    st.header("Batch")
    render_project_context(st)
    batch_rendered_this_run = False
    st.caption("Recommended workflow: choose one music folder, master everything sequentially, and replace only the audio on one shared YouTube MP4 base video.")

    st.subheader("Folder First")
    folder_path = st.text_input("Audio Folder Path", value="", placeholder=r"D:\Music\Album")
    recursive_scan = st.checkbox("Scan subfolders", value=True)
    st.caption("This is the primary batch input. Drag-and-drop uploads still work below as a fallback.")

    st.subheader("Fallback Upload")
    uploads = st.file_uploader(
        "Drop audio files for batch mastering",
        type=[suffix.lstrip(".") for suffix in supported_upload_extensions()],
        accept_multiple_files=True,
    )

    genre_options = load_genre_options()
    batch_genre_mode = st.segmented_control("Genre Input", options=["preset", "custom"], default="preset", key="batch_genre_mode")
    if batch_genre_mode == "custom":
        genre = st.text_input("Genre", value=st.session_state.get("batch_genre_custom", "hiphop"), key="batch_genre_custom")
    else:
        default_index = genre_options.index("hiphop") if "hiphop" in genre_options else 0
        genre = st.selectbox("Genre", genre_options, index=default_index, key="batch_genre")
    style = st.selectbox("Style", ["clean", "natural", "punch", "loud"], index=2, key="batch_style")
    processing_mode = st.segmented_control(
        "Processing Mode",
        options=["mastering_only", "stt_only", "mastering_and_stt"],
        default="mastering_only",
        format_func=lambda value: {
            "mastering_only": "Mastering Only",
            "stt_only": "STT Only",
            "mastering_and_stt": "Mastering + STT",
        }[value],
        key="batch_processing_mode",
    )

    include_mp3 = st.checkbox("Export MP3 after mastering", value=True)
    run_song_stt = processing_mode == "mastering_and_stt"
    song_stt_model_size = "base"
    if processing_mode in {"stt_only", "mastering_and_stt"}:
        song_stt_model_size = st.selectbox("Song STT Model", ["tiny", "base", "small", "medium", "large-v3"], index=1, key="batch_song_stt_model_size")
    render_video = st.checkbox("Render YouTube MP4 after mastering", value=True)
    st.caption("Choose the main background visual for the final YouTube MP4. Both still images and loopable video templates are supported.")
    visual_input_mode = st.segmented_control("Main Background", options=["preset", "custom"], default="preset", key="batch_visual_input_mode")
    visual_assets, visual_default_index = _build_visual_asset_options(genre)
    if visual_input_mode == "custom":
        video_template = st.text_input("Background Visual Path", value=str(DEFAULT_YOUTUBE_TEMPLATE), key="batch_video_template_custom")
    else:
        if visual_assets:
            selected_visual = st.selectbox(
                "Visual Asset",
                visual_assets,
                index=visual_default_index,
                format_func=lambda path: path.name,
                key="batch_visual_asset",
            )
            video_template = str(selected_visual)
            st.caption(f"Recommended for `{genre}`: `{Path(video_template).name}`")
        else:
            video_template = st.text_input("Background Visual Path", value=str(DEFAULT_YOUTUBE_TEMPLATE), key="batch_video_template_fallback")
    video_output_dir = st.text_input("YouTube MP4 Output Folder", value=str(DEFAULT_YOUTUBE_OUTPUT_DIR))

    folder_files = []
    if folder_path.strip():
        root = Path(folder_path).expanduser()
        if root.exists() and root.is_dir():
            folder_files = _collect_audio_files_from_folder(folder_path, recursive_scan)
        elif st.session_state.get("last_batch_result") is None:
            st.warning(f"Folder not found yet: {folder_path}")

    preview_entries = _build_preview_entries(uploads, folder_files)
    if preview_entries:
        _render_scan_summary(st, preview_entries)

    if st.button("Run Batch Mastering", use_container_width=True):
        folder_root = Path(folder_path).expanduser() if folder_path.strip() else None
        if folder_path.strip() and (not folder_root or not folder_root.exists() or not folder_root.is_dir()):
            st.error(f"Audio folder not found: {folder_path}")
            return
        if render_video and not Path(video_template).exists():
            st.error(f"Video template not found: {video_template}")
            return

        unique_inputs, unique_project_names = _resolve_batch_inputs(engine, uploads, folder_files)
        if not unique_inputs:
            st.error("No audio files found. Choose a folder with supported audio files or add uploads.")
            return

        progress = st.progress(0.0)
        current_stage = st.empty()
        status_metrics = st.empty()
        recent_activity = st.empty()
        recent_lines: list[str] = []
        started_at = time.monotonic()
        completed_tracks: set[int] = set()
        total = len(unique_inputs)

        def progress_callback(payload: dict) -> None:
            stage = payload.get("stage", "pending")
            current = int(payload.get("current", 0))
            remaining = max(total - current, 0)
            current_name = payload.get("project_name") or Path(payload.get("input_path", "")).name or "-"
            elapsed_seconds = time.monotonic() - started_at

            if stage in {"exported", "failed"} and current > 0:
                completed_tracks.add(current)
            completed_count = len(completed_tracks)
            average_seconds_per_track = (elapsed_seconds / completed_count) if completed_count > 0 else None
            eta_seconds = (average_seconds_per_track * max(total - completed_count, 0)) if average_seconds_per_track is not None else None

            if stage == "mastering":
                progress.progress((current - 1) / max(1, total))
                current_stage.info(f"Current stage: Mastering `{current_name}`")
            elif stage == "mastered":
                recent_lines.append(f"Mastered: {Path(payload['final_output_wav']).name}")
            elif stage == "rendering_video":
                progress.progress(((current - 1) + 0.7) / max(1, total))
                current_stage.info(f"Current stage: Rendering MP4 for `{current_name}`")
            elif stage == "exported":
                progress.progress(current / max(1, total))
                recent_lines.append(f"Exported: {current_name} | master={payload.get('master_status')} | mp4={payload.get('video_status')}")
            elif stage == "failed":
                progress.progress(current / max(1, total))
                recent_lines.append(f"Failed: {current_name}")

            with status_metrics.container():
                cols = st.columns(6)
                cols[0].metric("Current", f"{min(current, total)}/{total}")
                cols[1].metric("Completed", completed_count)
                cols[2].metric("Remaining", max(total - completed_count, 0))
                cols[3].metric("Elapsed", _format_duration(elapsed_seconds))
                cols[4].metric("ETA", _format_duration(eta_seconds))
                cols[5].metric("Current Stage", stage.replace("_", " ").title())
            recent_activity.code("\n".join(recent_lines[-10:]), language="text")

        if processing_mode == "stt_only":
            with st.spinner("Running batch song STT / lyrics export..."):
                track_results = []
                for index, input_path in enumerate(unique_inputs, start=1):
                    progress_callback(
                        {
                            "stage": "mastering",
                            "current": index,
                            "total": total,
                            "input_path": str(Path(input_path).resolve()),
                            "project_name": unique_project_names[index - 1],
                        }
                    )
                    stt_result = engine.run_stt_only(
                        input_path=input_path,
                        project_name=unique_project_names[index - 1],
                        model_size=song_stt_model_size,
                    )
                    selected_report = stt_result["song_stt"].get("selected_report", {})
                    track_results.append(
                        {
                            "order": index,
                            "source_name": Path(input_path).name,
                            "project_name": unique_project_names[index - 1],
                            "project_root": stt_result["import_report"]["project_root"],
                            "master_status": "stt_only",
                            "master_error": None,
                            "audio_restore_used": False,
                            "song_stt_status": "success",
                            "song_stt_result": stt_result["song_stt"].get("comparison_report") or stt_result["song_stt"].get("transcript"),
                            "lyrics_lrc": (stt_result["song_stt"].get("lyrics_files") or {}).get("lyrics_final_lrc"),
                            "aligned_srt": (stt_result["song_stt"].get("subtitle_files") or {}).get("srt"),
                            "wav": None,
                            "mp3_320": None,
                            "youtube_mp4": None,
                            "video_status": "disabled",
                            "video_error": None,
                            "word_count": selected_report.get("word_count", 0),
                            "overall_confidence": selected_report.get("overall_confidence", 0.0),
                            "language": selected_report.get("language"),
                        }
                    )
                    progress_callback(
                        {
                            "stage": "exported",
                            "current": index,
                            "total": total,
                            "project_name": unique_project_names[index - 1],
                            "master_status": "stt_only",
                            "video_status": "disabled",
                        }
                    )
                result = {
                    "album": {"album_name": "batch_song_stt", "album_root": "", "track_count": len(track_results)},
                    "album_report": {
                        "tracks": track_results,
                        "mastered_count": 0,
                        "video_rendered_count": 0,
                        "failed_count": 0,
                    },
                    "video_rendering": {"video_output_dir": None},
                    "mastered_count": 0,
                    "video_rendered_count": 0,
                    "failed_count": 0,
                }
        else:
            with st.spinner("Running batch mastering and YouTube MP4 rendering..."):
                result = engine.run_batch(
                    unique_inputs,
                    project_names=unique_project_names,
                    genre=genre,
                    style=style,
                    include_mp3=include_mp3,
                    run_song_stt=run_song_stt,
                    song_stt_model_size=song_stt_model_size,
                    render_youtube_video=render_video,
                    video_template_path=video_template,
                    video_output_dir=video_output_dir,
                    progress_callback=progress_callback,
                )

        st.session_state["last_batch_result"] = result
        progress.progress(1.0)
        current_stage.success("Current stage: Complete")
        _render_batch_dashboard(st, result)
        batch_rendered_this_run = True
        with st.expander("Batch Report JSON", expanded=False):
            st.json(result)

    last_batch_result = st.session_state.get("last_batch_result")
    if last_batch_result and not batch_rendered_this_run:
        _render_batch_dashboard(st, last_batch_result)
