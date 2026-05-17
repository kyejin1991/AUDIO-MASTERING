from __future__ import annotations

from datetime import datetime
from pathlib import Path

from community.core.audio_io import persist_uploaded_bytes
from community.ui.common import (
    ASSISTANT_VERSIONS,
    REFINEMENT_PROFILE_OPTIONS,
    extract_project_context,
    load_genre_options,
    render_project_context,
    resolve_refinement_ui_payload,
    set_current_project,
)


def _sanitize_project_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in name.strip())
    return safe.strip("_") or "queued_track"


def _queue_report_path(engine) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = engine.workspace.reports_dir / "upload_queue"
    report_dir.mkdir(parents=True, exist_ok=True)
    return report_dir / f"{timestamp}_queue_report.json"


def _as_file_url(path_like: str | Path | None) -> str:
    if not path_like:
        return ""
    return Path(path_like).resolve().as_uri()


def _render_queue_progress(st, *, current_index: int, total: int, file_name: str) -> None:
    remaining = max(total - current_index, 0)
    metric_cols = st.columns(3)
    metric_cols[0].metric("Current", f"{current_index}/{total}")
    metric_cols[1].metric("Remaining", remaining)
    metric_cols[2].metric("Completed", current_index - 1)
    st.info(f"Now rendering: `{file_name}`")


def _render_queue_results(st, queue_report: dict) -> None:
    results = queue_report.get("results", [])
    if not results:
        return

    st.subheader("Queue Results")
    st.caption(
        f"Processed `{queue_report.get('processed_count', len(results))}` file(s) "
        f"with `{queue_report.get('assistant_version', 'legacy')}` assistant."
    )
    for item in results:
        title = f"{item['index']:02d}. {item['source_name']}"
        with st.container(border=True):
            st.markdown(f"**{title}**")
            if item.get("processing_mode") == "stt_only":
                metric_cols = st.columns(3)
                metric_cols[0].metric("Words", item.get("word_count", 0))
                metric_cols[1].metric("Confidence", f"{float(item.get('overall_confidence', 0.0)):.2f}")
                metric_cols[2].metric("Language", item.get("language") or "-")
            else:
                metric_cols = st.columns(3)
                metric_cols[0].metric("LUFS", f"{item['integrated_lufs']:.2f}")
                metric_cols[1].metric("True Peak", f"{item['true_peak_dbtp']:.2f} dBTP")
                metric_cols[2].metric("Dyn Range", f"{item['dynamic_range_approx']:.2f}")
            if item.get("audio_restore_enabled"):
                st.caption("AudioRestore applied")

            link_cols = st.columns(5)
            output_url = _as_file_url(item.get("final_output_wav"))
            report_url = _as_file_url(item.get("report_json"))
            project_url = _as_file_url(item.get("project_root"))
            lrc_url = _as_file_url(item.get("lyrics_lrc"))
            srt_url = _as_file_url(item.get("aligned_srt"))
            if output_url:
                link_cols[0].link_button("Open WAV", output_url, use_container_width=True)
            if report_url:
                link_cols[1].link_button("Open Report", report_url, use_container_width=True)
            if lrc_url:
                link_cols[2].link_button("Open LRC", lrc_url, use_container_width=True)
            if srt_url:
                link_cols[3].link_button("Open SRT", srt_url, use_container_width=True)
            if project_url:
                link_cols[4].link_button("Open Folder", project_url, use_container_width=True)

            with st.expander("Paths", expanded=False):
                st.code(
                    "\n".join(
                        [
                            f"WAV: {item.get('final_output_wav', '')}",
                            f"Report: {item.get('report_json', '')}",
                            f"LRC: {item.get('lyrics_lrc', '')}",
                            f"SRT: {item.get('aligned_srt', '')}",
                            f"Project: {item.get('project_root', '')}",
                        ]
                    ),
                    language="text",
                )


def render_upload_page(engine):
    import streamlit as st

    st.header("Main Screen")
    render_project_context(st)
    queue_rendered_this_run = False

    st.subheader("Quick Mode")
    st.caption("Beginner flow: upload one or many tracks, choose a style, and process them one at a time.")
    uploads = st.file_uploader(
        "Choose mix files",
        type=["wav", "mp3", "flac", "m4a", "aac", "ogg", "aiff", "aif"],
        accept_multiple_files=True,
    )
    upload = uploads[0] if uploads else None
    project_name = st.text_input("Project Name")
    processing_mode = st.segmented_control(
        "Processing Mode",
        options=["mastering_only", "stt_only", "mastering_and_stt"],
        default="mastering_only",
        format_func=lambda value: {
            "mastering_only": "Mastering Only",
            "stt_only": "STT Only",
            "mastering_and_stt": "Mastering + STT",
        }[value],
    )
    genre_options = load_genre_options()
    genre = st.selectbox("Genre", genre_options, index=genre_options.index("hiphop") if "hiphop" in genre_options else 0)
    style = st.segmented_control("Style", options=["clean", "punch", "loud"], default="punch")
    run_song_stt = processing_mode == "mastering_and_stt"
    song_stt_model_size = "base"
    if processing_mode in {"stt_only", "mastering_and_stt"}:
        song_stt_model_size = st.selectbox("Song STT Model", ["tiny", "base", "small", "medium", "large-v3"], index=1)
    assistant_version = st.selectbox(
        "Assistant Engine",
        ASSISTANT_VERSIONS,
        index=ASSISTANT_VERSIONS.index(st.session_state.get("assistant_version", "legacy")),
        key="assistant_version",
    )
    refinement_profile = "auto"
    low_end_cut_scale = st.session_state.get("refinement_low_end_cut_scale", 0.48)
    presence_restraint = st.session_state.get("refinement_presence_restraint", 0.72)
    air_restraint = st.session_state.get("refinement_air_restraint", 0.64)
    stereo_restraint = st.session_state.get("refinement_stereo_restraint", 0.78)
    loudness_restraint = st.session_state.get("refinement_loudness_restraint", 0.74)
    if assistant_version == "refined":
        refinement_profile = st.selectbox(
            "Refined Profile",
            REFINEMENT_PROFILE_OPTIONS,
            index=REFINEMENT_PROFILE_OPTIONS.index(st.session_state.get("refinement_profile", "auto")),
            key="refinement_profile",
        )
        with st.expander("Refined Controls", expanded=False):
            low_end_cut_scale = st.slider("Low-End Cut Scale", min_value=0.2, max_value=1.3, value=float(low_end_cut_scale), step=0.02, key="refinement_low_end_cut_scale")
            presence_restraint = st.slider("Presence Restraint", min_value=0.2, max_value=1.3, value=float(presence_restraint), step=0.02, key="refinement_presence_restraint")
            air_restraint = st.slider("Air Restraint", min_value=0.2, max_value=1.3, value=float(air_restraint), step=0.02, key="refinement_air_restraint")
            stereo_restraint = st.slider("Stereo Restraint", min_value=0.2, max_value=1.3, value=float(stereo_restraint), step=0.02, key="refinement_stereo_restraint")
            loudness_restraint = st.slider("Loudness Restraint", min_value=0.2, max_value=1.3, value=float(loudness_restraint), step=0.02, key="refinement_loudness_restraint")
            resolved = resolve_refinement_ui_payload(
                genre,
                style or "punch",
                assistant_version,
                refinement_profile,
                low_end_cut_scale,
                presence_restraint,
                air_restraint,
                stereo_restraint,
                loudness_restraint,
            )["resolved_controls"]
            st.caption(f"Resolved profile: `{resolved['profile_name']}`")
            st.caption(resolved["description"])
    status_box = st.empty()
    if uploads:
        st.caption(f"`{len(uploads)}` file(s) ready. Processing queue always runs sequentially, one file at a time, to keep CPU load stable.")
        if len(uploads) > 1:
            with st.expander("Upload Queue", expanded=False):
                for index, queued in enumerate(uploads, start=1):
                    st.write(f"{index}. {queued.name}")
    if processing_mode == "stt_only":
        st.caption("`Quick Master Now` will skip mastering and run song STT / lyrics export only.")
    elif processing_mode == "mastering_and_stt":
        st.caption("`Quick Master Now` will master first, then run song STT / lyrics export.")
    else:
        st.caption("`Quick Master Now` can take a few minutes for longer songs. Wait until the spinner finishes and a result block appears.")

    col1, col2, col3 = st.columns(3)
    with col1:
        if upload and st.button("Create Project", use_container_width=True):
            with st.spinner("Creating project and importing audio..."):
                saved = persist_uploaded_bytes(upload.name, upload.getvalue(), engine.workspace.temp_dir / "uploads")
                result = engine.create_project(saved, project_name=project_name or None)
                set_current_project(
                    st,
                    project_json=result["project_root"] + "\\project.json",
                    project_root=result["project_root"],
                    project_name=project_name or upload.name,
                )
            status_box.success("Project created. You can now run Quick Master or move to Assistant Mode.")
            st.json(result)
    with col2:
        if upload and st.button("Quick Master Now", use_container_width=True):
            from ai_mastering_lab.core.render import QuickMasterOptions
            refinement_payload = resolve_refinement_ui_payload(
                genre,
                style or "punch",
                assistant_version,
                refinement_profile,
                low_end_cut_scale,
                presence_restraint,
                air_restraint,
                stereo_restraint,
                loudness_restraint,
            )

            saved = persist_uploaded_bytes(upload.name, upload.getvalue(), engine.workspace.temp_dir / "uploads")
            if processing_mode == "stt_only":
                with st.spinner("Running song STT / lyrics export..."):
                    result = engine.run_stt_only(
                        input_path=saved,
                        project_name=project_name or None,
                        model_size=song_stt_model_size,
                    )
                set_current_project(
                    st,
                    project_json=result["project_json"],
                    project_root=result["import_report"]["project_root"],
                    project_name=project_name or upload.name,
                )
                status_box.success("STT only flow finished. Lyrics and subtitle exports are ready.")
                st.json(result["song_stt"])
            else:
                with st.spinner(f"Running quick master ({style or 'punch'})... this can take a few minutes."):
                    result = engine.run_quick_master(
                        input_path=saved,
                        project_name=project_name or None,
                        options=QuickMasterOptions(
                            genre=genre,
                            style=style or "punch",
                            assistant_version=assistant_version,
                            refinement_profile=refinement_payload["refinement_profile"],
                            refinement_overrides=refinement_payload["refinement_overrides"],
                            run_song_stt=run_song_stt,
                            song_stt_model_size=song_stt_model_size,
                        ),
                    )
                context = extract_project_context(result)
                set_current_project(st, context["project_json"], context["project_root"], context["project_name"])
                st.session_state["last_suite_report"] = result["research_report"]
                status_box.success("Quick master finished. Scroll down for the report, or open Assistant / Export for the next step.")
                if result["assistant"].get("refinement_controls"):
                    st.info(
                        f"Refined profile `{result['assistant']['refinement_controls']['profile_name']}` applied: "
                        f"{result['assistant']['refinement_controls']['description']}"
                    )
                if result["assistant"]["module_parameter_draft"].get("AudioRestore", {}).get("enabled"):
                    st.warning(
                        "AudioRestore engaged for this track. "
                        "The source looked dull or noisy enough to run spectral repair before the main mastering chain."
                    )
                if result.get("song_stt"):
                    st.success("Song STT finished. Lyrics and subtitle exports were generated for this track.")
                st.json(result["research_report"])
    with col3:
        if upload and st.button("Quick Master 3 Styles", use_container_width=True):
            with st.spinner("Rendering clean, punch, and loud versions... this can take a while."):
                saved = persist_uploaded_bytes(upload.name, upload.getvalue(), engine.workspace.temp_dir / "uploads")
                result = engine.run_quick_master_suite(
                    input_path=saved,
                    project_name=project_name or None,
                    genre=genre,
                )
            set_current_project(
                st,
                project_json=result["import_report"]["project_root"] + "\\project.json",
                project_root=result["import_report"]["project_root"],
                project_name=project_name or upload.name,
            )
            st.session_state["last_suite_report"] = result["suite_report"]
            status_box.success("Quick Master 3 Styles finished. Compare the outputs in the report below.")
            st.json(result["suite_report"])

    if uploads and len(uploads) > 1 and st.button("Process Upload Queue Sequentially", use_container_width=True):
        from ai_mastering_lab.core.render import QuickMasterOptions
        import json

        refinement_payload = resolve_refinement_ui_payload(
            genre,
            style or "punch",
            assistant_version,
            refinement_profile,
            low_end_cut_scale,
            presence_restraint,
            air_restraint,
            stereo_restraint,
            loudness_restraint,
        )
        progress = st.progress(0.0)
        queue_status = st.empty()
        queue_metrics = st.empty()
        queue_results = []
        total = len(uploads)

        with st.spinner(f"Processing {total} queued file(s) one at a time..."):
            for index, queued in enumerate(uploads, start=1):
                queue_status.info(f"[{index}/{total}] Processing `{queued.name}`")
                with queue_metrics.container():
                    _render_queue_progress(st, current_index=index, total=total, file_name=queued.name)
                saved = persist_uploaded_bytes(queued.name, queued.getvalue(), engine.workspace.temp_dir / "uploads")
                inferred_name = _sanitize_project_name(Path(queued.name).stem)
                if processing_mode == "stt_only":
                    result = engine.run_stt_only(
                        input_path=saved,
                        project_name=f"{index:03d}_{inferred_name}",
                        model_size=song_stt_model_size,
                    )
                    set_current_project(st, result["project_json"], result["import_report"]["project_root"], f"{index:03d}_{inferred_name}")
                    song_stt_result = result["song_stt"]
                    selected_report = song_stt_result.get("selected_report", {})
                    queue_results.append(
                        {
                            "index": index,
                            "source_name": queued.name,
                            "processing_mode": "stt_only",
                            "project_root": result["import_report"]["project_root"],
                            "final_output_wav": None,
                            "report_json": song_stt_result.get("comparison_report") or song_stt_result.get("transcript"),
                            "word_count": selected_report.get("word_count", 0),
                            "overall_confidence": selected_report.get("overall_confidence", 0.0),
                            "language": selected_report.get("language"),
                            "lyrics_lrc": (song_stt_result.get("lyrics_files") or {}).get("lyrics_final_lrc"),
                            "aligned_srt": (song_stt_result.get("subtitle_files") or {}).get("srt"),
                        }
                    )
                else:
                    result = engine.run_quick_master(
                        input_path=saved,
                        project_name=f"{index:03d}_{inferred_name}",
                        options=QuickMasterOptions(
                            genre=genre,
                            style=style or "punch",
                            assistant_version=assistant_version,
                            refinement_profile=refinement_payload["refinement_profile"],
                            refinement_overrides=refinement_payload["refinement_overrides"],
                            run_song_stt=run_song_stt,
                            song_stt_model_size=song_stt_model_size,
                        ),
                    )
                    context = extract_project_context(result)
                    set_current_project(st, context["project_json"], context["project_root"], context["project_name"])
                    final_analysis = result["render"]["final_master_analysis"]
                    queue_results.append(
                        {
                            "index": index,
                            "source_name": queued.name,
                            "processing_mode": processing_mode,
                            "project_root": result["import_report"]["project_root"],
                            "final_output_wav": result["render"]["final_output_wav"],
                            "report_json": result["research_report"]["files"]["json"],
                            "integrated_lufs": final_analysis["loudness"]["integrated_lufs"],
                            "true_peak_dbtp": final_analysis["loudness"]["true_peak_dbtp"],
                            "dynamic_range_approx": final_analysis["dynamics"]["dynamic_range_approx"],
                            "audio_restore_enabled": bool(result["assistant"]["module_parameter_draft"].get("AudioRestore", {}).get("enabled")),
                            "assistant_version": assistant_version,
                            "refinement_profile": result["assistant"].get("refinement_controls", {}).get("profile_name"),
                            "song_stt_status": "success" if result.get("song_stt") else "disabled",
                            "lyrics_lrc": (result.get("song_stt") or {}).get("lyrics_files", {}).get("lyrics_final_lrc"),
                            "aligned_srt": (result.get("song_stt") or {}).get("subtitle_files", {}).get("srt"),
                        }
                    )
                progress.progress(index / total)

        queue_report = {
            "task": "Quick Upload Queue",
            "status": "success",
            "processed_count": len(queue_results),
            "genre": genre,
            "style": style or "punch",
            "processing_mode": processing_mode,
            "assistant_version": assistant_version,
            "refinement_profile": refinement_payload["resolved_controls"]["profile_name"] if refinement_payload["resolved_controls"] else None,
            "results": queue_results,
        }
        report_path = _queue_report_path(engine)
        report_path.write_text(json.dumps(queue_report, indent=2, ensure_ascii=False), encoding="utf-8")
        st.session_state["last_queue_report"] = queue_report
        queue_status.success(f"Queue finished. {len(queue_results)} file(s) processed sequentially.")
        with queue_metrics.container():
            metric_cols = st.columns(3)
            metric_cols[0].metric("Current", f"{total}/{total}")
            metric_cols[1].metric("Remaining", 0)
            metric_cols[2].metric("Completed", total)
        status_box.success(f"Queue report saved: {report_path}")
        _render_queue_results(st, queue_report)
        queue_rendered_this_run = True
        with st.expander("Queue Report JSON", expanded=False):
            st.json(queue_report)

    last_queue_report = st.session_state.get("last_queue_report")
    if last_queue_report and not queue_rendered_this_run:
        _render_queue_results(st, last_queue_report)

    st.subheader("Mode Guide")
    cols = st.columns(3)
    cols[0].info("Quick Mode\n\nUpload one or many files. Queue processing always runs one file at a time.")
    cols[1].info("Assistant Mode\n\nSee diagnosis, reasons, and suggested chain.")
    cols[2].info("Research / Compare\n\nReference match, meters, QC, and exports.")
