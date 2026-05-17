from __future__ import annotations

from pathlib import Path

from ai_mastering_lab.ui.common import render_project_context, set_current_project, summarize_qc_failures


def render_export_page(engine):
    import streamlit as st

    st.header("Export")
    render_project_context(st)

    default_project = st.session_state.get("current_project_json", "")
    project_json = st.text_input("Project JSON Path", value=default_project, key="export_project_json")
    if project_json:
        set_current_project(st, project_json, project_root=str(Path(project_json).parent), project_name=Path(project_json).parent.name)

    st.subheader("Formats")
    formats = []
    cols = st.columns(3)
    if cols[0].checkbox("WAV 48kHz 24-bit", value=True):
        formats.append("wav_48k_24")
    if cols[1].checkbox("WAV 44.1kHz 24-bit", value=False):
        formats.append("wav_44k1_24")
    if cols[2].checkbox("MP3 320", value=True):
        formats.append("mp3_320")
    cols2 = st.columns(3)
    if cols2[0].checkbox("AAC 192", value=False):
        formats.append("aac_192")
    if cols2[1].checkbox("YouTube Preset", value=False):
        formats.append("youtube_preset")
    if cols2[2].checkbox("Archive Preset", value=False):
        formats.append("archive_preset")

    if project_json and st.button("Run QC + Export", use_container_width=True):
        qc = engine.run_youtube_upload_qc(project_json)
        failures = summarize_qc_failures(qc)
        export = engine.export_project(project_json, formats or ["wav_48k_24"])
        st.session_state["last_export_report"] = export

        if failures:
            st.warning("QC flagged items before export:")
            for item in failures:
                st.write(f"- {item}")
        else:
            st.success("QC passed before export.")
        st.json(export)

    report = st.session_state.get("last_export_report")
    if report:
        st.subheader("Latest Export")
        st.json(report)
