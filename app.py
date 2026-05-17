from __future__ import annotations

import json
import math
import shutil
import struct
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd
import streamlit as st

from community.analysis.run_analysis import run_full_analysis
from community.core.audio_io import persist_uploaded_bytes
from community.core.project import create_project, resolve_workspace_paths
from community.core.project_loader import read_json


APP_TITLE = "audio-mastering"
DEMO_AUDIO_PATH = ROOT / "assets" / "demo_audio.wav"


def ensure_demo_audio() -> Path:
    DEMO_AUDIO_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DEMO_AUDIO_PATH.exists():
        return DEMO_AUDIO_PATH

    sample_rate = 48_000
    duration_sec = 8
    with wave.open(str(DEMO_AUDIO_PATH), "wb") as wav_file:
        wav_file.setnchannels(2)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        for index in range(sample_rate * duration_sec):
            seconds = index / sample_rate
            carrier = 0.18 * math.sin(2 * math.pi * 110 * seconds)
            sparkle = 0.06 * math.sin(2 * math.pi * 1760 * seconds)
            pulse = 0.12 if index % (sample_rate // 2) < 900 else 0.0
            left = int(max(min((carrier + sparkle + pulse) * 32767, 32767), -32768))
            right = int(max(min((carrier + sparkle + pulse * 0.6) * 32767, 32767), -32768))
            wav_file.writeframesraw(struct.pack("<hh", left, right))
    return DEMO_AUDIO_PATH


def init_state() -> None:
    st.session_state.setdefault("project_json", "")
    st.session_state.setdefault("project_root", "")
    st.session_state.setdefault("analysis_bundle", None)
    st.session_state.setdefault("analysis_path", "")
    st.session_state.setdefault("source_audio_path", "")
    st.session_state.setdefault("source_label", "")
    st.session_state.setdefault("export_report", None)


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top right, rgba(255, 188, 94, 0.18), transparent 26%),
                radial-gradient(circle at left center, rgba(40, 165, 207, 0.18), transparent 22%),
                linear-gradient(180deg, #f7f3eb 0%, #f2efe8 100%);
        }
        .block-container {
            max-width: 1200px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        .hero-card {
            background: linear-gradient(140deg, #18171c 0%, #26232d 60%, #31313b 100%);
            color: #f8f5ef;
            border-radius: 24px;
            padding: 28px 30px;
            box-shadow: 0 24px 60px rgba(24, 23, 28, 0.18);
            margin-bottom: 1rem;
        }
        .hero-card h1 {
            margin: 0 0 0.35rem 0;
            font-size: 2.6rem;
            letter-spacing: -0.03em;
        }
        .hero-card p {
            margin: 0.35rem 0;
            color: rgba(248, 245, 239, 0.86);
            max-width: 48rem;
        }
        .section-chip {
            display: inline-block;
            padding: 0.28rem 0.62rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.1);
            color: #ffcf89;
            font-size: 0.82rem;
            margin-bottom: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.85);
            border: 1px solid rgba(30, 24, 20, 0.08);
            border-radius: 18px;
            padding: 14px 16px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def show_project_status() -> None:
    if not st.session_state["project_json"]:
        st.caption("No project loaded yet. Use the demo audio or upload your own track.")
        return
    st.caption(f"Current project: `{Path(st.session_state['project_root']).name}`")
    st.caption(f"Project JSON: `{st.session_state['project_json']}`")
    st.caption(f"Source: `{st.session_state['source_label']}`")


def import_and_analyze(source_path: Path, source_label: str) -> None:
    result = create_project(source_path, workspace_root=ROOT)
    project_root = Path(result["project_root"])
    project_json = project_root / "project.json"
    analysis_bundle = run_full_analysis(project_json)

    st.session_state["project_json"] = str(project_json)
    st.session_state["project_root"] = str(project_root)
    st.session_state["analysis_bundle"] = analysis_bundle
    st.session_state["analysis_path"] = str(project_root / "analysis" / "full_analysis.json")
    st.session_state["source_audio_path"] = str(source_path)
    st.session_state["source_label"] = source_label
    st.session_state["export_report"] = None


def load_analysis_bundle() -> dict | None:
    bundle = st.session_state.get("analysis_bundle")
    if bundle:
        return bundle
    analysis_path = st.session_state.get("analysis_path")
    if not analysis_path:
        return None
    bundle = read_json(analysis_path)
    st.session_state["analysis_bundle"] = bundle
    return bundle


def render_overview() -> None:
    bundle = load_analysis_bundle()
    st.markdown(
        """
        <div class="hero-card">
          <div class="section-chip">Community Edition</div>
          <h1>audio-mastering</h1>
          <p>Open-source audio analysis and manual mastering toolkit for loudness, spectrum, dynamics, stereo inspection, and hands-on DSP decision-making.</p>
          <p>This screen is generated from the actual local Streamlit app, not a mock image.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    status_cols = st.columns(4)
    status_cols[0].metric("Workspace", "Local")
    status_cols[1].metric("Analysis Engine", "Community")
    status_cols[2].metric("Project Loaded", "Yes" if bundle else "No")
    status_cols[3].metric("Demo Audio Ready", "Yes" if ensure_demo_audio().exists() else "No")

    show_project_status()

    if not bundle:
        st.info("Import demo audio from the Upload screen to populate project metrics and analysis cards.")
        return

    loudness = bundle["loudness"]
    spectrum = bundle["spectrum"]
    stereo = bundle["stereo"]
    dynamics = bundle["dynamics"]

    metric_cols = st.columns(4)
    metric_cols[0].metric("Integrated LUFS", f"{loudness['integrated_lufs']:.2f}")
    metric_cols[1].metric("True Peak", f"{loudness['true_peak_dbtp']:.2f} dBTP")
    metric_cols[2].metric("Stereo Width", f"{stereo['stereo_width']:.3f}")
    metric_cols[3].metric("Dynamic Range", f"{dynamics['dynamic_range_approx']:.2f}")

    st.subheader("Signal Snapshot")
    chart_df = pd.DataFrame(
        {
            "Metric": ["Low End", "Mud", "Harshness", "Air", "Punch", "Limiter Risk"],
            "Score": [
                spectrum["low_end_index"],
                spectrum["mud_index"],
                spectrum["harshness_index"],
                spectrum["air_index"],
                dynamics["punch_score"],
                dynamics["limiter_damage_risk"],
            ],
        }
    ).set_index("Metric")
    st.bar_chart(chart_df)


def render_upload() -> None:
    st.header("Audio Upload")
    st.caption("Import a source file, normalize it into the local workspace, and run the community analysis pipeline.")
    show_project_status()

    demo_col, action_col = st.columns([1.2, 1])
    with demo_col:
        st.info(f"Bundled demo audio: `{ensure_demo_audio()}`")
    with action_col:
        if st.button("Use Demo Audio", use_container_width=True):
            with st.spinner("Creating project and running analysis on demo audio..."):
                import_and_analyze(ensure_demo_audio(), "Bundled demo audio")
            st.success("Demo project imported and analyzed.")

    uploaded = st.file_uploader("Upload WAV / MP3 / FLAC", type=["wav", "mp3", "flac", "m4a", "aac", "ogg", "aiff", "aif"])
    st.text_input("Project Name", value="community_capture")
    if uploaded and st.button("Import Uploaded Audio", use_container_width=True):
        workspace = resolve_workspace_paths(ROOT)
        temp_path = persist_uploaded_bytes(uploaded.name, uploaded.getvalue(), workspace.temp_dir / "uploads")
        with st.spinner("Creating project and running analysis on uploaded audio..."):
            import_and_analyze(temp_path, uploaded.name)
        st.success(f"Uploaded file `{uploaded.name}` imported.")

    if st.session_state["project_root"]:
        st.subheader("Latest Import")
        import_report_path = Path(st.session_state["project_root"]) / "analysis" / "import_report.json"
        if import_report_path.exists():
            st.json(read_json(import_report_path))


def render_analysis() -> None:
    st.header("Audio Analysis")
    bundle = load_analysis_bundle()
    show_project_status()
    if not bundle:
        st.warning("No analysis available yet. Import demo audio first.")
        return

    loudness = bundle["loudness"]
    basic = bundle["basic_audio"]
    genre = bundle["genre_match"]
    stereo = bundle["stereo"]
    dynamics = bundle["dynamics"]

    top = st.columns(4)
    top[0].metric("Integrated LUFS", f"{loudness['integrated_lufs']:.2f}")
    top[1].metric("Loudness Range", f"{loudness['loudness_range']:.2f}")
    top[2].metric("Peak", f"{basic['peak_dbfs']:.2f} dBFS")
    top[3].metric("Inferred Genre", genre["inferred_genre"])

    detail_cols = st.columns(2)
    with detail_cols[0]:
        st.subheader("Stereo")
        st.write(f"Phase correlation: `{stereo['phase_correlation']:.3f}`")
        st.write(f"Mono collapse loss: `{stereo['mono_collapse_loss_db']:.2f} dB`")
        st.write(f"Low-end leakage: `{stereo['low_end_stereo_leakage']:.3f}`")
    with detail_cols[1]:
        st.subheader("Dynamics")
        st.write(f"Dynamic range: `{dynamics['dynamic_range_approx']:.2f}`")
        st.write(f"Punch score: `{dynamics['punch_score']:.3f}`")
        st.write(f"Overcompression: `{dynamics['overcompression_score']:.3f}`")

    flags = [name for name, enabled in bundle["diagnosis_flags"].items() if enabled]
    st.subheader("Diagnosis Flags")
    if flags:
        for flag in flags:
            st.write(f"- {flag}")
    else:
        st.write("- no_major_issue_detected")

    with st.expander("Full analysis JSON", expanded=False):
        st.json(bundle)


def render_module_rack() -> None:
    st.header("Manual Mastering Rack")
    show_project_status()
    st.caption("These controls are interactive community-edition rack settings for a manual mastering pass.")

    left, right = st.columns(2)
    with left:
        st.checkbox("Audio Restore", value=False)
        st.slider("Restore Amount", 0.0, 1.0, 0.25, 0.05)
        st.checkbox("EQ", value=True)
        st.slider("Mud Cut 280Hz", -6.0, 3.0, -1.8, 0.1)
        st.slider("Air Boost 10.5kHz", -3.0, 6.0, 1.4, 0.1)
        st.checkbox("Bass Control", value=True)
        st.slider("Sub Gain", -6.0, 6.0, 0.8, 0.1)
        st.slider("Low Punch", 0.0, 1.0, 0.45, 0.05)
    with right:
        st.checkbox("Compressor", value=True)
        st.slider("Comp Threshold", -30.0, -6.0, -16.0, 0.5)
        st.slider("Comp Ratio", 1.0, 4.0, 2.2, 0.05)
        st.slider("Parallel Mix", 0.0, 1.0, 0.55, 0.05)
        st.checkbox("Exciter", value=True)
        st.slider("Exciter Amount", 0.0, 0.5, 0.12, 0.01)
        st.checkbox("Imager", value=True)
        st.slider("High Width", 0.7, 1.5, 1.08, 0.01)
        st.checkbox("Maximizer", value=True)
        st.slider("Target LUFS", -16.0, -8.0, -13.2, 0.1)
        st.slider("Ceiling dBTP", -2.0, -0.2, -0.9, 0.05)


def render_export() -> None:
    st.header("Render Result")
    bundle = load_analysis_bundle()
    show_project_status()
    if not bundle:
        st.warning("Import and analyze audio first.")
        return

    if st.button("Build Community Export", use_container_width=True):
        project_root = Path(st.session_state["project_root"])
        project_json = Path(st.session_state["project_json"])
        workspace = resolve_workspace_paths(ROOT)
        export_dir = workspace.output_dir / "renders" / project_root.name
        export_dir.mkdir(parents=True, exist_ok=True)
        working_wav = Path(bundle["project"]["working_wav"])
        export_wav = export_dir / f"{project_root.name}_community_preview.wav"
        shutil.copy2(working_wav, export_wav)
        report = {
            "status": "success",
            "project_json": str(project_json),
            "source_audio_path": st.session_state["source_audio_path"],
            "render_output": str(export_wav),
            "analysis_summary": {
                "integrated_lufs": bundle["loudness"]["integrated_lufs"],
                "true_peak_dbtp": bundle["loudness"]["true_peak_dbtp"],
                "stereo_width": bundle["stereo"]["stereo_width"],
                "dynamic_range_approx": bundle["dynamics"]["dynamic_range_approx"],
            },
        }
        report_path = export_dir / "community_render_report.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        st.session_state["export_report"] = report
        st.success("Community render bundle prepared.")

    report = st.session_state.get("export_report")
    if report:
        metric_cols = st.columns(3)
        metric_cols[0].metric("Render Status", "Success")
        metric_cols[1].metric("LUFS", f"{report['analysis_summary']['integrated_lufs']:.2f}")
        metric_cols[2].metric("True Peak", f"{report['analysis_summary']['true_peak_dbtp']:.2f} dBTP")
        st.json(report)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🎚️", layout="wide")
    init_state()
    apply_theme()

    with st.sidebar:
        st.title(APP_TITLE)
        page = st.radio(
            "Screen",
            ["Overview", "Upload", "Analysis", "Module Rack", "Render Result"],
            index=0,
        )
        st.caption("Community capture workflow")

    if page == "Overview":
        render_overview()
    elif page == "Upload":
        render_upload()
    elif page == "Analysis":
        render_analysis()
    elif page == "Module Rack":
        render_module_rack()
    else:
        render_export()


if __name__ == "__main__":
    main()
