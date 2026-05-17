from __future__ import annotations

from ai_mastering_lab.core.analyzer import load_analysis_bundle


def render_meter_page(engine):
    import streamlit as st

    st.header("Meters")
    project_json = st.text_input("Project JSON path", key="meters_project_json")
    reference_path = st.text_input("Optional Reference Path", key="meters_reference_path")
    duration_sec = st.slider("Preview Duration", 2.0, 12.0, 6.0, 0.5)

    col1, col2 = st.columns(2)
    with col1:
        if project_json and st.button("Load Analysis"):
            st.json(load_analysis_bundle(project_json))
    with col2:
        if project_json and st.button("Run Compare Suite"):
            st.json(
                engine.run_compare_suite(
                    project_json,
                    reference_path=reference_path or None,
                    duration_sec=duration_sec,
                )
            )
