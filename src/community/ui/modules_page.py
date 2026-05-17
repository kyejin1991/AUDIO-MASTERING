from __future__ import annotations

from pathlib import Path

from community.core.project_loader import read_json, write_json
from community.ui.common import list_preset_files, load_module_preset, render_project_context, save_module_preset, set_current_project
from pro.ui.viewmodels.mastering_summary import MasteringSummaryViewModel


def _analysis_dir_from_project_json(project_json: str | Path) -> Path:
    return Path(project_json).parent / "analysis"


def _load_module_draft(project_json: str | Path) -> tuple[dict, dict]:
    analysis_dir = _analysis_dir_from_project_json(project_json)
    params = read_json(analysis_dir / "module_parameter_draft.json")
    recommended = read_json(analysis_dir / "recommended_chain.json")
    return params, recommended


def _sync_chain_from_params(recommended_chain: dict, params: dict) -> dict:
    active = []
    for module in recommended_chain.get("active_chain", []):
        if params.get(module, {}).get("enabled", False):
            active.append(module)
    for module in ["AudioRestore", "Equalizer", "BassControl", "Compressor", "Exciter", "Imager", "Maximizer"]:
        if params.get(module, {}).get("enabled", False) and module not in active:
            active.append(module)
    recommended_chain["active_chain"] = active
    recommended_chain["disabled_modules"] = [name for name, cfg in params.items() if not cfg.get("enabled", False)]
    return recommended_chain


def render_modules_page(engine):
    import streamlit as st

    st.header("Advanced Modules")
    render_project_context(st)

    default_project = st.session_state.get("current_project_json", "")
    project_json = st.text_input("Project JSON Path", value=default_project, key="modules_project_json")
    if not project_json:
        return

    set_current_project(st, project_json, project_root=str(Path(project_json).parent), project_name=Path(project_json).parent.name)
    project_root = Path(project_json).parent
    params, recommended = _load_module_draft(project_json)

    preset_files = list_preset_files(project_root)
    preset_names = ["None"] + [path.stem for path in preset_files]
    selected_preset = st.selectbox("Load Preset", preset_names, index=0)
    if selected_preset != "None" and st.button("Apply Preset"):
        params = load_module_preset(next(path for path in preset_files if path.stem == selected_preset))

    st.subheader("Module Rack")
    audio_restore = params["AudioRestore"]
    eq = params["Equalizer"]
    bass = params["BassControl"]
    comp = params["Compressor"]
    exciter = params["Exciter"]
    imager = params["Imager"]
    maximizer = params["Maximizer"]

    rack_cols = st.columns(2)
    with rack_cols[0]:
        audio_restore["enabled"] = st.checkbox("Audio Restore", value=audio_restore["enabled"])
        audio_restore["amount"] = st.slider("Restore Amount", 0.0, 1.0, float(audio_restore["amount"]), 0.05)
        audio_restore["noise_suppress"] = st.slider("Noise Suppress", 0.0, 1.0, float(audio_restore["noise_suppress"]), 0.05)
        audio_restore["high_synth"] = st.slider("High Synth", 0.0, 1.0, float(audio_restore["high_synth"]), 0.05)
        audio_restore["transient_recovery"] = st.slider("Transient Recovery", 0.0, 1.0, float(audio_restore["transient_recovery"]), 0.05)

        eq["enabled"] = st.checkbox("EQ", value=eq["enabled"])
        eq["bands"][3]["gain_db"] = st.slider("Mud Cut 280Hz", -6.0, 3.0, float(eq["bands"][3]["gain_db"]), 0.1)
        eq["bands"][6]["gain_db"] = st.slider("Air Boost 10.5kHz", -3.0, 6.0, float(eq["bands"][6]["gain_db"]), 0.1)

        bass["enabled"] = st.checkbox("Bass Control", value=bass["enabled"])
        bass["sub_gain_db"] = st.slider("Sub Gain", -6.0, 6.0, float(bass["sub_gain_db"]), 0.1)
        bass["punch_amount"] = st.slider("Low Punch", 0.0, 1.0, float(bass["punch_amount"]), 0.05)

        comp["enabled"] = st.checkbox("Compressor", value=comp["enabled"])
        comp["threshold_db"] = st.slider("Comp Threshold", -30.0, -6.0, float(comp["threshold_db"]), 0.5)
        comp["ratio"] = st.slider("Comp Ratio", 1.0, 4.0, float(comp["ratio"]), 0.05)
        comp["mix"] = st.slider("Parallel Mix", 0.0, 1.0, float(comp["mix"]), 0.05)

    with rack_cols[1]:
        exciter["enabled"] = st.checkbox("Exciter", value=exciter["enabled"])
        exciter["amount"] = st.slider("Exciter Amount", 0.0, 0.5, float(exciter["amount"]), 0.01)
        exciter["bands"]["high"] = st.slider("High Harmonics", 0.0, 0.5, float(exciter["bands"]["high"]), 0.01)

        imager["enabled"] = st.checkbox("Imager", value=imager["enabled"])
        imager["mid_width"] = st.slider("Mid Width", 0.7, 1.3, float(imager["mid_width"]), 0.01)
        imager["high_width"] = st.slider("High Width", 0.7, 1.5, float(imager["high_width"]), 0.01)

        maximizer["enabled"] = st.checkbox("Maximizer", value=maximizer["enabled"])
        maximizer["target_lufs"] = st.slider("Target LUFS", -16.0, -8.0, float(maximizer["target_lufs"]), 0.1)
        maximizer["ceiling_dbtp"] = st.slider("Ceiling dBTP", -2.0, -0.2, float(maximizer["ceiling_dbtp"]), 0.05)

    st.subheader("Preset")
    preset_name = st.text_input("Preset Name", value="custom_rack")
    action_cols = st.columns(2)
    with action_cols[0]:
        if st.button("Save Preset", use_container_width=True):
            save_path = save_module_preset(project_root, preset_name, params)
            st.success(f"Saved preset: {save_path.name}")
    with action_cols[1]:
        if st.button("Re-render With Manual Rack", use_container_width=True):
            analysis_dir = _analysis_dir_from_project_json(project_json)
            synced_chain = _sync_chain_from_params(recommended, params)
            write_json(analysis_dir / "module_parameter_draft.json", params)
            write_json(analysis_dir / "recommended_chain.json", synced_chain)
            
            # Load the actual assistant report to pair with render report
            assistant_data = read_json(analysis_dir / "master_assistant.json")
            
            chain = engine.build_chain(project_json)
            render_report = engine.render(project_json)
            
            st.session_state["last_render_report"] = render_report
            
            vm = MasteringSummaryViewModel.from_reports(assistant_data, render_report)
            
            st.divider()
            st.subheader("Render Safety Status")
            r_col1, r_col2, r_col3 = st.columns(3)
            with r_col1:
                st.metric("Safety Status", vm.safety_status)
            with r_col2:
                st.metric("LUFS", f"{vm.final_lufs:.1f}" if vm.final_lufs else "N/A")
            with r_col3:
                st.metric("True Peak", f"{vm.final_true_peak:.2f}" if vm.final_true_peak else "N/A")
                
            if vm.guard_warning_count > 0 or vm.bypassed_step_count > 0:
                st.warning(f"Engine intervened: {vm.guard_warning_count} parameter corrections, {vm.bypassed_step_count} bypassed steps.")
            else:
                st.success("Render completed cleanly with no automated safety interventions.")
            
            with st.expander("Technical Render Report"):
                st.json({"chain": chain, "render": render_report})
