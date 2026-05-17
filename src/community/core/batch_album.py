from __future__ import annotations

from pathlib import Path
import shutil
from typing import Callable

from community.core.report import write_json
from community.core.youtube_video import render_mastered_video
from pro.qc.codec_preview import ffmpeg_convert


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def _successful_records(track_records: list[dict]) -> list[dict]:
    return [record for record in track_records if record.get("master_status") == "success" and record.get("run")]


def build_album_project_payload(album: dict, track_records: list[dict]) -> dict:
    tracks = []
    source_brightness = []
    source_low_end = []
    source_lufs = []
    for index, record in enumerate(track_records, start=1):
        run = record.get("run")
        analysis = run["analysis"] if run else {}
        spectrum = analysis.get("spectrum", {})
        loudness = analysis.get("loudness", {})
        if spectrum and loudness:
            source_brightness.append(float(spectrum["brightness_index"]))
            source_low_end.append(float(spectrum["low_end_index"]))
            source_lufs.append(float(loudness["integrated_lufs"]))
        tracks.append(
            {
                "order": index,
                "source_path": str(Path(record["source_path"]).resolve()),
                "source_name": record["source_name"],
                "project_root": record.get("project_root"),
                "project_name": record["project_name"],
                "master_status": record["master_status"],
                "master_error": record.get("master_error"),
                "source_brightness_index": spectrum.get("brightness_index"),
                "source_low_end_index": spectrum.get("low_end_index"),
                "source_integrated_lufs": loudness.get("integrated_lufs"),
            }
        )

    success_count = sum(1 for track in tracks if track["master_status"] == "success")
    payload = {
        "task": "Task 070 - Batch Import",
        "status": "success" if success_count == len(tracks) else ("partial_success" if success_count > 0 else "failed"),
        "album_name": album["album_name"],
        "album_root": album["album_root"],
        "track_count": len(tracks),
        "successful_track_count": success_count,
        "failed_track_count": len(tracks) - success_count,
        "track_order": [track["project_name"] for track in tracks],
        "tracks": tracks,
        "source_average_tone": {
            "brightness_index": round(_mean(source_brightness), 6) if source_brightness else None,
            "low_end_index": round(_mean(source_low_end), 6) if source_low_end else None,
            "integrated_lufs": round(_mean(source_lufs), 6) if source_lufs else None,
        },
    }
    return payload


def build_album_consistency_summary(album: dict, track_records: list[dict]) -> dict:
    success_records = _successful_records(track_records)
    final_lufs = []
    final_brightness = []
    final_low_end = []
    final_dynamic_range = []
    final_crest = []

    for record in success_records:
        final_analysis = record["run"]["render"]["final_master_analysis"]
        loudness = final_analysis["loudness"]
        spectrum = final_analysis["spectrum"]
        dynamics = final_analysis["dynamics"]

        final_lufs.append(float(loudness["integrated_lufs"]))
        final_brightness.append(float(spectrum["brightness_index"]))
        final_low_end.append(float(spectrum["low_end_index"]))
        final_dynamic_range.append(float(dynamics["dynamic_range_approx"]))
        final_crest.append(float(dynamics["crest_factor_db"]))

    averages = {
        "integrated_lufs": round(_mean(final_lufs), 6) if final_lufs else None,
        "brightness_index": round(_mean(final_brightness), 6) if final_brightness else None,
        "low_end_index": round(_mean(final_low_end), 6) if final_low_end else None,
        "dynamic_range_approx": round(_mean(final_dynamic_range), 6) if final_dynamic_range else None,
        "crest_factor_db": round(_mean(final_crest), 6) if final_crest else None,
    }

    tracks = []
    for index, record in enumerate(track_records, start=1):
        if record.get("master_status") != "success" or not record.get("run"):
            tracks.append(
                {
                    "order": index,
                    "project_name": record["project_name"],
                    "project_root": record.get("project_root"),
                    "master_status": record["master_status"],
                    "master_error": record.get("master_error"),
                    "correction_amount": None,
                    "outlier_flags": [],
                }
            )
            continue

        final_analysis = record["run"]["render"]["final_master_analysis"]
        loudness = final_analysis["loudness"]
        spectrum = final_analysis["spectrum"]
        dynamics = final_analysis["dynamics"]
        correction = {
            "lufs_delta_to_average": round(float(loudness["integrated_lufs"] - averages["integrated_lufs"]), 6),
            "brightness_delta_to_average": round(float(spectrum["brightness_index"] - averages["brightness_index"]), 6),
            "low_end_delta_to_average": round(float(spectrum["low_end_index"] - averages["low_end_index"]), 6),
            "dynamic_range_delta_to_average": round(float(dynamics["dynamic_range_approx"] - averages["dynamic_range_approx"]), 6),
            "crest_factor_delta_to_average": round(float(dynamics["crest_factor_db"] - averages["crest_factor_db"]), 6),
        }
        tracks.append(
            {
                "order": index,
                "project_name": record["project_name"],
                "project_root": record.get("project_root"),
                "final_output_wav": record["run"]["render"]["final_output_wav"],
                "integrated_lufs": loudness["integrated_lufs"],
                "brightness_index": spectrum["brightness_index"],
                "low_end_index": spectrum["low_end_index"],
                "dynamic_range_approx": dynamics["dynamic_range_approx"],
                "crest_factor_db": dynamics["crest_factor_db"],
                "master_status": "success",
                "correction_amount": correction,
                "outlier_flags": [
                    name
                    for name, value in correction.items()
                    if abs(value) >= {
                        "lufs_delta_to_average": 1.0,
                        "brightness_delta_to_average": 0.04,
                        "low_end_delta_to_average": 0.04,
                        "dynamic_range_delta_to_average": 0.75,
                        "crest_factor_delta_to_average": 1.0,
                    }[name]
                ],
            }
        )

    success_count = len(success_records)
    return {
        "task": "Task 071 - Album Consistency Engine",
        "status": "success" if success_count == len(track_records) else ("partial_success" if success_count > 0 else "failed"),
        "album_name": album["album_name"],
        "album_root": album["album_root"],
        "successful_track_count": success_count,
        "failed_track_count": len(track_records) - success_count,
        "average_master_profile": averages,
        "tracks": tracks,
    }


def export_album_outputs(
    album: dict,
    track_records: list[dict],
    include_mp3: bool = True,
    include_song_stt: bool = False,
    render_youtube_video: bool = False,
    video_template_path: str | Path | None = None,
    video_output_dir: str | Path | None = None,
    progress_callback: Callable[[dict], None] | None = None,
) -> dict:
    album_root = Path(album["album_root"])
    export_dir = album_root / "output"
    export_dir.mkdir(parents=True, exist_ok=True)
    if video_output_dir:
        Path(video_output_dir).mkdir(parents=True, exist_ok=True)

    exports = []
    video_exports = []
    total_tracks = len(track_records)

    for index, record in enumerate(track_records, start=1):
        entry = {
            "order": index,
            "source_name": record["source_name"],
            "project_name": record["project_name"],
            "project_root": record.get("project_root"),
            "master_status": record["master_status"],
            "master_error": record.get("master_error"),
            "audio_restore_used": False,
            "song_stt_status": "disabled",
            "song_stt_result": None,
            "lyrics_lrc": None,
            "aligned_srt": None,
            "video_status": "not_started" if render_youtube_video else "disabled",
            "video_error": None,
            "wav": None,
            "mp3_320": None,
            "youtube_mp4": None,
            "youtube_preset": {
                "target_lufs": -14.0,
                "ceiling_dbtp": -1.0,
            },
        }

        if record.get("master_status") != "success" or not record.get("run"):
            exports.append(entry)
            if progress_callback:
                progress_callback(
                    {
                        "stage": "failed",
                        "current": index,
                        "total": total_tracks,
                        "project_name": record["project_name"],
                        "master_status": record["master_status"],
                        "master_error": record.get("master_error"),
                    }
                )
            continue

        source_name = Path(record["project_name"]).stem
        wav_out = export_dir / f"{index:02d}_{source_name}_master.wav"
        shutil.copy2(record["run"]["render"]["final_output_wav"], wav_out)
        entry["wav"] = str(wav_out)
        entry["audio_restore_used"] = bool(
            record["run"]["assistant"]["module_parameter_draft"].get("AudioRestore", {}).get("enabled")
        )
        song_stt_result = record["run"].get("song_stt")
        if include_song_stt:
            if song_stt_result:
                entry["song_stt_status"] = "success"
                entry["song_stt_result"] = song_stt_result.get("comparison_report") or song_stt_result.get("transcript")
                entry["lyrics_lrc"] = (song_stt_result.get("lyrics_files") or {}).get("lyrics_final_lrc")
                entry["aligned_srt"] = (song_stt_result.get("subtitle_files") or {}).get("srt")
            else:
                entry["song_stt_status"] = "failed"

        if include_mp3:
            try:
                mp3_out = export_dir / f"{index:02d}_{source_name}_master.mp3"
                ffmpeg_convert(wav_out, mp3_out, "mp3_320")
                entry["mp3_320"] = str(mp3_out)
            except Exception as exc:
                entry["master_status"] = "partial_success"
                entry["master_error"] = str(exc)

        if render_youtube_video and video_template_path and video_output_dir:
            if progress_callback:
                progress_callback(
                    {
                        "stage": "rendering_video",
                        "current": index,
                        "total": total_tracks,
                        "project_name": record["project_name"],
                        "wav": str(wav_out),
                    }
                )
            try:
                video_result = render_mastered_video(
                    wav_out,
                    video_template_path=video_template_path,
                    output_dir=video_output_dir,
                    output_name=f"{wav_out.stem}.mp4",
                )
                entry["youtube_mp4"] = video_result["output_path"]
                entry["video_status"] = "success"
                video_exports.append(
                    {
                        "order": index,
                        "project_name": record["project_name"],
                        "status": "success",
                        "output_path": video_result["output_path"],
                        "mp4": video_result["output_path"],
                        "template_kind": video_result.get("template_kind"),
                        "video_template_path": str(video_template_path),
                    }
                )
            except Exception as exc:
                entry["video_status"] = "failed"
                entry["video_error"] = str(exc)
                video_exports.append(
                    {
                        "order": index,
                        "project_name": record["project_name"],
                        "status": "failed",
                        "output_path": None,
                        "mp4": None,
                        "template_kind": None,
                        "video_template_path": str(video_template_path),
                        "error": str(exc),
                    }
                )

        exports.append(entry)
        if progress_callback:
            progress_callback(
                {
                    "stage": "exported",
                    "current": index,
                    "total": total_tracks,
                    "project_name": record["project_name"],
                    "master_status": entry["master_status"],
                    "video_status": entry["video_status"],
                    "wav": entry["wav"],
                    "mp3_320": entry["mp3_320"],
                    "song_stt_status": entry["song_stt_status"],
                    "lyrics_lrc": entry["lyrics_lrc"],
                    "youtube_mp4": entry["youtube_mp4"],
                }
            )

    mastered_count = sum(1 for item in exports if item["master_status"] in {"success", "partial_success"})
    video_rendered_count = sum(1 for item in exports if item["video_status"] == "success")
    failed_count = sum(1 for item in exports if item["master_status"] == "failed" or item["video_status"] == "failed")
    report = {
        "task": "Task 072 - Batch Export",
        "status": "success" if failed_count == 0 else ("partial_success" if mastered_count > 0 else "failed"),
        "album_name": album["album_name"],
        "album_root": album["album_root"],
        "export_dir": str(export_dir),
        "naming_rule": "NN_projectname_master.ext",
        "mastered_count": mastered_count,
        "video_rendered_count": video_rendered_count,
        "failed_count": failed_count,
        "video_rendering": {
            "enabled": bool(render_youtube_video),
            "video_template_path": str(video_template_path) if video_template_path else None,
            "video_output_dir": str(video_output_dir) if video_output_dir else None,
            "rendered_count": video_rendered_count,
        },
        "song_stt": {
            "enabled": bool(include_song_stt),
        },
        "tracks": exports,
    }
    write_json(album_root / "album_report.json", report)
    if render_youtube_video:
        write_json(
            album_root / "album_video_report.json",
            {
                "task": "YouTube Video Render",
                "status": "success" if failed_count == 0 else ("partial_success" if video_rendered_count > 0 else "failed"),
                "album_name": album["album_name"],
                "album_root": album["album_root"],
                "video_template_path": str(video_template_path) if video_template_path else None,
                "video_output_dir": str(video_output_dir) if video_output_dir else None,
                "tracks": video_exports,
            },
        )
    return report

