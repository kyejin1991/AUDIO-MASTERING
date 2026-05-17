import json
import os
import subprocess
import textwrap


def run_song_stt_bridge(audio_path, vocals_path=None, mastered_path=None, config=None,
                        stt_repo_root=r"D:\SING VIDEO MACHINE\STT", output_dir=None):
    """
    Bridge adapter for calling the separate STT project from the mastering app.
    It uses a subprocess to avoid package-name collisions between the two repos.
    """
    if not os.path.isdir(stt_repo_root):
        raise FileNotFoundError(f"STT repo root not found: {stt_repo_root}")

    bridge_output_dir = output_dir or os.path.join(
        os.path.dirname(audio_path),
        "analysis",
        "song_stt",
    )
    os.makedirs(bridge_output_dir, exist_ok=True)

    payload = {
        "audio_path": audio_path,
        "vocals_path": vocals_path,
        "mastered_path": mastered_path,
        "config": config or {},
        "output_dir": bridge_output_dir,
    }
    payload_json = json.dumps(payload)

    code = textwrap.dedent(
        """
        import json
        from community.pipeline.song_stt_bridge import run_song_stt

        payload = json.loads(r'''__PAYLOAD__''')
        result = run_song_stt(
            payload["audio_path"],
            vocals_path=payload.get("vocals_path"),
            mastered_path=payload.get("mastered_path"),
            config=payload.get("config") or {},
            output_dir=payload["output_dir"],
        )
        print(json.dumps(result, ensure_ascii=False))
        """
    ).replace("__PAYLOAD__", payload_json.replace("\\", "\\\\"))

    result = subprocess.run(
        ["python", "-c", code],
        cwd=stt_repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Song STT bridge failed with exit code {result.returncode}: {result.stderr.strip()}"
        )

    result_path = os.path.join(bridge_output_dir, "song_stt_result.json")
    if os.path.exists(result_path):
        with open(result_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(result.stdout.strip())

