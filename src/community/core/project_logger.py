from pathlib import Path
from datetime import datetime

class ProjectLogger:
    def __init__(self, log_path: str | Path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, level: str, message: str) -> None:
        now = datetime.now().isoformat(timespec="seconds")
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(f"[{now}] [{level.upper()}] {message}\n")

    def info(self, message: str) -> None:
        self.write("info", message)

    def warning(self, message: str) -> None:
        self.write("warning", message)

    def error(self, message: str) -> None:
        self.write("error", message)

