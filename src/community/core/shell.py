import subprocess
from dataclasses import dataclass

@dataclass
class CommandResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str

def run_command(command: list[str], timeout: int | None = None) -> CommandResult:
    proc = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )
    result = CommandResult(command=command, returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
    if proc.returncode != 0:
        cmd = " ".join(command)
        raise RuntimeError(f"Command failed ({proc.returncode}): {cmd}\n{proc.stderr[-6000:]}")
    return result

