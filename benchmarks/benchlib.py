"""Shared helpers for OpenMeet's benchmarks: statistics, environment, servers, audio.

Standard library and numpy only, so every benchmark (and the smoke tests)
can import it without the sidecar's packages.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import io
import json
import math
import os
import platform
import subprocess
import sys
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path
from typing import Any, Iterator, Sequence

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
LOCAL_AI = REPO / "local-ai"
RESULTS = HERE / "results"

# Ports reserved for benchmark servers, so they never clash with a sidecar
# (8765) or a room (8443) someone is using.
SIDECAR_PORT = int(os.environ.get("OPENMEET_BENCH_SIDECAR_PORT", "8795"))
ROOM_PORT = int(os.environ.get("OPENMEET_BENCH_ROOM_PORT", "8463"))
OLLAMA_URL = os.environ.get("OPENMEET_BENCH_OLLAMA", "http://127.0.0.1:11434")


# --- statistics ---------------------------------------------------------------


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile (numpy's default), q in [0, 100]."""
    xs = sorted(float(v) for v in values)
    if not xs:
        return float("nan")
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def summary(values: Sequence[float], digits: int = 1) -> dict[str, Any]:
    """n, median, p90, p95, mean, min and max of a list of measurements."""
    xs = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not xs:
        return {"n": 0}
    r = lambda v: round(v, digits)  # noqa: E731
    return {"n": len(xs), "median": r(percentile(xs, 50)), "p90": r(percentile(xs, 90)), "p95": r(percentile(xs, 95)),
            "mean": r(sum(xs) / len(xs)), "min": r(min(xs)), "max": r(max(xs))}


def rate(k: int, n: int) -> float | None:
    return None if n == 0 else k / n


def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    """95% Wilson score interval of a proportion k/n."""
    if n == 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


# --- environment --------------------------------------------------------------


def _cpu_name() -> str:
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _ram_gb() -> float | None:
    if sys.platform == "win32":
        import ctypes

        class Status(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        s = Status()
        s.dwLength = ctypes.sizeof(Status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)):
            return round(s.ullTotalPhys / 2**30, 1)
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except (ValueError, OSError, AttributeError):
        return None


def free_ram_gb() -> float | None:
    if sys.platform == "win32":
        import ctypes

        class Status(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("pad", ctypes.c_ulonglong * 5)]

        s = Status()
        s.dwLength = ctypes.sizeof(Status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)):
            return round(s.ullAvailPhys / 2**30, 1)
    return None


def _gpu() -> str | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else None
    except (OSError, subprocess.SubprocessError, IndexError):
        return None


def package_versions(names: Sequence[str]) -> dict[str, str | None]:
    from importlib import metadata

    out: dict[str, str | None] = {}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def ollama_version(url: str = OLLAMA_URL) -> str | None:
    try:
        return get_json(f"{url}/api/version", timeout=5).get("version")
    except Exception:  # noqa: BLE001 - Ollama not running
        return None


def environment(packages: Sequence[str] = ()) -> dict[str, Any]:
    """Hardware, software and date of a run, stored with every result."""
    return {
        "date": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "cpu": _cpu_name(),
        "logical_cpus": os.cpu_count(),
        "ram_gb": _ram_gb(),
        "gpu": _gpu(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "packages": package_versions(packages),
        "repo_commit": git_commit(),
    }


# --- results --------------------------------------------------------------------


def write_json(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return path


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --- HTTP -------------------------------------------------------------------------


def get_json(url: str, timeout: float = 30) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def post_json(url: str, body: Any, timeout: float = 600) -> Any:
    request = urllib.request.Request(url, json.dumps(body).encode("utf-8"), {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def post_bytes(url: str, data: bytes, content_type: str, timeout: float = 600) -> Any:
    request = urllib.request.Request(url, data, {"Content-Type": content_type})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def multipart(fields: list[tuple[str, str]], files: list[tuple[str, str, bytes, str]]) -> tuple[bytes, str]:
    """A multipart/form-data body: `fields` (name, value), `files` (name, filename, data, type)."""
    boundary = "----openmeet-bench-" + os.urandom(8).hex()
    out = io.BytesIO()
    for name, value in fields:
        out.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode())
    for name, filename, data, ctype in files:
        out.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\n"
                  f"Content-Type: {ctype}\r\n\r\n".encode())
        out.write(data)
        out.write(b"\r\n")
    out.write(f"--{boundary}--\r\n".encode())
    return out.getvalue(), f"multipart/form-data; boundary={boundary}"


def wait_for(url: str, timeout: float, process: subprocess.Popen | None = None) -> None:
    """Polls `url` until it answers 200, the process exits, or `timeout` passes."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"server exited with code {process.returncode} before it was ready")
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            pass
        time.sleep(0.5)
    raise TimeoutError(f"{url} not ready after {timeout:.0f} s")


@contextlib.contextmanager
def server(args: list[str], ready_url: str, log_path: Path, env: dict[str, str] | None = None,
           cwd: Path = LOCAL_AI, timeout: float = 300) -> Iterator[subprocess.Popen]:
    """Runs a Python server (sidecar or room) for the duration of a block, then stops it."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    full_env = {**os.environ, **(env or {}), "PYTHONUNBUFFERED": "1"}
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, *args], cwd=str(cwd), env=full_env, stdout=log, stderr=subprocess.STDOUT)
        try:
            wait_for(ready_url, timeout, process)
            yield process
        finally:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=20)


def sidecar_env(device: str | None = None, cuda_libs: Path | None = None, model_dir: Path | None = None) -> dict[str, str]:
    """Environment for a sidecar subprocess: Whisper device, model folder, optional CUDA DLL folders on PATH."""
    env: dict[str, str] = {}
    if device:
        env["OPENMEET_WHISPER_DEVICE"] = device
    if model_dir:
        env["OPENMEET_MODEL_DIR"] = str(model_dir)
    if cuda_libs:
        bins = sorted(str(p) for p in Path(cuda_libs).glob("nvidia/*/bin") if p.is_dir())
        if not bins:
            raise FileNotFoundError(f"no nvidia/*/bin folders under {cuda_libs}")
        env["PATH"] = os.pathsep.join(bins + [os.environ.get("PATH", "")])
    return env


# --- audio -------------------------------------------------------------------------


def wav_bytes(samples: Sequence[float] | Any, rate: int = 16000) -> bytes:
    """Mono float samples in [-1, 1] as 16-bit PCM WAV bytes."""
    import numpy as np

    pcm = (np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0) * 32767.0).astype("<i2")
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return out.getvalue()


def wav_samples(path: Path) -> Any:
    """A 16 kHz mono 16-bit WAV file as float32 samples in [-1, 1]."""
    import numpy as np

    with wave.open(io.BytesIO(Path(path).read_bytes()), "rb") as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (16000, 1, 2):
            raise ValueError(f"{path}: need 16 kHz mono 16-bit PCM")
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0


def wav_duration(data: bytes) -> float:
    with wave.open(io.BytesIO(data), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def load_wav(path: Path) -> bytes:
    data = Path(path).read_bytes()
    if data[:4] != b"RIFF":
        raise ValueError(f"{path} is not a WAV file")
    return data


def jpeg_frame(path: Path, max_side: int = 640, quality: int = 80) -> bytes:
    """An image as the room's host page sends video frames: at most 640 px on the long side, JPEG quality 0.8."""
    from PIL import Image

    image = Image.open(path).convert("RGB")
    scale = min(1.0, max_side / max(image.size))
    if scale < 1.0:
        image = image.resize((round(image.width * scale), round(image.height * scale)))
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=quality)
    return out.getvalue()


class FrameFeed:
    """Posts one frame at a time to a gaze endpoint at `fps`, like the host page (never two in flight)."""

    def __init__(self, url: str, frame: bytes, fps: float = 10.0) -> None:
        import threading

        self.url, self.frame, self.period = url, frame, 1.0 / fps
        self.latencies_ms: list[float] = []
        self.face_found = 0
        self.errors = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="frame-feed")

    def _run(self) -> None:
        while not self._stop.is_set():
            started = time.perf_counter()
            try:
                out = post_bytes(self.url, self.frame, "image/jpeg", timeout=30)
                self.latencies_ms.append((time.perf_counter() - started) * 1000.0)
                self.face_found += bool(out.get("face_found"))
            except Exception:  # noqa: BLE001 - counted, the feed goes on
                self.errors += 1
            self._stop.wait(max(0.0, self.period - (time.perf_counter() - started)))

    def __enter__(self) -> "FrameFeed":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=10)


def tmp_dir() -> Path:
    """Where benchmark logs go: OPENMEET_BENCH_TMP, else the system temp folder."""
    import tempfile

    folder = Path(os.environ.get("OPENMEET_BENCH_TMP") or Path(tempfile.gettempdir()) / "openmeet-bench")
    folder.mkdir(parents=True, exist_ok=True)
    return folder
