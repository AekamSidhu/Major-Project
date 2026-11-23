"""Auto-download cloudflared and manage a quick tunnel.

    tunnel = CloudflareTunnel(local_url="https://localhost:8443")
    tunnel.start()          # downloads cloudflared on first run, starts the tunnel
    print(tunnel.public_url)  # e.g. https://abc-def-ghi.trycloudflare.com
    tunnel.stop()

Quick tunnels (trycloudflare.com) need no account, no config, and no signup.
The binary is cached in the models directory so it's downloaded once.
"""

from __future__ import annotations

import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

LOG = logging.getLogger("openmeet.tunnel")

_DOWNLOAD_URLS = {
    ("Windows", "AMD64"): "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe",
    ("Windows", "x86_64"): "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe",
    ("Linux", "x86_64"): "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
    ("Linux", "aarch64"): "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64",
    ("Darwin", "x86_64"): "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-amd64.tgz",
    ("Darwin", "arm64"): "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-amd64.tgz",
}

TUNNEL_URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


def _bin_dir(model_dir: Path) -> Path:
    return model_dir / "bin"


def _exe_name() -> str:
    return "cloudflared.exe" if platform.system() == "Windows" else "cloudflared"


def find_cloudflared(model_dir: Path) -> Path | None:
    """Return the path to cloudflared if it exists (system PATH or our download)."""
    local = _bin_dir(model_dir) / _exe_name()
    if local.exists():
        return local
    system = shutil.which("cloudflared")
    if system:
        return Path(system)
    return None


def download_cloudflared(model_dir: Path) -> Path:
    """Download cloudflared for this platform into the models directory."""
    key = (platform.system(), platform.machine())
    url = _DOWNLOAD_URLS.get(key)
    if not url:
        raise RuntimeError(f"No cloudflared download available for {key[0]} {key[1]}")

    bin_dir = _bin_dir(model_dir)
    bin_dir.mkdir(parents=True, exist_ok=True)
    exe = bin_dir / _exe_name()

    if exe.exists():
        return exe

    LOG.info("Downloading cloudflared from %s ...", url)
    tmp = exe.with_suffix(".download")
    try:
        urlretrieve(url, tmp)
        if url.endswith(".tgz"):
            import tarfile
            with tarfile.open(tmp) as tar:
                for member in tar.getmembers():
                    if member.name.endswith("cloudflared"):
                        f = tar.extractfile(member)
                        if f:
                            exe.write_bytes(f.read())
                            break
            tmp.unlink(missing_ok=True)
        else:
            tmp.rename(exe)

        if platform.system() != "Windows":
            exe.chmod(0o755)

        LOG.info("cloudflared saved to %s", exe)
        return exe
    except Exception:
        tmp.unlink(missing_ok=True)
        exe.unlink(missing_ok=True)
        raise


class CloudflareTunnel:
    """Manages a cloudflared quick tunnel as a subprocess."""

    def __init__(self, local_url: str, model_dir: Path) -> None:
        self.local_url = local_url
        self.model_dir = model_dir
        self.public_url: str | None = None
        self._process: subprocess.Popen | None = None  # type: ignore[type-arg]
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._error: str | None = None

    def start(self, timeout: float = 30) -> str | None:
        """Start the tunnel. Returns the public URL, or None on failure."""
        exe = find_cloudflared(self.model_dir)
        if not exe:
            try:
                exe = download_cloudflared(self.model_dir)
            except Exception as e:
                self._error = str(e)
                LOG.warning("Could not get cloudflared: %s", e)
                return None

        cmd = [str(exe), "tunnel", "--url", self.local_url, "--no-tls-verify"]
        LOG.info("Starting tunnel: %s", " ".join(cmd))

        try:
            self._process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
                creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0,
            )
        except Exception as e:
            self._error = str(e)
            LOG.warning("Could not start cloudflared: %s", e)
            return None

        self._thread = threading.Thread(target=self._read_output, daemon=True)
        self._thread.start()

        if self._ready.wait(timeout=timeout):
            LOG.info("Tunnel ready: %s -> %s", self.public_url, self.local_url)
            return self.public_url
        else:
            self._error = self._error or "Tunnel did not produce a URL within the timeout"
            LOG.warning("Tunnel startup timed out")
            return None

    def _read_output(self) -> None:
        assert self._process and self._process.stdout
        for line in self._process.stdout:
            line = line.strip()
            if line:
                LOG.debug("cloudflared: %s", line)
            m = TUNNEL_URL_RE.search(line)
            if m:
                self.public_url = m.group(0)
                self._ready.set()

    def stop(self) -> None:
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def error(self) -> str | None:
        return self._error
