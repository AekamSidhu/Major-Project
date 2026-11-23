"""Auto-download cloudflared and manage a quick tunnel.

    tunnel = CloudflareTunnel(local_url="https://localhost:8443")
    tunnel.start()          # downloads cloudflared on first run, starts the tunnel
    print(tunnel.public_url)  # e.g. https://abc-def-ghi.trycloudflare.com
    tunnel.stop()

Quick tunnels (trycloudflare.com) need no account, no config, and no signup.
The binary is cached in the models directory so it's downloaded once.
"""

from __future__ import annotations

import hashlib
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

# One pinned cloudflared release. Every value is checked before anything is
# extracted, renamed, made executable or run. For each platform:
#   asset        the file on the release page
#   asset_sha    SHA-256 of that file (GitHub's asset digest, confirmed by download)
#   binary_sha   SHA-256 of the cloudflared program itself (the release notes list
#                these for the Darwin archives; for other platforms it equals asset_sha)
#   is_archive   the asset is a .tgz holding the program
CLOUDFLARED_VERSION = "2026.11.1"
_RELEASE_URL = f"https://github.com/cloudflare/cloudflared/releases/download/{CLOUDFLARED_VERSION}"

_WINDOWS_AMD64 = "341deda3db5ec498a0482254d6c38c0b7cf01a30c37bb5dd95d2216d08955e35"
_LINUX_AMD64 = "97bb3b02d23a5980fc9c15a1a270e692da1923f103f920d5b16d4931d5ab55e7"
_LINUX_ARM64 = "b155cec3e48e489cfb8e5c602769fb8c771083018d8881d68b4d18a791567c5a"
_ASSETS: dict[tuple[str, str], tuple[str, str, str, bool]] = {
    ("Windows", "AMD64"): ("cloudflared-windows-amd64.exe", _WINDOWS_AMD64, _WINDOWS_AMD64, False),
    ("Windows", "x86_64"): ("cloudflared-windows-amd64.exe", _WINDOWS_AMD64, _WINDOWS_AMD64, False),
    ("Linux", "x86_64"): ("cloudflared-linux-amd64", _LINUX_AMD64, _LINUX_AMD64, False),
    ("Linux", "aarch64"): ("cloudflared-linux-arm64", _LINUX_ARM64, _LINUX_ARM64, False),
    ("Darwin", "x86_64"): ("cloudflared-darwin-amd64.tgz",
                           "5e0403b188dfee71fbd9b995e7dc9910d61b661c52371935ab6475d41185632f",
                           "52686df40889be19803c368470bd46d5228737462f10a40380bedb6af8ed49a2", True),
    ("Darwin", "arm64"): ("cloudflared-darwin-arm64.tgz",
                          "02fe4bcf08f2c09699cff30658228a739a0c47fb75f3653d1b8772a27ade4462",
                          "1cd7041838e5b0943378763ca86ddc0af896d1727875f36b29f5cc35eb2c4df7", True),
}

TUNNEL_URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


def _bin_dir(model_dir: Path) -> Path:
    return model_dir / "bin"


def _exe_name() -> str:
    """Our cached copy is named for the pinned version, so an older unpinned copy is never used."""
    suffix = ".exe" if platform.system() == "Windows" else ""
    return f"cloudflared-{CLOUDFLARED_VERSION}{suffix}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, expected: str) -> None:
    """Raise unless `path` has the expected SHA-256. A file that fails is deleted."""
    actual = sha256_file(path)
    if actual != expected:
        path.unlink(missing_ok=True)
        raise RuntimeError(
            f"cloudflared failed its checksum (expected {expected}, got {actual}); "
            "the file was deleted. Check the network or install cloudflared yourself."
        )


def find_cloudflared(model_dir: Path) -> Path | None:
    """cloudflared from the system PATH, else our verified copy in the models directory."""
    system = shutil.which("cloudflared")
    if system:
        return Path(system)
    entry = _ASSETS.get((platform.system(), platform.machine()))
    if entry is None:
        return None
    local = _bin_dir(model_dir) / _exe_name()
    if not local.exists():
        return None
    try:
        verify_file(local, entry[2])  # re-checked on every start, whatever the platform
    except RuntimeError as error:
        LOG.warning("%s", error)
        return None
    return local


def download_cloudflared(model_dir: Path) -> Path:
    """Download the pinned cloudflared for this platform, verify it, and keep it in the models directory."""
    key = (platform.system(), platform.machine())
    entry = _ASSETS.get(key)
    if entry is None:
        raise RuntimeError(f"No cloudflared download available for {key[0]} {key[1]}")
    asset, asset_sha, binary_sha, is_archive = entry

    bin_dir = _bin_dir(model_dir)
    bin_dir.mkdir(parents=True, exist_ok=True)
    exe = bin_dir / _exe_name()
    if exe.exists():
        try:
            verify_file(exe, binary_sha)
            return exe
        except RuntimeError as error:
            LOG.warning("%s", error)

    url = f"{_RELEASE_URL}/{asset}"
    LOG.info("Downloading cloudflared %s from %s ...", CLOUDFLARED_VERSION, url)
    tmp = bin_dir / (asset + ".download")
    part = exe.with_name(exe.name + ".part")
    try:
        urlretrieve(url, tmp)
        verify_file(tmp, asset_sha)  # before anything is extracted, renamed or made executable
        if is_archive:
            import tarfile

            with tarfile.open(tmp) as tar:
                member = next((m for m in tar.getmembers() if m.isfile() and m.name.split("/")[-1] == "cloudflared"), None)
                extracted = tar.extractfile(member) if member is not None else None
                if extracted is None:
                    raise RuntimeError(f"{asset} does not contain a cloudflared binary")
                data = extracted.read()
            actual = hashlib.sha256(data).hexdigest()
            if actual != binary_sha:
                raise RuntimeError(f"cloudflared inside {asset} failed its checksum (expected {binary_sha}, got {actual})")
            part.write_bytes(data)
            part.replace(exe)
            tmp.unlink(missing_ok=True)
        else:
            tmp.replace(exe)

        if platform.system() != "Windows":
            exe.chmod(0o755)

        LOG.info("cloudflared saved to %s", exe)
        return exe
    except Exception:
        tmp.unlink(missing_ok=True)
        part.unlink(missing_ok=True)
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
