"""OpenMeet Room server: host a 1:1 interview from this computer.

    python room_server.py                 # https://localhost:8443, reachable on the local network
    python room_server.py --no-https      # http://localhost:8443 (this computer only: browsers
                                          # allow camera access on plain http only for localhost)

Open https://localhost:8443 (accept the self-signed certificate once),
create a room and send the invite link to the candidate. On a phone or a
second laptop on the same Wi-Fi, the link uses this computer's network
address; across networks, run a tunnel such as
`cloudflared tunnel --url https://localhost:8443 --no-tls-verify` and
share its address instead.

Transcription (faster-whisper) and both integrity models run here; see
room/__init__.py for what each side can reach.

Follow-up questions come from a chat model: Ollama on this computer by
default (`ollama pull llama3.2`), or any OpenAI-compatible endpoint set with
OPENMEET_ROOM_LLM_URL, OPENMEET_ROOM_LLM_MODEL and OPENMEET_ROOM_LLM_KEY.
Without one, simple rules suggest follow-ups and the page says so.
"""

from __future__ import annotations

import argparse
import datetime as dt
import ipaddress
import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request

import server
from integrity import add_routes as add_integrity_routes
from integrity.models import EchoModel, ReadingModel
from room import add_room_routes, is_loopback, lan_addresses

CERT_DIR = server.MODEL_DIR / "room"


def self_signed_cert(folder: Path) -> tuple[Path, Path]:
    """A certificate for localhost and this computer's network addresses, made once."""
    cert, key = folder / "cert.pem", folder / "key.pem"
    addresses = ["127.0.0.1", *lan_addresses()]
    stamp = folder / "addresses.txt"
    if cert.exists() and key.exists() and stamp.exists() and stamp.read_text() == ",".join(addresses):
        return cert, key
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    folder.mkdir(parents=True, exist_ok=True)
    private = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "OpenMeet Room (this computer)")])
    now = dt.datetime.now(dt.timezone.utc)
    alt = [x509.DNSName("localhost"), *(x509.IPAddress(ipaddress.ip_address(a)) for a in addresses)]
    certificate = (
        x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(private.public_key())
        .serial_number(x509.random_serial_number()).not_valid_before(now - dt.timedelta(days=1))
        .not_valid_after(now + dt.timedelta(days=365)).add_extension(x509.SubjectAlternativeName(alt), critical=False)
        .sign(private, hashes.SHA256())
    )
    key.write_bytes(private.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    cert.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    stamp.write_text(",".join(addresses))
    return cert, key


def create_room_app(port: int, scheme: str, whisper_model: str, transcriber=None, tracker_factory=None, clock=None,  # type: ignore[no-untyped-def]
                    followup_writer=None) -> FastAPI:
    import time

    app = FastAPI(title="OpenMeet Room", docs_url=None, redoc_url=None)

    @app.middleware("http")
    async def no_store(request, call_next):  # type: ignore[no-untyped-def]
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    def host_only_auth(request: Request) -> None:  # integrity API: this computer only
        if not is_loopback(request.client.host if request.client else None):
            raise HTTPException(status_code=403, detail="only the interviewer's computer can use this")

    monitor = add_integrity_routes(app, host_only_auth, server.MODEL_DIR, tracker_factory=tracker_factory, clock=clock or time.time)
    add_room_routes(
        app,
        transcriber=transcriber or server.WhisperTranscriber(whisper_model),
        monitor=monitor,
        echo_factory=EchoModel,
        reading_factory=ReadingModel,
        port=port,
        scheme=scheme,
        whisper_model=whisper_model,
        followup_writer=followup_writer,
    )
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=int(os.environ.get("OPENMEET_ROOM_PORT", "8443")))
    parser.add_argument("--no-https", action="store_true", help="plain http on localhost only")
    parser.add_argument("--whisper-model", default=os.environ.get("OPENMEET_ROOM_WHISPER", "base.en"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpx2", "httpcore", "huggingface_hub", "urllib3", "faster_whisper"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    import uvicorn

    scheme = "http" if args.no_https else "https"
    app = create_room_app(args.port, scheme, args.whisper_model)
    host = "127.0.0.1" if args.no_https else "0.0.0.0"
    ssl = {}
    if not args.no_https:
        cert, key = self_signed_cert(CERT_DIR)
        ssl = {"ssl_certfile": str(cert), "ssl_keyfile": str(key)}
    print(f"\nOpenMeet Room: open {scheme}://localhost:{args.port} on this computer.")
    if not args.no_https:
        for ip in lan_addresses():
            print(f"  candidates on this network join through https://{ip}:{args.port}/join/... (the invite link)")
    uvicorn.run(app, host=host, port=args.port, log_level="warning", access_log=False, **ssl)


if __name__ == "__main__":
    main()
