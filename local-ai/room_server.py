"""OpenMeet Room server: host a 1:1 interview from this computer.

    python room_server.py                 # https://localhost:8443, reachable on the local network
    python room_server.py --tunnel        # also a public Cloudflare address (opt-in, see below)
    python room_server.py --no-https      # http://localhost:8443 (this computer only: browsers
                                          # allow camera access on plain http only for localhost)

Open https://localhost:8443 (accept the self-signed certificate once),
create a room and send the invite link to the candidate. On a phone or a
second laptop on the same Wi-Fi, the link uses this computer's network
address. Across networks, add --tunnel: the server then starts a Cloudflare
quick tunnel (cloudflared, downloaded once and checked against a pinned
SHA-256) that publishes this room's join page to the internet. It is off by
default, because anyone with an invite link can then reach the join page.

Transcription runs here (faster-whisper, the default) or, with
`--transcription groq`, on Groq's cloud API, which then receives the candidate's
audio (needs OPENMEET_GROQ_API_KEY). The integrity signals run here; see
room/__init__.py for what each side can reach and what the join page tells
the candidate.

Follow-up questions (room/followups.py, a LangChain chain with guardrails)
and the reference answers for the echo-similarity signal (room/similarity.py)
come from the same model:
    --followups local    Ollama on this computer (default; `ollama pull llama3.2`)
    --followups online   Google Gemini; set OPENMEET_GEMINI_API_KEY in the environment
                         or in ~/.openmeet/room.env. The candidate's join page says so.
    --followups rules    no model, simple rules (no echo similarity)
If the model is unreachable, the rules answer and the page says so.

    --study DIR          study mode (room/study.py): with the candidate's separate consent,
                         save each labelled answer's transcript, timings, gaze and scores
                         under DIR, which must be outside the repository
"""

from __future__ import annotations

import argparse
import datetime as dt
import ipaddress
import logging
import os
from pathlib import Path

from fastapi import FastAPI

import server
from integrity import add_routes as add_integrity_routes
from integrity.models import EchoModel, ReadingModel
from room import add_room_routes, interviewer_names, interviewer_only, lan_addresses
from room.tunnel import CloudflareTunnel

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
                    followup_writer=None, followups: str | None = None, reference_writer=None, embedder=None,
                    embed_model: str = server.DEFAULT_EMBED_MODEL, study_dir: Path | None = None,
                    tunnel_url: str | None = None, cloud_transcription: bool = False,
                    gaze_model: Path | None = None, gaze_tta: bool = False) -> FastAPI:
    from room.followups import FollowupWriter
    from room.similarity import ReferenceWriter, SimilarityChecker
    from room.study import StudyRecorder

    import time

    writer = followup_writer or FollowupWriter(mode=followups)
    references = reference_writer or ReferenceWriter(writer.mode, writer.settings)
    similarity = SimilarityChecker(references, embedder or server.FastEmbedder(embed_model), embed_model)
    study = None
    if study_dir is not None:
        study = StudyRecorder(study_dir, info={"whisper_model": whisper_model, "followup_mode": writer.mode,
                                               "reference_model": references.model_name, "embedding_model": embed_model})

    app = FastAPI(title="OpenMeet Room", docs_url=None, redoc_url=None)

    @app.middleware("http")
    async def no_store(request, call_next):  # type: ignore[no-untyped-def]
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    host_only_auth = interviewer_only(interviewer_names())  # integrity API: this computer only

    monitor = add_integrity_routes(app, host_only_auth, server.MODEL_DIR, tracker_factory=tracker_factory, clock=clock or time.time,
                                   gaze_model=gaze_model, gaze_tta=gaze_tta)
    app.state.rooms = add_room_routes(
        app,
        transcriber=transcriber or server.WhisperTranscriber(whisper_model),
        monitor=monitor,
        echo_factory=EchoModel,
        reading_factory=ReadingModel,
        port=port,
        scheme=scheme,
        whisper_model=whisper_model,
        followup_writer=writer,
        similarity=similarity,
        study=study,
        tunnel_url=tunnel_url,
        cloud_transcription=cloud_transcription,
    )
    return app


def main() -> None:
    from integrity.learned_gaze import tta_requested

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=int(os.environ.get("OPENMEET_ROOM_PORT", "8443")))
    parser.add_argument("--no-https", action="store_true", help="plain http on localhost only")
    parser.add_argument("--host", default=None, help="override bind address (default: 127.0.0.1 for http, 0.0.0.0 for https)")
    parser.add_argument("--whisper-model", default=os.environ.get("OPENMEET_ROOM_WHISPER", "small.en"))
    parser.add_argument("--transcription", choices=("local", "groq"), default=os.environ.get("OPENMEET_ROOM_TRANSCRIPTION", "local"),
                        help="transcription backend: local (faster-whisper on this computer) or groq (Groq cloud API, needs OPENMEET_GROQ_API_KEY)")
    parser.add_argument("--followups", choices=("local", "online", "rules"), default=None,
                        help="where follow-up questions and reference answers come from (default: OPENMEET_ROOM_FOLLOWUPS or local)")
    parser.add_argument("--study", type=Path, default=None, metavar="DIR",
                        help="study mode: with the candidate's consent, save labelled answers under DIR (outside the repository)")
    parser.add_argument("--tunnel", action="store_true",
                        help="publish the join page through a Cloudflare quick tunnel (off by default)")
    parser.add_argument("--no-tunnel", action="store_true",
                        help="no-op: the tunnel is off unless --tunnel is given")
    parser.add_argument("--gaze-model", type=Path, default=os.environ.get("OPENMEET_GAZE_MODEL") or None, metavar="PATH",
                        help="use the learned gaze model (ONNX file you provide, non-commercial licence: see integrity/README.md) "
                             "instead of the MediaPipe iris offsets (default: OPENMEET_GAZE_MODEL or none)")
    parser.add_argument("--gaze-tta", action="store_true", default=tta_requested(),
                        help="average the mirrored face into each learned gaze estimate (about twice the CPU; "
                             "the 4.54 degree figure used it). Default off; OPENMEET_GAZE_TTA=1 also turns it on")
    args = parser.parse_args()
    if args.tunnel and args.no_https:
        parser.error("--tunnel needs https (drop --no-https)")
    if args.gaze_model is not None and not args.gaze_model.is_file():
        parser.error(f"--gaze-model: no file at {args.gaze_model}")
    if args.gaze_tta and args.gaze_model is None:
        parser.error("--gaze-tta needs --gaze-model")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpx2", "httpcore", "huggingface_hub", "urllib3", "faster_whisper"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    import uvicorn

    scheme = "http" if args.no_https else "https"

    # Start the tunnel early so the URL is ready before the app serves requests.
    tunnel = None
    tunnel_url = None
    if args.tunnel:
        print("\nStarting Cloudflare tunnel (downloading cloudflared on first run)...")
        tunnel = CloudflareTunnel(f"https://localhost:{args.port}", server.MODEL_DIR)
        tunnel_url = tunnel.start(timeout=30)
        if tunnel_url:
            print(f"  Tunnel ready: {tunnel_url}")
        else:
            print(f"  Tunnel failed: {tunnel.error or 'unknown error'}. Falling back to LAN only.")

    transcriber = None
    if args.transcription == "groq":
        from room.followups import _env_file_values, ENV_FILE
        env = {**_env_file_values(ENV_FILE), **os.environ}
        groq_key = env.get("OPENMEET_GROQ_API_KEY", "")
        if not groq_key:
            raise SystemExit("--transcription groq needs OPENMEET_GROQ_API_KEY (environment or ~/.openmeet/room.env)")
        groq_model = env.get("OPENMEET_GROQ_WHISPER_MODEL", "whisper-large-v3")
        transcriber = server.GroqTranscriber(groq_key, model=groq_model)
        print(f"  transcription: {groq_model} (Groq cloud API)")
    else:
        print(f"  transcription: {args.whisper_model} (local, faster-whisper)")

    try:
        app = create_room_app(args.port, scheme, args.whisper_model, transcriber=transcriber,
                              followups=args.followups, study_dir=args.study, tunnel_url=tunnel_url,
                              cloud_transcription=args.transcription == "groq", gaze_model=args.gaze_model,
                              gaze_tta=args.gaze_tta)
    except ValueError as error:  # a study folder inside the repository
        parser.error(str(error))
    host = args.host or ("127.0.0.1" if args.no_https else "0.0.0.0")
    ssl = {}
    if not args.no_https:
        cert, key = self_signed_cert(CERT_DIR)
        ssl = {"ssl_certfile": str(cert), "ssl_keyfile": str(key)}
    from room.followups import Settings

    settings = Settings.load()
    mode = args.followups or settings.mode
    if mode == "online" and not settings.gemini_key:
        raise SystemExit("--followups online needs OPENMEET_GEMINI_API_KEY (environment or ~/.openmeet/room.env)")
    where = {"local": f"{settings.ollama_model} on this computer (Ollama)", "online": f"{settings.gemini_model} (Google, online)",
             "rules": "simple rules"}[mode]
    print(f"\nOpenMeet Room: open {scheme}://localhost:{args.port} on this computer.")
    print(f"  follow-up questions and reference answers: {where}")
    if tunnel_url:
        print(f"  Anyone with the link can reach this room's join page: {tunnel_url}")
    if args.study:
        print(f"  STUDY MODE: answers of candidates who tick the study consent are saved under {args.study.resolve()}")
    if not args.no_https:
        for ip in lan_addresses():
            print(f"  candidates on this network join through {scheme}://{ip}:{args.port}/join/... (the invite link)")
    try:
        uvicorn.run(app, host=host, port=args.port, log_level="warning", access_log=False, **ssl)
    finally:
        if tunnel:
            tunnel.stop()


if __name__ == "__main__":
    main()
