# OpenMeet Room

A 1:1 interview room in the browser, hosted from the interviewer's own
computer. The candidate only opens a link; nothing to install on their side.
While they talk, the interviewer's computer transcribes both sides and runs
the answer-integrity checks on the candidate (reading gaze and AI-written
answers, see `../integrity/README.md`).

```
candidate's browser  ◄──── video + audio (WebRTC, direct) ────►  interviewer's browser
        │                                                              │
        └──── signaling only ────►  room_server.py  ◄──── speech segments, video frames,
                                    (interviewer's computer)   answer text (this computer only)
                                    Whisper · reading CNN · AI-answer model
```

## Run it

From `local-ai/` with the sidecar's environment (`requirements.txt` +
`requirements-integrity.txt`):

```bash
python room_server.py                 # https://localhost:8443, reachable on this network
```

1. Open **https://localhost:8443** on the interviewer's computer and accept
   the certificate warning once (the server made the certificate itself).
2. Send the candidate the first invite link (this computer's Wi-Fi address).
   Their browser warns about the certificate once too: *Advanced → Proceed*.
3. Talk. The transcript fills in, the reading meter follows the candidate's
   eyes, and each answer gets a card when the candidate stops and you speak
   again (or press *End answer now*). *Download report* saves a Markdown
   summary.

**Testing alone:** open the host page and the invite link in two tabs or two
browsers on the same computer, or use your phone on the same Wi-Fi. Add
`?fake=1` to both URLs to send a generated picture and tone instead of a
camera and microphone.

**Across networks** (candidate at home): run a tunnel such as
`cloudflared tunnel --url https://localhost:8443 --no-tls-verify` and swap
the invite link's address for the tunnel's. Video still goes browser to
browser; a TURN server is only needed if both sides are behind strict
corporate firewalls.

## Who can reach what

| | Interviewer (this computer) | Candidate (any network device) |
|---|---|---|
| `/host`, `/api/*`, `/v1/integrity/*` | yes | no (403) |
| `/join/<room>` page, `/room.js` | yes | yes |
| `/ws/<room>` signaling | with the room's host key | with the guest key from the invite |

The candidate sees a notice of what is analysed and must agree before
joining. Audio segments and frames are processed and dropped; transcripts and
scores stay in memory until the server stops.

## Limits (v1)

- One candidate per room, one room at a time for the gaze check.
- Speech is cut by loudness in the browser; use headphones so the
  candidate's voice doesn't reach the interviewer's microphone.
- Transcription runs on the CPU (`base.en`, a few seconds behind). A GPU or
  a smaller model (`--whisper-model tiny.en`) keeps it closer to live.
- Follow-up suggestions from an LLM are not wired in yet (the desktop app
  has them; next step).
