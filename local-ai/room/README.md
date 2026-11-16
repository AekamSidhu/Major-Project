# OpenMeet Room

A 1:1 interview room in the browser, hosted from the interviewer's own
computer. The candidate only opens a link; nothing to install on their side.
While they talk, the interviewer's computer transcribes both sides, runs the
answer-integrity checks on the candidate (reading gaze and AI-written
answers, see `../integrity/README.md`) and suggests follow-up questions.

```
candidate's browser  ◄──── video + audio (WebRTC, direct) ────►  interviewer's browser
        │                                                              │
        └──── signaling only ────►  room_server.py  ◄──── speech segments, video frames,
                                    (interviewer's computer)   answer text (this computer only)
                                    Whisper, reading CNN, AI-answer model,
                                    follow-ups (Llama 3.2 here, or Gemini online)
```

## Install once

From `local-ai/`, with Python 3.10 to 3.12:

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS / Linux: source .venv/bin/activate
pip install -r requirements.txt -r requirements-integrity.txt
```

For AI follow-up questions on this computer, install [Ollama](https://ollama.com)
and download the model once: `ollama pull llama3.2`.

For the online mode, put a Google Gemini key in `~/.openmeet/room.env`
(outside the repository; never commit it):

```
OPENMEET_GEMINI_API_KEY=your-key
```

## Run it

```bash
python room_server.py                      # follow-ups from Llama 3.2 on this computer
python room_server.py --followups online   # follow-ups from Gemini; the candidate is told
python room_server.py --followups rules    # no AI model, simple rules
```

The first start downloads the Whisper speech model (about 140 MB).

1. Open **https://localhost:8443** on the interviewer's computer and accept
   the certificate warning once (the server made the certificate itself).
2. Send the candidate the first invite link (this computer's Wi-Fi address).
   Their browser warns about the certificate once too: *Advanced → Proceed*.
3. Talk. The transcript fills in, the reading meter follows the candidate's
   eyes, and each answer gets a card when the candidate stops and you speak
   again (or press *End answer now*). *Ask next* then lists follow-up
   questions. *Report* saves a Markdown summary.

**Testing alone:** open the host page and the invite link in two windows (for
example a normal and a private window), or use your phone on the same Wi-Fi.
Add `?fake=1` to both URLs to send a moving test picture and a tone instead of
a camera and microphone, so two windows don't fight over one camera.

**Across networks** (candidate at home): run a tunnel such as
`cloudflared tunnel --url https://localhost:8443 --no-tls-verify` and swap
the invite link's address for the tunnel's. Video still goes browser to
browser; a TURN server is only needed if both sides are behind strict
corporate firewalls.

### What to try

| Feature | How to see it |
|---|---|
| Video call | Both sides see each other; the clock starts when the candidate connects. |
| Live transcript | Speak on either side; lines appear a few seconds later. |
| AI-written check | As the candidate, read out an answer written by ChatGPT; the answer card shows the score. |
| Reading check | As the candidate, read a paragraph from the screen for 10 seconds or more, then talk freely, and watch *Reading, last 10 seconds*. The camera must see the face clearly. |
| Follow-up questions | After each answer, *Ask next* lists up to three. Click one to copy it, or press *Suggest follow-ups* for a new set. The line underneath says which model wrote them and what the guardrails held back. |
| Report | *Report* downloads answers, scores, follow-ups and the transcript. |

## Follow-up questions

`followups.py` is a LangChain chain:

```
input guardrails -> prompt -> model (structured output) -> parse -> output guardrails
```

- **Context.** The system prompt says the model's only job is follow-up
  questions for a job interview, grounded in what the candidate said, deeper
  rather than wider, neutral, one sentence each (the same rules as the
  desktop app's interviewer prompt).
- **Input guardrails** (`guardrails.py`). The transcript is data, never
  instructions: control characters and anything that could close the prompt's
  tags are removed, lengths are capped, and text that looks like a prompt
  injection is flagged to the model. In online mode, emails, phone numbers and
  the candidate's name are replaced before anything leaves this computer.
- **Output guardrails.** A suggestion is dropped when it is not a question,
  is assistant chatter, asks about a protected characteristic (age, family,
  religion, health...), judges or accuses the candidate, repeats an earlier
  question, or has nothing to do with the answer. A quote the candidate never
  said is removed.
- **Fallback.** If the model is missing, unreachable or nothing passes the
  guardrails, `followup_rules.py` writes grounded follow-ups by simple rules
  and the page says so.

| Mode | Model | Where it runs |
|---|---|---|
| `local` (default) | Llama 3.2 through Ollama (`OPENMEET_ROOM_OLLAMA_MODEL`) | This computer |
| `online` | Gemini 3.5 Flash (`OPENMEET_GEMINI_MODEL`, key in `OPENMEET_GEMINI_API_KEY`) | Google; answer text only |
| `rules` | none | This computer |

## Who can reach what

| | Interviewer (this computer) | Candidate (any network device) |
|---|---|---|
| `/host`, `/api/*`, `/v1/integrity/*` | yes | no (403) |
| `/join/<room>` page, `/room.js` | yes | yes |
| `/ws/<room>` signaling | with the room's host key | with the guest key from the invite |

The candidate sees a notice of what is analysed and must agree before
joining; in online mode it says that answer text goes to Google Gemini.
Audio segments and frames are processed and dropped; transcripts and scores
stay in memory until the server stops.

## Limits (v1)

- One candidate per room, one room at a time for the gaze check.
- Speech is cut by loudness in the browser; use headphones so the
  candidate's voice doesn't reach the interviewer's microphone.
- Transcription runs on the CPU (`base.en`, a few seconds behind). A GPU or
  a smaller model (`--whisper-model tiny.en`) keeps it closer to live.
- Llama 3.2 on a laptop CPU takes several seconds to tens of seconds per set
  of follow-ups; the online mode takes about 5 seconds.
