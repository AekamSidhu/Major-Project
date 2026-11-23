# OpenMeet Room

A 1:1 interview room in the browser, hosted from the interviewer's own
computer. The candidate only opens a link; nothing to install on their side.
While they talk, the interviewer's computer transcribes both sides (with
word timings), runs the answer-integrity signals on each of the candidate's
answers (see [Integrity signals](#integrity-signals) and
`../integrity/README.md`) and suggests follow-up questions.

```
candidate's browser  ◄──── video + audio (WebRTC, direct) ────►  interviewer's browser
        │                                                              │
        └──── signaling only ────►  room_server.py  ◄──── speech segments, video frames,
                                    (interviewer's computer)   answer text (this computer only)
                                    Whisper, reading CNN, AI-answer model, embeddings,
                                    follow-ups and reference answers (Llama 3.2 here, or Gemini online)
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
python room_server.py --followups rules    # no AI model, simple rules (no echo similarity)
python room_server.py --study D:/study     # study mode, see "Study mode" below
```

The first start downloads the Whisper speech model (about 140 MB); the
first answer downloads the embedding model (about 70 MB).

1. Open **https://localhost:8443** on the interviewer's computer and accept
   the certificate warning once (the server made the certificate itself).
2. Send the candidate the first invite link (this computer's Wi-Fi address).
   Their browser warns about the certificate once too: *Advanced → Proceed*.
3. Talk. The transcript fills in, the reading meter follows the candidate's
   eyes, and each answer gets a card when the candidate stops and you speak
   again (or press *End answer now*): one combined signal, the four
   integrity signals and, under *Why*, their evidence. *Ask next* then lists
   follow-up questions. *Report* saves a Markdown summary.

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
| Similar to an AI answer | Ask a question, then answer by reading what Llama 3.2 (or any assistant) says to it; the card shows the cosine to the room model's own answers a few seconds later. |
| Delivery | Answer the first three questions normally (they set the candidate's usual pace), then wait a few seconds before answering the fourth and read it out; the card shows the delay and pace against the first answers. |
| Reading check | As the candidate, read a paragraph from the screen for 10 seconds or more, then talk freely, and watch *Reading, last 10 seconds*. The camera must see the face clearly. |
| Follow-up questions | After each answer, *Ask next* lists up to three. Click one to copy it, or press *Suggest follow-ups* for a new set. The line underneath says which model wrote them and what the guardrails held back. |
| Report | *Report* downloads answers, signals with their evidence, follow-ups and the transcript (written by the server, `report.py`). |

## Integrity signals

Each answer gets four signals (definitions in `spec/integrity/README.md`,
implemented in `../integrity/signals.py`), fused into one flag
(`checks.py`):

| Signal | How the room computes it | Calibrated |
|---|---|---|
| AI-written text | TF-IDF + logistic regression on the answer's transcript | on HC3 (public data) |
| Similar to an AI answer | When the candidate starts answering, the follow-up model writes 2 answers to the interviewer's question (`similarity.py`, the recipe in `signals.py`); the answer and those are embedded with `BAAI/bge-small-en-v1.5`; the highest cosine counts | on HC3, see `thresholds.json` |
| Delivery | From Whisper's word timings: delay after the question, words a minute, pauses a minute, fillers, against the candidate's own first three answers | no |
| Reading | The reading network on the gaze during the answer | on GazeBase (public data) |

The flag is **a signal for the interviewer, never a verdict**: at most
*Worth a follow-up*. It is *Worth a follow-up: several signals agree* only
when a calibrated signal is high and another signal is raised too; an
uncalibrated signal counts at most as *possible*. The card and the report
say which signals were raised, why, and which are uncalibrated. None of the
thresholds has been calibrated on real interviews yet (the collection kit in
`training/data-collection` is for that).

Echo similarity is *unavailable* in rules mode, when the model doesn't
answer, or when the question wasn't transcribed. Its thresholds were
measured with Llama 3.2's answers, so with Gemini (online mode) or another
Ollama model it is still shown but counts as uncalibrated, and says so.

## Study mode

`python room_server.py --study DIR` saves labelled answers for calibrating
the signals on our own data (`study.py`, protocol in
`training/data-collection/README.md`). DIR must be outside the repository;
the server refuses anything inside it.

- The join page shows a second, optional consent box, and its notice says
  nothing is recorded unless the candidate joins the study. The server, not
  the host page, decides: only a guest who joined with that box ticked
  turns saving on, and joining again without it turns it off.
- The host page gets a *Study* card: the participant id (P07, never a
  name), the next answer's condition (`honest`, `read_script`, `read_ai`,
  `paraphrase_ai`), and a label select on every answer card.
- Per answer it saves the condition, question, transcript with word
  timings, the gaze series (10 per second) and every score, under an
  anonymous session id, in `DIR/session-<id>/`. Never audio, video or the
  candidate's name.

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
| `online` | Gemini 3.5 Flash (`OPENMEET_GEMINI_MODEL`, key in `OPENMEET_GEMINI_API_KEY`) | Google; answer text for follow-ups, the interviewer's question alone for reference answers |
| `rules` | none | This computer |

The same model writes the reference answers for echo similarity. In online
mode only the interviewer's question is sent for those (with the
candidate's name, emails and phone numbers removed), and the join notice
says that the questions go to Gemini too.

## Who can reach what

| | Interviewer (this computer) | Candidate (any network device) |
|---|---|---|
| `/host`, `/api/*`, `/v1/integrity/*` | yes | no (403) |
| `/join/<room>` page, `/room.js` | yes | yes |
| `/ws/<room>` signaling | with the room's host key | with the guest key from the invite |

The candidate sees a notice of what is analysed (transcription with word
timings, reading gaze, AI-written text and comparison with an AI model's
answers, pace and pauses against their first answers) and must agree before
joining; in online mode it says that the questions and answer text go to
Google Gemini. No score or flag ever reaches the candidate's page or
socket. Audio segments and frames are processed and dropped; transcripts
and scores stay in memory until the server stops (study mode, with the
candidate's separate consent, is the exception).

## Limits (v1)

- One candidate per room, one room at a time for the gaze check.
- Speech is cut by loudness in the browser; use headphones so the
  candidate's voice doesn't reach the interviewer's microphone.
- Transcription runs on the CPU (`base.en`, a few seconds behind). A GPU or
  a smaller model (`--whisper-model tiny.en`) keeps it closer to live.
- Llama 3.2 on a laptop CPU takes several seconds to tens of seconds per set
  of follow-ups; the online mode takes about 5 seconds. Reference answers
  are written while the candidate talks (about 2 s each on an RTX 4060), so
  echo similarity usually arrives a moment after the answer card.
- Delivery needs three answers of at least 10 words before it scores
  anything, cannot tell a slow reader from a thoughtful speaker, and on
  public data (SpiCE) barely separated read from spontaneous speech; see
  `training/echo-detection/README.md`.
