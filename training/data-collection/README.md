# Own-data collection kit (Phase 3, shared by gaze and echo)

The answer-integrity signals were built on public data (GazeBase, HC3,
SpiCE). None of it shows the case we care about: a person in a video
interview reading or paraphrasing an answer. This kit collects that, once,
for both the reading-gaze signal (`training/gaze-detection`) and the
AI-answer, echo-similarity and delivery signals (`training/echo-detection`).

| File | What |
|---|---|
| `README.md` | This protocol |
| `consent_form.md` | Participant information and consent, including whether cloud processing is allowed |
| `question_bank.csv` | Warm-up, technical, open and behavioural questions |
| `conditions.md` | The four answer conditions, with the exact instructions |
| `randomise.py` | Each participant's questions and condition order, as a pre-filled label sheet |
| `label_sheet_template.csv` | One row per answer: condition, materials, exclusions |
| `data_card_template.md` | The data card to fill in after collection (it goes in git; the data never does) |
| `calibrate.py` | Thresholds for each signal from the saved sessions, split by person |

## What is recorded, and where

Sessions run in the OpenMeet Room in **study mode**, on the study team's
computer:

```bash
cd local-ai
python room_server.py --study D:/openmeet-study     # any folder OUTSIDE the repository
```

The join page then shows a second, optional consent box. Only if the
participant ticks it does the room save, for each answer, under an
anonymous session id (`room/study.py`):

- the condition label the interviewer set,
- the question and the transcript, with the time of every word,
- the gaze series (two numbers per frame, 10 frames a second),
- every signal's score and the combined flag.

It never saves audio, video frames or the participant's name. Nothing from
the study folder is ever committed: only code, this kit and the filled-in
data card (aggregate numbers) go into git.

## Roles

- **Interviewer**: runs the room, asks the questions, sets each answer's
  condition on the host page (the *Study* card) before the participant
  answers, and fills in the label sheet.
- **Participant**: joins from their own laptop with a webcam, on a separate
  network if possible (as a real candidate would).
- **Study lead**: keeps the consent forms and the participant id list,
  prepares the materials, runs `randomise.py`, and is the contact for
  withdrawals.

## Before the session

1. Run `python randomise.py --participants P01 P02 ... --out sheets/`. Each
   participant gets 3 warm-up questions (always answered honestly; they form
   the delivery baseline) and 12 study questions, one technical, one open
   and one behavioural question per condition, in a random order.
2. Prepare the materials listed in each participant's sheet
   (`conditions.md`): their own written script for the `read_script`
   questions (sent to them a few days before, written by them), and
   AI-written answers for the `read_ai` and `paraphrase_ai` questions.
   Record which assistant wrote each AI answer (`ai_model`).
3. Send the consent form. Collect it signed before the session.

## The session (about 40 minutes)

1. **Set-up (5 min).** Participant joins the invite link, ticks the
   ordinary consent and, if they agree, the study consent. Check on the
   host page that the *Study* card says "candidate consented: saving", then
   enter the participant id (P07, never a name). Check the face is in view
   and the light is even.
2. **Warm-up (5 min).** The 3 warm-up questions, honest answers. Condition
   `honest`.
3. **Study questions (25 min).** For each of the 12 questions, in the
   sheet's order:
   - set the *Next answer's condition* on the host page;
   - tell the participant the condition (the sheet has the wording) and
     give them the material, if any, in a window on their screen at the
     position the sheet names (next to the call, or below the camera);
   - ask the question; let them answer fully; don't interrupt;
   - press *End answer now* if the turn doesn't end by itself.
   Correct a wrong label on the answer card at once (the label select).
   Copy the session id the *Study* card shows into the sheet's `session`
   column. If the room split one answer into two cards (or merged two), fix
   the sheet's `answer_no` to the card numbers: `calibrate.py` matches the
   sheet to the saved answers on session (or participant) and answer number,
   and warns about sheet rows it cannot match. Answers given before the
   participant ticked the study consent are never saved, even if relabelled.
4. **Debrief (5 min).** Explain what each signal showed (the host page and
   report), ask how natural each condition felt, note anything unusual
   (glasses glare, a noisy room, the participant reading in an `honest`
   answer) in the sheet's `notes`, and mark answers to exclude.

Run 2-3 pilot sessions with team members first, fix whatever was awkward,
and don't use pilot sessions for calibration.

## Who to recruit

Aim for at least 20 participants for a first calibration (each gives 12
labelled answers plus 3 baseline answers), more for subgroup checks. Mix:
native and non-native English speakers, glasses and no glasses, different
laptops and webcams, nervous and relaxed speakers. Record these in the
label sheet's participant columns only as categories, never as free text
about a person.

## After collection

```bash
python calibrate.py D:/openmeet-study --labels sheets/all.csv --out results/own_data_thresholds.json
```

`calibrate.py` splits **people** (not answers) into development (70%) and
test (30%), chooses each signal's `possible` and `high` thresholds on the
development people only, and scores the test people once. It writes a
proposal; nobody's thresholds change until a person reviews it and edits
`spec/integrity/thresholds.json` (then reruns `spec/integrity/reference.py`).
Fill in `data_card_template.md` as `DATA_CARD.md` with the aggregate numbers.

## Withdrawal and storage

- A participant can withdraw at any time, during or after the session. The
  study lead finds their session id in the label sheet and deletes the
  `session-<id>` folder and their sheet rows.
- Keep the study folder, the label sheets and the consent forms in the
  team's private storage, access limited to the study team, and apart from
  each other: the consent form has the name, the sheet only the participant
  id.
- Delete the data when the project ends, unless participants agreed to a
  longer period on the consent form.
