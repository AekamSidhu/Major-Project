# Data card: OpenMeet interview study [version, date]

*Copy to `DATA_CARD.md` and fill in after collection. Aggregate numbers only:
no names, no participant ids, no transcripts. This file may be committed;
the data itself never is.*

## Summary

- **What:** labelled answers from mock video interviews, recorded with the
  OpenMeet Room in study mode (`local-ai/room/study.py`), for calibrating
  the answer-integrity signals.
- **Collected:** [dates], by [team], at/over [place / network setup].
- **Protocol:** `training/data-collection/README.md` at commit [hash].
- **Consent:** written consent from every participant
  (`consent_form.md` version [x]); [n] allowed cloud processing.

## Contents

| | Count |
|---|---|
| Participants (sessions) | [n] ([n] pilots excluded) |
| Answers saved | [n] |
| ... honest (incl. warm-up) | [n] |
| ... read_script | [n] |
| ... read_ai | [n] |
| ... paraphrase_ai | [n] |
| Answers excluded, and why | [n]: [reasons] |
| Median answer length | [s] seconds, [n] words |

Per answer: condition, question, transcript with word timings, gaze series
(10 per second), every signal's score and the fused flag. No audio, no video.

## Participants (categories only)

| Group | Count |
|---|---|
| Native / non-native English | [n] / [n] |
| Glasses / no glasses | [n] / [n] |
| Laptop camera / external camera | [n] / [n] |

## Materials

- AI answers by: [assistant: n answers, ...]
- Material position: side [n], below camera [n]

## Known issues

- [noisy rooms, bad light, a condition misunderstood, ...]
- The interviewer knew each answer's condition (needed to label it).
- Scripted conditions are acted: a real cheater may behave differently.

## Storage and access

- Stored at: [private location], access: [who].
- Delete by: [date]. Withdrawals so far: [n] (data deleted).
- Sharing: [none / research only, as consented].

## Results

See `results/own_data_thresholds.json` (from `calibrate.py`): per signal,
development / test people, ROC-AUC and false-positive rate on test people.
