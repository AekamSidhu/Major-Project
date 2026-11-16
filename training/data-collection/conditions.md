# Answer conditions

Every study question is answered in one of four conditions. The interviewer
sets it on the host page (*Study* card, *Next answer's condition*) before the
question, and it is saved with the answer. Warm-up questions are always
`honest`.

| Condition | What the participant does | Material on their screen | Expected signals |
|---|---|---|---|
| `honest` | Answers in their own words, without notes | none | none (the negatives) |
| `read_script` | Reads aloud a script **they wrote themselves** before the session | their own script | reading gaze, read delivery; not AI text |
| `read_ai` | Reads aloud an answer an AI assistant wrote | the AI answer | reading gaze, read delivery, AI text, echo similarity |
| `paraphrase_ai` | Reads an AI answer silently for 30 s, then answers in their own words; the answer stays visible and they may glance at it | the AI answer | AI content (weaker), some reading |

`read_script` is the hard negative for the content signals (human-written,
but read), and `paraphrase_ai` the hard positive (the adaptive cheater).

## What to say

- **honest:** "Please answer this one in your own words, as you would in a
  real interview."
- **read_script:** "For this one, please read out the answer you prepared,
  from your screen, as naturally as you can."
- **read_ai:** "For this one, an AI wrote the answer. Please read it out from
  your screen, as naturally as you can, as if it were your own."
- **paraphrase_ai:** "Here is an AI's answer. Take 30 seconds to read it.
  Then answer in your own words; you can keep it open and glance at it."

Ask participants not to say which condition they are in, and not to
apologise or comment while answering.

## Materials

- **Scripts (`read_script`).** Send the participant their `read_script`
  questions a few days before. They write each answer in their own words,
  100-200 words, without AI help. Collect the scripts before the session.
- **AI answers (`read_ai`, `paraphrase_ai`).** The study team generates them
  before the session, with the plain question as the prompt plus "Answer as
  a candidate in a job interview, in about 150 words." Rotate assistants
  across participants (for example ChatGPT, Claude, Gemini, Llama) and
  write the one used in the sheet's `ai_model` column: the echo-similarity
  thresholds were measured against Llama 3.2's answers, and the study shows
  whether they hold for other assistants. Keep the generated texts with the
  materials (they contain no personal data).
- **Position.** The sheet's `screen_position` says where the material
  window goes: `side` (next to the call window) or `below_camera` (just
  under the webcam, the hardest case for the gaze signal). Half of each
  participant's read answers use each, as randomised.
