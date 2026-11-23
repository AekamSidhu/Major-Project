# Workplace mode

Workplace mode is for ordinary team meetings (stand-ups, reviews, planning,
1:1s) where nobody is interviewed. It keeps a running record of what the team
decided and committed to, and writes the minutes when the meeting ends.

## What it does

During the meeting, the main window shows the **Board**:

- **Decisions**: claims of kind decision, as said ("We decided to freeze the
  schema.").
- **Action items**: commitments and requests with the owner and the due date
  when they were stated ("Priya will update the runbook by Friday" gives owner
  Priya and due "by Friday"). A later reply can fill in a missing owner or
  date. Dates are kept as said, never resolved to a calendar date.
- **Open questions**: questions asked in the meeting, marked answered when
  someone replies, or deferred ("let's take that offline").
- **Clarify**: suggested clarifying questions for decisions or action items that
  are vague: no owner, no due date, unclear scope ("look into it"), or
  tentative ("probably", "for now").

When someone asks **you** a question, OpenMeet suggests an answer in the
overlay, worded for a colleague and grounded in the transcript and your notes.
Questions asked by you are tracked on the board but do not get suggestions.

**Catch me up** streams a summary of the last five minutes into the overlay:
what was said, the decisions and action items in that window, and the open
questions still waiting for you.

When the meeting ends, the **minutes** are written: an overview, decisions,
action items with owner and due date, open questions, and a draft follow-up
message you can copy. The minutes are stored with the meeting and appear in
History (Summary tab) and in Markdown, text and JSON export. Without an AI
provider, or if the model's answer cannot be used, basic minutes are built from
the board without a model.

## How to use it

1. Choose **Workplace** on the start screen, in Settings → General, or with the
   mode switch in the overlay (it cycles Candidate, Interviewer, Workplace).
   The keyboard shortcut for switching modes also cycles through all three.
2. Start the meeting as usual. The board and the questions fill in as people
   speak.
3. Use **Catch me up** in the main window or the overlay when you step back in.
4. End the meeting. Open it in History to read the minutes and copy the
   follow-up message.

Demo: "Sprint planning (workplace)" in the demo list runs the meeting without
devices.

## Privacy and limits

- Local-only rules apply unchanged: audio, transcripts and minutes stay on the
  device unless you choose a cloud AI provider. Retention, delete and
  delete-all cover workplace meetings like any other meeting.
- Answer-integrity checks never run in workplace mode. Speakers are labelled
  "You" and "Others". No interview scoring or hiring language is produced.
- Vagueness detection and the board are rule-based and can miss or over-flag
  phrasing; the clarifying questions are suggestions for you to use or ignore.
- Action items and decisions come from what is said. Items said by someone
  else are attributed only when the speaker is named.
- Voice commands do not switch modes or ask for a catch-up. Use the mode switch,
  the shortcut or the Catch me up button.
- Follow-up question suggestions (the interviewer feature) are not offered in
  workplace mode.
- Switching into workplace mode in the middle of a meeting starts the board
  from that point; earlier speech is not added to it.
