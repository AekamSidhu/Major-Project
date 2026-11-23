-- OpenMeet schema version 3: workplace mode.
--
-- action_items.due keeps the due date as it was said in the meeting ("by
-- Friday"), never resolved to a calendar date. NULL for interview meetings.
--
-- meeting_summaries.workplace holds the workplace minutes as JSON (structured
-- action items with owner and due date, the follow-up message draft). NULL
-- for every other meeting, including meetings recorded before this version.

ALTER TABLE action_items ADD COLUMN due TEXT;
ALTER TABLE meeting_summaries ADD COLUMN workplace TEXT;
