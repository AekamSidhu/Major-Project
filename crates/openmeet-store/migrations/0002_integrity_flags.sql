-- OpenMeet schema version 2: answer-integrity flags.
--
-- Interviewer mode, only when the user turned the answer-integrity check on.
-- One row per checked candidate answer; a re-check of the same answer (the
-- candidate kept talking) replaces the row. A flag is a signal for the
-- interviewer, never a verdict. The signals and their evidence are a JSON
-- list; no audio, video frames or gaze samples are ever stored.
--
-- created_at is when the answer was first checked, checked_at when the
-- stored revision was computed; updated_at is the write time.

CREATE TABLE integrity_flags (
    id               TEXT    PRIMARY KEY NOT NULL,
    meeting_id       TEXT    NOT NULL REFERENCES meetings (id) ON DELETE CASCADE,
    question_id      TEXT,
    question_text    TEXT,
    answer_start_ms  INTEGER NOT NULL,
    answer_end_ms    INTEGER NOT NULL,
    answer_words     INTEGER NOT NULL,
    level            TEXT    NOT NULL,
    elevated         TEXT    NOT NULL DEFAULT '[]',
    headline         TEXT    NOT NULL,
    signals          TEXT    NOT NULL DEFAULT '[]',
    revision         INTEGER NOT NULL,
    spec_version     INTEGER NOT NULL,
    created_at       INTEGER NOT NULL,
    checked_at       INTEGER NOT NULL,
    updated_at       INTEGER NOT NULL
) STRICT;

CREATE INDEX idx_integrity_flags_meeting_time ON integrity_flags (meeting_id, answer_start_ms);
