-- OpenMeet schema version 1.
--
-- Conventions
-- * Timestamps are Unix milliseconds (INTEGER). Transcript start_time /
--   end_time are milliseconds relative to the meeting start.
-- * Enum values are stored as their snake_case serde names (TEXT).
-- * Lists (segment_ids, parts, entities, followups, summary lists) are JSON
--   text columns.
-- * Embeddings are little-endian f32 BLOBs (embeddings.vector is the only
--   BLOB column in the schema).
-- * Every meeting-owned row references meetings(id) with ON DELETE CASCADE,
--   so deleting a meeting removes everything recorded about it.
-- * No raw audio and no screenshots are ever stored.
--
-- created_at / updated_at columns that have no domain counterpart are
-- maintained by the store (wall clock at write time).

CREATE TABLE meetings (
    id          TEXT    PRIMARY KEY NOT NULL,
    title       TEXT    NOT NULL,
    mode        TEXT    NOT NULL,
    state       TEXT    NOT NULL,
    started_at  INTEGER NOT NULL,
    ended_at    INTEGER,
    persisted   INTEGER NOT NULL DEFAULT 1,
    created_at  INTEGER NOT NULL,
    updated_at  INTEGER NOT NULL
) STRICT;

CREATE INDEX idx_meetings_started_at ON meetings (started_at);

-- Participant list of a meeting, replaced as a whole on every save_meeting.
-- Keyed by position so the list round-trips exactly.
CREATE TABLE participants (
    meeting_id  TEXT    NOT NULL REFERENCES meetings (id) ON DELETE CASCADE,
    position    INTEGER NOT NULL,
    id          TEXT    NOT NULL,
    role        TEXT    NOT NULL,
    label       TEXT    NOT NULL,
    updated_at  INTEGER NOT NULL,
    PRIMARY KEY (meeting_id, position)
) STRICT;

CREATE TABLE transcript_segments (
    id            TEXT    PRIMARY KEY NOT NULL,
    meeting_id    TEXT    NOT NULL REFERENCES meetings (id) ON DELETE CASCADE,
    utterance_id  TEXT    NOT NULL,
    speaker_id    TEXT,
    speaker_role  TEXT    NOT NULL,
    source        TEXT    NOT NULL,
    start_time    INTEGER NOT NULL,
    end_time      INTEGER NOT NULL,
    text          TEXT    NOT NULL,
    is_final      INTEGER NOT NULL,
    confidence    REAL,
    created_at    INTEGER NOT NULL,
    updated_at    INTEGER NOT NULL
) STRICT;

CREATE INDEX idx_transcript_segments_meeting_time
    ON transcript_segments (meeting_id, start_time);

CREATE TABLE questions (
    id                 TEXT    PRIMARY KEY NOT NULL,
    meeting_id         TEXT    NOT NULL REFERENCES meetings (id) ON DELETE CASCADE,
    question           TEXT    NOT NULL,
    parts              TEXT    NOT NULL DEFAULT '[]',
    segment_ids        TEXT    NOT NULL DEFAULT '[]',
    is_question        INTEGER NOT NULL,
    confidence         REAL    NOT NULL,
    category           TEXT    NOT NULL,
    speaker_id         TEXT,
    speaker_role       TEXT    NOT NULL,
    requires_response  INTEGER NOT NULL,
    method             TEXT    NOT NULL,
    supersedes         TEXT,
    detected_at        INTEGER NOT NULL,
    created_at         INTEGER NOT NULL,
    updated_at         INTEGER NOT NULL
) STRICT;

CREATE INDEX idx_questions_meeting_time ON questions (meeting_id, detected_at);

CREATE TABLE answers (
    id            TEXT    PRIMARY KEY NOT NULL,
    meeting_id    TEXT    NOT NULL REFERENCES meetings (id) ON DELETE CASCADE,
    question_id   TEXT,
    speaker_role  TEXT    NOT NULL,
    text          TEXT    NOT NULL,
    segment_ids   TEXT    NOT NULL DEFAULT '[]',
    created_at    INTEGER NOT NULL,
    updated_at    INTEGER NOT NULL
) STRICT;

CREATE INDEX idx_answers_meeting_time ON answers (meeting_id, created_at);

CREATE TABLE ai_suggestions (
    id              TEXT    PRIMARY KEY NOT NULL,
    meeting_id      TEXT    NOT NULL REFERENCES meetings (id) ON DELETE CASCADE,
    kind            TEXT    NOT NULL,
    question_id     TEXT,
    question_text   TEXT,
    category        TEXT,
    style           TEXT,
    parent_id       TEXT,
    prompt_id       TEXT    NOT NULL,
    prompt_version  INTEGER NOT NULL,
    content         TEXT    NOT NULL,
    followups       TEXT    NOT NULL DEFAULT '[]',
    provider        TEXT    NOT NULL,
    model           TEXT    NOT NULL,
    pinned          INTEGER NOT NULL DEFAULT 0,
    created_at      INTEGER NOT NULL,
    completed_at    INTEGER,
    updated_at      INTEGER NOT NULL
) STRICT;

CREATE INDEX idx_ai_suggestions_meeting_time ON ai_suggestions (meeting_id, created_at);

-- Topic list of a meeting, replaced as a whole on every save_topics. Topic
-- ids are slugs that are only unique within a meeting; rows are keyed by
-- position so the tracker's ordering round-trips exactly.
CREATE TABLE topics (
    meeting_id          TEXT    NOT NULL REFERENCES meetings (id) ON DELETE CASCADE,
    position            INTEGER NOT NULL,
    id                  TEXT    NOT NULL,
    name                TEXT    NOT NULL,
    status              TEXT    NOT NULL,
    mentions            INTEGER NOT NULL DEFAULT 0,
    first_mentioned_at  INTEGER,
    last_mentioned_at   INTEGER,
    planned             INTEGER NOT NULL DEFAULT 0,
    summary             TEXT,
    updated_at          INTEGER NOT NULL,
    PRIMARY KEY (meeting_id, position)
) STRICT;

CREATE TABLE claims (
    id            TEXT    PRIMARY KEY NOT NULL,
    meeting_id    TEXT    NOT NULL REFERENCES meetings (id) ON DELETE CASCADE,
    text          TEXT    NOT NULL,
    kind          TEXT    NOT NULL,
    segment_id    TEXT,
    speaker_role  TEXT    NOT NULL,
    source        TEXT    NOT NULL,
    created_at    INTEGER NOT NULL,
    updated_at    INTEGER NOT NULL
) STRICT;

CREATE INDEX idx_claims_meeting_time ON claims (meeting_id, created_at);

-- Structured screen observations only; screenshots are never stored.
CREATE TABLE screen_observations (
    id            TEXT    PRIMARY KEY NOT NULL,
    meeting_id    TEXT    NOT NULL REFERENCES meetings (id) ON DELETE CASCADE,
    observed_at   INTEGER NOT NULL,
    source        TEXT    NOT NULL,
    application   TEXT,
    title         TEXT,
    description   TEXT    NOT NULL,
    visible_text  TEXT,
    entities      TEXT    NOT NULL DEFAULT '[]',
    content_type  TEXT,
    frame_hash    TEXT,
    created_at    INTEGER NOT NULL,
    updated_at    INTEGER NOT NULL
) STRICT;

CREATE INDEX idx_screen_observations_meeting_time
    ON screen_observations (meeting_id, observed_at);

CREATE TABLE action_items (
    id                 TEXT    PRIMARY KEY NOT NULL,
    meeting_id         TEXT    NOT NULL REFERENCES meetings (id) ON DELETE CASCADE,
    text               TEXT    NOT NULL,
    owner              TEXT,
    source_segment_id  TEXT,
    created_at         INTEGER NOT NULL,
    updated_at         INTEGER NOT NULL
) STRICT;

CREATE INDEX idx_action_items_meeting_time ON action_items (meeting_id, created_at);

-- At most one summary per meeting (keyed by its meeting).
CREATE TABLE meeting_summaries (
    meeting_id             TEXT    PRIMARY KEY NOT NULL
                                   REFERENCES meetings (id) ON DELETE CASCADE,
    overview               TEXT    NOT NULL,
    key_topics             TEXT    NOT NULL DEFAULT '[]',
    questions_asked        TEXT    NOT NULL DEFAULT '[]',
    answers                TEXT    NOT NULL DEFAULT '[]',
    decisions              TEXT    NOT NULL DEFAULT '[]',
    action_items           TEXT    NOT NULL DEFAULT '[]',
    open_questions         TEXT    NOT NULL DEFAULT '[]',
    technical_topics       TEXT    NOT NULL DEFAULT '[]',
    follow_up_suggestions  TEXT    NOT NULL DEFAULT '[]',
    interviewer            TEXT,
    markdown               TEXT    NOT NULL,
    generated_at           INTEGER NOT NULL,
    generated_by           TEXT    NOT NULL,
    created_at             INTEGER NOT NULL,
    updated_at             INTEGER NOT NULL
) STRICT;

-- One vector per (item, kind, model) so switching embedding models never
-- mixes incomparable vectors; vector holds `dims` little-endian f32 values.
CREATE TABLE embeddings (
    id          INTEGER PRIMARY KEY,
    meeting_id  TEXT    NOT NULL REFERENCES meetings (id) ON DELETE CASCADE,
    item_id     TEXT    NOT NULL,
    kind        TEXT    NOT NULL,
    model       TEXT    NOT NULL,
    dims        INTEGER NOT NULL,
    vector      BLOB    NOT NULL,
    created_at  INTEGER NOT NULL,
    updated_at  INTEGER NOT NULL,
    UNIQUE (meeting_id, item_id, kind, model)
) STRICT;

-- Not meeting data: kept by delete_all_meeting_data and retention.
CREATE TABLE privacy_rules (
    id           TEXT    PRIMARY KEY NOT NULL,
    rule_type    TEXT    NOT NULL,
    value        TEXT    NOT NULL,
    rule_action  TEXT    NOT NULL,
    created_at   INTEGER NOT NULL,
    updated_at   INTEGER NOT NULL
) STRICT;

CREATE INDEX idx_privacy_rules_created_at ON privacy_rules (created_at);

-- Not meeting data: kept by delete_all_meeting_data and retention.
-- The application settings document is stored as JSON under name = 'app'.
CREATE TABLE settings (
    name        TEXT    PRIMARY KEY NOT NULL,
    value       TEXT    NOT NULL,
    updated_at  INTEGER NOT NULL
) STRICT;
