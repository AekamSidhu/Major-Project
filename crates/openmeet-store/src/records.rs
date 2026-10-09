//! [`MeetingStore`] for [`Database`]: upserts of everything recorded during a
//! meeting and [`MeetingStore::load_meeting`].
//!
//! Every write is an upsert keyed by id (`INSERT … ON CONFLICT DO UPDATE`,
//! never `INSERT OR REPLACE`, which would delete the old row and cascade to
//! its children). Writes of meeting-owned rows for an unknown meeting fail
//! with `StoreError::NotFound(meeting_id)`, like `MemoryStore`.

use openmeet_core::StoreError;
use openmeet_core::now_ms;
use openmeet_core::store::{EmbeddingRecord, MeetingStore};
use openmeet_core::types::{
    ActionItem, AnswerRecord, Claim, DetectedQuestion, InterviewerSummary, Meeting, MeetingRecord,
    MeetingSummary, Participant, ScreenObservation, Suggestion, Topic, TranscriptSegment,
};
use rusqlite::{Connection, OptionalExtension, Row, TransactionBehavior, params};
use tracing::debug;

use crate::codec::{Json, SqlEnum, finite_or_zero, vector_to_blob};
use crate::database::Database;
use crate::error::{SqlResultExt, meeting_write_error};

// ---------------------------------------------------------------------------
// Statements
// ---------------------------------------------------------------------------

const UPSERT_MEETING: &str = "
    INSERT INTO meetings (id, title, mode, state, started_at, ended_at, persisted,
                          created_at, updated_at)
    VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?8)
    ON CONFLICT (id) DO UPDATE SET
        title = excluded.title,
        mode = excluded.mode,
        state = excluded.state,
        started_at = excluded.started_at,
        ended_at = excluded.ended_at,
        persisted = excluded.persisted,
        updated_at = excluded.updated_at";

const DELETE_PARTICIPANTS: &str = "DELETE FROM participants WHERE meeting_id = ?1";

const INSERT_PARTICIPANT: &str = "
    INSERT INTO participants (meeting_id, position, id, role, label, updated_at)
    VALUES (?1, ?2, ?3, ?4, ?5, ?6)";

const UPSERT_SEGMENT: &str = "
    INSERT INTO transcript_segments (id, meeting_id, utterance_id, speaker_id, speaker_role,
                                     source, start_time, end_time, text, is_final, confidence,
                                     created_at, updated_at)
    VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?12)
    ON CONFLICT (id) DO UPDATE SET
        meeting_id = excluded.meeting_id,
        utterance_id = excluded.utterance_id,
        speaker_id = excluded.speaker_id,
        speaker_role = excluded.speaker_role,
        source = excluded.source,
        start_time = excluded.start_time,
        end_time = excluded.end_time,
        text = excluded.text,
        is_final = excluded.is_final,
        confidence = excluded.confidence,
        updated_at = excluded.updated_at";

const UPSERT_QUESTION: &str = "
    INSERT INTO questions (id, meeting_id, question, parts, segment_ids, is_question, confidence,
                           category, speaker_id, speaker_role, requires_response, method,
                           supersedes, detected_at, created_at, updated_at)
    VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?15)
    ON CONFLICT (id) DO UPDATE SET
        meeting_id = excluded.meeting_id,
        question = excluded.question,
        parts = excluded.parts,
        segment_ids = excluded.segment_ids,
        is_question = excluded.is_question,
        confidence = excluded.confidence,
        category = excluded.category,
        speaker_id = excluded.speaker_id,
        speaker_role = excluded.speaker_role,
        requires_response = excluded.requires_response,
        method = excluded.method,
        supersedes = excluded.supersedes,
        detected_at = excluded.detected_at,
        updated_at = excluded.updated_at";

const UPSERT_ANSWER: &str = "
    INSERT INTO answers (id, meeting_id, question_id, speaker_role, text, segment_ids,
                         created_at, updated_at)
    VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
    ON CONFLICT (id) DO UPDATE SET
        meeting_id = excluded.meeting_id,
        question_id = excluded.question_id,
        speaker_role = excluded.speaker_role,
        text = excluded.text,
        segment_ids = excluded.segment_ids,
        created_at = excluded.created_at,
        updated_at = excluded.updated_at";

const UPSERT_SUGGESTION: &str = "
    INSERT INTO ai_suggestions (id, meeting_id, kind, question_id, question_text, category,
                                style, parent_id, prompt_id, prompt_version, content, followups,
                                provider, model, pinned, created_at, completed_at, updated_at)
    VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18)
    ON CONFLICT (id) DO UPDATE SET
        meeting_id = excluded.meeting_id,
        kind = excluded.kind,
        question_id = excluded.question_id,
        question_text = excluded.question_text,
        category = excluded.category,
        style = excluded.style,
        parent_id = excluded.parent_id,
        prompt_id = excluded.prompt_id,
        prompt_version = excluded.prompt_version,
        content = excluded.content,
        followups = excluded.followups,
        provider = excluded.provider,
        model = excluded.model,
        pinned = excluded.pinned,
        created_at = excluded.created_at,
        completed_at = excluded.completed_at,
        updated_at = excluded.updated_at";

const SET_SUGGESTION_PINNED: &str =
    "UPDATE ai_suggestions SET pinned = ?2, updated_at = ?3 WHERE id = ?1";

const MEETING_EXISTS: &str = "SELECT EXISTS (SELECT 1 FROM meetings WHERE id = ?1)";

const DELETE_TOPICS: &str = "DELETE FROM topics WHERE meeting_id = ?1";

const INSERT_TOPIC: &str = "
    INSERT INTO topics (meeting_id, position, id, name, status, mentions, first_mentioned_at,
                        last_mentioned_at, planned, summary, updated_at)
    VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)";

const UPSERT_CLAIM: &str = "
    INSERT INTO claims (id, meeting_id, text, kind, segment_id, speaker_role, source,
                        created_at, updated_at)
    VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
    ON CONFLICT (id) DO UPDATE SET
        meeting_id = excluded.meeting_id,
        text = excluded.text,
        kind = excluded.kind,
        segment_id = excluded.segment_id,
        speaker_role = excluded.speaker_role,
        source = excluded.source,
        created_at = excluded.created_at,
        updated_at = excluded.updated_at";

const UPSERT_OBSERVATION: &str = "
    INSERT INTO screen_observations (id, meeting_id, observed_at, source, application, title,
                                     description, visible_text, entities, content_type,
                                     frame_hash, created_at, updated_at)
    VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?12)
    ON CONFLICT (id) DO UPDATE SET
        meeting_id = excluded.meeting_id,
        observed_at = excluded.observed_at,
        source = excluded.source,
        application = excluded.application,
        title = excluded.title,
        description = excluded.description,
        visible_text = excluded.visible_text,
        entities = excluded.entities,
        content_type = excluded.content_type,
        frame_hash = excluded.frame_hash,
        updated_at = excluded.updated_at";

const UPSERT_ACTION_ITEM: &str = "
    INSERT INTO action_items (id, meeting_id, text, owner, source_segment_id, created_at,
                              updated_at)
    VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
    ON CONFLICT (id) DO UPDATE SET
        meeting_id = excluded.meeting_id,
        text = excluded.text,
        owner = excluded.owner,
        source_segment_id = excluded.source_segment_id,
        created_at = excluded.created_at,
        updated_at = excluded.updated_at";

const UPSERT_SUMMARY: &str = "
    INSERT INTO meeting_summaries (meeting_id, overview, key_topics, questions_asked, answers,
                                   decisions, action_items, open_questions, technical_topics,
                                   follow_up_suggestions, interviewer, markdown, generated_at,
                                   generated_by, created_at, updated_at)
    VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?15)
    ON CONFLICT (meeting_id) DO UPDATE SET
        overview = excluded.overview,
        key_topics = excluded.key_topics,
        questions_asked = excluded.questions_asked,
        answers = excluded.answers,
        decisions = excluded.decisions,
        action_items = excluded.action_items,
        open_questions = excluded.open_questions,
        technical_topics = excluded.technical_topics,
        follow_up_suggestions = excluded.follow_up_suggestions,
        interviewer = excluded.interviewer,
        markdown = excluded.markdown,
        generated_at = excluded.generated_at,
        generated_by = excluded.generated_by,
        updated_at = excluded.updated_at";

const UPSERT_EMBEDDING: &str = "
    INSERT INTO embeddings (meeting_id, item_id, kind, model, dims, vector, created_at,
                            updated_at)
    VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
    ON CONFLICT (meeting_id, item_id, kind, model) DO UPDATE SET
        dims = excluded.dims,
        vector = excluded.vector,
        created_at = excluded.created_at,
        updated_at = excluded.updated_at";

const SELECT_MEETING: &str = "
    SELECT id, title, mode, state, started_at, ended_at, persisted
    FROM meetings WHERE id = ?1";

const SELECT_PARTICIPANTS: &str = "
    SELECT id, role, label FROM participants WHERE meeting_id = ?1 ORDER BY position";

// Chronological; ties broken by first write, then id (UUIDv7, time-ordered).
const SELECT_SEGMENTS: &str = "
    SELECT id, meeting_id, utterance_id, speaker_id, speaker_role, source, start_time,
           end_time, text, is_final, confidence
    FROM transcript_segments WHERE meeting_id = ?1
    ORDER BY start_time, end_time, created_at, id";

const SELECT_QUESTIONS: &str = "
    SELECT id, meeting_id, question, parts, segment_ids, is_question, confidence, category,
           speaker_id, speaker_role, requires_response, method, supersedes, detected_at
    FROM questions WHERE meeting_id = ?1
    ORDER BY detected_at, created_at, id";

const SELECT_ANSWERS: &str = "
    SELECT id, meeting_id, question_id, speaker_role, text, segment_ids, created_at
    FROM answers WHERE meeting_id = ?1
    ORDER BY created_at, id";

const SELECT_SUGGESTIONS: &str = "
    SELECT id, meeting_id, kind, question_id, question_text, category, style, parent_id,
           prompt_id, prompt_version, content, followups, provider, model, pinned,
           created_at, completed_at
    FROM ai_suggestions WHERE meeting_id = ?1
    ORDER BY created_at, id";

const SELECT_TOPICS: &str = "
    SELECT id, meeting_id, name, status, mentions, first_mentioned_at, last_mentioned_at,
           planned, summary
    FROM topics WHERE meeting_id = ?1
    ORDER BY position";

const SELECT_CLAIMS: &str = "
    SELECT id, meeting_id, text, kind, segment_id, speaker_role, source, created_at
    FROM claims WHERE meeting_id = ?1
    ORDER BY created_at, id";

const SELECT_OBSERVATIONS: &str = "
    SELECT id, meeting_id, observed_at, source, application, title, description,
           visible_text, entities, content_type, frame_hash
    FROM screen_observations WHERE meeting_id = ?1
    ORDER BY observed_at, id";

const SELECT_ACTION_ITEMS: &str = "
    SELECT id, meeting_id, text, owner, source_segment_id, created_at
    FROM action_items WHERE meeting_id = ?1
    ORDER BY created_at, id";

const SELECT_SUMMARY: &str = "
    SELECT meeting_id, overview, key_topics, questions_asked, answers, decisions, action_items,
           open_questions, technical_topics, follow_up_suggestions, interviewer, markdown,
           generated_at, generated_by
    FROM meeting_summaries WHERE meeting_id = ?1";

// ---------------------------------------------------------------------------
// Writes
// ---------------------------------------------------------------------------

/// List positions are stored as INTEGER.
fn position(index: usize) -> i64 {
    i64::try_from(index).unwrap_or(i64::MAX)
}

fn meeting_exists(conn: &Connection, meeting_id: &str) -> Result<bool, StoreError> {
    conn.prepare_cached(MEETING_EXISTS)
        .db()?
        .query_row([meeting_id], |row| row.get(0))
        .db()
}

impl MeetingStore for Database {
    /// Inserts or updates the meeting row and replaces its participant list.
    fn save_meeting(&self, meeting: &Meeting) -> Result<(), StoreError> {
        let now = now_ms();
        let mut conn = self.lock();
        let tx = conn
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .db()?;
        tx.prepare_cached(UPSERT_MEETING)
            .db()?
            .execute(params![
                meeting.id,
                meeting.title,
                SqlEnum(meeting.mode),
                SqlEnum(meeting.state),
                meeting.started_at,
                meeting.ended_at,
                meeting.persisted,
                now,
            ])
            .db()?;
        tx.prepare_cached(DELETE_PARTICIPANTS)
            .db()?
            .execute([&meeting.id])
            .db()?;
        {
            let mut insert = tx.prepare_cached(INSERT_PARTICIPANT).db()?;
            for (index, participant) in meeting.participants.iter().enumerate() {
                insert
                    .execute(params![
                        meeting.id,
                        position(index),
                        participant.id,
                        SqlEnum(participant.role),
                        participant.label,
                        now,
                    ])
                    .db()?;
            }
        }
        tx.commit().db()
    }

    fn save_segment(&self, segment: &TranscriptSegment) -> Result<(), StoreError> {
        let conn = self.lock();
        conn.prepare_cached(UPSERT_SEGMENT)
            .db()?
            .execute(params![
                segment.id,
                segment.meeting_id,
                segment.utterance_id,
                segment.speaker_id,
                SqlEnum(segment.speaker_role),
                SqlEnum(segment.source),
                segment.start_time,
                segment.end_time,
                segment.text,
                segment.is_final,
                segment.confidence.map(f64::from),
                now_ms(),
            ])
            .map_err(|e| meeting_write_error(e, &segment.meeting_id))?;
        Ok(())
    }

    fn save_question(&self, question: &DetectedQuestion) -> Result<(), StoreError> {
        let conn = self.lock();
        conn.prepare_cached(UPSERT_QUESTION)
            .db()?
            .execute(params![
                question.id,
                question.meeting_id,
                question.question,
                Json(&question.parts),
                Json(&question.segment_ids),
                question.is_question,
                finite_or_zero(question.confidence),
                SqlEnum(question.category),
                question.speaker_id,
                SqlEnum(question.speaker_role),
                question.requires_response,
                SqlEnum(question.method),
                question.supersedes,
                question.detected_at,
                now_ms(),
            ])
            .map_err(|e| meeting_write_error(e, &question.meeting_id))?;
        Ok(())
    }

    fn save_answer(&self, answer: &AnswerRecord) -> Result<(), StoreError> {
        let conn = self.lock();
        conn.prepare_cached(UPSERT_ANSWER)
            .db()?
            .execute(params![
                answer.id,
                answer.meeting_id,
                answer.question_id,
                SqlEnum(answer.speaker_role),
                answer.text,
                Json(&answer.segment_ids),
                answer.created_at,
                now_ms(),
            ])
            .map_err(|e| meeting_write_error(e, &answer.meeting_id))?;
        Ok(())
    }

    /// Suggestions without a meeting (e.g. "Ask OpenMeet" outside a meeting)
    /// are not part of any meeting history and are not persisted.
    fn save_suggestion(&self, suggestion: &Suggestion) -> Result<(), StoreError> {
        let Some(meeting_id) = suggestion.meeting_id.as_deref() else {
            debug!(suggestion_id = %suggestion.id, "suggestion without a meeting not persisted");
            return Ok(());
        };
        let conn = self.lock();
        conn.prepare_cached(UPSERT_SUGGESTION)
            .db()?
            .execute(params![
                suggestion.id,
                meeting_id,
                SqlEnum(suggestion.kind),
                suggestion.question_id,
                suggestion.question_text,
                suggestion.category.map(SqlEnum),
                suggestion.style.map(SqlEnum),
                suggestion.parent_id,
                suggestion.prompt_id,
                suggestion.prompt_version,
                suggestion.content,
                Json(&suggestion.followups),
                suggestion.provider,
                suggestion.model,
                suggestion.pinned,
                suggestion.created_at,
                suggestion.completed_at,
                now_ms(),
            ])
            .map_err(|e| meeting_write_error(e, meeting_id))?;
        Ok(())
    }

    fn set_suggestion_pinned(&self, suggestion_id: &str, pinned: bool) -> Result<(), StoreError> {
        let conn = self.lock();
        let updated = conn
            .prepare_cached(SET_SUGGESTION_PINNED)
            .db()?
            .execute(params![suggestion_id, pinned, now_ms()])
            .db()?;
        if updated == 0 {
            return Err(StoreError::NotFound(suggestion_id.to_string()));
        }
        Ok(())
    }

    /// Replaces the meeting's topic list; the order of `topics` is kept and
    /// every stored topic is attributed to `meeting_id`.
    fn save_topics(&self, meeting_id: &str, topics: &[Topic]) -> Result<(), StoreError> {
        let now = now_ms();
        let mut conn = self.lock();
        let tx = conn
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .db()?;
        if !meeting_exists(&tx, meeting_id)? {
            return Err(StoreError::NotFound(meeting_id.to_string()));
        }
        tx.prepare_cached(DELETE_TOPICS)
            .db()?
            .execute([meeting_id])
            .db()?;
        {
            let mut insert = tx.prepare_cached(INSERT_TOPIC).db()?;
            for (index, topic) in topics.iter().enumerate() {
                insert
                    .execute(params![
                        meeting_id,
                        position(index),
                        topic.id,
                        topic.name,
                        SqlEnum(topic.status),
                        topic.mentions,
                        topic.first_mentioned_at,
                        topic.last_mentioned_at,
                        topic.planned,
                        topic.summary,
                        now,
                    ])
                    .db()?;
            }
        }
        tx.commit().db()
    }

    /// Upserts all claims atomically: if one refers to an unknown meeting,
    /// none are written.
    fn save_claims(&self, claims: &[Claim]) -> Result<(), StoreError> {
        if claims.is_empty() {
            return Ok(());
        }
        let now = now_ms();
        let mut conn = self.lock();
        let tx = conn
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .db()?;
        {
            let mut upsert = tx.prepare_cached(UPSERT_CLAIM).db()?;
            for claim in claims {
                upsert
                    .execute(params![
                        claim.id,
                        claim.meeting_id,
                        claim.text,
                        SqlEnum(claim.kind),
                        claim.segment_id,
                        SqlEnum(claim.speaker_role),
                        SqlEnum(claim.source),
                        claim.created_at,
                        now,
                    ])
                    .map_err(|e| meeting_write_error(e, &claim.meeting_id))?;
            }
        }
        tx.commit().db()
    }

    /// Observations without a meeting (screen context outside a meeting) are
    /// not persisted. Only the structured observation is stored, never the
    /// frame itself.
    fn save_observation(&self, observation: &ScreenObservation) -> Result<(), StoreError> {
        let Some(meeting_id) = observation.meeting_id.as_deref() else {
            debug!(observation_id = %observation.id, "observation without a meeting not persisted");
            return Ok(());
        };
        let conn = self.lock();
        conn.prepare_cached(UPSERT_OBSERVATION)
            .db()?
            .execute(params![
                observation.id,
                meeting_id,
                observation.timestamp,
                observation.source,
                observation.application,
                observation.title,
                observation.description,
                observation.visible_text,
                Json(&observation.entities),
                observation.content_type,
                observation.frame_hash,
                now_ms(),
            ])
            .map_err(|e| meeting_write_error(e, meeting_id))?;
        Ok(())
    }

    /// Upserts all action items atomically (all or none).
    fn save_action_items(&self, items: &[ActionItem]) -> Result<(), StoreError> {
        if items.is_empty() {
            return Ok(());
        }
        let now = now_ms();
        let mut conn = self.lock();
        let tx = conn
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .db()?;
        {
            let mut upsert = tx.prepare_cached(UPSERT_ACTION_ITEM).db()?;
            for item in items {
                upsert
                    .execute(params![
                        item.id,
                        item.meeting_id,
                        item.text,
                        item.owner,
                        item.source_segment_id,
                        item.created_at,
                        now,
                    ])
                    .map_err(|e| meeting_write_error(e, &item.meeting_id))?;
            }
        }
        tx.commit().db()
    }

    /// A meeting has at most one summary; saving again replaces it.
    fn save_summary(&self, summary: &MeetingSummary) -> Result<(), StoreError> {
        let conn = self.lock();
        conn.prepare_cached(UPSERT_SUMMARY)
            .db()?
            .execute(params![
                summary.meeting_id,
                summary.overview,
                Json(&summary.key_topics),
                Json(&summary.questions_asked),
                Json(&summary.answers),
                Json(&summary.decisions),
                Json(&summary.action_items),
                Json(&summary.open_questions),
                Json(&summary.technical_topics),
                Json(&summary.follow_up_suggestions),
                summary.interviewer.as_ref().map(Json),
                summary.markdown,
                summary.generated_at,
                summary.generated_by,
                now_ms(),
            ])
            .map_err(|e| meeting_write_error(e, &summary.meeting_id))?;
        Ok(())
    }

    /// Keyed by (meeting, item, kind, model): re-embedding an item with the
    /// same model replaces its vector; other models keep their own rows.
    fn save_embedding(&self, embedding: &EmbeddingRecord) -> Result<(), StoreError> {
        let dims = i64::try_from(embedding.vector.len())
            .map_err(|_| StoreError::Serialization("embedding vector is too large".to_string()))?;
        let blob = vector_to_blob(&embedding.vector);
        let conn = self.lock();
        conn.prepare_cached(UPSERT_EMBEDDING)
            .db()?
            .execute(params![
                embedding.meeting_id,
                embedding.item_id,
                embedding.kind,
                embedding.model,
                dims,
                blob,
                embedding.created_at,
                now_ms(),
            ])
            .map_err(|e| meeting_write_error(e, &embedding.meeting_id))?;
        Ok(())
    }

    /// Everything stored about a meeting, read from one consistent snapshot.
    /// Lists are chronological (segments by start time, questions by
    /// detection time, observations by capture time, the rest by creation
    /// time); topics and participants keep the order they were saved in.
    fn load_meeting(&self, meeting_id: &str) -> Result<MeetingRecord, StoreError> {
        let mut conn = self.lock();
        let tx = conn.transaction().db()?;
        let Some(mut meeting) = tx
            .prepare_cached(SELECT_MEETING)
            .db()?
            .query_row([meeting_id], meeting_from_row)
            .optional()
            .db()?
        else {
            return Err(StoreError::NotFound(meeting_id.to_string()));
        };
        meeting.participants =
            query_list(&tx, SELECT_PARTICIPANTS, meeting_id, participant_from_row)?;
        let record = MeetingRecord {
            segments: query_list(&tx, SELECT_SEGMENTS, meeting_id, segment_from_row)?,
            questions: query_list(&tx, SELECT_QUESTIONS, meeting_id, question_from_row)?,
            answers: query_list(&tx, SELECT_ANSWERS, meeting_id, answer_from_row)?,
            suggestions: query_list(&tx, SELECT_SUGGESTIONS, meeting_id, suggestion_from_row)?,
            topics: query_list(&tx, SELECT_TOPICS, meeting_id, topic_from_row)?,
            claims: query_list(&tx, SELECT_CLAIMS, meeting_id, claim_from_row)?,
            observations: query_list(&tx, SELECT_OBSERVATIONS, meeting_id, observation_from_row)?,
            action_items: query_list(&tx, SELECT_ACTION_ITEMS, meeting_id, action_item_from_row)?,
            summary: tx
                .prepare_cached(SELECT_SUMMARY)
                .db()?
                .query_row([meeting_id], summary_from_row)
                .optional()
                .db()?,
            meeting,
        };
        tx.commit().db()?;
        Ok(record)
    }
}

// ---------------------------------------------------------------------------
// Reads
// ---------------------------------------------------------------------------

type RowMapper<T> = fn(&Row<'_>) -> rusqlite::Result<T>;

/// Runs a per-meeting list query and maps every row.
fn query_list<T>(
    conn: &Connection,
    sql: &str,
    meeting_id: &str,
    map: RowMapper<T>,
) -> Result<Vec<T>, StoreError> {
    let mut stmt = conn.prepare_cached(sql).db()?;
    let rows = stmt.query_map([meeting_id], map).db()?;
    rows.collect::<rusqlite::Result<Vec<T>>>().db()
}

fn meeting_from_row(row: &Row<'_>) -> rusqlite::Result<Meeting> {
    Ok(Meeting {
        id: row.get("id")?,
        title: row.get("title")?,
        mode: row.get::<_, SqlEnum<_>>("mode")?.0,
        state: row.get::<_, SqlEnum<_>>("state")?.0,
        started_at: row.get("started_at")?,
        ended_at: row.get("ended_at")?,
        participants: Vec::new(),
        persisted: row.get("persisted")?,
    })
}

fn participant_from_row(row: &Row<'_>) -> rusqlite::Result<Participant> {
    Ok(Participant {
        id: row.get("id")?,
        role: row.get::<_, SqlEnum<_>>("role")?.0,
        label: row.get("label")?,
    })
}

fn segment_from_row(row: &Row<'_>) -> rusqlite::Result<TranscriptSegment> {
    Ok(TranscriptSegment {
        id: row.get("id")?,
        meeting_id: row.get("meeting_id")?,
        utterance_id: row.get("utterance_id")?,
        speaker_id: row.get("speaker_id")?,
        speaker_role: row.get::<_, SqlEnum<_>>("speaker_role")?.0,
        source: row.get::<_, SqlEnum<_>>("source")?.0,
        start_time: row.get("start_time")?,
        end_time: row.get("end_time")?,
        text: row.get("text")?,
        is_final: row.get("is_final")?,
        confidence: row.get("confidence")?,
    })
}

fn question_from_row(row: &Row<'_>) -> rusqlite::Result<DetectedQuestion> {
    Ok(DetectedQuestion {
        id: row.get("id")?,
        meeting_id: row.get("meeting_id")?,
        segment_ids: row.get::<_, Json<_>>("segment_ids")?.0,
        is_question: row.get("is_question")?,
        confidence: row.get("confidence")?,
        question: row.get("question")?,
        parts: row.get::<_, Json<_>>("parts")?.0,
        category: row.get::<_, SqlEnum<_>>("category")?.0,
        speaker_id: row.get("speaker_id")?,
        speaker_role: row.get::<_, SqlEnum<_>>("speaker_role")?.0,
        requires_response: row.get("requires_response")?,
        method: row.get::<_, SqlEnum<_>>("method")?.0,
        supersedes: row.get("supersedes")?,
        detected_at: row.get("detected_at")?,
    })
}

fn answer_from_row(row: &Row<'_>) -> rusqlite::Result<AnswerRecord> {
    Ok(AnswerRecord {
        id: row.get("id")?,
        meeting_id: row.get("meeting_id")?,
        question_id: row.get("question_id")?,
        speaker_role: row.get::<_, SqlEnum<_>>("speaker_role")?.0,
        text: row.get("text")?,
        segment_ids: row.get::<_, Json<_>>("segment_ids")?.0,
        created_at: row.get("created_at")?,
    })
}

fn suggestion_from_row(row: &Row<'_>) -> rusqlite::Result<Suggestion> {
    Ok(Suggestion {
        id: row.get("id")?,
        meeting_id: row.get("meeting_id")?,
        kind: row.get::<_, SqlEnum<_>>("kind")?.0,
        question_id: row.get("question_id")?,
        question_text: row.get("question_text")?,
        category: row.get::<_, Option<SqlEnum<_>>>("category")?.map(|c| c.0),
        style: row.get::<_, Option<SqlEnum<_>>>("style")?.map(|s| s.0),
        parent_id: row.get("parent_id")?,
        prompt_id: row.get("prompt_id")?,
        prompt_version: row.get("prompt_version")?,
        content: row.get("content")?,
        followups: row.get::<_, Json<_>>("followups")?.0,
        provider: row.get("provider")?,
        model: row.get("model")?,
        created_at: row.get("created_at")?,
        completed_at: row.get("completed_at")?,
        pinned: row.get("pinned")?,
    })
}

fn topic_from_row(row: &Row<'_>) -> rusqlite::Result<Topic> {
    Ok(Topic {
        id: row.get("id")?,
        meeting_id: row.get("meeting_id")?,
        name: row.get("name")?,
        status: row.get::<_, SqlEnum<_>>("status")?.0,
        mentions: row.get("mentions")?,
        first_mentioned_at: row.get("first_mentioned_at")?,
        last_mentioned_at: row.get("last_mentioned_at")?,
        planned: row.get("planned")?,
        summary: row.get("summary")?,
    })
}

fn claim_from_row(row: &Row<'_>) -> rusqlite::Result<Claim> {
    Ok(Claim {
        id: row.get("id")?,
        meeting_id: row.get("meeting_id")?,
        text: row.get("text")?,
        kind: row.get::<_, SqlEnum<_>>("kind")?.0,
        segment_id: row.get("segment_id")?,
        speaker_role: row.get::<_, SqlEnum<_>>("speaker_role")?.0,
        source: row.get::<_, SqlEnum<_>>("source")?.0,
        created_at: row.get("created_at")?,
    })
}

fn observation_from_row(row: &Row<'_>) -> rusqlite::Result<ScreenObservation> {
    Ok(ScreenObservation {
        id: row.get("id")?,
        meeting_id: row.get("meeting_id")?,
        timestamp: row.get("observed_at")?,
        source: row.get("source")?,
        application: row.get("application")?,
        title: row.get("title")?,
        description: row.get("description")?,
        visible_text: row.get("visible_text")?,
        entities: row.get::<_, Json<_>>("entities")?.0,
        content_type: row.get("content_type")?,
        frame_hash: row.get("frame_hash")?,
    })
}

fn action_item_from_row(row: &Row<'_>) -> rusqlite::Result<ActionItem> {
    Ok(ActionItem {
        id: row.get("id")?,
        meeting_id: row.get("meeting_id")?,
        text: row.get("text")?,
        owner: row.get("owner")?,
        source_segment_id: row.get("source_segment_id")?,
        created_at: row.get("created_at")?,
    })
}

fn summary_from_row(row: &Row<'_>) -> rusqlite::Result<MeetingSummary> {
    Ok(MeetingSummary {
        meeting_id: row.get("meeting_id")?,
        overview: row.get("overview")?,
        key_topics: row.get::<_, Json<_>>("key_topics")?.0,
        questions_asked: row.get::<_, Json<_>>("questions_asked")?.0,
        answers: row.get::<_, Json<_>>("answers")?.0,
        decisions: row.get::<_, Json<_>>("decisions")?.0,
        action_items: row.get::<_, Json<_>>("action_items")?.0,
        open_questions: row.get::<_, Json<_>>("open_questions")?.0,
        technical_topics: row.get::<_, Json<_>>("technical_topics")?.0,
        follow_up_suggestions: row.get::<_, Json<_>>("follow_up_suggestions")?.0,
        interviewer: row
            .get::<_, Option<Json<InterviewerSummary>>>("interviewer")?
            .map(|i| i.0),
        markdown: row.get("markdown")?,
        generated_at: row.get("generated_at")?,
        generated_by: row.get("generated_by")?,
    })
}
