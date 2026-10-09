//! Meeting history: listing, deletion, retention, embeddings and crash
//! recovery.

use openmeet_core::StoreError;
use openmeet_core::settings::RetentionPolicy;
use openmeet_core::store::EmbeddingRecord;
use openmeet_core::types::{MeetingListItem, MeetingState};
use rusqlite::types::Type;
use rusqlite::{Connection, Row, TransactionBehavior, params};
use tracing::info;

use crate::codec::{SqlEnum, blob_to_vector};
use crate::database::{Database, truncate_wal, vacuum};
use crate::error::{SqlResultExt, count_u32};

const LIST_MEETINGS: &str = "
    SELECT m.id, m.title, m.mode, m.state, m.started_at, m.ended_at,
           (SELECT COUNT(*) FROM transcript_segments s WHERE s.meeting_id = m.id)
               AS segment_count,
           (SELECT COUNT(*) FROM questions q WHERE q.meeting_id = m.id) AS question_count,
           EXISTS (SELECT 1 FROM meeting_summaries ms WHERE ms.meeting_id = m.id)
               AS has_summary
    FROM meetings m
    ORDER BY m.started_at DESC, m.id DESC
    LIMIT ?1 OFFSET ?2";

const DELETE_MEETING: &str = "DELETE FROM meetings WHERE id = ?1";

const DELETE_MEETINGS_STARTED_BEFORE: &str = "DELETE FROM meetings WHERE started_at < ?1";

/// Deletes every meeting-owned row, children first. Deleting the child tables
/// explicitly (instead of relying on the cascade alone) also removes rows
/// orphaned while foreign keys were not enforced, e.g. by external tools.
const DELETE_MEETING_CHILDREN: &str = "
    DELETE FROM embeddings;
    DELETE FROM meeting_summaries;
    DELETE FROM action_items;
    DELETE FROM screen_observations;
    DELETE FROM claims;
    DELETE FROM topics;
    DELETE FROM ai_suggestions;
    DELETE FROM answers;
    DELETE FROM questions;
    DELETE FROM transcript_segments;
    DELETE FROM participants;";

const DELETE_ALL_MEETINGS: &str = "DELETE FROM meetings";

const SELECT_EMBEDDINGS: &str = "
    SELECT item_id, meeting_id, kind, model, dims, vector, created_at
    FROM embeddings WHERE meeting_id = ?1
    ORDER BY created_at, id";

/// States in which a meeting is still in progress. Finding one at startup
/// means the app stopped (crash, kill, power loss) before finishing it.
const IN_PROGRESS_STATES: [MeetingState; 5] = [
    MeetingState::Starting,
    MeetingState::Active,
    MeetingState::Paused,
    MeetingState::Stopping,
    MeetingState::Summarizing,
];

const SELECT_IN_PROGRESS: &str = "
    SELECT id, started_at, ended_at FROM meetings
    WHERE state IN (?1, ?2, ?3, ?4, ?5)
    ORDER BY started_at, id";

/// Latest recorded activity of a meeting: the end of its last transcript
/// segment (relative to the meeting start) and the latest absolute timestamp
/// of anything else recorded for it.
const LAST_ACTIVITY: &str = "
    SELECT
        (SELECT MAX(end_time) FROM transcript_segments WHERE meeting_id = ?1),
        (SELECT MAX(t) FROM (
            SELECT MAX(detected_at) AS t FROM questions WHERE meeting_id = ?1
            UNION ALL SELECT MAX(created_at) FROM answers WHERE meeting_id = ?1
            UNION ALL SELECT MAX(COALESCE(completed_at, created_at))
                      FROM ai_suggestions WHERE meeting_id = ?1
            UNION ALL SELECT MAX(created_at) FROM claims WHERE meeting_id = ?1
            UNION ALL SELECT MAX(observed_at) FROM screen_observations WHERE meeting_id = ?1
            UNION ALL SELECT MAX(created_at) FROM action_items WHERE meeting_id = ?1
            UNION ALL SELECT MAX(last_mentioned_at) FROM topics WHERE meeting_id = ?1
        ))";

const COMPLETE_MEETING: &str =
    "UPDATE meetings SET state = ?2, ended_at = ?3, updated_at = ?4 WHERE id = ?1";

impl Database {
    /// Meetings for the history view, newest first (by start time), with
    /// the number of stored transcript segments and questions and whether a
    /// summary exists.
    pub fn list_meetings(
        &self,
        limit: u32,
        offset: u32,
    ) -> Result<Vec<MeetingListItem>, StoreError> {
        let conn = self.lock();
        let mut stmt = conn.prepare_cached(LIST_MEETINGS).db()?;
        let rows = stmt
            .query_map(params![limit, offset], list_item_from_row)
            .db()?;
        let items = rows.collect::<rusqlite::Result<Vec<_>>>().db()?;
        Ok(items)
    }

    /// Deletes a meeting and, through `ON DELETE CASCADE`, everything stored
    /// about it (transcript, questions, answers, suggestions, topics, claims,
    /// observations, action items, summary, embeddings, participants).
    /// Returns [`StoreError::NotFound`] when no such meeting exists.
    pub fn delete_meeting(&self, id: &str) -> Result<(), StoreError> {
        let conn = self.lock();
        let deleted = conn
            .prepare_cached(DELETE_MEETING)
            .db()?
            .execute([id])
            .db()?;
        if deleted == 0 {
            return Err(StoreError::NotFound(id.to_string()));
        }
        truncate_wal(&conn);
        info!(meeting_id = %id, "deleted meeting");
        Ok(())
    }

    /// Deletes all meeting data (keeps settings and privacy rules), then
    /// VACUUMs so the file no longer contains freed pages.
    pub fn delete_all_meeting_data(&self) -> Result<(), StoreError> {
        let mut conn = self.lock();
        let deleted = delete_all_meetings(&mut conn)?;
        vacuum(&conn)?;
        truncate_wal(&conn);
        info!(meetings = deleted, "deleted all meeting data");
        Ok(())
    }

    /// Deletes meetings that started before `now_ms - max_age`, regardless
    /// of their state. [`RetentionPolicy::None`] deletes all meetings (and
    /// compacts the file); [`RetentionPolicy::Forever`] deletes nothing.
    /// Returns the number of meetings deleted.
    pub fn apply_retention(&self, policy: RetentionPolicy, now_ms: i64) -> Result<u32, StoreError> {
        let mut conn = self.lock();
        let deleted = if !policy.persists() {
            delete_all_meetings(&mut conn)?
        } else if let Some(max_age) = policy.max_age_ms() {
            let cutoff = now_ms.saturating_sub(max_age);
            conn.prepare_cached(DELETE_MEETINGS_STARTED_BEFORE)
                .db()?
                .execute([cutoff])
                .db()?
        } else {
            0
        };
        if deleted > 0 {
            if !policy.persists() {
                vacuum(&conn)?;
            }
            truncate_wal(&conn);
            info!(?policy, meetings = deleted, "retention deleted meetings");
        }
        Ok(count_u32(deleted))
    }

    /// Stored embeddings of a meeting, oldest first. Empty when the meeting
    /// has none (or does not exist).
    pub fn load_embeddings(&self, meeting_id: &str) -> Result<Vec<EmbeddingRecord>, StoreError> {
        let conn = self.lock();
        let mut stmt = conn.prepare_cached(SELECT_EMBEDDINGS).db()?;
        let rows = stmt.query_map([meeting_id], embedding_from_row).db()?;
        let embeddings = rows.collect::<rusqlite::Result<Vec<_>>>().db()?;
        Ok(embeddings)
    }

    /// Crash recovery: meetings left in progress (starting, active, paused,
    /// stopping or summarizing) become `Completed`. A missing `ended_at` is
    /// set to the meeting's last recorded activity, clamped to
    /// `[started_at, max(now_ms, started_at)]`. Returns the number of
    /// meetings recovered.
    pub fn recover_interrupted_meetings(&self, now_ms: i64) -> Result<u32, StoreError> {
        let mut conn = self.lock();
        let tx = conn
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .db()?;
        let states = IN_PROGRESS_STATES.map(SqlEnum);
        let interrupted: Vec<(String, i64, Option<i64>)> = {
            let mut stmt = tx.prepare_cached(SELECT_IN_PROGRESS).db()?;
            let rows = stmt
                .query_map(
                    params![states[0], states[1], states[2], states[3], states[4]],
                    |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
                )
                .db()?;
            rows.collect::<rusqlite::Result<_>>().db()?
        };
        for (id, started_at, ended_at) in &interrupted {
            let ended_at = match ended_at {
                Some(ended_at) => *ended_at,
                None => {
                    last_activity(&tx, id, *started_at)?.clamp(*started_at, now_ms.max(*started_at))
                }
            };
            tx.prepare_cached(COMPLETE_MEETING)
                .db()?
                .execute(params![
                    id,
                    SqlEnum(MeetingState::Completed),
                    ended_at,
                    now_ms
                ])
                .db()?;
        }
        tx.commit().db()?;
        if !interrupted.is_empty() {
            info!(
                meetings = interrupted.len(),
                "recovered interrupted meetings"
            );
        }
        Ok(count_u32(interrupted.len()))
    }
}

/// Deletes every meeting and all meeting-owned rows in one transaction;
/// returns the number of meetings deleted.
fn delete_all_meetings(conn: &mut Connection) -> Result<usize, StoreError> {
    let tx = conn
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .db()?;
    tx.execute_batch(DELETE_MEETING_CHILDREN).db()?;
    let deleted = tx.execute(DELETE_ALL_MEETINGS, []).db()?;
    tx.commit().db()?;
    Ok(deleted)
}

/// Latest activity timestamp of a meeting (never earlier than its start).
fn last_activity(conn: &Connection, meeting_id: &str, started_at: i64) -> Result<i64, StoreError> {
    let (last_segment_end, last_event): (Option<i64>, Option<i64>) = conn
        .prepare_cached(LAST_ACTIVITY)
        .db()?
        .query_row([meeting_id], |row| Ok((row.get(0)?, row.get(1)?)))
        .db()?;
    let segment_end = last_segment_end.map(|end| started_at.saturating_add(end));
    Ok([Some(started_at), segment_end, last_event]
        .into_iter()
        .flatten()
        .max()
        .unwrap_or(started_at))
}

fn list_item_from_row(row: &Row<'_>) -> rusqlite::Result<MeetingListItem> {
    Ok(MeetingListItem {
        id: row.get("id")?,
        title: row.get("title")?,
        mode: row.get::<_, SqlEnum<_>>("mode")?.0,
        state: row.get::<_, SqlEnum<_>>("state")?.0,
        started_at: row.get("started_at")?,
        ended_at: row.get("ended_at")?,
        segment_count: count_u32(row.get::<_, i64>("segment_count")?),
        question_count: count_u32(row.get::<_, i64>("question_count")?),
        has_summary: row.get("has_summary")?,
    })
}

fn embedding_from_row(row: &Row<'_>) -> rusqlite::Result<EmbeddingRecord> {
    let dims: i64 = row.get("dims")?;
    let vector_index = row.as_ref().column_index("vector")?;
    let vector = blob_to_vector(row.get_ref(vector_index)?.as_blob()?, dims).map_err(|e| {
        rusqlite::Error::FromSqlConversionFailure(vector_index, Type::Blob, Box::new(e))
    })?;
    Ok(EmbeddingRecord {
        item_id: row.get("item_id")?,
        meeting_id: row.get("meeting_id")?,
        kind: row.get("kind")?,
        model: row.get("model")?,
        vector,
        created_at: row.get("created_at")?,
    })
}
