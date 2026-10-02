//! Conversions between domain values and SQLite column values.
//!
//! Error messages produced here never contain the stored value itself (it may
//! be meeting content); they only name the expected type or a position.

use std::fmt;

use rusqlite::types::{FromSql, FromSqlError, FromSqlResult, ToSql, ToSqlOutput, ValueRef};
use serde::Serialize;
use serde::de::{DeserializeOwned, IntoDeserializer};

/// Why a column value could not be converted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CodecError {
    /// The text is not a known variant of the enum.
    UnknownEnumValue { type_name: &'static str },
    /// The value does not serialize to a plain string (not a unit enum).
    NotAUnitEnum { type_name: &'static str },
    /// The column does not hold valid JSON of the expected shape.
    InvalidJson { line: usize, column: usize },
    /// The value could not be serialized to JSON.
    JsonEncoding,
    /// An embedding BLOB whose length is not a whole number of f32 values or
    /// does not match the recorded dimension count.
    InvalidVector { bytes: usize, dims: i64 },
}

impl fmt::Display for CodecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CodecError::UnknownEnumValue { type_name } => {
                write!(f, "unrecognized value for {type_name}")
            }
            CodecError::NotAUnitEnum { type_name } => {
                write!(f, "{type_name} cannot be stored as text")
            }
            CodecError::InvalidJson { line, column } => {
                write!(f, "invalid JSON at line {line}, column {column}")
            }
            CodecError::JsonEncoding => write!(f, "value cannot be encoded as JSON"),
            CodecError::InvalidVector { bytes, dims } => {
                write!(
                    f,
                    "invalid embedding vector: {bytes} bytes for {dims} dimensions"
                )
            }
        }
    }
}

impl std::error::Error for CodecError {}

/// Last path segment of a type name, e.g. `Mode` for `openmeet_core::types::Mode`.
fn short_type_name<T: ?Sized>() -> &'static str {
    let full = std::any::type_name::<T>();
    full.rsplit("::").next().unwrap_or(full)
}

/// The serde name of a unit enum variant, e.g. `"system_design"`.
pub(crate) fn enum_name<T: Serialize>(value: &T) -> Result<String, CodecError> {
    match serde_json::to_value(value) {
        Ok(serde_json::Value::String(name)) => Ok(name),
        _ => Err(CodecError::NotAUnitEnum {
            type_name: short_type_name::<T>(),
        }),
    }
}

/// Parses a unit enum variant from its serde name.
pub(crate) fn parse_enum<T: DeserializeOwned>(name: &str) -> Result<T, CodecError> {
    let deserializer: serde::de::value::StrDeserializer<'_, serde::de::value::Error> =
        name.into_deserializer();
    T::deserialize(deserializer).map_err(|_| CodecError::UnknownEnumValue {
        type_name: short_type_name::<T>(),
    })
}

/// A unit enum stored as its serde (snake_case) name, so the database uses
/// the same vocabulary as the JSON/TypeScript contract
/// (`Mode::Interviewer` <-> `'interviewer'`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SqlEnum<T>(pub T);

impl<T: Serialize> ToSql for SqlEnum<T> {
    fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>> {
        enum_name(&self.0)
            .map(ToSqlOutput::from)
            .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))
    }
}

impl<T: DeserializeOwned> FromSql for SqlEnum<T> {
    fn column_result(value: ValueRef<'_>) -> FromSqlResult<Self> {
        parse_enum(value.as_str()?)
            .map(SqlEnum)
            .map_err(FromSqlError::other)
    }
}

/// A value stored as JSON text (lists such as `segment_ids`, nested objects
/// such as follow-ups or the interviewer summary).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Json<T>(pub T);

impl<T: Serialize> ToSql for Json<T> {
    fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>> {
        serde_json::to_string(&self.0)
            .map(ToSqlOutput::from)
            .map_err(|_| rusqlite::Error::ToSqlConversionFailure(Box::new(CodecError::JsonEncoding)))
    }
}

impl<T: DeserializeOwned> FromSql for Json<T> {
    fn column_result(value: ValueRef<'_>) -> FromSqlResult<Self> {
        serde_json::from_str(value.as_str()?)
            .map(Json)
            .map_err(|e| {
                FromSqlError::other(CodecError::InvalidJson {
                    line: e.line(),
                    column: e.column(),
                })
            })
    }
}

/// Encodes an embedding as consecutive little-endian f32 values.
pub(crate) fn vector_to_blob(vector: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(vector.len() * 4);
    for value in vector {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// Decodes a little-endian f32 BLOB holding exactly `dims` values.
pub(crate) fn blob_to_vector(bytes: &[u8], dims: i64) -> Result<Vec<f32>, CodecError> {
    let expected = usize::try_from(dims)
        .ok()
        .and_then(|d| d.checked_mul(4));
    if expected != Some(bytes.len()) {
        return Err(CodecError::InvalidVector {
            bytes: bytes.len(),
            dims,
        });
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// SQLite stores NaN as NULL, which a NOT NULL REAL column rejects; a NaN
/// score is meaningless anyway, so it is persisted as 0.
pub(crate) fn finite_or_zero(value: f32) -> f64 {
    if value.is_nan() { 0.0 } else { f64::from(value) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openmeet_core::types::{
        AnswerStyle, ClaimKind, DetectionMethod, MeetingState, Mode, PrivacyRuleType,
        QuestionCategory, SpeakerRole,
    };

    #[test]
    fn enum_names_match_serde_contract() {
        assert_eq!(enum_name(&Mode::Interviewer).unwrap(), "interviewer");
        assert_eq!(
            enum_name(&QuestionCategory::SystemDesign).unwrap(),
            "system_design"
        );
        assert_eq!(enum_name(&QuestionCategory::FollowUp).unwrap(), "follow_up");
        assert_eq!(enum_name(&AnswerStyle::StepByStep).unwrap(), "step_by_step");
        assert_eq!(enum_name(&MeetingState::Summarizing).unwrap(), "summarizing");
    }

    #[test]
    fn enums_round_trip_through_names() {
        for category in QuestionCategory::ALL {
            let name = enum_name(&category).unwrap();
            assert_eq!(parse_enum::<QuestionCategory>(&name).unwrap(), category);
        }
        for style in AnswerStyle::ALL {
            let name = enum_name(&style).unwrap();
            assert_eq!(parse_enum::<AnswerStyle>(&name).unwrap(), style);
        }
        for role in [SpeakerRole::User, SpeakerRole::Remote, SpeakerRole::Unknown] {
            let name = enum_name(&role).unwrap();
            assert_eq!(parse_enum::<SpeakerRole>(&name).unwrap(), role);
        }
        assert_eq!(
            parse_enum::<DetectionMethod>("semantic").unwrap(),
            DetectionMethod::Semantic
        );
        assert_eq!(
            parse_enum::<PrivacyRuleType>("domain").unwrap(),
            PrivacyRuleType::Domain
        );
        assert_eq!(parse_enum::<ClaimKind>("risk").unwrap(), ClaimKind::Risk);
    }

    #[test]
    fn unknown_enum_value_error_names_type_but_not_value() {
        let err = parse_enum::<Mode>("secret meeting words").unwrap_err();
        let message = err.to_string();
        assert!(message.contains("Mode"), "{message}");
        assert!(!message.contains("secret"), "{message}");
    }

    #[test]
    fn non_unit_values_are_rejected_as_enums() {
        assert!(matches!(
            enum_name(&vec![1, 2]),
            Err(CodecError::NotAUnitEnum { .. })
        ));
    }

    #[test]
    fn vector_blob_round_trip_is_bit_exact() {
        let vector = vec![
            0.0,
            -0.0,
            1.5,
            -2.25,
            f32::MIN_POSITIVE,
            f32::MAX,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ];
        let blob = vector_to_blob(&vector);
        assert_eq!(blob.len(), vector.len() * 4);
        assert_eq!(&blob[8..12], &1.5f32.to_le_bytes());
        let back = blob_to_vector(&blob, vector.len() as i64).unwrap();
        let bits: Vec<u32> = vector.iter().map(|v| v.to_bits()).collect();
        let back_bits: Vec<u32> = back.iter().map(|v| v.to_bits()).collect();
        assert_eq!(bits, back_bits);
        assert!(blob_to_vector(&[], 0).unwrap().is_empty());
    }

    #[test]
    fn malformed_vector_blobs_are_rejected() {
        assert!(blob_to_vector(&[0, 0, 0], 1).is_err());
        assert!(blob_to_vector(&[0; 8], 1).is_err());
        assert!(blob_to_vector(&[0; 4], -1).is_err());
    }

    #[test]
    fn nan_scores_become_zero() {
        assert_eq!(finite_or_zero(f32::NAN), 0.0);
        assert_eq!(finite_or_zero(0.75), 0.75);
    }
}
