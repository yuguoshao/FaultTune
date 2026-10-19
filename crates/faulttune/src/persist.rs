//! Persistence and checkpoints (ARCHITECTURE §10). Requires the `serde` feature.
//!
//! **Saved calibrations** are JSON keyed by element label (format `faulttune.parameters/1`):
//!
//! ```json
//! { "format": "faulttune.parameters/1",
//!   "graph": { "fingerprint": "9f2c…", "elements": 214 },
//!   "values": { "q1_3": { "value": 6.12, "option": 56, "step": 17 } },
//!   "history": [ { "type": "step", "index": 17, "pass": 0, "subgoal": 2, "centre": "q1_3",
//!                  "parameters": ["q1_3", "q1_3-q1_4"], "options": [56, 43],
//!                  "constraints": ["q0_3", "q1_2"],
//!                  "error": 0.0031, "evaluations": 1200, "duration_ms": 4 },
//!                { "type": "invalidated", "elements": ["q1_3"] } ] }
//! ```
//!
//! * `fingerprint` is the graph fingerprint as 16 lower-case hexadecimal digits; `values` is
//!   written in ascending element-ID order, so files diff well.
//! * Durations are stored as fractional milliseconds (`duration_ms`, and the optional
//!   `prepare_ms` and `evaluate_ms`); sub-nanosecond rounding is not preserved.
//! * An `error` that is not finite (`+∞`, which is a legal result meaning "nothing feasible
//!   was found", or NaN) is stored as `null`, because JSON has no infinity. `null` is read
//!   back as `+∞`; NaN and `-∞` therefore do not survive a round trip. A step error is never
//!   NaN (the engine rejects it) and never `-∞` in practice.
//! * Values go through `serde`, so `V` must round-trip through JSON; `f64` values round-trip
//!   bit for bit when `serde_json` is built with its `float_roundtrip` feature (FaultTune's
//!   `Cargo.toml` enables it). A value that would not read back (for example a non-finite
//!   `f64`, which becomes `null`, alone or inside a vector) is refused when saving with a
//!   `Format` error, so that a file written by `save` can always be loaded.
//!
//! The `format` field is versioned (`faulttune.parameters/<n>`); newer versions read older
//! formats. Loading checks the format name, the graph fingerprint (`GraphMismatch` on a
//! different device) and that every label exists in the graph (`Format` otherwise).
//!
//! **Checkpoints** are an append-only journal written by [`CheckpointObserver`]: one JSON line
//! per event, flushed as it is written. A `StepCommitted` event becomes a `step` line and an
//! `Invalidated` event (what `Snake::recalibrate` announces after it removed elements from the
//! database) an `invalidated` line; the other events are not journaled:
//!
//! ```json
//! {"type":"step","index":0,"pass":3,"subgoal":2,"centre":"q1_3","parameters":["q1_3","q1_3-q1_4"],"options":[56,43],"constraints":["q0_3","q1_2"],"error":0.0031,"evaluations":1200,"duration_ms":4.0}
//! {"type":"invalidated","pass":4,"elements":["q1_3","q1_3-q1_4"]}
//! ```
//!
//! `pass` and `subgoal` are those of the step (`StepRecord::pass` and `StepRecord::subgoal`),
//! and `index` is the step's ordinal *within its subgoal and pass*, which is what
//! `Event::StepCommitted` carries (the global index of the database is assigned when the
//! engine merges the subgoals). An invalidated line carries the number the next pass will
//! have (`Event::Invalidated::pass`). Pass numbers continue across the `calibrate_*` calls on
//! one database, so `(pass, subgoal, centre)` identifies a step for the whole life of a
//! database. Every one of these fields is required: a line that lacks one is a `Format`
//! error.
//!
//! **Canonical order.** The events of subgoals that run in parallel interleave, so the order of
//! the lines in the file is *not* the order of the history of the database, in which the
//! engine merges the subgoals of a pass one after the other. The canonical order of journal
//! entries is the order of the history: by `pass`; inside a pass an invalidation comes first
//! (it was announced before the pass started), then the steps by `(subgoal, index)`; entries
//! that tie keep the order of the file. [`read_journal_entries`] and [`read_journal`] return the
//! entries in canonical order, and the replay sorts into it too, so the file order matters
//! nowhere. The journal of a run therefore replays into the database of that run, whatever the
//! number of threads was.
//!
//! A torn last line (JSON that stops short) is ignored on replay, so a crash loses at most the
//! line being written. Read a journal with [`read_journal_entries`], which returns the steps
//! and the invalidations ([`JournalEntry`]), and re-apply it with
//! [`ParameterDatabase::replay_entries`]: a replay onto an older saved database then
//! reproduces the removals of a `recalibrate` as well as its steps. [`read_journal`] and
//! [`ParameterDatabase::replay_journal`] work on the steps alone (the former skips the
//! invalidations), for journals that never had invalidations or when forgetting is not wanted.
//!
//! **Replay** makes no assumption about how the journal relates to the database. A step is
//! skipped if the history of the database already contains a step with the same
//! `(pass, subgoal, centre)`, and applied otherwise; an invalidation removes the elements that
//! are calibrated by a step of an earlier pass than its own and ignores the others. So a replay
//! onto an empty database, onto a save taken at any time, onto the result of an earlier replay,
//! and the replay of the journals of several runs of one database (a run that was resumed may
//! start a new journal, or append to the old one) all give the database of the live run, and
//! replaying twice changes nothing. Recovery after a crash is: load the last save, replay the
//! journal (or the journals, oldest first), and continue with `calibrate_into`.
//! A crash in the middle of a parallel pass leaves some subgoals with their steps in the
//! journal and others without; the replay applies the survivors in canonical order (a valid
//! database, though not a prefix of the history the uninterrupted run would have had), and the
//! resumed run calibrates what is still missing in later passes.
//!
//! The journal is handed to the operating system with every line, so it survives the death of
//! the process; use [`CheckpointObserver::durable`] to survive a crash of the machine too, and
//! check [`CheckpointObserver::take_error`] after a run.

use std::collections::HashMap;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, Read, Seek, SeekFrom, Write};
use std::marker::PhantomData;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use serde::de::{DeserializeOwned, Deserializer, MapAccess, Visitor};
use serde::ser::{SerializeMap, Serializer};
use serde::{Deserialize, Serialize};

use crate::database::{Entry, HistoryRecord, ParameterDatabase, StepRecord};
use crate::error::FaultTuneError;
use crate::events::{Event, Observer};
use crate::graph::{ElementId, ProcessorGraph};

/// Name and version of the saved-calibration format written today.
pub const PARAMETERS_FORMAT: &str = "faulttune.parameters/1";

/// The part of the format string before the version number.
const PARAMETERS_FORMAT_NAME: &str = "faulttune.parameters/";

// ---------------------------------------------------------------------------------------
// JSON shapes
// ---------------------------------------------------------------------------------------

/// A step as written to a saved calibration or to the journal. `index` is the global index in
/// a saved calibration and the ordinal within the subgoal in the journal (that is
/// `StepRecord::index` in both cases). The timing details are written to saved calibrations
/// only.
#[derive(Serialize)]
struct StepOut<'a> {
    index: u64,
    pass: u32,
    subgoal: u32,
    centre: &'a str,
    parameters: Vec<&'a str>,
    options: &'a [u32],
    constraints: Vec<&'a str>,
    /// `None` (JSON `null`) stands for a non-finite error.
    error: Option<f64>,
    evaluations: u64,
    duration_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    prepare_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    evaluate_ms: Option<f64>,
}

/// An entry of the saved `history` array and the shape of a journal line.
#[derive(Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum LineOut<'a> {
    Step(StepOut<'a>),
    Invalidated {
        /// Present in journal lines; the history of a database does not record it.
        #[serde(skip_serializing_if = "Option::is_none")]
        pass: Option<u32>,
        elements: Vec<&'a str>,
    },
}

#[derive(Serialize)]
struct GraphInfoOut {
    fingerprint: String,
    elements: usize,
}

#[derive(Serialize)]
struct EntryOut<'a, V> {
    value: &'a V,
    option: u32,
    step: u64,
}

/// The `values` object, written in ascending element-ID order.
struct ValuesOut<'a, V> {
    graph: &'a ProcessorGraph,
    db: &'a ParameterDatabase<V>,
}

impl<V: Serialize> Serialize for ValuesOut<'_, V> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(self.db.len()))?;
        for (id, entry) in self.db.iter() {
            map.serialize_entry(
                self.graph.label(id),
                &EntryOut {
                    value: &entry.value,
                    option: entry.option_index,
                    step: entry.step,
                },
            )?;
        }
        map.end()
    }
}

#[derive(Serialize)]
struct FileOut<'a, V> {
    format: &'static str,
    graph: GraphInfoOut,
    values: ValuesOut<'a, V>,
    history: Vec<LineOut<'a>>,
}

/// Only the `format` field: read first to choose the reader for the file's version.
#[derive(Deserialize)]
struct Header {
    format: String,
}

#[derive(Deserialize)]
struct GraphInfoIn {
    fingerprint: String,
    elements: usize,
}

#[derive(Deserialize)]
struct EntryIn<V> {
    value: V,
    option: u32,
    step: u64,
}

/// A JSON object read as a list of pairs, so that duplicate keys can be detected.
struct Pairs<T>(Vec<(String, T)>);

impl<'de, T: Deserialize<'de>> Deserialize<'de> for Pairs<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct PairsVisitor<T>(PhantomData<T>);

        impl<'de, T: Deserialize<'de>> Visitor<'de> for PairsVisitor<T> {
            type Value = Pairs<T>;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("an object keyed by element label")
            }

            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Pairs<T>, A::Error> {
                let mut pairs = Vec::with_capacity(map.size_hint().unwrap_or(0).min(1 << 20));
                while let Some(pair) = map.next_entry::<String, T>()? {
                    pairs.push(pair);
                }
                Ok(Pairs(pairs))
            }
        }

        deserializer.deserialize_map(PairsVisitor(PhantomData))
    }
}

/// Reads `error`: `null` stands for `+∞`. The key itself is required.
fn deserialize_error<'de, D: Deserializer<'de>>(deserializer: D) -> Result<f64, D::Error> {
    Ok(Option::<f64>::deserialize(deserializer)?.unwrap_or(f64::INFINITY))
}

#[derive(Deserialize)]
struct StepIn {
    /// The global index in a saved calibration (required there, and checked), the ordinal
    /// within the subgoal in a journal line (required there too).
    #[serde(default)]
    index: Option<u64>,
    /// Required in journal lines; a saved calibration may omit it (0).
    #[serde(default)]
    pass: Option<u32>,
    /// Required in journal lines; a saved calibration may omit it (0).
    #[serde(default)]
    subgoal: Option<u32>,
    centre: String,
    parameters: Vec<String>,
    options: Vec<u32>,
    constraints: Vec<String>,
    #[serde(deserialize_with = "deserialize_error")]
    error: f64,
    evaluations: u64,
    duration_ms: f64,
    #[serde(default)]
    prepare_ms: Option<f64>,
    #[serde(default)]
    evaluate_ms: Option<f64>,
}

/// An entry of the saved `history` array, and equally a journal line (the two share their
/// record types). Other record types are not part of the format today.
#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum HistoryIn {
    Step(StepIn),
    Invalidated { elements: Vec<String> },
}

/// A journal line: the two record types of the saved history, with the fields that only the
/// journal requires.
#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum JournalLineIn {
    Step(StepIn),
    Invalidated { pass: u32, elements: Vec<String> },
}

#[derive(Deserialize)]
struct FileV1<V> {
    graph: GraphInfoIn,
    values: Pairs<EntryIn<V>>,
    history: Vec<HistoryIn>,
}

// ---------------------------------------------------------------------------------------
// Conversions
// ---------------------------------------------------------------------------------------

fn format_err(msg: impl Into<String>) -> FaultTuneError {
    FaultTuneError::Format(msg.into())
}

fn to_ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

fn from_ms(ms: f64, field: &str) -> Result<Duration, FaultTuneError> {
    if ms.is_nan() || ms < 0.0 {
        return Err(format_err(format!("invalid {field}: {ms}")));
    }
    Duration::try_from_secs_f64(ms / 1000.0)
        .map_err(|_| format_err(format!("invalid {field}: {ms}")))
}

fn hex16(value: u64) -> String {
    format!("{value:016x}")
}

fn parse_hex16(text: &str) -> Result<u64, FaultTuneError> {
    if text.len() != 16 || !text.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(format_err(format!(
            "graph fingerprint {text:?} is not 16 hexadecimal digits"
        )));
    }
    u64::from_str_radix(text, 16)
        .map_err(|_| format_err(format!("graph fingerprint {text:?} is not hexadecimal")))
}

/// Looks a label up in the graph.
fn lookup(graph: &ProcessorGraph, label: &str, context: &str) -> Result<ElementId, FaultTuneError> {
    graph
        .id_of(label)
        .ok_or_else(|| format_err(format!("unknown element label {label:?} in {context}")))
}

fn lookup_all(
    graph: &ProcessorGraph,
    labels: &[String],
    context: &str,
) -> Result<Vec<ElementId>, FaultTuneError> {
    labels
        .iter()
        .map(|label| lookup(graph, label, context))
        .collect()
}

/// A label for rendering; fails (instead of panicking) for an ID outside the graph.
fn label_of(labels: &[String], id: ElementId) -> io::Result<&str> {
    labels.get(id.index()).map(String::as_str).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "element {} is not in the journal's graph ({} elements)",
                id.index(),
                labels.len()
            ),
        )
    })
}

fn step_out<'a>(
    labels: &'a [String],
    record: &'a StepRecord,
    saved: bool,
) -> io::Result<StepOut<'a>> {
    let names = |ids: &[ElementId]| {
        ids.iter()
            .map(|&id| label_of(labels, id))
            .collect::<io::Result<Vec<&'a str>>>()
    };
    Ok(StepOut {
        index: record.index,
        pass: record.pass,
        subgoal: record.subgoal,
        centre: label_of(labels, record.centre)?,
        parameters: names(&record.parameters)?,
        options: &record.option_indices,
        constraints: names(&record.constraints)?,
        error: record.error.is_finite().then_some(record.error),
        evaluations: record.evaluations,
        duration_ms: to_ms(record.duration),
        prepare_ms: saved.then(|| to_ms(record.prepare_time)),
        evaluate_ms: saved.then(|| to_ms(record.error_time)),
    })
}

/// Converts a parsed step, resolving labels. `index` is the step's index in its container (a
/// saved calibration), or `None` for a journal line, which must carry its `index`, `pass` and
/// `subgoal` itself.
fn step_in(
    graph: &ProcessorGraph,
    step: StepIn,
    index: Option<u64>,
    context: &str,
) -> Result<StepRecord, FaultTuneError> {
    let missing = |field: &str| format_err(format!("{context}: missing field `{field}`"));
    let (index, pass, subgoal) = match index {
        Some(index) => (
            index,
            step.pass.unwrap_or_default(),
            step.subgoal.unwrap_or_default(),
        ),
        None => (
            step.index.ok_or_else(|| missing("index"))?,
            step.pass.ok_or_else(|| missing("pass"))?,
            step.subgoal.ok_or_else(|| missing("subgoal"))?,
        ),
    };
    if step.parameters.len() != step.options.len() {
        return Err(format_err(format!(
            "{context}: {} parameters but {} options",
            step.parameters.len(),
            step.options.len()
        )));
    }
    let optional_ms =
        |value: Option<f64>, field: &str| value.map_or(Ok(Duration::ZERO), |ms| from_ms(ms, field));
    Ok(StepRecord {
        index,
        pass,
        subgoal,
        centre: lookup(graph, &step.centre, context)?,
        parameters: lookup_all(graph, &step.parameters, context)?,
        option_indices: step.options,
        constraints: lookup_all(graph, &step.constraints, context)?,
        error: step.error,
        evaluations: step.evaluations,
        prepare_time: optional_ms(step.prepare_ms, "prepare_ms")?,
        error_time: optional_ms(step.evaluate_ms, "evaluate_ms")?,
        duration: from_ms(step.duration_ms, "duration_ms")?,
    })
}

/// The version number of a `faulttune.parameters/<n>` format string.
fn parse_format_version(format: &str) -> Result<u32, FaultTuneError> {
    let version = format.strip_prefix(PARAMETERS_FORMAT_NAME).ok_or_else(|| {
        format_err(format!(
            "unknown format {format:?} (expected \"{PARAMETERS_FORMAT_NAME}<version>\")"
        ))
    })?;
    // Plain digits only: `str::parse` would also accept a leading `+`.
    if version.is_empty() || !version.bytes().all(|b| b.is_ascii_digit()) {
        return Err(format_err(format!("invalid format version in {format:?}")));
    }
    version
        .parse::<u32>()
        .map_err(|_| format_err(format!("invalid format version in {format:?}")))
}

/// Reader of version 1. A future version 2 gets its own `read_v2` next to it, and
/// [`ParameterDatabase::from_json_str`] dispatches on the version.
fn read_v1<V: DeserializeOwned>(
    graph: &ProcessorGraph,
    json: &str,
) -> Result<ParameterDatabase<V>, FaultTuneError> {
    let file: FileV1<V> = serde_json::from_str(json)?;

    let found = parse_hex16(&file.graph.fingerprint)?;
    if found != graph.fingerprint() {
        return Err(FaultTuneError::GraphMismatch {
            expected: graph.fingerprint(),
            found,
        });
    }
    if file.graph.elements != graph.len() {
        return Err(format_err(format!(
            "the file records {} elements but the graph has {}",
            file.graph.elements,
            graph.len()
        )));
    }

    // History first: it tells how many steps exist and what each wrote.
    let mut history = Vec::with_capacity(file.history.len());
    let mut step_positions: Vec<usize> = Vec::new();
    for (position, record) in file.history.into_iter().enumerate() {
        let context = format!("history record {position}");
        match record {
            HistoryIn::Step(step) => {
                let expected = step_positions.len() as u64;
                if step.index != Some(expected) {
                    return Err(format_err(format!(
                        "{context}: step index {:?} where {expected} was expected \
                         (steps must be numbered 0, 1, 2, … in order)",
                        step.index
                    )));
                }
                step_positions.push(position);
                history.push(HistoryRecord::Step(step_in(
                    graph,
                    step,
                    Some(expected),
                    &context,
                )?));
            }
            HistoryIn::Invalidated { elements } => {
                history.push(HistoryRecord::Invalidated(lookup_all(
                    graph, &elements, &context,
                )?));
            }
        }
    }

    let mut entries: Vec<Option<Entry<V>>> = (0..graph.len()).map(|_| None).collect();
    for (label, entry) in file.values.0 {
        let id = lookup(graph, &label, "values")?;
        if entries[id.index()].is_some() {
            return Err(format_err(format!("duplicate value for {label:?}")));
        }
        // The entry must be backed by the step it names.
        let consistent = usize::try_from(entry.step)
            .ok()
            .and_then(|s| step_positions.get(s))
            .and_then(|&position| match &history[position] {
                HistoryRecord::Step(step) => Some(step),
                HistoryRecord::Invalidated(_) => None,
            })
            .and_then(|step| {
                step.parameters
                    .iter()
                    .rposition(|&p| p == id)
                    .map(|k| step.option_indices[k] == entry.option)
            })
            .unwrap_or(false);
        if !consistent {
            return Err(format_err(format!(
                "the value of {label:?} does not match history step {} (option {})",
                entry.step, entry.option
            )));
        }
        entries[id.index()] = Some(Entry {
            value: entry.value,
            option_index: entry.option,
            step: entry.step,
        });
    }
    Ok(ParameterDatabase::from_parts(graph, entries, history))
}

// ---------------------------------------------------------------------------------------
// Saved calibrations
// ---------------------------------------------------------------------------------------

impl<V: Serialize + DeserializeOwned + Clone> ParameterDatabase<V> {
    /// Serialises to the JSON format described in the module docs.
    ///
    /// Fails with `GraphMismatch` if `graph` is not the graph this database belongs to, and
    /// with `Format` if a value would not read back from the text (for example a non-finite
    /// `f64`, which JSON writes as `null`, alone or inside a vector or a struct).
    pub fn to_json_string(&self, graph: &ProcessorGraph) -> Result<String, FaultTuneError> {
        self.check_graph(graph)?;
        self.check_values_read_back(graph)?;
        let labels: Vec<String> = graph.ids().map(|id| graph.label(id).to_owned()).collect();
        let history = self
            .history()
            .iter()
            .map(|record| match record {
                HistoryRecord::Step(step) => Ok(LineOut::Step(step_out(&labels, step, true)?)),
                HistoryRecord::Invalidated(elements) => Ok(LineOut::Invalidated {
                    pass: None,
                    elements: elements
                        .iter()
                        .map(|&id| label_of(&labels, id))
                        .collect::<io::Result<_>>()?,
                }),
            })
            .collect::<io::Result<Vec<_>>>()
            .map_err(|err| format_err(format!("the history cannot be saved: {err}")))?;
        let file = FileOut {
            format: PARAMETERS_FORMAT,
            graph: GraphInfoOut {
                fingerprint: hex16(graph.fingerprint()),
                elements: graph.len(),
            },
            values: ValuesOut { graph, db: self },
            history,
        };
        Ok(serde_json::to_string_pretty(&file)?)
    }

    /// Refuses a value that would be written in a form `load` cannot read back, so that
    /// `save` never writes a file `load` rejects. The typical case is a non-finite `f64`,
    /// which JSON writes as `null`, on its own or inside a vector, tuple or struct.
    fn check_values_read_back(&self, graph: &ProcessorGraph) -> Result<(), FaultTuneError> {
        for (id, entry) in self.iter() {
            let written = serde_json::to_value(&entry.value)?;
            if let Err(err) = serde_json::from_value::<V>(written) {
                return Err(format_err(format!(
                    "the value of {:?} would not read back from the saved file ({err}); \
                     a non-finite number?",
                    graph.label(id)
                )));
            }
        }
        Ok(())
    }

    /// Parses the JSON format. Fails with `GraphMismatch` if `graph` is not the graph it was saved for.
    ///
    /// Fails with `Format` if the text is not JSON, the format name or version is not
    /// supported, a label is not an element of `graph`, or the file is inconsistent (for
    /// example a value that no history step explains).
    pub fn from_json_str(graph: &ProcessorGraph, json: &str) -> Result<Self, FaultTuneError> {
        let header: Header = serde_json::from_str(json)?;
        match parse_format_version(&header.format)? {
            1 => read_v1(graph, json),
            version => Err(format_err(format!(
                "unsupported {PARAMETERS_FORMAT_NAME}{version} (this version of FaultTune \
                 reads versions up to 1)"
            ))),
        }
    }

    /// Writes the JSON file atomically (temporary file, then rename).
    ///
    /// The temporary file lives in the same directory and is synced before the rename, so a
    /// crash leaves either the old file or the new one, never a partial one.
    pub fn save(
        &self,
        graph: &ProcessorGraph,
        path: impl AsRef<Path>,
    ) -> Result<(), FaultTuneError> {
        let mut json = self.to_json_string(graph)?;
        json.push('\n');
        write_atomically(path.as_ref(), json.as_bytes())?;
        Ok(())
    }

    /// Reads a file written by [`save`](Self::save).
    pub fn load(graph: &ProcessorGraph, path: impl AsRef<Path>) -> Result<Self, FaultTuneError> {
        let bytes = fs::read(path.as_ref())?;
        let text =
            String::from_utf8(bytes).map_err(|_| format_err("the file is not valid UTF-8 text"))?;
        Self::from_json_str(graph, &text)
    }
}

/// Writes `bytes` to a temporary file next to `path`, then renames it over `path`.
fn write_atomically(path: &Path, bytes: &[u8]) -> io::Result<()> {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    if path.file_name().is_none() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{} has no file name", path.display()),
        ));
    }
    // A short name of fixed length, so that a target whose own name is close to the limit of
    // the file system can still be saved.
    let tmp_name = format!(
        ".faulttune-{}-{}.tmp",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    );
    let tmp = path.with_file_name(tmp_name);

    let result = (|| {
        let mut file = File::create_new(&tmp)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        drop(file);
        fs::rename(&tmp, path)
    })();
    if result.is_err() {
        // Best effort: the original error is the one worth reporting.
        let _ = fs::remove_file(&tmp);
    }
    result
}

// ---------------------------------------------------------------------------------------
// Journal replay
// ---------------------------------------------------------------------------------------

/// One entry of a checkpoint journal (see [`read_journal_entries`]).
///
/// The canonical order of entries is the order of the history of the database they describe:
/// by `pass`; inside a pass an invalidation first, then the steps by `(subgoal, index)`.
#[derive(Clone, Debug, PartialEq)]
pub enum JournalEntry {
    /// A committed step, as `Event::StepCommitted` announced it: `index` is the ordinal within
    /// the subgoal and pass, and the global index is not known (a replay assigns it). The
    /// timings `prepare_time` and `error_time` are zero (the journal does not record them).
    Step(StepRecord),
    /// The elements `recalibrate` removed from the database (`Event::Invalidated`), as
    /// written: ascending, normally without duplicates.
    Invalidated {
        /// The number the next pass had when the elements were removed: the elements were
        /// removed after every step of an earlier pass and before the steps of this one.
        pass: u32,
        /// The elements that were forgotten.
        elements: Vec<ElementId>,
    },
}

/// A borrowed journal entry: what the replay works on, so that
/// [`ParameterDatabase::replay_journal`] does not have to copy its steps.
#[derive(Clone, Copy)]
enum EntryRef<'a> {
    Step(&'a StepRecord),
    Invalidated {
        pass: u32,
        elements: &'a [ElementId],
    },
}

impl EntryRef<'_> {
    /// The position of the entry in the canonical order (see [`JournalEntry`]).
    fn canonical_key(&self) -> (u32, u8, u32, u64) {
        match *self {
            EntryRef::Invalidated { pass, .. } => (pass, 0, 0, 0),
            EntryRef::Step(step) => (step.pass, 1, step.subgoal, step.index),
        }
    }
}

impl JournalEntry {
    fn as_ref(&self) -> EntryRef<'_> {
        match self {
            JournalEntry::Step(step) => EntryRef::Step(step),
            JournalEntry::Invalidated { pass, elements } => EntryRef::Invalidated {
                pass: *pass,
                elements,
            },
        }
    }
}

/// What a replay will do, worked out (and validated) before anything is changed.
enum Planned<'a, V> {
    /// Commit this step with these values.
    Step(&'a StepRecord, Vec<V>),
    /// Apply this invalidation.
    Invalidate {
        pass: u32,
        elements: &'a [ElementId],
    },
}

impl<V: Clone> ParameterDatabase<V> {
    /// Re-applies journal steps to this database, resolving values with `options(element)`
    /// (the same function the model's `options` provides).
    ///
    /// This is [`replay_entries`](Self::replay_entries) for a journal of steps alone: it
    /// follows the same rules, and there are no invalidations to apply. Use
    /// [`read_journal_entries`] and `replay_entries` instead to also restore what a
    /// `recalibrate` removed.
    ///
    /// Returns the number of steps applied. Fails with `GraphMismatch` if `graph` is not this
    /// database's graph, and with `Format` for a malformed step (parameters and option
    /// indices of different lengths, an element outside the graph, or an option index out
    /// of range of `options(element)`), in which case nothing is applied.
    pub fn replay_journal(
        &mut self,
        graph: &ProcessorGraph,
        steps: &[StepRecord],
        options: impl FnMut(ElementId) -> Vec<V>,
    ) -> Result<usize, FaultTuneError> {
        let entries: Vec<EntryRef<'_>> = steps.iter().map(EntryRef::Step).collect();
        self.replay_refs(graph, &entries, options)
    }

    /// Re-applies a journal of steps and invalidations to this database, resolving values
    /// with `options(element)` (the same function the model's `options` provides). `entries`
    /// normally comes from [`read_journal_entries`].
    ///
    /// The entries are processed in canonical order (see [`JournalEntry`]; they are sorted
    /// first, stably, so the order of the slice does not matter). The journal and the database
    /// are related by nothing but the identity of the steps:
    ///
    /// * A **step** is skipped if the history of this database already contains a step with
    ///   the same `(pass, subgoal, centre)` (pass numbers continue across the `calibrate_*`
    ///   calls on one database, so that triple identifies a step for the life of the database;
    ///   if a journal holds the same triple several times, that many matching steps of the
    ///   database are skipped). Otherwise it is applied whole: each value is
    ///   `options(element)[option_index]`, and the step is committed with
    ///   [`commit_step`](Self::commit_step) keeping its `pass` and `subgoal`, so it gets the
    ///   next global step index and overwrites the current values of its parameters.
    /// * An **invalidation** `(pass, elements)` is applied to each element that is currently
    ///   calibrated by a step of a pass *before* `pass`, and ignored for the others: they are
    ///   not calibrated (the database was saved after the removal, or never had them), or they
    ///   were learned again by a step of `pass` or later. It counts as applied only if
    ///   something was removed.
    ///
    /// Nothing else is assumed. A replay onto an empty database, onto a save taken at any time
    /// of the run (before, inside or after a `recalibrate`), onto a database that is ahead of
    /// the journal, or onto the result of an earlier replay applies exactly what is missing;
    /// the journals of the several runs of one database (for example a run and its resumed
    /// part, with a journal of its own) can be replayed one after the other, and so can their
    /// concatenation. The result of replaying the complete journal of a run onto an empty
    /// database is [`same_outcome`](Self::same_outcome)-equal to the database of that run, at
    /// any thread count, and replaying again applies nothing. A step is *recognised* by its
    /// triple alone: replaying a journal onto a database with the same triples but another
    /// content (the history of a different run) skips them.
    ///
    /// `options` is called at most once per element during one replay, and only for elements
    /// of steps that are applied. Returns the number of entries applied (steps, plus
    /// invalidations that removed something). Fails with `GraphMismatch` if `graph` is not
    /// this database's graph, and with `Format` for a malformed entry (a step whose
    /// parameters and option indices have different lengths, a step or an invalidation that
    /// names an element outside the graph, or an option index out of range of
    /// `options(element)`). All entries are checked before the first one is applied, so
    /// nothing is applied if there is an error.
    pub fn replay_entries(
        &mut self,
        graph: &ProcessorGraph,
        entries: &[JournalEntry],
        options: impl FnMut(ElementId) -> Vec<V>,
    ) -> Result<usize, FaultTuneError> {
        let entries: Vec<EntryRef<'_>> = entries.iter().map(JournalEntry::as_ref).collect();
        self.replay_refs(graph, &entries, options)
    }

    /// The common implementation of [`replay_journal`](Self::replay_journal) and
    /// [`replay_entries`](Self::replay_entries).
    fn replay_refs(
        &mut self,
        graph: &ProcessorGraph,
        entries: &[EntryRef<'_>],
        mut options: impl FnMut(ElementId) -> Vec<V>,
    ) -> Result<usize, FaultTuneError> {
        self.check_graph(graph)?;
        let mut ordered = entries.to_vec();
        ordered.sort_by_key(EntryRef::canonical_key);

        // First pass: check every entry, decide which steps are new and resolve their values.
        // The database is not touched, so a bad entry leaves it as it was.
        let mut known: HashMap<(u32, u32, ElementId), usize> = HashMap::new();
        for step in self.steps() {
            *known
                .entry((step.pass, step.subgoal, step.centre))
                .or_default() += 1;
        }
        let outside = |id: &ElementId| id.index() >= graph.len();
        let mut tables: Vec<Option<Vec<V>>> = (0..graph.len()).map(|_| None).collect();
        let mut plan: Vec<Planned<'_, V>> = Vec::with_capacity(ordered.len());
        let mut step_number = 0usize;
        for (position, entry) in ordered.iter().enumerate() {
            match *entry {
                EntryRef::Step(step) => {
                    let n = step_number;
                    step_number += 1;
                    if step.parameters.len() != step.option_indices.len() {
                        return Err(format_err(format!(
                            "journal step {n}: {} parameters but {} option indices",
                            step.parameters.len(),
                            step.option_indices.len()
                        )));
                    }
                    let mut elements = step
                        .parameters
                        .iter()
                        .chain(std::iter::once(&step.centre))
                        .chain(&step.constraints);
                    if let Some(bad) = elements.find(|id| outside(id)) {
                        return Err(format_err(format!(
                            "journal step {n}: element {} is not in the graph",
                            bad.index()
                        )));
                    }
                    let key = (step.pass, step.subgoal, step.centre);
                    match known.get_mut(&key) {
                        Some(count) if *count > 0 => {
                            *count -= 1; // in the database already
                            continue;
                        }
                        _ => {}
                    }
                    let mut values = Vec::with_capacity(step.parameters.len());
                    for (&p, &k) in step.parameters.iter().zip(&step.option_indices) {
                        let table = tables[p.index()].get_or_insert_with(|| options(p));
                        let value = table.get(k as usize).ok_or_else(|| {
                            format_err(format!(
                                "journal step {n}: option index {k} is out of range for \
                                 element {:?} ({} options)",
                                graph.label(p),
                                table.len()
                            ))
                        })?;
                        values.push(value.clone());
                    }
                    plan.push(Planned::Step(step, values));
                }
                EntryRef::Invalidated { pass, elements } => {
                    if let Some(bad) = elements.iter().find(|id| outside(id)) {
                        return Err(format_err(format!(
                            "journal entry {position}: invalidated element {} is not in the \
                             graph",
                            bad.index()
                        )));
                    }
                    plan.push(Planned::Invalidate { pass, elements });
                }
            }
        }

        // Second pass: apply. `step_pass[s]` is the pass of the step with global index `s`.
        let mut step_pass: Vec<u32> = self.steps().map(|step| step.pass).collect();
        let mut applied = 0;
        for item in plan {
            match item {
                Planned::Step(step, values) => {
                    step_pass.push(step.pass);
                    self.commit_step(step.clone(), values);
                    applied += 1;
                }
                Planned::Invalidate { pass, elements } => {
                    let removable: Vec<ElementId> = elements
                        .iter()
                        .copied()
                        .filter(|&element| {
                            self.get(element).is_some_and(|entry| {
                                usize::try_from(entry.step)
                                    .ok()
                                    .and_then(|s| step_pass.get(s))
                                    .is_some_and(|&learned| learned < pass)
                            })
                        })
                        .collect();
                    if !removable.is_empty() {
                        self.invalidate(&removable);
                        applied += 1;
                    }
                }
            }
        }
        Ok(applied)
    }
}

// ---------------------------------------------------------------------------------------
// Checkpoint journal
// ---------------------------------------------------------------------------------------

/// The writer behind a [`CheckpointObserver`], with its first error.
struct Sink {
    writer: Box<dyn Write + Send>,
    error: Option<io::Error>,
    /// Set after the first failure: later events are not written, so that a partly written
    /// line can only ever be the last one.
    failed: bool,
    /// A second handle on the journal file, if there is one, for `sync_data`.
    file: Option<Box<dyn SyncData>>,
    /// Whether every line is synced to the disk after it was written.
    durable: bool,
}

/// Something whose written data can be forced to the disk: the journal file (a trait so that
/// the tests can watch the calls).
trait SyncData: Send {
    /// `File::sync_data`.
    fn sync_data(&self) -> io::Result<()>;
}

impl SyncData for File {
    fn sync_data(&self) -> io::Result<()> {
        File::sync_data(self)
    }
}

/// Writes the events of a run as flushed JSON lines (see the module docs for the formats):
/// every `StepCommitted` event as a `step` line and every `Invalidated` event as an
/// `invalidated` line. Other events are ignored.
///
/// Events may arrive from several threads; each line is written under a lock with a single
/// `write_all` followed by a flush, so lines never interleave. The lines appear in the order
/// in which the observer received the events, which with parallel subgoals is *not* the order
/// of the database's history: [`read_journal_entries`] and the replay restore that order from
/// the `pass`, `subgoal` and `index` fields of the lines. The observer cannot return errors
/// from [`on_event`](Observer::on_event): the first I/O error is remembered (see
/// [`take_error`](Self::take_error)) and the observer stops writing, so a half-written line
/// can only be the last line of the journal.
///
/// A journal can be started at any time, also for a run that resumes a database that already
/// has steps (see [`create`](Self::create)): lines identify their steps by themselves.
///
/// **Durability.** A line is handed to the operating system when its event is delivered, so
/// the journal survives a crash or a kill of the process. It does not necessarily survive a
/// crash of the machine or a power failure; for that, ask for [`durable`](Self::durable),
/// which makes every line wait for the disk. **After the run, check
/// [`take_error`](Self::take_error)**: a failed write is otherwise visible only in the log
/// (with the `tracing` feature) and as a journal that ends early.
pub struct CheckpointObserver {
    labels: Vec<String>,
    sink: Mutex<Sink>,
}

impl CheckpointObserver {
    fn with_writer(
        writer: Box<dyn Write + Send>,
        file: Option<Box<dyn SyncData>>,
        graph: &ProcessorGraph,
    ) -> Self {
        Self {
            labels: graph.ids().map(|id| graph.label(id).to_owned()).collect(),
            sink: Mutex::new(Sink {
                writer,
                error: None,
                failed: false,
                file,
                durable: false,
            }),
        }
    }

    fn with_file(file: File, graph: &ProcessorGraph) -> Result<Self, FaultTuneError> {
        let handle = file.try_clone()?;
        Ok(Self::with_writer(
            Box::new(file),
            Some(Box::new(handle)),
            graph,
        ))
    }

    /// Creates (truncating) a journal file for `graph`.
    ///
    /// It can be used on a fresh run and on a run that resumes a database that already has
    /// steps (for example after a crash, with the earlier journal kept): every line says which
    /// step it is, so [`ParameterDatabase::replay_entries`] needs no information about where
    /// the journal begins. To continue the journal of the earlier part instead, use
    /// [`append`](Self::append).
    pub fn create(path: impl AsRef<Path>, graph: &ProcessorGraph) -> Result<Self, FaultTuneError> {
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path.as_ref())?;
        Self::with_file(file, graph)
    }

    /// Opens a journal for appending (creating it if missing), for resumed runs.
    ///
    /// To resume a run, load the last saved database, replay the journal onto it
    /// ([`ParameterDatabase::replay_entries`]), append to the same journal with this
    /// function and continue the run.
    ///
    /// If the existing file ends in a torn line (a crash while writing), that line is dropped
    /// first, or just terminated if it turns out to be complete, so that the new lines start
    /// on a line of their own and [`read_journal`] still sees a valid journal. Nothing is
    /// changed, and a `Format` error is returned, if the file is not a journal of `graph`:
    /// its first non-empty line is not a journal line of this graph (a step or an
    /// invalidation naming only elements of `graph`), or its unterminated last line is
    /// neither complete nor the beginning of a journal line. A file is never truncated unless
    /// it passes these checks.
    pub fn append(path: impl AsRef<Path>, graph: &ProcessorGraph) -> Result<Self, FaultTuneError> {
        let path = path.as_ref();
        repair_torn_tail(path, graph)?;
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        Self::with_file(file, graph)
    }

    /// Makes every line wait until the operating system has written it to the disk
    /// (`File::sync_data` after each flush), so that the journal also survives a crash of the
    /// machine, not only of the process. It is off by default: it costs a disk round trip
    /// per step, which is nothing next to a step that measures hardware and a lot next to a
    /// step that is a pure computation. Has no effect on an observer that does not write to
    /// a file of its own.
    pub fn durable(self) -> Self {
        self.lock().durable = true;
        self
    }

    /// Takes the first I/O error that happened while writing, if any.
    ///
    /// After an error the observer stops writing; the events since then are not in the
    /// journal. Call it when the run is over (the error cannot be raised from inside the
    /// run).
    pub fn take_error(&self) -> Option<io::Error> {
        self.lock().error.take()
    }

    fn lock(&self) -> MutexGuard<'_, Sink> {
        // A panic while holding the lock cannot leave the sink in a bad state.
        self.sink.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// One journal line, including the final newline.
    fn line_text(&self, line: &LineOut<'_>) -> io::Result<String> {
        let mut text = serde_json::to_string(line).map_err(io::Error::other)?;
        text.push('\n');
        Ok(text)
    }

    /// The journal line of an event that is journaled.
    fn render(&self, event: &Event) -> Option<io::Result<String>> {
        let line = match event {
            Event::StepCommitted(record) => {
                step_out(&self.labels, record, false).map(LineOut::Step)
            }
            Event::Invalidated { pass, elements } => elements
                .iter()
                .map(|&id| label_of(&self.labels, id))
                .collect::<io::Result<Vec<_>>>()
                .map(|elements| LineOut::Invalidated {
                    pass: Some(*pass),
                    elements,
                }),
            _ => return None,
        };
        Some(line.and_then(|line| self.line_text(&line)))
    }

    /// Writes one rendered line, or remembers why it could not be rendered or written.
    fn write_line(&self, line: io::Result<String>) {
        let mut guard = self.lock();
        let sink = &mut *guard;
        if sink.failed {
            return;
        }
        let outcome = line.and_then(|line| {
            sink.writer.write_all(line.as_bytes())?;
            sink.writer.flush()?;
            match &sink.file {
                Some(file) if sink.durable => file.sync_data(),
                _ => Ok(()),
            }
        });
        if let Err(err) = outcome {
            sink.failed = true;
            #[cfg(feature = "tracing")]
            tracing::error!(error = %err, "the checkpoint journal can no longer be written");
            sink.error.get_or_insert(err);
        }
    }
}

impl fmt::Debug for CheckpointObserver {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CheckpointObserver")
            .field("elements", &self.labels.len())
            .finish_non_exhaustive()
    }
}

impl Observer for CheckpointObserver {
    fn on_event(&self, event: &Event) {
        if let Some(line) = self.render(event) {
            self.write_line(line);
        }
    }
}

impl Drop for CheckpointObserver {
    fn drop(&mut self) {
        let sink = self.sink.get_mut().unwrap_or_else(PoisonError::into_inner);
        // Nobody can receive an error here.
        let _ = sink.writer.flush();
    }
}

/// Why a journal line could not be parsed.
enum LineError {
    /// The JSON stops short: the line is cut off, as by a crash while it was written.
    Incomplete(FaultTuneError),
    /// Anything else: not JSON, wrong shape, unknown label, bad values.
    Invalid(FaultTuneError),
}

impl LineError {
    fn into_error(self) -> FaultTuneError {
        match self {
            LineError::Incomplete(err) | LineError::Invalid(err) => err,
        }
    }
}

/// Parses one journal line (without its newline).
fn parse_line(graph: &ProcessorGraph, line: &[u8]) -> Result<JournalEntry, LineError> {
    let parsed: Result<JournalLineIn, _> = serde_json::from_slice(line);
    match parsed {
        Ok(JournalLineIn::Step(step)) => step_in(graph, step, None, "journal line")
            .map(JournalEntry::Step)
            .map_err(LineError::Invalid),
        Ok(JournalLineIn::Invalidated { pass, elements }) => {
            lookup_all(graph, &elements, "journal line")
                .map(|elements| JournalEntry::Invalidated { pass, elements })
                .map_err(LineError::Invalid)
        }
        Err(err) if err.is_eof() => Err(LineError::Incomplete(err.into())),
        Err(err) => Err(LineError::Invalid(err.into())),
    }
}

/// How many bytes `last_line_start` reads at a time.
const TAIL_CHUNK: usize = 4096;

/// The longest first line `append` examines. A journal line holds a few labels per parameter;
/// a "line" longer than this is not a journal.
const MAX_HEAD_LINE: u64 = 1 << 26;

/// The offset where the last line of `file` (of `len` bytes, not ending in a newline) starts,
/// found by reading backwards from the end.
fn last_line_start(file: &mut File, len: u64) -> io::Result<u64> {
    let mut end = len;
    let mut buf = vec![0u8; TAIL_CHUNK];
    while end > 0 {
        let start = end.saturating_sub(TAIL_CHUNK as u64);
        let chunk = &mut buf[..(end - start) as usize];
        file.seek(SeekFrom::Start(start))?;
        file.read_exact(chunk)?;
        if let Some(p) = chunk.iter().rposition(|&b| b == b'\n') {
            return Ok(start + p as u64 + 1);
        }
        end = start;
    }
    Ok(0)
}

/// The beginnings of the lines this module writes (`serde` writes the `type` tag first).
const LINE_STARTS: [&[u8]; 2] = [b"{\"type\":\"step\"", b"{\"type\":\"invalidated\""];

/// Whether `tail` (an unterminated last line whose JSON stops short) can be the cut-off
/// beginning of a line written by [`CheckpointObserver`]. Only such a line may be dropped by
/// `append`: anything else is somebody's data.
fn looks_like_a_torn_journal_line(tail: &[u8]) -> bool {
    tail.iter().all(u8::is_ascii_whitespace)
        || LINE_STARTS
            .iter()
            .any(|start| tail.starts_with(start) || start.starts_with(tail))
}

/// Refuses a non-empty file whose first non-empty line is not a journal line of `graph`.
///
/// A first line that is the unterminated last line of the file is left to the tail check
/// (it may be torn). Reads only the first non-empty line.
fn check_journal_head(
    file: &mut File,
    path: &Path,
    graph: &ProcessorGraph,
) -> Result<(), FaultTuneError> {
    file.seek(SeekFrom::Start(0))?;
    let mut reader = io::BufReader::new(Read::take(&*file, MAX_HEAD_LINE));
    let mut line = Vec::new();
    loop {
        line.clear();
        if reader.read_until(b'\n', &mut line)? == 0 {
            return Ok(()); // blank lines only
        }
        let terminated = line.last() == Some(&b'\n');
        if terminated {
            line.pop();
        }
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        if !terminated && (line.len() as u64) < MAX_HEAD_LINE {
            return Ok(()); // the last line of the file: judged with the tail
        }
        let refuse = |why: String| {
            format_err(format!(
                "{} is not a journal of this graph ({why}); refusing to append to it",
                path.display()
            ))
        };
        if !terminated {
            return Err(refuse(
                "its first line is longer than any journal line".into(),
            ));
        }
        return match parse_line(graph, &line) {
            Ok(_) => Ok(()),
            Err(err) => Err(refuse(format!("first line: {}", err.into_error()))),
        };
    }
}

/// Makes the end of an existing journal safe to append to.
///
/// A file whose first non-empty line is not a journal line of `graph` is refused (`Format`
/// error, file untouched). Otherwise a last line without a newline is completed if it
/// parses, dropped if it stops short and can be the beginning of a journal line, and
/// otherwise refused in the same way. Only the first and the last line are read.
fn repair_torn_tail(path: &Path, graph: &ProcessorGraph) -> Result<(), FaultTuneError> {
    let mut file = match OpenOptions::new().read(true).write(true).open(path) {
        Ok(file) => file,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(err) => return Err(err.into()),
    };
    let len = file.metadata()?.len();
    if len == 0 {
        return Ok(());
    }
    check_journal_head(&mut file, path, graph)?;
    file.seek(SeekFrom::Start(len - 1))?;
    let mut last = [0u8; 1];
    file.read_exact(&mut last)?;
    if last[0] == b'\n' {
        return Ok(());
    }
    let start = last_line_start(&mut file, len)?;
    file.seek(SeekFrom::Start(start))?;
    let mut tail = Vec::with_capacity((len - start) as usize);
    file.read_to_end(&mut tail)?;
    let refuse = |err: FaultTuneError| {
        format_err(format!(
            "{} does not end in a journal line of this graph ({err}); \
             refusing to append to it",
            path.display()
        ))
    };
    match parse_line(graph, &tail) {
        Ok(_) => {
            // Complete but for its newline.
            file.seek(SeekFrom::End(0))?;
            file.write_all(b"\n")?;
        }
        Err(LineError::Incomplete(_)) if looks_like_a_torn_journal_line(&tail) => {
            file.set_len(start)?;
        }
        Err(err) => return Err(refuse(err.into_error())),
    }
    Ok(())
}

/// Reads a journal written by [`CheckpointObserver`]: its steps and invalidations, in
/// canonical order. A torn (incomplete) last line is ignored; a malformed line anywhere else
/// is a `Format` error.
///
/// **Order.** The lines are in the order in which the events arrived. With several threads
/// the subgoals of a pass commit their steps at the same time, so their lines interleave,
/// whereas the engine merges the subgoals into the database one after the other, in subgoal
/// order (which is what makes a run independent of the thread count). The entries are
/// therefore returned in the order of the history they describe, the canonical order: by
/// `pass`; inside a pass an invalidation first, then the steps by `(subgoal, index)`. The sort
/// is stable, so entries that tie keep the order of the file. Nothing relies on the order of
/// the file.
///
/// Only a last line without a newline whose JSON stops short counts as torn. A last line
/// that is complete but wrong (an unknown label, for example because the journal belongs to
/// another graph, or a missing field) is an error like any other line.
///
/// The steps are as the lines record them (`index` is the ordinal within the subgoal and pass,
/// not a global index), with the timings `prepare_time` and `error_time` set to zero and
/// `duration` as written. Blank lines are skipped. A missing file is an `Io` error.
pub fn read_journal_entries(
    path: impl AsRef<Path>,
    graph: &ProcessorGraph,
) -> Result<Vec<JournalEntry>, FaultTuneError> {
    entries_from_bytes(&fs::read(path.as_ref())?, graph)
}

/// The entries of the journal text `bytes` (see [`read_journal_entries`]).
fn entries_from_bytes(
    bytes: &[u8],
    graph: &ProcessorGraph,
) -> Result<Vec<JournalEntry>, FaultTuneError> {
    let mut entries: Vec<JournalEntry> = Vec::new();
    let mut pieces = bytes.split(|&b| b == b'\n').enumerate().peekable();
    while let Some((n, line)) = pieces.next() {
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        // The piece after the last newline is unterminated (empty if the file ends with one).
        let unterminated = pieces.peek().is_none();
        match parse_line(graph, line) {
            Ok(entry) => entries.push(entry),
            Err(LineError::Incomplete(_)) if unterminated => {}
            Err(err) => {
                return Err(match err.into_error() {
                    FaultTuneError::Format(msg) => {
                        format_err(format!("journal line {}: {msg}", n + 1))
                    }
                    other => other,
                });
            }
        }
    }
    entries.sort_by_key(|entry| entry.as_ref().canonical_key());
    Ok(entries)
}

/// Reads the steps of a journal written by [`CheckpointObserver`], in canonical order;
/// invalidation lines are skipped (use [`read_journal_entries`] to get them). Torn tails,
/// malformed lines and the order are treated as in [`read_journal_entries`], and every line is
/// checked, invalidations included.
///
/// The records are as the lines record them (`index` is the ordinal within the subgoal and
/// pass), with the timings `prepare_time` and `error_time` set to zero and `duration` as
/// written. A missing file is an `Io` error.
pub fn read_journal(
    path: impl AsRef<Path>,
    graph: &ProcessorGraph,
) -> Result<Vec<StepRecord>, FaultTuneError> {
    Ok(read_journal_entries(path, graph)?
        .into_iter()
        .filter_map(|entry| match entry {
            JournalEntry::Step(record) => Some(record),
            JournalEntry::Invalidated { .. } => None,
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::EdgeType;
    use crate::model::ErrorModel;
    use proptest::prelude::*;
    use serde_json::{json, Value};
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::thread;

    fn id(i: u32) -> ElementId {
        ElementId::new(i)
    }

    fn ids(v: &[u32]) -> Vec<ElementId> {
        v.iter().map(|&i| id(i)).collect()
    }

    fn lattice() -> ProcessorGraph {
        ProcessorGraph::square_lattice(3, 4)
    }

    /// The options of an element: a deterministic table of awkward `f64`s.
    fn options_of(e: ElementId) -> Vec<f64> {
        (0..16)
            .map(|k| ((e.raw() as f64 + 1.0) * 0.1 + k as f64 * 0.37).sqrt() * 6.0)
            .collect()
    }

    fn step_record(
        centre: u32,
        params: &[u32],
        opts: &[u32],
        constraints: &[u32],
        error: f64,
    ) -> StepRecord {
        StepRecord {
            index: 0,
            pass: 1,
            subgoal: 3,
            centre: id(centre),
            parameters: ids(params),
            option_indices: opts.to_vec(),
            constraints: ids(constraints),
            error,
            evaluations: 1234,
            prepare_time: Duration::from_micros(1500),
            error_time: Duration::from_micros(2500),
            duration: Duration::from_micros(4321),
        }
    }

    /// Commits a step, resolving values from `options_of`.
    fn commit(db: &mut ParameterDatabase<f64>, record: StepRecord) -> u64 {
        let values = record
            .parameters
            .iter()
            .zip(&record.option_indices)
            .map(|(&p, &k)| options_of(p)[k as usize])
            .collect();
        db.commit_step(record, values)
    }

    /// A database with overwrites, an invalidation and an infinite error.
    fn sample_db(graph: &ProcessorGraph) -> ParameterDatabase<f64> {
        let mut db = ParameterDatabase::new(graph);
        commit(&mut db, step_record(0, &[0, 12], &[3, 5], &[1, 4], 0.0031));
        commit(
            &mut db,
            step_record(1, &[1, 13, 14], &[0, 15, 7], &[0, 2], 1e-9),
        );
        commit(&mut db, step_record(2, &[2], &[9], &[], f64::INFINITY));
        db.invalidate(&ids(&[12, 2, 40]));
        commit(&mut db, step_record(4, &[4, 12, 0], &[2, 2, 2], &[5], 0.5));
        db.invalidate(&ids(&[99, 98])); // nothing to remove: no record
        commit(&mut db, step_record(5, &[5], &[1], &[4, 6], 123456.789));
        db
    }

    fn assert_close(a: &ParameterDatabase<f64>, b: &ParameterDatabase<f64>) {
        for (x, y) in a.history().iter().zip(b.history()) {
            if let (HistoryRecord::Step(x), HistoryRecord::Step(y)) = (x, y) {
                assert!(x.same_outcome(y), "step differs {x:?} {y:?}");
            } else {
                assert_eq!(x, y);
            }
        }
        assert_eq!(a.iter().collect::<Vec<_>>(), b.iter().collect::<Vec<_>>());
        assert!(a.same_outcome(b), "databases differ:\n{a:#?}\n{b:#?}");
        // Timings are not part of same_outcome; they survive to well under a microsecond.
        for (x, y) in a.steps().zip(b.steps()) {
            for (p, q) in [
                (x.duration, y.duration),
                (x.prepare_time, y.prepare_time),
                (x.error_time, y.error_time),
            ] {
                let diff = p.abs_diff(q);
                assert!(diff < Duration::from_nanos(100), "{p:?} vs {q:?}");
            }
        }
    }

    fn parse(text: &str) -> Value {
        serde_json::from_str(text).unwrap()
    }

    // -------------------------------------------------------------------------------
    // Saved calibrations
    // -------------------------------------------------------------------------------

    /// Saved values must come back bit for bit. serde_json's default float parser can be off by
    /// one unit in the last place; the `float_roundtrip` feature of `serde_json` (see
    /// `Cargo.toml`) fixes that, and everything below that compares `f64` values relies on it.
    #[test]
    fn serde_json_parses_floats_exactly() {
        for text in [
            "15.817711591756881",
            "5.499090833947008",
            "0.1",
            "2.2250738585072014e-308",
        ] {
            let expected: f64 = text.parse().unwrap();
            let got: f64 = serde_json::from_str(text).unwrap();
            assert_eq!(
                got.to_bits(),
                expected.to_bits(),
                "enable the `float_roundtrip` feature of serde_json in Cargo.toml"
            );
            let written = serde_json::to_string(&expected).unwrap();
            let back: f64 = serde_json::from_str(&written).unwrap();
            assert_eq!(back.to_bits(), expected.to_bits());
        }
    }

    #[test]
    fn json_round_trip_on_a_lattice() {
        let graph = lattice();
        let db = sample_db(&graph);
        let text = db.to_json_string(&graph).unwrap();
        let back = ParameterDatabase::<f64>::from_json_str(&graph, &text).unwrap();
        assert_close(&db, &back);
        assert_eq!(back.step_count(), db.step_count());
        assert_eq!(back.calibrated(), db.calibrated());
        assert_eq!(back.len(), db.len());
        // Exact f64 values.
        for (e, entry) in db.iter() {
            assert_eq!(back.get(e).unwrap().value.to_bits(), entry.value.to_bits());
        }
        // The loaded database is a full database: it keeps working.
        let mut back = back;
        let next = commit(&mut back, step_record(7, &[7], &[0], &[], 0.1));
        assert_eq!(next, db.step_count());
        // Serialising again gives the same text.
        let again = db.to_json_string(&graph).unwrap();
        assert_eq!(text, again);
    }

    #[test]
    fn json_layout() {
        let graph = lattice();
        let db = sample_db(&graph);
        let doc = parse(&db.to_json_string(&graph).unwrap());
        assert_eq!(doc["format"], "faulttune.parameters/1");
        assert_eq!(doc["format"], PARAMETERS_FORMAT);
        let fp = doc["graph"]["fingerprint"].as_str().unwrap();
        assert_eq!(fp.len(), 16);
        assert_eq!(u64::from_str_radix(fp, 16).unwrap(), graph.fingerprint());
        assert_eq!(doc["graph"]["elements"], graph.len());
        // Values are keyed by label.
        let values = doc["values"].as_object().unwrap();
        assert_eq!(values.len(), db.len());
        let q = &values[graph.label(id(0))];
        assert_eq!(q["option"], 2);
        assert_eq!(q["step"], 3);
        assert_eq!(
            q["value"].as_f64().unwrap(),
            db.value(id(0)).copied().unwrap()
        );
        // History: steps and invalidations, in order.
        let history = doc["history"].as_array().unwrap();
        assert_eq!(history.len(), 6);
        assert_eq!(history[0]["type"], "step");
        assert_eq!(history[0]["index"], 0);
        assert_eq!(history[0]["centre"], graph.label(id(0)));
        assert_eq!(
            history[0]["parameters"],
            json!([graph.label(id(0)), graph.label(id(12))])
        );
        assert_eq!(history[0]["options"], json!([3, 5]));
        assert_eq!(history[0]["evaluations"], 1234);
        assert!((history[0]["duration_ms"].as_f64().unwrap() - 4.321).abs() < 1e-9);
        assert_eq!(history[3]["type"], "invalidated");
        assert_eq!(
            history[3]["elements"],
            json!([graph.label(id(2)), graph.label(id(12))])
        );
        // The infinite error is null.
        assert_eq!(history[2]["error"], Value::Null);
        assert_eq!(history[1]["error"].as_f64(), Some(1e-9));
        // The index counts steps only.
        assert_eq!(history[4]["index"], 3);
        assert_eq!(history[5]["index"], 4);
    }

    #[test]
    fn values_are_written_in_id_order() {
        let graph = lattice();
        let mut db = ParameterDatabase::<f64>::new(&graph);
        // Committed in descending ID order; labels would sort differently from IDs too.
        commit(
            &mut db,
            step_record(14, &[14, 10, 2, 0], &[0, 0, 0, 0], &[], 0.0),
        );
        let text = db.to_json_string(&graph).unwrap();
        let positions: Vec<usize> = [0u32, 2, 10, 14]
            .iter()
            .map(|&e| {
                text.find(&format!("\"{}\": {{", graph.label(id(e))))
                    .unwrap()
            })
            .collect();
        assert!(positions.windows(2).all(|w| w[0] < w[1]), "{positions:?}");
    }

    #[test]
    fn non_finite_errors_are_null_and_read_back_as_infinity() {
        let graph = lattice();
        let mut db = ParameterDatabase::<f64>::new(&graph);
        commit(&mut db, step_record(0, &[0], &[0], &[], f64::INFINITY));
        commit(&mut db, step_record(1, &[1], &[0], &[], f64::NAN));
        commit(&mut db, step_record(2, &[2], &[0], &[], f64::NEG_INFINITY));
        commit(&mut db, step_record(3, &[3], &[0], &[], 0.0));
        let text = db.to_json_string(&graph).unwrap();
        let doc = parse(&text);
        for i in 0..3 {
            assert_eq!(doc["history"][i]["error"], Value::Null);
        }
        assert_eq!(doc["history"][3]["error"].as_f64(), Some(0.0));
        let back = ParameterDatabase::<f64>::from_json_str(&graph, &text).unwrap();
        let errors: Vec<f64> = back.steps().map(|s| s.error).collect();
        assert_eq!(
            errors,
            vec![f64::INFINITY; 3]
                .into_iter()
                .chain([0.0])
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn empty_databases_round_trip() {
        let graph = lattice();
        let db = ParameterDatabase::<f64>::new(&graph);
        let text = db.to_json_string(&graph).unwrap();
        let back = ParameterDatabase::<f64>::from_json_str(&graph, &text).unwrap();
        assert!(back.same_outcome(&db));
        assert!(back.is_empty());
        for (rows, cols) in [(0, 0), (1, 1), (1, 3), (3, 1)] {
            let graph = ProcessorGraph::square_lattice(rows, cols);
            let mut db = ParameterDatabase::<f64>::new(&graph);
            if !graph.is_empty() {
                commit(&mut db, step_record(0, &[0], &[0], &[], 1.0));
            }
            let text = db.to_json_string(&graph).unwrap();
            let back = ParameterDatabase::<f64>::from_json_str(&graph, &text).unwrap();
            assert_close(&db, &back);
        }
    }

    #[test]
    fn other_value_types_round_trip() {
        let graph = lattice();
        let mut db = ParameterDatabase::<(String, Vec<u32>)>::new(&graph);
        db.commit_step(
            step_record(0, &[0, 1], &[1, 2], &[], 0.5),
            vec![
                ("a \"quoted\"\nvalue".into(), vec![1, 2]),
                ("é".into(), vec![]),
            ],
        );
        let text = db.to_json_string(&graph).unwrap();
        let back = ParameterDatabase::<(String, Vec<u32>)>::from_json_str(&graph, &text).unwrap();
        assert!(back.same_outcome(&db));
    }

    #[test]
    fn awkward_labels_round_trip() {
        let mut b = GraphBuilder::new();
        let a = b.add_node("a \"b\"").unwrap();
        let c = b.add_node("é\n\\ü").unwrap();
        b.add_edge(a, c, EdgeType::Engineered).unwrap();
        let graph = b.build().unwrap();
        let mut db = ParameterDatabase::<f64>::new(&graph);
        db.commit_step(step_record(2, &[0, 2], &[1, 1], &[1], 0.25), vec![1.5, 2.5]);
        db.invalidate(&[id(1)]);
        db.invalidate(&[id(0)]);
        let text = db.to_json_string(&graph).unwrap();
        let back = ParameterDatabase::<f64>::from_json_str(&graph, &text).unwrap();
        assert_close(&db, &back);
    }

    use crate::graph::GraphBuilder;

    #[test]
    fn serialising_for_the_wrong_graph_fails() {
        let graph = lattice();
        let db = sample_db(&graph);
        let other = ProcessorGraph::square_lattice(4, 3);
        assert!(matches!(
            db.to_json_string(&other),
            Err(FaultTuneError::GraphMismatch { .. })
        ));
    }

    #[test]
    fn loading_onto_a_different_graph_is_a_mismatch() {
        let graph = lattice();
        let text = sample_db(&graph).to_json_string(&graph).unwrap();
        let other = ProcessorGraph::square_lattice(4, 3);
        assert_ne!(other.fingerprint(), graph.fingerprint());
        match ParameterDatabase::<f64>::from_json_str(&other, &text) {
            Err(FaultTuneError::GraphMismatch { expected, found }) => {
                assert_eq!(expected, other.fingerprint());
                assert_eq!(found, graph.fingerprint());
            }
            r => panic!("expected GraphMismatch, got {:?}", r.err()),
        }
        let tiny = ProcessorGraph::square_lattice(1, 1);
        assert!(matches!(
            ParameterDatabase::<f64>::from_json_str(&tiny, &text),
            Err(FaultTuneError::GraphMismatch { .. })
        ));
    }

    /// Edits the JSON of a valid file and tries to load it.
    fn load_edited(
        edit: impl FnOnce(&mut Value),
    ) -> Result<ParameterDatabase<f64>, FaultTuneError> {
        let graph = lattice();
        let mut doc = parse(&sample_db(&graph).to_json_string(&graph).unwrap());
        edit(&mut doc);
        ParameterDatabase::<f64>::from_json_str(&graph, &doc.to_string())
    }

    fn assert_format(result: Result<ParameterDatabase<f64>, FaultTuneError>, needle: &str) {
        match result {
            Err(FaultTuneError::Format(msg)) => {
                assert!(msg.contains(needle), "message {msg:?} lacks {needle:?}")
            }
            Err(e) => panic!("expected Format error containing {needle:?}, got {e:?}"),
            Ok(_) => panic!("expected Format error containing {needle:?}, got Ok"),
        }
    }

    #[test]
    fn unedited_file_loads() {
        assert!(load_edited(|_| {}).is_ok());
    }

    #[test]
    fn unknown_labels_are_format_errors() {
        assert_format(
            load_edited(|d| {
                let v = d["values"].as_object_mut().unwrap();
                let entry = v.remove("q0_0").unwrap();
                v.insert("nope".into(), entry);
            }),
            "nope",
        );
        for (field, new) in [
            ("centre", json!("ghost")),
            ("parameters", json!(["q0_0", "ghost"])),
            ("constraints", json!(["ghost"])),
        ] {
            assert_format(load_edited(|d| d["history"][0][field] = new), "ghost");
        }
        assert_format(
            load_edited(|d| d["history"][3]["elements"] = json!(["ghost"])),
            "ghost",
        );
    }

    #[test]
    fn garbage_is_a_format_error() {
        let graph = lattice();
        for text in [
            "",
            "   ",
            "garbage",
            "[]",
            "null",
            "42",
            "{}",
            "{\"format\": 3}",
            "{\"format\": \"faulttune.parameters/1\"",
            "{\"format\": \"faulttune.parameters/1\"}",
            "\u{0}\u{1}",
        ] {
            let r = ParameterDatabase::<f64>::from_json_str(&graph, text);
            assert!(
                matches!(r, Err(FaultTuneError::Format(_))),
                "{text:?} gave {:?}",
                r.err()
            );
        }
        // A valid file cut short anywhere is an error, never a panic or a wrong success.
        let text = sample_db(&graph).to_json_string(&graph).unwrap();
        for cut in (0..text.len()).step_by(7) {
            if !text.is_char_boundary(cut) {
                continue;
            }
            let r = ParameterDatabase::<f64>::from_json_str(&graph, &text[..cut]);
            assert!(matches!(r, Err(FaultTuneError::Format(_))), "cut at {cut}");
        }
    }

    #[test]
    fn format_name_and_version_are_checked() {
        for (format, needle) in [
            ("faulttune.graph/1", "unknown format"),
            ("something/1", "unknown format"),
            ("", "unknown format"),
            ("faulttune.parameters", "unknown format"),
            ("faulttune.parameters/", "invalid format version"),
            ("faulttune.parameters/x", "invalid format version"),
            ("faulttune.parameters/-1", "invalid format version"),
            ("faulttune.parameters/0", "unsupported"),
            ("faulttune.parameters/2", "unsupported"),
            ("faulttune.parameters/99", "unsupported"),
        ] {
            assert_format(load_edited(|d| d["format"] = json!(format)), needle);
        }
    }

    #[test]
    fn bad_fingerprints_are_format_errors() {
        for bad in [
            "",
            "xyz",
            "9f2c",
            "0123456789abcdeg",
            "+123456789abcdef",
            "0123456789abcdef0",
        ] {
            assert_format(
                load_edited(|d| d["graph"]["fingerprint"] = json!(bad)),
                "fingerprint",
            );
        }
        // Upper-case digits are accepted: only the value matters.
        let graph = lattice();
        let upper = hex16(graph.fingerprint()).to_uppercase();
        assert!(load_edited(|d| d["graph"]["fingerprint"] = json!(upper)).is_ok());
    }

    #[test]
    fn inconsistent_files_are_rejected() {
        let n = lattice().len();
        assert_format(
            load_edited(|d| d["graph"]["elements"] = json!(n + 1)),
            "elements",
        );
        // Step numbers must be 0, 1, 2, … in order.
        assert_format(
            load_edited(|d| d["history"][1]["index"] = json!(7)),
            "step index",
        );
        assert_format(
            load_edited(|d| {
                d["history"].as_array_mut().unwrap().swap(0, 1);
            }),
            "step index",
        );
        assert_format(
            load_edited(|d| {
                d["history"][0].as_object_mut().unwrap().remove("index");
            }),
            "step index",
        );
        // Parameters and options must align.
        assert_format(
            load_edited(|d| d["history"][0]["options"] = json!([1])),
            "options",
        );
        // A value must be backed by the step it names, with the same option.
        assert_format(
            load_edited(|d| d["values"]["q0_0"]["step"] = json!(1)),
            "does not match",
        );
        assert_format(
            load_edited(|d| d["values"]["q0_0"]["step"] = json!(4000)),
            "does not match",
        );
        assert_format(
            load_edited(|d| d["values"]["q0_0"]["option"] = json!(0)),
            "does not match",
        );
        // Durations must be sensible.
        for bad in [json!(-1.0), json!(1e300)] {
            assert_format(
                load_edited(|d| d["history"][0]["duration_ms"] = bad),
                "duration_ms",
            );
        }
        // Required fields.
        for field in ["error", "evaluations", "duration_ms", "options", "centre"] {
            assert_format(
                load_edited(|d| {
                    d["history"][0].as_object_mut().unwrap().remove(field);
                }),
                field,
            );
        }
        assert_format(
            load_edited(|d| d["history"][0]["type"] = json!("mystery")),
            "mystery",
        );
        assert_format(
            load_edited(|d| d["values"]["q0_0"]["value"] = json!("text")),
            "invalid type",
        );
    }

    #[test]
    fn duplicate_value_labels_are_rejected() {
        let graph = lattice();
        let text = sample_db(&graph).to_json_string(&graph).unwrap();
        let entry = r#"{ "value": 1.0, "option": 3, "step": 0 }"#;
        let dup = text.replacen(
            "\"values\": {",
            &format!("\"values\": {{ \"q0_0\": {entry}, "),
            1,
        );
        assert_ne!(dup, text);
        assert_format(
            ParameterDatabase::<f64>::from_json_str(&graph, &dup),
            "duplicate",
        );
    }

    #[test]
    fn optional_timing_fields_may_be_missing_and_extra_fields_are_ignored() {
        let db = load_edited(|d| {
            let step = d["history"][0].as_object_mut().unwrap();
            step.remove("prepare_ms");
            step.remove("evaluate_ms");
            step.insert("comment".into(), json!("hello"));
            d["extra"] = json!({"anything": [1, 2, 3]});
        })
        .unwrap();
        let first = db.steps().next().unwrap();
        assert_eq!(first.prepare_time, Duration::ZERO);
        assert_eq!(first.error_time, Duration::ZERO);
        assert!(first.duration > Duration::ZERO);
    }

    #[test]
    fn save_and_load_through_a_file() {
        let graph = lattice();
        let db = sample_db(&graph);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("params.json");
        db.save(&graph, &path).unwrap();
        let back = ParameterDatabase::<f64>::load(&graph, &path).unwrap();
        assert_close(&db, &back);
        // The file ends with a newline and is the same text as to_json_string.
        let text = fs::read_to_string(&path).unwrap();
        assert_eq!(text, format!("{}\n", db.to_json_string(&graph).unwrap()));
        // Overwriting works and leaves no temporary file behind.
        let mut db2 = db.clone();
        commit(&mut db2, step_record(9, &[9], &[4], &[], 0.2));
        db2.save(&graph, &path).unwrap();
        let back2 = ParameterDatabase::<f64>::load(&graph, &path).unwrap();
        assert!(back2.same_outcome(&db2));
        let names: Vec<_> = fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .collect();
        assert_eq!(names, vec!["params.json".to_string()]);
    }

    #[test]
    fn save_failures_are_io_errors_and_keep_the_old_file() {
        let graph = lattice();
        let db = sample_db(&graph);
        let dir = tempfile::tempdir().unwrap();
        // Directory does not exist.
        let missing = dir.path().join("no_such_dir").join("p.json");
        assert!(matches!(
            db.save(&graph, &missing),
            Err(FaultTuneError::Io(_))
        ));
        // No file name.
        assert!(matches!(db.save(&graph, "/"), Err(FaultTuneError::Io(_))));
        // The target is a directory: the rename fails, the temp file is cleaned up.
        let target = dir.path().join("sub");
        fs::create_dir(&target).unwrap();
        assert!(matches!(
            db.save(&graph, &target),
            Err(FaultTuneError::Io(_))
        ));
        let names: Vec<_> = fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .collect();
        assert_eq!(names, vec!["sub".to_string()]);
        // Saving for the wrong graph fails before touching the file.
        let path = dir.path().join("keep.json");
        db.save(&graph, &path).unwrap();
        let before = fs::read(&path).unwrap();
        let other = ProcessorGraph::square_lattice(2, 2);
        assert!(matches!(
            db.save(&other, &path),
            Err(FaultTuneError::GraphMismatch { .. })
        ));
        assert_eq!(fs::read(&path).unwrap(), before);
    }

    #[test]
    fn concurrent_saves_to_one_path_never_corrupt_it() {
        let graph = lattice();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("p.json");
        let mut dbs = Vec::new();
        for k in 0..4u32 {
            let mut db = ParameterDatabase::<f64>::new(&graph);
            for j in 0..=k {
                commit(&mut db, step_record(j, &[j], &[j], &[], 0.1));
            }
            dbs.push(db);
        }
        let dbs = &dbs;
        thread::scope(|scope| {
            for db in dbs {
                let (graph, path) = (&graph, &path);
                scope.spawn(move || {
                    for _ in 0..20 {
                        db.save(graph, path).unwrap();
                        let loaded = ParameterDatabase::<f64>::load(graph, path).unwrap();
                        assert!(dbs.iter().any(|d| d.same_outcome(&loaded)));
                    }
                });
            }
        });
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 1);
    }

    #[test]
    fn load_errors() {
        let graph = lattice();
        let dir = tempfile::tempdir().unwrap();
        // Missing file: Io.
        match ParameterDatabase::<f64>::load(&graph, dir.path().join("absent.json")) {
            Err(FaultTuneError::Io(e)) => assert_eq!(e.kind(), io::ErrorKind::NotFound),
            r => panic!("expected Io, got {:?}", r.err()),
        }
        // Garbage: Format.
        let junk = dir.path().join("junk.json");
        fs::write(&junk, "this is not json").unwrap();
        assert!(matches!(
            ParameterDatabase::<f64>::load(&graph, &junk),
            Err(FaultTuneError::Format(_))
        ));
        // Not UTF-8: Format.
        fs::write(&junk, [0xff, 0xfe, 0x00]).unwrap();
        assert!(matches!(
            ParameterDatabase::<f64>::load(&graph, &junk),
            Err(FaultTuneError::Format(_))
        ));
        // A file saved for another graph: GraphMismatch.
        let p = dir.path().join("p.json");
        sample_db(&graph).save(&graph, &p).unwrap();
        let other = ProcessorGraph::square_lattice(2, 2);
        assert!(matches!(
            ParameterDatabase::<f64>::load(&other, &p),
            Err(FaultTuneError::GraphMismatch { .. })
        ));
    }

    #[derive(Clone, Debug)]
    enum Op {
        Commit(Vec<(u32, u32, f64)>, f64, u32, u32),
        Invalidate(Vec<u32>),
    }

    fn op_strategy() -> impl Strategy<Value = Op> {
        prop_oneof![
            (
                prop::collection::vec(
                    (
                        0u32..29,
                        0u32..16,
                        prop::num::f64::NORMAL | prop::num::f64::ZERO
                    ),
                    0..5
                ),
                prop_oneof![Just(f64::INFINITY), 0.0f64..1e6, prop::num::f64::POSITIVE],
                0u32..5,
                0u32..5,
            )
                .prop_map(|(p, e, pass, sg)| Op::Commit(p, e, pass, sg)),
            prop::collection::vec(0u32..29, 0..5).prop_map(Op::Invalidate),
        ]
    }

    proptest! {
        #[test]
        fn json_round_trip_property(ops in prop::collection::vec(op_strategy(), 0..25)) {
            let graph = ProcessorGraph::square_lattice(3, 3); // 9 + 12 + 8 = 29 elements
            let mut db = ParameterDatabase::<f64>::new(&graph);
            for op in &ops {
                match op {
                    Op::Commit(params, error, pass, subgoal) => {
                        let mut r = step_record(
                            params.first().map_or(0, |p| p.0),
                            &params.iter().map(|p| p.0).collect::<Vec<_>>(),
                            &params.iter().map(|p| p.1).collect::<Vec<_>>(),
                            &[1, 2],
                            *error,
                        );
                        r.pass = *pass;
                        r.subgoal = *subgoal;
                        db.commit_step(r, params.iter().map(|p| p.2).collect());
                    }
                    Op::Invalidate(elements) => db.invalidate(&ids(elements)),
                }
            }
            let text = db.to_json_string(&graph).unwrap();
            let back = ParameterDatabase::<f64>::from_json_str(&graph, &text).unwrap();
            prop_assert!(db.same_outcome(&back));
            prop_assert_eq!(back.to_json_string(&graph).unwrap(), text);
        }
    }

    // -------------------------------------------------------------------------------
    // Checkpoint journal: helpers
    // -------------------------------------------------------------------------------

    fn emit(observer: &CheckpointObserver, record: &StepRecord) {
        observer.on_event(&Event::StepCommitted(record.clone()));
    }

    /// `record` as the step `index` of `subgoal` in `pass`, with the timings the journal does
    /// not keep zeroed (so that a step read from a journal equals the one written).
    fn at(mut record: StepRecord, pass: u32, subgoal: u32, index: u64) -> StepRecord {
        record.pass = pass;
        record.subgoal = subgoal;
        record.index = index;
        record.prepare_time = Duration::ZERO;
        record.error_time = Duration::ZERO;
        record.duration = Duration::ZERO;
        record
    }

    fn step_event(record: &StepRecord) -> Event {
        Event::StepCommitted(record.clone())
    }

    fn forget_event(pass: u32, elements: &[u32]) -> Event {
        Event::Invalidated {
            pass,
            elements: ids(elements),
        }
    }

    /// Six distinct steps, in canonical order.
    fn sample_steps() -> Vec<StepRecord> {
        vec![
            at(step_record(0, &[0, 12], &[3, 5], &[1, 4], 0.0031), 0, 0, 0),
            at(
                step_record(1, &[1, 13, 14], &[0, 15, 7], &[0, 2], 1e-9),
                0,
                0,
                1,
            ),
            at(step_record(2, &[2], &[9], &[], f64::INFINITY), 0, 1, 0),
            at(step_record(4, &[4, 5], &[2, 2], &[5], 0.5), 1, 0, 0),
            at(step_record(6, &[6], &[1], &[4, 7], 123456.789), 1, 0, 1),
            at(step_record(8, &[8, 9, 10], &[1, 2, 3], &[7], 0.0), 2, 0, 0),
        ]
    }

    /// The canonical order, written down independently of the implementation: by pass; in a
    /// pass the invalidations, then the steps by subgoal and index; ties in the given order.
    fn canonical(mut entries: Vec<JournalEntry>) -> Vec<JournalEntry> {
        let key = |e: &JournalEntry| match e {
            JournalEntry::Invalidated { pass, .. } => (*pass, 0u8, 0u32, 0u64),
            JournalEntry::Step(s) => (s.pass, 1, s.subgoal, s.index),
        };
        entries.sort_by_key(key);
        entries
    }

    /// The entry a journaled event stands for (the timings of `at` steps survive).
    fn entry_of(event: &Event) -> JournalEntry {
        match event {
            Event::StepCommitted(record) => JournalEntry::Step(record.clone()),
            Event::Invalidated { pass, elements } => JournalEntry::Invalidated {
                pass: *pass,
                elements: elements.clone(),
            },
            other => panic!("not a journaled event: {other:?}"),
        }
    }

    /// Journals `events` (in this order) to `path`.
    fn write_events(path: &Path, graph: &ProcessorGraph, events: &[Event]) {
        let o = CheckpointObserver::create(path, graph).unwrap();
        for event in events {
            o.on_event(event);
        }
        assert!(o.take_error().is_none());
    }

    /// The keys that identify a step, for comparing journals whose timings differ.
    fn content(entries: &[JournalEntry]) -> Vec<String> {
        entries
            .iter()
            .map(|e| match e {
                JournalEntry::Step(s) => format!(
                    "step {} {} {} {:?} {:?} {:?} {:?} {}",
                    s.pass,
                    s.subgoal,
                    s.index,
                    s.centre,
                    s.parameters,
                    s.option_indices,
                    s.constraints,
                    s.error
                ),
                JournalEntry::Invalidated { pass, elements } => {
                    format!("invalidated {pass} {elements:?}")
                }
            })
            .collect()
    }

    /// One pass of a hand-built run: what `recalibrate` forgot just before it, and the steps of
    /// each subgoal (centre, parameters, option indices), in commit order.
    /// A step of a hand-built run: centre, parameters, option indices.
    type StepSpec = (u32, Vec<u32>, Vec<u32>);

    #[derive(Clone, Debug)]
    struct PassSpec {
        forget: Vec<u32>,
        subgoals: Vec<Vec<StepSpec>>,
    }

    fn st(centre: u32, params: &[u32], opts: &[u32]) -> StepSpec {
        (centre, params.to_vec(), opts.to_vec())
    }

    fn pass_spec(forget: &[u32], subgoals: Vec<Vec<StepSpec>>) -> PassSpec {
        PassSpec {
            forget: forget.to_vec(),
            subgoals,
        }
    }

    /// Plays `passes` on `db` like the engine does, numbering them from `first_pass`: the
    /// events of the steps of a pass arrive interleaved across its subgoals (round-robin, the
    /// highest subgoal first, so that the file order is as far from the merge order as can be),
    /// then the database merges the subgoals in order. An invalidation is announced, and
    /// recorded, only if it removed something. `saves` collects a copy of the database after
    /// every change of its history.
    fn run_passes(
        db: &mut ParameterDatabase<f64>,
        observer: Option<&CheckpointObserver>,
        first_pass: u32,
        passes: &[PassSpec],
        mut saves: Option<&mut Vec<ParameterDatabase<f64>>>,
    ) {
        let mut save = |db: &ParameterDatabase<f64>| {
            if let Some(saves) = saves.as_deref_mut() {
                saves.push(db.clone());
            }
        };
        for (k, spec) in passes.iter().enumerate() {
            let pass = first_pass + k as u32;
            let mut removed: Vec<ElementId> = spec
                .forget
                .iter()
                .map(|&e| id(e))
                .filter(|&e| db.is_calibrated(e))
                .collect();
            removed.sort_unstable();
            removed.dedup();
            if !removed.is_empty() {
                db.invalidate(&removed);
                if let Some(o) = observer {
                    o.on_event(&Event::Invalidated {
                        pass,
                        elements: removed,
                    });
                }
                save(db);
            }
            let records: Vec<Vec<StepRecord>> = spec
                .subgoals
                .iter()
                .enumerate()
                .map(|(sg, steps)| {
                    steps
                        .iter()
                        .enumerate()
                        .map(|(i, (c, p, o))| {
                            at(step_record(*c, p, o, &[], 0.25), pass, sg as u32, i as u64)
                        })
                        .collect()
                })
                .collect();
            if let Some(o) = observer {
                let longest = records.iter().map(Vec::len).max().unwrap_or(0);
                for i in 0..longest {
                    for subgoal in records.iter().rev() {
                        if let Some(record) = subgoal.get(i) {
                            emit(o, record);
                        }
                    }
                }
            }
            for record in records.into_iter().flatten() {
                commit(db, record);
                save(db);
            }
        }
    }

    /// The journal entries of a hand-built run, and its final database.
    fn journal_of(
        graph: &ProcessorGraph,
        passes: &[PassSpec],
    ) -> (Vec<JournalEntry>, ParameterDatabase<f64>) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("j.jsonl");
        let mut live = ParameterDatabase::<f64>::new(graph);
        {
            let o = CheckpointObserver::create(&path, graph).unwrap();
            run_passes(&mut live, Some(&o), 0, passes, None);
            assert!(o.take_error().is_none());
        }
        (read_journal_entries(&path, graph).unwrap(), live)
    }

    /// A run with several subgoals per pass, and a recalibration at the start of pass 2 that
    /// forgets `{1, 2, 3, 7}` (and element 15, which is not calibrated), learns them again
    /// with other values, and a no-op recalibration at the start of pass 3.
    fn recalibration_passes() -> Vec<PassSpec> {
        vec![
            pass_spec(
                &[],
                vec![
                    vec![st(0, &[0, 1], &[1, 1]), st(1, &[1, 2], &[2, 2])],
                    vec![st(6, &[6, 7], &[3, 3])],
                    vec![st(10, &[10], &[4])],
                ],
            ),
            pass_spec(
                &[],
                vec![
                    vec![st(2, &[2, 3], &[3, 3]), st(3, &[3, 4], &[4, 4])],
                    vec![st(7, &[7, 8], &[5, 5])],
                ],
            ),
            pass_spec(
                &[3, 1, 2, 7, 15],
                vec![
                    vec![
                        st(1, &[1, 2], &[5, 6]),
                        st(2, &[2], &[6]),
                        st(3, &[3, 5], &[8, 9]),
                    ],
                    vec![st(7, &[7], &[2])],
                ],
            ),
            pass_spec(&[14, 15], vec![vec![st(5, &[5, 6], &[10, 11])]]),
        ]
    }

    fn assert_same_values(a: &ParameterDatabase<f64>, b: &ParameterDatabase<f64>) {
        assert!(a.same_outcome(b), "databases differ:\n{a:#?}\n{b:#?}");
        for (e, entry) in a.iter() {
            assert_eq!(b.get(e).unwrap().value.to_bits(), entry.value.to_bits());
        }
        assert_eq!(a.len(), b.len());
    }

    fn options_of_fn(e: ElementId) -> Vec<f64> {
        options_of(e)
    }

    type Calls = Arc<Mutex<HashMap<u32, usize>>>;

    fn counting_options() -> (Calls, impl FnMut(ElementId) -> Vec<f64>) {
        let calls = Arc::new(Mutex::new(HashMap::new()));
        let c = calls.clone();
        (calls, move |e: ElementId| {
            *c.lock().unwrap().entry(e.raw()).or_insert(0) += 1;
            options_of(e)
        })
    }

    /// A writer that keeps its bytes where the test can see them.
    #[derive(Clone, Default)]
    struct SharedBuf(Arc<Mutex<Vec<u8>>>);

    impl Write for SharedBuf {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    // -------------------------------------------------------------------------------
    // Checkpoint journal: format, order, torn tails
    // -------------------------------------------------------------------------------

    #[test]
    fn journal_lines_have_the_documented_shape() {
        let graph = lattice();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("j.jsonl");
        let step = at(step_record(0, &[0, 12], &[3, 5], &[1, 4], 0.0031), 3, 2, 5);
        {
            let o = CheckpointObserver::create(&path, &graph).unwrap();
            emit(&o, &step);
            o.on_event(&forget_event(4, &[1, 2]));
            o.on_event(&forget_event(7, &[]));
            // Other events are not journaled.
            o.on_event(&Event::PassStarted {
                pass: 0,
                subgoals: 1,
                uncalibrated: 3,
            });
            o.on_event(&Event::PassFinished {
                pass: 0,
                calibrated: 3,
            });
            o.on_event(&Event::StepFailed {
                centre: id(0),
                error: "boom".into(),
            });
            o.on_event(&Event::Finished(Default::default()));
            assert!(o.take_error().is_none());
        }
        let text = fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert!(text.ends_with('\n'));
        assert_eq!(lines.len(), 3);
        let label12 = graph.label(id(12));
        assert_eq!(
            lines[0],
            format!(
                r#"{{"type":"step","index":5,"pass":3,"subgoal":2,"centre":"q0_0","parameters":["q0_0","{label12}"],"options":[3,5],"constraints":["q0_1","{}"],"error":0.0031,"evaluations":1234,"duration_ms":0.0}}"#,
                graph.label(id(4))
            )
        );
        assert_eq!(
            lines[1],
            format!(
                r#"{{"type":"invalidated","pass":4,"elements":["{}","{}"]}}"#,
                graph.label(id(1)),
                graph.label(id(2))
            )
        );
        assert_eq!(lines[2], r#"{"type":"invalidated","pass":7,"elements":[]}"#);
        // The invalidation of any pass is written (it used to be written for pass 0 only).
        assert_eq!(
            read_journal_entries(&path, &graph).unwrap()[1],
            JournalEntry::Invalidated {
                pass: 4,
                elements: ids(&[1, 2])
            }
        );
    }

    #[test]
    fn journal_round_trip() {
        let graph = lattice();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("j.jsonl");
        let mut steps = sample_steps();
        // Real durations do not survive to the last nanosecond; the rest is exact.
        for s in &mut steps {
            s.duration = Duration::from_micros(4321);
        }
        {
            let o = CheckpointObserver::create(&path, &graph).unwrap();
            for step in &steps {
                emit(&o, step);
            }
        }
        let text = fs::read_to_string(&path).unwrap();
        assert_eq!(parse(text.lines().nth(2).unwrap())["error"], Value::Null);
        let read = read_journal(&path, &graph).unwrap();
        assert_eq!(read.len(), steps.len());
        for (got, want) in read.iter().zip(&steps) {
            // Index, pass and subgoal are those of the event, not renumbered.
            assert!(got.same_outcome(want), "{got:?} vs {want:?}");
            assert_eq!(
                (got.pass, got.subgoal, got.index),
                (want.pass, want.subgoal, want.index)
            );
            assert_eq!(got.prepare_time, Duration::ZERO);
            assert_eq!(got.error_time, Duration::ZERO);
            assert!(got.duration.abs_diff(want.duration) < Duration::from_nanos(100));
        }
        let entries = read_journal_entries(&path, &graph).unwrap();
        assert_eq!(
            entries,
            read.into_iter().map(JournalEntry::Step).collect::<Vec<_>>()
        );
    }

    #[test]
    fn entries_are_returned_in_canonical_order_whatever_the_file_order() {
        let graph = lattice();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("j.jsonl");
        let s = |centre: u32, pass: u32, subgoal: u32, index: u64| {
            at(
                step_record(centre, &[centre], &[1], &[], 0.5),
                pass,
                subgoal,
                index,
            )
        };
        // As the lines of a parallel run could arrive: subgoals interleaved, passes in order,
        // and an invalidation announced before the steps of its pass.
        let events = vec![
            step_event(&s(0, 0, 2, 0)),
            step_event(&s(1, 0, 0, 0)),
            step_event(&s(2, 0, 1, 0)),
            step_event(&s(3, 0, 0, 1)),
            step_event(&s(4, 0, 2, 1)),
            forget_event(1, &[1]),
            step_event(&s(5, 1, 1, 0)),
            step_event(&s(6, 1, 0, 0)),
            // The same key twice and two invalidations of one pass keep their file order.
            forget_event(2, &[5]),
            forget_event(2, &[2]),
            step_event(&s(7, 2, 0, 0)),
            step_event(&s(8, 2, 0, 0)),
        ];
        write_events(&path, &graph, &events);
        let file_order: Vec<JournalEntry> = events.iter().map(entry_of).collect();
        let entries = read_journal_entries(&path, &graph).unwrap();
        assert_eq!(entries, canonical(file_order.clone()));
        assert_ne!(entries, file_order, "the test needs a file out of order");
        // In pass 0: subgoal 0 (index 0, 1), subgoal 1, subgoal 2 (index 0, 1).
        let centres: Vec<u32> = entries
            .iter()
            .filter_map(|e| match e {
                JournalEntry::Step(s) => Some(s.centre.raw()),
                JournalEntry::Invalidated { .. } => None,
            })
            .collect();
        assert_eq!(centres, [1, 3, 2, 0, 4, 6, 5, 7, 8]);
        // Invalidations come first in their pass.
        assert_eq!(
            entries[5],
            JournalEntry::Invalidated {
                pass: 1,
                elements: ids(&[1])
            }
        );
        assert_eq!(
            entries[8..10],
            [
                JournalEntry::Invalidated {
                    pass: 2,
                    elements: ids(&[5])
                },
                JournalEntry::Invalidated {
                    pass: 2,
                    elements: ids(&[2])
                }
            ]
        );
        // read_journal returns the steps in the same order.
        let steps = read_journal(&path, &graph).unwrap();
        assert_eq!(
            steps.iter().map(|s| s.centre.raw()).collect::<Vec<_>>(),
            centres
        );
    }

    proptest! {
        #[test]
        fn any_file_order_reads_back_in_canonical_order(
            keys in prop::collection::btree_set((0u32..4, 0u32..4, 0u64..4), 0..25),
            forgets in prop::collection::vec(0u32..5, 0..4),
            shuffle in any::<u64>(),
        ) {
            let graph = lattice();
            let mut events: Vec<Event> = keys
                .iter()
                .enumerate()
                .map(|(k, &(pass, subgoal, index))| {
                    let c = (k % 12) as u32;
                    step_event(&at(step_record(c, &[c], &[1], &[], 0.5), pass, subgoal, index))
                })
                .collect();
            events.extend(forgets.iter().enumerate().map(|(k, &p)| forget_event(p, &[k as u32])));
            // A deterministic shuffle.
            let mut x = shuffle | 1;
            for i in (1..events.len()).rev() {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                events.swap(i, (x % (i as u64 + 1)) as usize);
            }
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("j.jsonl");
            write_events(&path, &graph, &events);
            let expected = canonical(events.iter().map(entry_of).collect());
            prop_assert_eq!(read_journal_entries(&path, &graph).unwrap(), expected);
        }
    }

    #[test]
    fn lines_without_pass_subgoal_or_index_are_format_errors() {
        let graph = lattice();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("j.jsonl");
        let good_step = {
            write_events(&path, &graph, &[step_event(&sample_steps()[0])]);
            fs::read_to_string(&path).unwrap().trim_end().to_owned()
        };
        let good_inval = r#"{"type":"invalidated","pass":2,"elements":["q0_0"]}"#;
        let write = |lines: &[&str], terminated: bool| {
            let mut s = lines.join("\n");
            if terminated {
                s.push('\n');
            }
            fs::write(&path, s).unwrap();
        };
        write(&[&good_step, good_inval, &good_step], true);
        assert_eq!(read_journal_entries(&path, &graph).unwrap().len(), 3);
        let without = |text: &str, field: &str| {
            let mut value = parse(text);
            value.as_object_mut().unwrap().remove(field).unwrap();
            value.to_string()
        };
        let bad = [
            (without(&good_step, "index"), "index"),
            (without(&good_step, "pass"), "pass"),
            (without(&good_step, "subgoal"), "subgoal"),
            (without(good_inval, "pass"), "pass"),
        ];
        for (line, field) in &bad {
            for terminated in [true, false] {
                // In the middle, and as the last line, with or without its newline: an error,
                // never a torn line.
                write(&[&good_step, line, &good_step], true);
                match read_journal_entries(&path, &graph) {
                    Err(FaultTuneError::Format(msg)) => {
                        assert!(
                            msg.contains("journal line 2") && msg.contains(field),
                            "{msg}"
                        )
                    }
                    r => panic!("expected Format for {line}, got {r:?}"),
                }
                write(&[&good_step, line], terminated);
                assert!(
                    matches!(read_journal(&path, &graph), Err(FaultTuneError::Format(_))),
                    "{line}"
                );
                // And append refuses a journal that starts with such a line.
                write(&[line, &good_step], terminated);
                assert!(matches!(
                    CheckpointObserver::append(&path, &graph),
                    Err(FaultTuneError::Format(_))
                ));
            }
        }
        // Negative or fractional values are not pass numbers either.
        for line in [
            good_inval.replace("\"pass\":2", "\"pass\":-1"),
            good_inval.replace("\"pass\":2", "\"pass\":1.5"),
            good_inval.replace("\"pass\":2", "\"pass\":\"2\""),
            good_step.replace("\"index\":0", "\"index\":-1"),
        ] {
            write(&[&good_step, &line], true);
            assert!(read_journal_entries(&path, &graph).is_err(), "{line}");
        }
    }

    #[test]
    fn journal_lines_are_flushed_one_by_one() {
        #[derive(Clone, Default)]
        struct Probe {
            data: Arc<Mutex<Vec<u8>>>,
            flushes: Arc<Mutex<Vec<usize>>>,
            writes: Arc<Mutex<usize>>,
        }
        impl Write for Probe {
            fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
                *self.writes.lock().unwrap() += 1;
                self.data.lock().unwrap().extend_from_slice(buf);
                Ok(buf.len())
            }
            fn flush(&mut self) -> io::Result<()> {
                let len = self.data.lock().unwrap().len();
                self.flushes.lock().unwrap().push(len);
                Ok(())
            }
        }
        let graph = lattice();
        let probe = Probe::default();
        let observer = CheckpointObserver::with_writer(Box::new(probe.clone()), None, &graph);
        let steps = sample_steps();
        for (k, step) in steps.iter().enumerate() {
            // Steps and invalidations alike.
            if k % 2 == 0 {
                emit(&observer, step);
            } else {
                observer.on_event(&forget_event(step.pass, &[1]));
            }
            // Visible (flushed) as soon as the event returns.
            let flushes = probe.flushes.lock().unwrap().clone();
            assert_eq!(flushes.len(), k + 1);
            let data = probe.data.lock().unwrap().clone();
            assert_eq!(*flushes.last().unwrap(), data.len());
            assert_eq!(data.iter().filter(|&&b| b == b'\n').count(), k + 1);
            assert_eq!(data.last(), Some(&b'\n'));
        }
        // One write per line (the line and its newline are written together).
        assert_eq!(*probe.writes.lock().unwrap(), steps.len());
        // Dropping flushes once more.
        drop(observer);
        assert_eq!(probe.flushes.lock().unwrap().len(), steps.len() + 1);
    }

    #[test]
    fn io_errors_are_remembered_and_stop_writing() {
        struct Failing {
            ok_writes: usize,
            attempts: Arc<Mutex<usize>>,
        }
        impl Write for Failing {
            fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
                *self.attempts.lock().unwrap() += 1;
                if self.ok_writes == 0 {
                    return Err(io::Error::new(io::ErrorKind::StorageFull, "disk full"));
                }
                self.ok_writes -= 1;
                Ok(buf.len())
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        let graph = lattice();
        let attempts = Arc::new(Mutex::new(0));
        let observer = CheckpointObserver::with_writer(
            Box::new(Failing {
                ok_writes: 2,
                attempts: attempts.clone(),
            }),
            None,
            &graph,
        );
        let steps = sample_steps();
        for step in &steps[..2] {
            emit(&observer, step);
        }
        assert!(observer.take_error().is_none());
        emit(&observer, &steps[2]);
        // An invalidation line is stopped like a step line.
        observer.on_event(&forget_event(3, &[0]));
        emit(&observer, &steps[4]);
        // Only the first failure counts, and nothing is written after it.
        assert_eq!(*attempts.lock().unwrap(), 3);
        let err = observer.take_error().expect("an error was recorded");
        assert_eq!(err.kind(), io::ErrorKind::StorageFull);
        assert!(observer.take_error().is_none(), "taking clears the error");
        emit(&observer, &steps[5]);
        assert_eq!(*attempts.lock().unwrap(), 3);
    }

    #[test]
    fn flush_errors_are_remembered() {
        struct FlushFails;
        impl Write for FlushFails {
            fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
                Ok(buf.len())
            }
            fn flush(&mut self) -> io::Result<()> {
                Err(io::Error::other("flush failed"))
            }
        }
        let graph = lattice();
        let observer = CheckpointObserver::with_writer(Box::new(FlushFails), None, &graph);
        emit(&observer, &sample_steps()[0]);
        let err = observer.take_error().unwrap();
        assert_eq!(err.to_string(), "flush failed");
    }

    #[test]
    fn events_outside_the_graph_are_reported_not_panicked_and_write_nothing() {
        let graph = lattice();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("j.jsonl");
        let step = step_record(9999, &[0], &[0], &[], 0.0);
        let mut constraint = step_record(0, &[0], &[0], &[], 0.0);
        constraint.constraints = ids(&[9999]);
        let mut parameter = step_record(0, &[0], &[0], &[], 0.0);
        parameter.parameters = ids(&[9999]);
        for event in [
            step_event(&step),
            step_event(&constraint),
            step_event(&parameter),
            forget_event(1, &[0, 9999]),
        ] {
            let observer = CheckpointObserver::create(&path, &graph).unwrap();
            observer.on_event(&event);
            let err = observer.take_error().unwrap();
            assert_eq!(err.kind(), io::ErrorKind::InvalidData);
            assert_eq!(fs::read(&path).unwrap(), b"", "{event:?}");
        }
    }

    #[test]
    fn four_threads_write_intact_lines_that_read_back_in_canonical_order() {
        const THREADS: u32 = 4;
        const PER_THREAD: u32 = 60;
        let graph = lattice();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("j.jsonl");
        let observer = CheckpointObserver::create(&path, &graph).unwrap();
        let n = graph.len() as u32;
        // Thread `t` is subgoal `t`; every ten steps make a pass. Long lines (many parameters
        // and constraints) make interleaving likely if the writes were not guarded.
        let step = |t: u32, k: u32| {
            let params: Vec<u32> = (0..25).map(|j| (t * 5 + k + j) % n).collect();
            let mut unique = params;
            unique.sort_unstable();
            unique.dedup();
            let opts: Vec<u32> = unique.iter().map(|&p| (p + k) % 16).collect();
            at(
                step_record(unique[0], &unique, &opts, &unique, k as f64 / 7.0),
                k / 10,
                t,
                u64::from(k % 10),
            )
        };
        // At the start of every pass thread `t` announces an invalidation of its own element.
        let invalidated = |t: u32, k: u32| -> Vec<u32> { vec![t, THREADS + k / 10 + t] };
        thread::scope(|scope| {
            for t in 0..THREADS {
                let observer = &observer;
                scope.spawn(move || {
                    for k in 0..PER_THREAD {
                        if k % 10 == 0 {
                            observer.on_event(&forget_event(k / 10, &invalidated(t, k)));
                        }
                        emit(observer, &step(t, k));
                    }
                });
            }
        });
        assert!(observer.take_error().is_none());
        drop(observer);

        let text = fs::read_to_string(&path).unwrap();
        let lines = (THREADS * (PER_THREAD + PER_THREAD / 10)) as usize;
        assert_eq!(text.lines().count(), lines);
        for line in text.lines() {
            assert!(matches!(
                parse(line)["type"].as_str(),
                Some("step" | "invalidated")
            ));
        }
        let entries = read_journal_entries(&path, &graph).unwrap();
        assert_eq!(entries.len(), lines);
        // Nothing lost, duplicated or changed.
        let mut expected: Vec<JournalEntry> = Vec::new();
        for t in 0..THREADS {
            for k in 0..PER_THREAD {
                if k % 10 == 0 {
                    expected.push(JournalEntry::Invalidated {
                        pass: k / 10,
                        elements: ids(&invalidated(t, k)),
                    });
                }
                expected.push(JournalEntry::Step(step(t, k)));
            }
        }
        // The steps are exactly in canonical order, whatever the threads did; the four
        // invalidations of a pass (which tie) come first in the pass, in any order.
        let steps_of = |entries: Vec<JournalEntry>| -> Vec<StepRecord> {
            entries
                .into_iter()
                .filter_map(|e| match e {
                    JournalEntry::Step(s) => Some(s),
                    JournalEntry::Invalidated { .. } => None,
                })
                .collect()
        };
        assert_eq!(
            steps_of(entries.clone()),
            steps_of(canonical(expected.clone()))
        );
        let mut have = content(&entries);
        let mut want = content(&expected);
        have.sort();
        want.sort();
        assert_eq!(have, want);
        // Every pass starts with its four invalidations.
        for (position, entry) in entries.iter().enumerate() {
            if let JournalEntry::Invalidated { pass, .. } = entry {
                let first = (*pass * (THREADS * 10 + THREADS)) as usize;
                assert!(
                    (first..first + THREADS as usize).contains(&position),
                    "pass {pass}: invalidation at {position}"
                );
            }
        }
    }

    /// A graph with non-ASCII labels, and a journal of it in a scrambled file order.
    fn unicode_journal() -> (ProcessorGraph, Vec<Event>) {
        let mut b = GraphBuilder::new();
        let a = b.add_node("é1").unwrap();
        let c = b.add_node("ü2").unwrap();
        b.add_edge(a, c, EdgeType::Engineered).unwrap();
        let graph = b.build().unwrap();
        let s = |centre, params: &[u32], opts: &[u32], pass, subgoal, index| {
            step_event(&at(
                step_record(centre, params, opts, &[], 0.25),
                pass,
                subgoal,
                index,
            ))
        };
        let events = vec![
            s(1, &[1], &[3], 0, 1, 0),
            s(0, &[0, 1], &[1, 2], 0, 0, 0),
            forget_event(1, &[1, 0]),
            s(2, &[2, 0], &[4, 5], 1, 0, 0),
            forget_event(2, &[1, 2]),
            s(0, &[0], &[6], 2, 0, 0),
        ];
        (graph, events)
    }

    /// The number of lines of `full[..cut]` that count: the complete ones, and a last one that
    /// lacks nothing but its newline.
    fn surviving_lines(full: &[u8], cut: usize) -> usize {
        let complete = full[..cut].iter().filter(|&&b| b == b'\n').count();
        let only_newline_missing =
            cut > 0 && full[cut - 1] != b'\n' && full.get(cut) == Some(&b'\n');
        complete + usize::from(only_newline_missing)
    }

    #[test]
    fn a_torn_last_line_is_ignored_at_every_cut() {
        let (graph, events) = unicode_journal();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("j.jsonl");
        write_events(&path, &graph, &events);
        let file_order: Vec<JournalEntry> = events.iter().map(entry_of).collect();
        let full = fs::read(&path).unwrap();
        for cut in 0..=full.len() {
            fs::write(&path, &full[..cut]).unwrap();
            let kept = surviving_lines(&full, cut);
            let read = read_journal_entries(&path, &graph).unwrap();
            assert_eq!(
                read,
                canonical(file_order[..kept].to_vec()),
                "cut at {cut} of {}",
                full.len()
            );
            let steps = read_journal(&path, &graph).unwrap();
            let want = read
                .iter()
                .filter(|e| matches!(e, JournalEntry::Step(_)))
                .count();
            assert_eq!(steps.len(), want, "cut at {cut}");
        }
    }

    #[test]
    fn append_repairs_a_torn_line_of_either_type_at_every_cut() {
        let (graph, events) = unicode_journal();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("j.jsonl");
        write_events(&path, &graph, &events);
        let file_order: Vec<JournalEntry> = events.iter().map(entry_of).collect();
        let full = fs::read(&path).unwrap();
        let fresh = forget_event(5, &[2]);
        for cut in 0..=full.len() {
            fs::write(&path, &full[..cut]).unwrap();
            {
                let o = CheckpointObserver::append(&path, &graph).unwrap();
                o.on_event(&fresh);
                assert!(o.take_error().is_none());
            }
            // What survived the cut, then the new line; nothing is glued to a torn line.
            let kept = surviving_lines(&full, cut);
            let mut want = file_order[..kept].to_vec();
            want.push(entry_of(&fresh));
            assert_eq!(
                read_journal_entries(&path, &graph).unwrap(),
                canonical(want),
                "cut at {cut}"
            );
            assert!(fs::read(&path).unwrap().ends_with(b"\n"));
        }
    }

    #[test]
    fn malformed_lines_elsewhere_are_format_errors() {
        let graph = lattice();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("j.jsonl");
        let events: Vec<Event> = sample_steps()[..3].iter().map(step_event).collect();
        write_events(&path, &graph, &events);
        let text = fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        let write = |lines: &[&str], terminated: bool| {
            let mut s = lines.join("\n");
            if terminated {
                s.push('\n');
            }
            fs::write(&path, s).unwrap();
        };
        let expect_format = |needle: &str| match read_journal(&path, &graph) {
            Err(FaultTuneError::Format(msg)) => {
                assert!(msg.contains(needle), "{msg:?} lacks {needle:?}")
            }
            r => panic!("expected Format error, got {r:?}"),
        };
        // Garbage in the middle.
        write(&[lines[0], "{\"type\":\"step\",", lines[2]], true);
        expect_format("journal line 2");
        write(&[lines[0], "not json", lines[2]], true);
        expect_format("journal line 2");
        // Garbage as the first line.
        write(&["xxx", lines[1]], true);
        expect_format("journal line 1");
        // A terminated (complete) last line that is malformed is an error too.
        write(&[lines[0], lines[1], "garbage"], true);
        expect_format("journal line 3");
        // The same garbage without the newline is still garbage: only a line that stops short
        // (valid JSON cut off) counts as torn.
        write(&[lines[0], lines[1], "garbage"], false);
        expect_format("journal line 3");
        write(&[lines[0], lines[1], "{\"type\":\"step\",\"pa"], false);
        assert_eq!(read_journal(&path, &graph).unwrap().len(), 2);
        // A complete last line without its newline that names an unknown label (a journal of
        // another graph) or has the wrong shape is an error, not a torn line.
        let unterminated_bad_label = lines[2].replace("q0_2", "zzz");
        write(&[lines[0], lines[1], &unterminated_bad_label], false);
        expect_format("zzz");
        write(&[lines[0], r#"{"type":"step","pass":"x"}"#], false);
        expect_format("journal line 2");
        write(&[lines[0], r#"{"type":"other"}"#], false);
        expect_format("journal line 2");
        // Lines of the types the first design had are not part of the format.
        for old in [
            r#"{"type":"pass","pass":0}"#,
            r#"{"type":"base","steps":3}"#,
        ] {
            write(&[old, lines[0]], true);
            expect_format("journal line 1");
        }
        // Unknown label / type / bad option count in a middle line.
        let bad_label = lines[1].replace("q0_1", "zzz");
        write(&[lines[0], &bad_label, lines[2]], true);
        expect_format("zzz");
        write(&[lines[0], r#"{"type":"other"}"#, lines[2]], true);
        expect_format("journal line 2");
        let bad_options = lines[1].replace("\"options\":[0,15,7]", "\"options\":[0]");
        assert_ne!(bad_options, lines[1]);
        write(&[lines[0], &bad_options, lines[2]], true);
        expect_format("options");
        // Blank lines are fine.
        write(&[lines[0], "", "  ", lines[1], "", lines[2]], true);
        assert_eq!(read_journal(&path, &graph).unwrap().len(), 3);
        // An empty file is an empty journal; a missing file is an Io error.
        fs::write(&path, "").unwrap();
        assert!(read_journal(&path, &graph).unwrap().is_empty());
        assert!(matches!(
            read_journal(dir.path().join("absent"), &graph),
            Err(FaultTuneError::Io(_))
        ));
        // Invalid UTF-8 is a Format error in the middle and at the end.
        let mut bytes = lines[0].as_bytes().to_vec();
        bytes.extend_from_slice(b"\n\xff\xfe\n");
        bytes.extend_from_slice(lines[1].as_bytes());
        bytes.push(b'\n');
        fs::write(&path, &bytes).unwrap();
        expect_format("journal line 2");
        let mut bytes = lines[0].as_bytes().to_vec();
        bytes.extend_from_slice(b"\n\xff\xfe");
        fs::write(&path, &bytes).unwrap();
        expect_format("journal line 2");
    }

    #[test]
    fn malformed_invalidation_lines_are_format_errors() {
        let graph = lattice();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("j.jsonl");
        write_events(&path, &graph, &[step_event(&sample_steps()[0])]);
        let step = fs::read_to_string(&path).unwrap();
        let step = step.trim_end();
        let good = r#"{"type":"invalidated","pass":1,"elements":["q0_0","q0_1"]}"#;
        let write = |lines: &[&str], terminated: bool| {
            let mut s = lines.join("\n");
            if terminated {
                s.push('\n');
            }
            fs::write(&path, s).unwrap();
        };
        let check = |needle: &str| match read_journal_entries(&path, &graph) {
            Err(FaultTuneError::Format(msg)) => {
                assert!(msg.contains(needle), "{msg:?} lacks {needle:?}")
            }
            r => panic!("expected Format error containing {needle:?}, got {r:?}"),
        };
        write(&[step, good, step], true);
        assert_eq!(read_journal_entries(&path, &graph).unwrap().len(), 3);
        let bad_lines = [
            // Unknown label.
            r#"{"type":"invalidated","pass":1,"elements":["q0_0","ghost"]}"#,
            // Wrong shapes.
            r#"{"type":"invalidated","pass":1}"#,
            r#"{"type":"invalidated","elements":["q0_0"]}"#,
            r#"{"type":"invalidated","pass":1,"elements":"q0_0"}"#,
            r#"{"type":"invalidated","pass":1,"elements":[0,1]}"#,
            r#"{"type":"invalidated","pass":1,"elements":null}"#,
            r#"{"type":"invalidated","pass":null,"elements":[]}"#,
            r#"{"type":"Invalidated","pass":1,"elements":[]}"#,
            r#"[]"#,
        ];
        for bad in bad_lines {
            // In the middle and as a terminated last line: an error in the right line.
            write(&[step, bad, step], true);
            check("journal line 2");
            write(&[step, bad], true);
            check("journal line 2");
            // read_journal reads the same lines and fails the same way.
            assert!(matches!(
                read_journal(&path, &graph),
                Err(FaultTuneError::Format(_))
            ));
        }
        // An unterminated last line that is complete but wrong is an error, not a torn tail.
        for bad in [
            r#"{"type":"invalidated","pass":1,"elements":["ghost"]}"#,
            r#"{"type":"invalidated","pass":1,"elements":"q0_0"}"#,
            r#"{"type":"invalidated","pass":1}"#,
        ] {
            write(&[step, bad], false);
            check("journal line 2");
        }
        write(
            &[
                step,
                r#"{"type":"invalidated","pass":1,"elements":["ghost"]}"#,
            ],
            false,
        );
        check("ghost");
        // A cut-off one is torn and ignored, in the middle it is an error.
        write(&[step, r#"{"type":"invalidated","pa"#], false);
        assert_eq!(read_journal_entries(&path, &graph).unwrap().len(), 1);
        write(&[step, r#"{"type":"invalidated","pa"#, step], true);
        check("journal line 2");
        write(
            &[step, r#"{"type":"invalidated","pass":1,"elements":["q0_0""#],
            false,
        );
        assert_eq!(read_journal_entries(&path, &graph).unwrap().len(), 1);
        write(
            &[step, r#"{"type":"invalidated","pass":1,"elements":["q0_0""#],
            true,
        );
        check("journal line 2");
    }

    #[test]
    fn create_truncates_and_append_extends() {
        let graph = lattice();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("j.jsonl");
        let steps = sample_steps();
        // append creates a missing file.
        {
            let o = CheckpointObserver::append(&path, &graph).unwrap();
            emit(&o, &steps[0]);
        }
        assert_eq!(read_journal(&path, &graph).unwrap().len(), 1);
        {
            let o = CheckpointObserver::append(&path, &graph).unwrap();
            emit(&o, &steps[1]);
            emit(&o, &steps[2]);
        }
        let read = read_journal(&path, &graph).unwrap();
        assert_eq!(read, steps[..3]);
        // create truncates, whatever is there.
        {
            let o = CheckpointObserver::create(&path, &graph).unwrap();
            emit(&o, &steps[3]);
        }
        assert_eq!(read_journal(&path, &graph).unwrap(), steps[3..4]);
        fs::write(&path, "not a journal\n").unwrap();
        drop(CheckpointObserver::create(&path, &graph).unwrap());
        assert_eq!(fs::read(&path).unwrap(), b"");
        // Errors opening: the directory does not exist.
        let bad = dir.path().join("nope").join("j");
        assert!(matches!(
            CheckpointObserver::create(&bad, &graph),
            Err(FaultTuneError::Io(_))
        ));
        assert!(matches!(
            CheckpointObserver::append(&bad, &graph),
            Err(FaultTuneError::Io(_))
        ));
    }

    #[test]
    fn append_never_damages_a_file_that_is_not_a_journal_of_this_graph() {
        let graph = lattice();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("notes.txt");
        let good_first = {
            write_events(&path, &graph, &[step_event(&sample_steps()[0])]);
            fs::read_to_string(&path).unwrap().trim_end().to_owned()
        };
        let wrong_graph_line = {
            let other = ProcessorGraph::square_lattice(5, 5);
            let p = dir.path().join("other.jsonl");
            write_events(
                &p,
                &other,
                &[step_event(&at(
                    step_record(40, &[40], &[1], &[], 0.5),
                    0,
                    0,
                    0,
                ))],
            );
            fs::read_to_string(&p).unwrap().trim_end().to_owned()
        };
        for content in [
            "important notes line1\nimportant line 2 without newline".to_owned(),
            "no newline at all".to_owned(),
            // Complete line, but for another graph.
            format!("{wrong_graph_line}\n{wrong_graph_line}"),
            wrong_graph_line.clone(),
            "{\"type\":\"other\"}".to_owned(),
            // Not journals, but terminated by a newline.
            "important notes\n".to_owned(),
            "{\"some\": \"json\"}\n".to_owned(),
            "\n\n  \nnot a journal\nmore\n".to_owned(),
            // A saved calibration (pretty JSON) is not a journal.
            "{\n  \"format\": \"faulttune.parameters/1\"\n}\n".to_owned(),
            // A first line that is cut off but terminated.
            format!("{}\n{good_first}\n", &good_first[..20]),
            // An invalidation line with a label of another graph, or without its pass.
            "{\"type\":\"invalidated\",\"pass\":0,\"elements\":[\"zz\"]}\n".to_owned(),
            "{\"type\":\"invalidated\",\"elements\":[]}\n".to_owned(),
            // The lines of the first design of the journal.
            "{\"type\":\"pass\",\"pass\":0}\n".to_owned(),
            "{\"type\":\"base\",\"steps\":2}\n".to_owned(),
            // The first line is fine but the file ends in somebody's text.
            format!("{good_first}\nthe end"),
            // Looks like the start of a torn line only by accident of JSON, but is not one.
            "{\"hello\":".to_owned(),
            "{\"type\":\"stepper\"".to_owned(),
            "[1, 2".to_owned(),
        ] {
            fs::write(&path, &content).unwrap();
            match CheckpointObserver::append(&path, &graph) {
                Err(FaultTuneError::Format(msg)) => {
                    assert!(msg.contains("refusing"), "{msg}");
                }
                r => panic!("expected Format for {content:?}, got {:?}", r.map(|_| ())),
            }
            assert_eq!(fs::read_to_string(&path).unwrap(), content);
        }
        // Journals, however they end, are accepted: blank lines first, a journal that
        // starts with an invalidation, and a genuinely torn line of either type (dropped,
        // together with trailing blanks).
        let inv = "{\"type\":\"invalidated\",\"pass\":2,\"elements\":[\"q0_0\"]}\n";
        let blank_first = format!("\n  \n{good_first}\n");
        for (content, expected) in [
            (blank_first.clone(), blank_first),
            (inv.to_owned(), inv.to_owned()),
            (format!("{inv}{{\"type\":\"inval"), inv.to_owned()),
            (format!("{inv}{{\"type\":\"step\",\"pa"), inv.to_owned()),
            ("{\"type\":\"".to_owned(), String::new()),
            ("{\"type\":".to_owned(), String::new()),
            ("{".to_owned(), String::new()),
            ("   ".to_owned(), String::new()),
            ("\n\n".to_owned(), "\n\n".to_owned()),
            (String::new(), String::new()),
        ] {
            fs::write(&path, &content).unwrap();
            let o = CheckpointObserver::append(&path, &graph);
            assert!(o.is_ok(), "{content:?}: {:?}", o.err());
            drop(o);
            assert_eq!(fs::read_to_string(&path).unwrap(), expected, "{content:?}");
            assert!(read_journal_entries(&path, &graph).is_ok(), "{content:?}");
        }
    }

    #[test]
    fn append_repairs_a_torn_tail_longer_than_a_read_chunk() {
        // One element with a very long label: the journal line is much longer than any chunk
        // the tail search reads at a time.
        let long = "x".repeat(20_000);
        let mut b = GraphBuilder::new();
        let a = b.add_node(&long).unwrap();
        let c = b.add_node("y").unwrap();
        b.add_edge(a, c, EdgeType::Engineered).unwrap();
        let graph = b.build().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("j.jsonl");
        write_events(
            &path,
            &graph,
            &[
                step_event(&at(step_record(0, &[0, 1], &[1, 2], &[], 0.5), 0, 0, 0)),
                step_event(&at(step_record(1, &[1, 0], &[3, 4], &[], 0.25), 0, 0, 1)),
            ],
        );
        let full = fs::read(&path).unwrap();
        let first_len = full.iter().position(|&b| b == b'\n').unwrap() + 1;
        // Cuts inside the long second line, far from both ends and close to the end.
        for cut in [
            first_len + 5,
            first_len + 15_000,
            full.len() - 100,
            full.len() - 2,
        ] {
            fs::write(&path, &full[..cut]).unwrap();
            drop(CheckpointObserver::append(&path, &graph).unwrap());
            assert_eq!(fs::read(&path).unwrap(), &full[..first_len], "cut {cut}");
        }
        // A complete long line missing only its newline gets the newline.
        fs::write(&path, &full[..full.len() - 1]).unwrap();
        drop(CheckpointObserver::append(&path, &graph).unwrap());
        assert_eq!(fs::read(&path).unwrap(), full);
    }

    #[test]
    fn a_durable_observer_syncs_after_every_line_and_a_plain_one_never() {
        struct Spy {
            written: SharedBuf,
            /// The number of bytes written at each sync.
            syncs: Arc<Mutex<Vec<usize>>>,
            /// Fail the sync with this ordinal.
            fail_at: Option<usize>,
        }
        impl SyncData for Spy {
            fn sync_data(&self) -> io::Result<()> {
                let mut syncs = self.syncs.lock().unwrap();
                if self.fail_at == Some(syncs.len()) {
                    return Err(io::Error::other("sync failed"));
                }
                syncs.push(self.written.0.lock().unwrap().len());
                Ok(())
            }
        }
        let graph = lattice();
        let steps = sample_steps();
        let build = |fail_at: Option<usize>| {
            let written = SharedBuf::default();
            let syncs = Arc::new(Mutex::new(Vec::new()));
            let spy = Spy {
                written: written.clone(),
                syncs: syncs.clone(),
                fail_at,
            };
            let o = CheckpointObserver::with_writer(
                Box::new(written.clone()),
                Some(Box::new(spy)),
                &graph,
            );
            (o, written, syncs)
        };
        // Off by default.
        let (o, _written, syncs) = build(None);
        for step in &steps {
            emit(&o, step);
        }
        assert!(syncs.lock().unwrap().is_empty());
        // On request: one sync after each line, when the whole line is written.
        let (o, written, syncs) = build(None);
        let o = o.durable();
        for (k, step) in steps.iter().enumerate() {
            if k == 2 {
                o.on_event(&forget_event(1, &[0]));
            }
            emit(&o, step);
        }
        let syncs = syncs.lock().unwrap().clone();
        assert_eq!(syncs.len(), steps.len() + 1);
        let bytes = written.0.lock().unwrap().clone();
        let ends: Vec<usize> = bytes
            .iter()
            .enumerate()
            .filter(|(_, &b)| b == b'\n')
            .map(|(i, _)| i + 1)
            .collect();
        assert_eq!(syncs, ends);
        assert!(o.take_error().is_none());
        // A failing sync is remembered like any I/O error and stops the journal.
        let (o, written, syncs) = build(Some(1));
        let o = o.durable();
        for step in &steps {
            emit(&o, step);
        }
        assert_eq!(syncs.lock().unwrap().len(), 1);
        assert_eq!(
            written
                .0
                .lock()
                .unwrap()
                .iter()
                .filter(|&&b| b == b'\n')
                .count(),
            2
        );
        assert_eq!(o.take_error().unwrap().to_string(), "sync failed");
        // A file journal accepts the option and writes the same bytes.
        let dir = tempfile::tempdir().unwrap();
        let (a, b) = (dir.path().join("a.jsonl"), dir.path().join("b.jsonl"));
        let events: Vec<Event> = steps.iter().map(step_event).collect();
        write_events(&a, &graph, &events);
        {
            let o = CheckpointObserver::create(&b, &graph).unwrap().durable();
            for event in &events {
                o.on_event(event);
            }
            assert!(o.take_error().is_none());
        }
        assert_eq!(fs::read(&a).unwrap(), fs::read(&b).unwrap());
        let o = CheckpointObserver::append(&b, &graph).unwrap().durable();
        o.on_event(&forget_event(9, &[1]));
        assert!(o.take_error().is_none());
        // An observer without a file of its own ignores the option.
        let buf = SharedBuf::default();
        let o = CheckpointObserver::with_writer(Box::new(buf.clone()), None, &graph).durable();
        o.on_event(&forget_event(3, &[1]));
        assert!(o.take_error().is_none());
        assert_eq!(
            buf.0
                .lock()
                .unwrap()
                .iter()
                .filter(|&&b| b == b'\n')
                .count(),
            1
        );
    }

    // -------------------------------------------------------------------------------
    // Replay
    // -------------------------------------------------------------------------------

    /// Replays `entries` onto `start` and checks the contract against the `live` database
    /// they were journaled from: the result is the live database, exactly the missing history
    /// was applied, the order of the slice is irrelevant, and a second replay applies nothing.
    fn assert_replay_reproduces(
        graph: &ProcessorGraph,
        entries: &[JournalEntry],
        start: &ParameterDatabase<f64>,
        live: &ParameterDatabase<f64>,
        what: &str,
    ) {
        let missing = live.history().len() - start.history().len();
        let mut db = start.clone();
        let applied = db.replay_entries(graph, entries, options_of_fn).unwrap();
        assert_eq!(applied, missing, "{what}: entries applied");
        assert_same_values(&db, live);
        assert_eq!(
            db.replay_entries(graph, entries, options_of_fn).unwrap(),
            0,
            "{what}: second replay"
        );
        assert_same_values(&db, live);
        // The slice may be in any order: replay sorts.
        let mut reversed = entries.to_vec();
        reversed.reverse();
        let mut db = start.clone();
        db.replay_entries(graph, &reversed, options_of_fn).unwrap();
        assert_same_values(&db, live);
        let rotated: Vec<JournalEntry> = entries
            .iter()
            .cycle()
            .skip(entries.len() / 3)
            .take(entries.len())
            .cloned()
            .collect();
        let mut db = start.clone();
        db.replay_entries(graph, &rotated, options_of_fn).unwrap();
        assert_same_values(&db, live);
    }

    #[test]
    fn replay_onto_an_empty_database_reproduces_the_run() {
        let graph = lattice();
        let (entries, live) = journal_of(&graph, &recalibration_passes());
        assert_eq!(entries.len(), live.history().len());
        let mut db = ParameterDatabase::<f64>::new(&graph);
        let (calls, options) = counting_options();
        let applied = db.replay_entries(&graph, &entries, options).unwrap();
        assert_eq!(applied, entries.len());
        assert_same_values(&db, &live);
        // `options` is called at most once per element, and only for elements that were used.
        let calls = calls.lock().unwrap();
        assert!(calls.values().all(|&c| c == 1));
        let used: std::collections::BTreeSet<u32> = entries
            .iter()
            .filter_map(|e| match e {
                JournalEntry::Step(s) => Some(s.parameters.iter().map(|p| p.raw())),
                JournalEntry::Invalidated { .. } => None,
            })
            .flatten()
            .collect();
        assert_eq!(
            calls
                .keys()
                .copied()
                .collect::<std::collections::BTreeSet<_>>(),
            used
        );
        // The history is the live one: the journal entries line up with it one to one, and
        // the global indices were assigned in canonical order.
        for (entry, record) in entries.iter().zip(live.history()) {
            match (entry, record) {
                (JournalEntry::Step(a), HistoryRecord::Step(b)) => {
                    assert_eq!((a.pass, a.subgoal, a.centre), (b.pass, b.subgoal, b.centre));
                    assert_eq!(
                        (&a.parameters, &a.option_indices),
                        (&b.parameters, &b.option_indices)
                    );
                }
                (
                    JournalEntry::Invalidated { elements, .. },
                    HistoryRecord::Invalidated(removed),
                ) => assert_eq!(elements, removed),
                other => panic!("{other:?}"),
            }
        }
        assert_replay_reproduces(
            &graph,
            &entries,
            &ParameterDatabase::new(&graph),
            &live,
            "empty",
        );
    }

    #[test]
    fn replay_from_a_save_taken_at_any_point_of_the_run() {
        let graph = lattice();
        let passes = recalibration_passes();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("j.jsonl");
        let mut live = ParameterDatabase::<f64>::new(&graph);
        let mut saves = vec![live.clone()];
        {
            let o = CheckpointObserver::create(&path, &graph).unwrap();
            run_passes(&mut live, Some(&o), 0, &passes, Some(&mut saves));
        }
        let entries = read_journal_entries(&path, &graph).unwrap();
        assert_eq!(saves.len(), live.history().len() + 1);
        for (k, saved) in saves.iter().enumerate() {
            assert_eq!(saved.history().len(), k);
            // Through a real save and load, too.
            let text = saved.to_json_string(&graph).unwrap();
            let loaded = ParameterDatabase::<f64>::from_json_str(&graph, &text).unwrap();
            assert_replay_reproduces(&graph, &entries, saved, &live, &format!("save {k}"));
            assert_replay_reproduces(&graph, &entries, &loaded, &live, &format!("loaded {k}"));
        }
    }

    #[test]
    fn replaying_the_steps_alone_cannot_reproduce_a_recalibration() {
        let graph = lattice();
        let (entries, live) = journal_of(&graph, &recalibration_passes());
        let steps: Vec<StepRecord> = entries
            .iter()
            .filter_map(|e| match e {
                JournalEntry::Step(s) => Some(s.clone()),
                JournalEntry::Invalidated { .. } => None,
            })
            .collect();
        let mut db = ParameterDatabase::<f64>::new(&graph);
        assert_eq!(
            db.replay_journal(&graph, &steps, options_of_fn).unwrap(),
            steps.len()
        );
        assert!(!db.same_outcome(&live));
        assert_eq!(db.step_count(), live.step_count());
        // On a journal without invalidations the two replays agree.
        let plain: Vec<JournalEntry> = steps.iter().cloned().map(JournalEntry::Step).collect();
        let mut a = ParameterDatabase::<f64>::new(&graph);
        let mut b = ParameterDatabase::<f64>::new(&graph);
        a.replay_journal(&graph, &steps, options_of_fn).unwrap();
        b.replay_entries(&graph, &plain, options_of_fn).unwrap();
        assert!(a.same_outcome(&b));
    }

    #[test]
    fn later_steps_overwrite_earlier_ones_in_canonical_order() {
        let graph = lattice();
        let steps = vec![
            at(step_record(0, &[0], &[1], &[], 0.1), 0, 0, 0), // x = option 1
            at(step_record(1, &[1], &[2], &[], 0.1), 0, 1, 0), // y
            at(step_record(0, &[0], &[7], &[], 0.1), 1, 0, 0), // x overwritten: option 7
        ];
        let want = |db: &ParameterDatabase<f64>| {
            assert_eq!(db.get(id(0)).unwrap().option_index, 7);
            assert_eq!(db.get(id(0)).unwrap().value, options_of(id(0))[7]);
            assert_eq!(db.get(id(1)).unwrap().option_index, 2);
            assert_eq!(db.get(id(0)).unwrap().step, 2);
        };
        let mut db = ParameterDatabase::<f64>::new(&graph);
        assert_eq!(db.replay_journal(&graph, &steps, options_of_fn).unwrap(), 3);
        want(&db);
        // The order of the slice does not matter: the passes decide.
        let swapped = [steps[2].clone(), steps[1].clone(), steps[0].clone()];
        let mut other = ParameterDatabase::<f64>::new(&graph);
        other
            .replay_journal(&graph, &swapped, options_of_fn)
            .unwrap();
        assert!(other.same_outcome(&db));
        // From a database that already has all three steps nothing is missing, although step 0
        // alone looks out of date (step 2 overwrote it).
        let mut saved = db.clone();
        assert_eq!(
            saved.replay_journal(&graph, &steps, options_of_fn).unwrap(),
            0
        );
        assert!(saved.same_outcome(&db));
        // From a database saved after the second step only the last step is missing.
        let mut mid = ParameterDatabase::<f64>::new(&graph);
        mid.replay_journal(&graph, &steps[..2], options_of_fn)
            .unwrap();
        assert_eq!(
            mid.replay_journal(&graph, &steps, options_of_fn).unwrap(),
            1
        );
        want(&mid);
    }

    #[test]
    fn a_step_is_recognised_by_pass_subgoal_and_centre_alone() {
        let graph = lattice();
        let first = at(step_record(0, &[0, 1], &[1, 1], &[], 0.1), 4, 2, 0);
        let mut db = ParameterDatabase::<f64>::new(&graph);
        db.replay_journal(&graph, std::slice::from_ref(&first), options_of_fn)
            .unwrap();
        let replay = |db: &mut ParameterDatabase<f64>, step: StepRecord| {
            db.replay_journal(&graph, &[step], options_of_fn).unwrap()
        };
        // Same key, other content (the history of a different run): skipped.
        let other_content = at(step_record(0, &[0, 5], &[9, 9], &[3], 0.7), 4, 2, 7);
        assert_eq!(replay(&mut db, other_content), 0);
        assert_eq!(db.step_count(), 1);
        // Another pass, subgoal or centre: a different step, applied, keeping pass and subgoal.
        for (centre, pass, subgoal) in [(0, 5, 2), (0, 4, 3), (1, 4, 2)] {
            let n = db.step_count();
            let step = at(
                step_record(centre, &[centre], &[2], &[], 0.1),
                pass,
                subgoal,
                0,
            );
            assert_eq!(replay(&mut db, step), 1, "{centre} {pass} {subgoal}");
            let Some(HistoryRecord::Step(record)) = db.history().last() else {
                panic!("no step")
            };
            assert_eq!(
                (record.pass, record.subgoal, record.index),
                (pass, subgoal, n)
            );
        }
        // The same key several times in a journal: that many matching steps are skipped.
        let twice = [
            at(step_record(2, &[2], &[1], &[], 0.1), 6, 0, 0),
            at(step_record(2, &[2], &[2], &[], 0.1), 6, 0, 1),
        ];
        let mut a = ParameterDatabase::<f64>::new(&graph);
        assert_eq!(a.replay_journal(&graph, &twice, options_of_fn).unwrap(), 2);
        assert_eq!(a.replay_journal(&graph, &twice, options_of_fn).unwrap(), 0);
        let mut b = ParameterDatabase::<f64>::new(&graph);
        b.replay_journal(&graph, &twice[..1], options_of_fn)
            .unwrap();
        assert_eq!(b.replay_journal(&graph, &twice, options_of_fn).unwrap(), 1);
        assert!(a.same_outcome(&b));
        // A step without parameters changes no value, but it is a step of the run.
        let empty = at(step_record(0, &[], &[], &[], 0.1), 8, 0, 0);
        let len = db.len();
        assert_eq!(replay(&mut db, empty.clone()), 1);
        assert_eq!(replay(&mut db, empty), 0);
        assert_eq!(db.len(), len);
    }

    #[test]
    fn invalidations_remove_only_what_earlier_passes_calibrated() {
        let graph = lattice();
        let passes = vec![
            pass_spec(&[], vec![vec![st(0, &[0, 1, 2], &[1, 1, 1])]]),
            pass_spec(&[0, 1], vec![vec![st(1, &[1, 3], &[2, 2])]]),
        ];
        let (entries, live) = journal_of(&graph, &passes);
        assert_eq!(entries.len(), 3);
        assert!(matches!(
            entries[1],
            JournalEntry::Invalidated { pass: 1, .. }
        ));
        // Live: 0 is forgotten for good, 1 is learned again by a step of pass 1.
        assert!(!live.is_calibrated(id(0)));
        assert_eq!(live.get(id(1)).unwrap().step, 1);

        // Saved before the invalidation: it removes both.
        let mut before = ParameterDatabase::<f64>::new(&graph);
        before
            .replay_entries(&graph, &entries[..1], options_of_fn)
            .unwrap();
        assert_replay_reproduces(&graph, &entries, &before, &live, "before");
        // Saved after the invalidation, before 1 was learned again: nothing is removed
        // (neither is counted), and the step is applied.
        let mut between = before.clone();
        between.invalidate(&ids(&[0, 1]));
        let mut db = between.clone();
        assert_eq!(
            db.replay_entries(&graph, &entries, options_of_fn).unwrap(),
            1
        );
        assert_same_values(&db, &live);
        // Saved after everything: element 1 was learned by a step of pass 1, which the
        // invalidation of pass 1 must not touch.
        let mut done = live.clone();
        assert_eq!(
            done.replay_entries(&graph, &entries, options_of_fn)
                .unwrap(),
            0
        );
        assert_same_values(&done, &live);
        // Elements that were never calibrated are no-ops, and are not counted.
        let lonely = [JournalEntry::Invalidated {
            pass: 9,
            elements: ids(&[10, 11]),
        }];
        let mut db = live.clone();
        assert_eq!(
            db.replay_entries(&graph, &lonely, options_of_fn).unwrap(),
            0
        );
        assert_same_values(&db, &live);
        // An invalidation never removes what a step of its own pass or a later one learned,
        // wherever the entry stands in the slice.
        let late = [JournalEntry::Invalidated {
            pass: 1,
            elements: ids(&[1, 3]),
        }];
        let mut db = live.clone();
        assert_eq!(db.replay_entries(&graph, &late, options_of_fn).unwrap(), 0);
        let early = [JournalEntry::Invalidated {
            pass: 2,
            elements: ids(&[1, 3, 2]),
        }];
        assert_eq!(db.replay_entries(&graph, &early, options_of_fn).unwrap(), 1);
        assert!(!db.is_calibrated(id(1)) && !db.is_calibrated(id(3)) && !db.is_calibrated(id(2)));
        let Some(HistoryRecord::Invalidated(removed)) = db.history().last() else {
            panic!("no invalidation")
        };
        assert_eq!(removed, &ids(&[1, 2, 3]));
    }

    #[test]
    fn a_database_ahead_of_the_journal_is_left_alone() {
        // The journal lost its tail: it ends before the invalidation of pass 2, which the
        // database (saved later) has seen, and the elements were learned again afterwards.
        let graph = lattice();
        let (entries, live) = journal_of(&graph, &recalibration_passes());
        let cut = entries
            .iter()
            .position(|e| matches!(e, JournalEntry::Invalidated { .. }))
            .unwrap();
        for keep in [0, 1, cut, cut + 1, entries.len() - 1] {
            let mut db = live.clone();
            assert_eq!(
                db.replay_entries(&graph, &entries[..keep], options_of_fn)
                    .unwrap(),
                0,
                "journal of {keep} entries"
            );
            assert_same_values(&db, &live);
        }
        // The full journal, with the invalidation, is no harm to the later database either.
        let mut db = live.clone();
        assert_eq!(
            db.replay_entries(&graph, &entries, options_of_fn).unwrap(),
            0
        );
    }

    #[test]
    fn a_resumed_run_may_start_a_journal_of_its_own_with_create() {
        // The first part of a run, saved; the process dies; the run resumes from the save with
        // a new journal made by `create` (not `append`), is saved again, and ends.
        let graph = lattice();
        let dir = tempfile::tempdir().unwrap();
        let (j1, j2) = (dir.path().join("1.jsonl"), dir.path().join("2.jsonl"));
        let passes = recalibration_passes();
        let mut live = ParameterDatabase::<f64>::new(&graph);
        {
            let o = CheckpointObserver::create(&j1, &graph).unwrap();
            run_passes(&mut live, Some(&o), 0, &passes[..2], None);
        }
        let first_part = live.clone();
        let next = live.steps().map(|s| s.pass).max().unwrap() + 1;
        let mut saves = vec![live.clone()];
        {
            let o = CheckpointObserver::create(&j2, &graph).unwrap();
            run_passes(&mut live, Some(&o), next, &passes[2..], Some(&mut saves));
        }
        let (e1, e2) = (
            read_journal_entries(&j1, &graph).unwrap(),
            read_journal_entries(&j2, &graph).unwrap(),
        );
        // The second journal alone, onto the save that preceded it and onto any save taken
        // inside the resumed run (this is the case that used to apply steps twice).
        for (k, saved) in saves.iter().enumerate() {
            let mut db = saved.clone();
            let missing = live.history().len() - saved.history().len();
            assert_eq!(
                db.replay_entries(&graph, &e2, options_of_fn).unwrap(),
                missing
            );
            assert_same_values(&db, &live);
            // Both journals, oldest first, are all the same to it.
            let both: Vec<JournalEntry> = e1.iter().chain(&e2).cloned().collect();
            assert_replay_reproduces(&graph, &both, saved, &live, &format!("resumed save {k}"));
        }
        // Onto an empty database: the first journal and then the second, or both at once.
        let mut db = ParameterDatabase::<f64>::new(&graph);
        db.replay_entries(&graph, &e1, options_of_fn).unwrap();
        assert_same_values(&db, &first_part);
        db.replay_entries(&graph, &e2, options_of_fn).unwrap();
        assert_same_values(&db, &live);
        // Onto a save taken before the resumed run, replaying the old journal first is a no-op.
        let mut db = first_part.clone();
        db.replay_entries(&graph, &e1, options_of_fn).unwrap();
        db.replay_entries(&graph, &e2, options_of_fn).unwrap();
        assert_same_values(&db, &live);
    }

    #[test]
    fn a_resumed_run_may_append_to_the_old_journal() {
        let graph = lattice();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("j.jsonl");
        let passes = recalibration_passes();
        let mut live = ParameterDatabase::<f64>::new(&graph);
        {
            let o = CheckpointObserver::create(&path, &graph).unwrap();
            run_passes(&mut live, Some(&o), 0, &passes[..2], None);
        }
        let saved = live.clone();
        // A torn line from the crash.
        {
            let mut f = OpenOptions::new().append(true).open(&path).unwrap();
            f.write_all(b"{\"type\":\"step\",\"index\":0,\"pa").unwrap();
        }
        let mut resumed = saved.clone();
        resumed
            .replay_entries(
                &graph,
                &read_journal_entries(&path, &graph).unwrap(),
                options_of_fn,
            )
            .unwrap();
        assert_same_values(&resumed, &saved);
        let next = resumed.steps().map(|s| s.pass).max().unwrap() + 1;
        {
            let o = CheckpointObserver::append(&path, &graph).unwrap();
            run_passes(&mut resumed, Some(&o), next, &passes[2..], None);
        }
        let entries = read_journal_entries(&path, &graph).unwrap();
        assert_eq!(entries.len(), resumed.history().len());
        assert_replay_reproduces(
            &graph,
            &entries,
            &ParameterDatabase::new(&graph),
            &resumed,
            "appended",
        );
        assert_replay_reproduces(
            &graph,
            &entries,
            &saved,
            &resumed,
            "appended, from the save",
        );
    }

    #[test]
    fn replay_validates_its_input_and_applies_nothing_on_error() {
        let graph = lattice();
        let outside = graph.len() as u32;
        let mut db = ParameterDatabase::<f64>::new(&graph);
        let good = at(step_record(0, &[0], &[1], &[], 0.1), 0, 0, 0);
        // No entries.
        assert_eq!(db.replay_journal(&graph, &[], options_of_fn).unwrap(), 0);
        assert_eq!(db.replay_entries(&graph, &[], options_of_fn).unwrap(), 0);
        let mut mismatched = at(step_record(0, &[0, 1], &[1], &[], 0.1), 1, 0, 0);
        mismatched.parameters = ids(&[0, 1]);
        let mut bad_centre = at(step_record(0, &[0], &[1], &[], 0.1), 1, 0, 0);
        bad_centre.centre = id(outside);
        let mut bad_constraint = at(step_record(0, &[0], &[1], &[], 0.1), 1, 0, 0);
        bad_constraint.constraints = ids(&[0, outside]);
        let bad_parameter = at(step_record(0, &[outside], &[1], &[], 0.1), 1, 0, 0);
        let bad_option = at(step_record(1, &[1], &[16], &[], 0.1), 1, 0, 0);
        let bad_forget = JournalEntry::Invalidated {
            pass: 5,
            elements: ids(&[0, outside]),
        };
        let cases: Vec<(JournalEntry, &str)> = vec![
            (JournalEntry::Step(mismatched), "option indices"),
            (JournalEntry::Step(bad_centre), "not in the graph"),
            (JournalEntry::Step(bad_constraint), "not in the graph"),
            (JournalEntry::Step(bad_parameter), "not in the graph"),
            (JournalEntry::Step(bad_option), "out of range"),
            (bad_forget, "not in the graph"),
        ];
        for (bad, needle) in cases {
            // The bad entry comes after a good step, a bad entry of the other kind, and last.
            for entries in [
                vec![JournalEntry::Step(good.clone()), bad.clone()],
                vec![bad.clone(), JournalEntry::Step(good.clone())],
                vec![bad.clone()],
            ] {
                match db.replay_entries(&graph, &entries, options_of_fn) {
                    Err(FaultTuneError::Format(msg)) => {
                        assert!(msg.contains(needle), "{msg:?} lacks {needle:?}")
                    }
                    r => panic!("expected Format, got {r:?} for {bad:?}"),
                }
                // Nothing was applied, not even the good step.
                assert!(db.is_empty() && db.history().is_empty(), "{bad:?}");
            }
            if let JournalEntry::Step(step) = &bad {
                assert!(matches!(
                    db.replay_journal(&graph, &[good.clone(), step.clone()], options_of_fn),
                    Err(FaultTuneError::Format(_))
                ));
                assert!(db.history().is_empty());
            }
        }
        // Wrong graph.
        let other = ProcessorGraph::square_lattice(2, 2);
        assert!(matches!(
            db.replay_journal(&other, &[], options_of_fn),
            Err(FaultTuneError::GraphMismatch { .. })
        ));
        assert!(matches!(
            db.replay_entries(&other, &[], options_of_fn),
            Err(FaultTuneError::GraphMismatch { .. })
        ));
        // A bad entry is reported even where the database already has the step, but a skipped
        // step's options are not looked at.
        db.replay_journal(&graph, std::slice::from_ref(&good), options_of_fn)
            .unwrap();
        let mut broken = good.clone();
        broken.option_indices = vec![99];
        assert_eq!(
            db.replay_journal(&graph, &[broken.clone()], |_| panic!(
                "options of a skipped step"
            ))
            .unwrap(),
            0
        );
        broken.centre = id(outside);
        assert!(db.replay_journal(&graph, &[broken], options_of_fn).is_err());
        assert_eq!(db.step_count(), 1);
    }

    #[test]
    fn a_database_that_holds_a_step_outside_the_graph_cannot_be_saved() {
        // Replay refuses such steps (above); a hand-built database that holds one reports a
        // Format error when it is saved, not an I/O error.
        let graph = lattice();
        let outside = graph.len() as u32;
        for (centre, constraints) in [(outside, vec![]), (0, vec![outside])] {
            let mut step = step_record(0, &[0], &[1], &[], 0.5);
            step.centre = id(centre);
            step.constraints = ids(&constraints);
            let mut bad = ParameterDatabase::<f64>::new(&graph);
            bad.commit_step(step, vec![1.0]);
            match bad.to_json_string(&graph) {
                Err(FaultTuneError::Format(msg)) => assert!(msg.contains("history"), "{msg}"),
                r => panic!("expected Format, got {r:?}"),
            }
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("p.json");
            assert!(matches!(
                bad.save(&graph, &path),
                Err(FaultTuneError::Format(_))
            ));
            assert!(!path.exists());
        }
    }

    fn pass_spec_strategy() -> impl Strategy<Value = Vec<PassSpec>> {
        let step = prop::collection::vec((0u32..29, 0u32..16), 1..4).prop_map(|v| {
            st(
                v[0].0,
                &v.iter().map(|x| x.0).collect::<Vec<_>>(),
                &v.iter().map(|x| x.1).collect::<Vec<_>>(),
            )
        });
        let subgoals = prop::collection::vec(prop::collection::vec(step, 0..4), 0..4);
        let pass = (prop::collection::vec(0u32..29, 0..5), subgoals)
            .prop_map(|(forget, subgoals)| PassSpec { forget, subgoals });
        prop::collection::vec(pass, 0..7)
    }

    proptest! {
        #[test]
        fn replay_property(passes in pass_spec_strategy(), split in 0usize..7) {
            let graph = ProcessorGraph::square_lattice(3, 3); // 9 + 12 + 8 = 29 elements
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("j.jsonl");
            let mut live = ParameterDatabase::<f64>::new(&graph);
            let mut saves = vec![live.clone()];
            {
                let o = CheckpointObserver::create(&path, &graph).unwrap();
                run_passes(&mut live, Some(&o), 0, &passes, Some(&mut saves));
            }
            let entries = read_journal_entries(&path, &graph).unwrap();
            prop_assert_eq!(entries.len(), live.history().len());
            // From scratch and from every save the replay ends in the live database, applying
            // what is missing, and a second one applies nothing.
            for saved in &saves {
                let mut db = saved.clone();
                let applied = db.replay_entries(&graph, &entries, options_of_fn).unwrap();
                prop_assert_eq!(applied, live.history().len() - saved.history().len());
                prop_assert!(db.same_outcome(&live));
                prop_assert_eq!(db.replay_entries(&graph, &entries, options_of_fn).unwrap(), 0);
                prop_assert!(db.same_outcome(&live));
                let mut reversed = entries.clone();
                reversed.reverse();
                let mut db = saved.clone();
                db.replay_entries(&graph, &reversed, options_of_fn).unwrap();
                prop_assert!(db.same_outcome(&live));
            }
            // A resumed run with a journal of its own: split the run after some pass.
            let split = split.min(passes.len());
            let mut first = ParameterDatabase::<f64>::new(&graph);
            run_passes(&mut first, None, 0, &passes[..split], None);
            let mut second_saves = vec![first.clone()];
            let second_path = dir.path().join("2.jsonl");
            let mut resumed = first.clone();
            {
                let o = CheckpointObserver::create(&second_path, &graph).unwrap();
                run_passes(&mut resumed, Some(&o), split as u32, &passes[split..], Some(&mut second_saves));
            }
            let second = read_journal_entries(&second_path, &graph).unwrap();
            prop_assert!(resumed.same_outcome(&live));
            for saved in &second_saves {
                let mut db = saved.clone();
                db.replay_entries(&graph, &second, options_of_fn).unwrap();
                prop_assert!(db.same_outcome(&live));
            }
        }
    }

    // -------------------------------------------------------------------------------
    // With the engine
    // -------------------------------------------------------------------------------

    use crate::engine::{CancellationToken, Snake, SnakeConfig};
    use crate::graph::ElementKind;
    use crate::sets::ElementSet;
    use crate::toy::CollisionModel;

    /// The model of the engine tests, and the same device after it drifted: the same
    /// options, other preferred frequencies.
    fn engine_models(graph: &ProcessorGraph) -> (CollisionModel, CollisionModel) {
        (
            CollisionModel::new(graph, 150, 5.0, 7.0, 0.01, 1, 3),
            CollisionModel::new(graph, 150, 5.0, 7.0, 0.01, 1, 4),
        )
    }

    fn goal_of(graph: &ProcessorGraph) -> ElementSet {
        graph.goal_excluding(&[ElementKind::Edge(EdgeType::Parasitic)])
    }

    fn engine_config(threads: usize) -> SnakeConfig {
        SnakeConfig {
            threads: Some(threads),
            ..SnakeConfig::default()
        }
    }

    /// With `d_P = 2` a pass calibrates a few elements per subgoal, so that a stitched run takes
    /// many passes, most of them with several subgoals.
    fn stitch_config(threads: usize) -> SnakeConfig {
        SnakeConfig {
            parameter_distance: 2,
            ..engine_config(threads)
        }
    }

    /// Four quadrants of an `n x n` lattice, as goal subsets.
    fn quadrants(graph: &ProcessorGraph, goal: &ElementSet, n: usize) -> Vec<ElementSet> {
        let mut regions = vec![ElementSet::new(graph.len()); 4];
        let position = |label: &str| -> (usize, usize) {
            // Labels are q{r}_{c} or q{r}_{c}-q{r2}_{c2}: use the first endpoint.
            let first = label.split('-').next().unwrap();
            let (r, c) = first[1..].split_once('_').unwrap();
            (r.parse().unwrap(), c.parse().unwrap())
        };
        for element in goal.iter() {
            let (r, c) = position(graph.label(element));
            regions[usize::from(r >= n / 2) * 2 + usize::from(c >= n / 2)].insert(element);
        }
        regions
    }

    /// Checks that `entries` (the canonical journal of `live`) line up one to one with the
    /// history of `live`.
    fn assert_entries_are_the_history(entries: &[JournalEntry], live: &ParameterDatabase<f64>) {
        assert_eq!(entries.len(), live.history().len());
        let mut local_index = HashMap::new();
        for (entry, record) in entries.iter().zip(live.history()) {
            match (entry, record) {
                (JournalEntry::Step(a), HistoryRecord::Step(b)) => {
                    assert_eq!((a.pass, a.subgoal, a.centre), (b.pass, b.subgoal, b.centre));
                    assert_eq!(a.parameters, b.parameters);
                    assert_eq!(a.option_indices, b.option_indices);
                    assert_eq!(a.constraints, b.constraints);
                    assert_eq!(a.evaluations, b.evaluations);
                    assert!(a.error == b.error || (a.error.is_nan() && b.error.is_nan()));
                    // The journal carries the ordinal within the subgoal.
                    let n = local_index.entry((a.pass, a.subgoal)).or_insert(0u64);
                    assert_eq!(a.index, *n);
                    *n += 1;
                }
                (
                    JournalEntry::Invalidated { elements, .. },
                    HistoryRecord::Invalidated(removed),
                ) => assert_eq!(elements, removed),
                other => panic!("the journal and the history differ: {other:?}"),
            }
        }
    }

    /// The database a replay of the first `k` entries gives: the live database as it was
    /// after `k` changes of its history.
    fn prefix_of(
        graph: &ProcessorGraph,
        entries: &[JournalEntry],
        live: &ParameterDatabase<f64>,
        k: usize,
        options: &impl Fn(ElementId) -> Vec<f64>,
    ) -> ParameterDatabase<f64> {
        let mut db = ParameterDatabase::<f64>::new(graph);
        assert_eq!(db.replay_entries(graph, &entries[..k], options).unwrap(), k);
        for (a, b) in db.history().iter().zip(live.history()) {
            match (a, b) {
                (HistoryRecord::Step(a), HistoryRecord::Step(b)) => assert!(a.same_outcome(b)),
                (a, b) => assert_eq!(a, b),
            }
        }
        db
    }

    /// The contract of a replay against a live run: onto an empty database, onto the databases
    /// in `saves` (taken at any time of the run) and onto prefixes of the history, the result
    /// is the live database, exactly the missing history is applied, and replaying again
    /// applies nothing.
    fn assert_engine_replay(
        graph: &ProcessorGraph,
        model: &CollisionModel,
        entries: &[JournalEntry],
        live: &ParameterDatabase<f64>,
        saves: &[(&str, ParameterDatabase<f64>)],
        context: &str,
    ) {
        let options = |e: ElementId| model.options(e);
        let check = |start: &ParameterDatabase<f64>, what: &str| {
            let mut db = start.clone();
            let applied = db.replay_entries(graph, entries, options).unwrap();
            assert_eq!(
                applied,
                live.history().len() - start.history().len(),
                "{context}: {what}: entries applied"
            );
            assert!(db.same_outcome(live), "{context}: {what}");
            assert_eq!(
                db.replay_entries(graph, entries, options).unwrap(),
                0,
                "{context}: {what}: twice"
            );
            assert!(db.same_outcome(live), "{context}: {what}: twice");
        };
        check(&ParameterDatabase::new(graph), "empty");
        for (name, saved) in saves {
            check(saved, name);
            // A save written to disk and read back is the same.
            let text = saved.to_json_string(graph).unwrap();
            check(
                &ParameterDatabase::<f64>::from_json_str(graph, &text).unwrap(),
                name,
            );
        }
        check(live, "final");
        let n = entries.len();
        for k in (0..=n).step_by((n / 9).max(1)).chain([n.saturating_sub(1)]) {
            check(
                &prefix_of(graph, entries, live, k, &options),
                &format!("prefix {k}"),
            );
        }
    }

    /// A stitched calibration with `threads` workers and a journal: the database, the journal
    /// entries and the database as saved after the first pass.
    fn stitched_run(
        graph: &ProcessorGraph,
        threads: usize,
        journal: &Path,
    ) -> (ParameterDatabase<f64>, Vec<JournalEntry>) {
        let goal = goal_of(graph);
        let (model, _) = engine_models(graph);
        let observer = Arc::new(CheckpointObserver::create(journal, graph).unwrap());
        let snake =
            Snake::new(graph, goal.clone(), stitch_config(threads)).observer(observer.clone());
        let plan = snake.stitch_plan(quadrants(graph, &goal, 12)).unwrap();
        let mut db = ParameterDatabase::new(graph);
        snake.stitch(&mut db, &plan, &model).unwrap();
        assert!(observer.take_error().is_none());
        drop(snake);
        drop(observer);
        let entries = read_journal_entries(journal, graph).unwrap();
        (db, entries)
    }

    #[test]
    fn a_stitched_run_replays_exactly_at_any_thread_count() {
        let graph = ProcessorGraph::square_lattice(12, 12);
        let model = engine_models(&graph).0;
        let mut first: Option<(ParameterDatabase<f64>, Vec<String>)> = None;
        for threads in [1, 2, 4, 8] {
            let dir = tempfile::tempdir().unwrap();
            let (live, entries) = stitched_run(&graph, threads, &dir.path().join("j.jsonl"));
            let context = format!("{threads} threads");
            // The test is only worth something if passes have several subgoals.
            assert!(
                live.steps().any(|s| s.subgoal >= 3),
                "{context}: the stitched run should have four subgoals in a pass"
            );
            assert_entries_are_the_history(&entries, &live);
            assert_engine_replay(&graph, &model, &entries, &live, &[], &context);
            // The run, its history and its canonical journal do not depend on the threads.
            let keys = content(&entries);
            match &first {
                None => first = Some((live, keys)),
                Some((reference, reference_keys)) => {
                    assert!(live.same_outcome(reference), "{context}");
                    assert_eq!(&keys, reference_keys, "{context}");
                }
            }
        }
    }

    /// Calibrates a lattice, saves, and recalibrates scattered elements (several subgoals per
    /// pass) with `threads` workers. With `split`, the recalibration has a journal of its own,
    /// created after the calibration.
    fn recalibrated_run(
        graph: &ProcessorGraph,
        threads: usize,
        split: bool,
        dir: &Path,
    ) -> (
        ParameterDatabase<f64>,
        ParameterDatabase<f64>,
        Vec<JournalEntry>,
        Vec<JournalEntry>,
    ) {
        let goal = goal_of(graph);
        let (model, drifted) = engine_models(graph);
        let journal =
            |name: &str| Arc::new(CheckpointObserver::create(dir.join(name), graph).unwrap());
        let first = journal("1.jsonl");
        let snake = Snake::new(graph, goal.clone(), engine_config(threads)).observer(first.clone());
        let mut db = snake.calibrate(&model).unwrap();
        // Saved before the recalibration, as a periodic save would have been.
        let path = dir.join("p.json");
        db.save(graph, &path).unwrap();
        let saved = ParameterDatabase::<f64>::load(graph, &path).unwrap();
        let second = if split {
            journal("2.jsonl")
        } else {
            first.clone()
        };
        let snake = if split {
            Snake::new(graph, goal, engine_config(threads)).observer(second.clone())
        } else {
            snake
        };
        let spots = [(0, 0), (0, 11), (11, 0), (11, 11), (5, 5)];
        let centres: Vec<ElementId> = spots
            .iter()
            .map(|&(r, c)| graph.lattice_node(r, c).unwrap())
            .collect();
        snake.recalibrate(&mut db, &centres, &drifted).unwrap();
        assert!(first.take_error().is_none() && second.take_error().is_none());
        drop(snake);
        drop((first, second));
        let e1 = read_journal_entries(dir.join("1.jsonl"), graph).unwrap();
        let e2 = if split {
            read_journal_entries(dir.join("2.jsonl"), graph).unwrap()
        } else {
            Vec::new()
        };
        (db, saved, e1, e2)
    }

    #[test]
    fn a_recalibration_of_scattered_elements_replays_exactly_at_any_thread_count() {
        let graph = ProcessorGraph::square_lattice(12, 12);
        let model = engine_models(&graph).0;
        for threads in [1, 2, 4, 8] {
            for attempt in 0..2 {
                let dir = tempfile::tempdir().unwrap();
                let (live, saved, entries, _) =
                    recalibrated_run(&graph, threads, false, dir.path());
                let context = format!("{threads} threads, attempt {attempt}");
                assert_entries_are_the_history(&entries, &live);
                let invalidations = entries
                    .iter()
                    .filter(|e| matches!(e, JournalEntry::Invalidated { .. }))
                    .count();
                assert_eq!(
                    invalidations, 1,
                    "{context}: recalibrate announces one removal"
                );
                assert!(
                    live.steps().filter(|s| s.subgoal > 0).count() > 0,
                    "{context}: several subgoals"
                );
                assert_engine_replay(
                    &graph,
                    &model,
                    &entries,
                    &live,
                    &[("before the recalibration", saved.clone())],
                    &context,
                );
                // Replaying the steps alone does not reproduce the recalibration.
                let steps: Vec<StepRecord> = entries
                    .iter()
                    .filter_map(|e| match e {
                        JournalEntry::Step(s) => Some(s.clone()),
                        JournalEntry::Invalidated { .. } => None,
                    })
                    .collect();
                let mut steps_only = saved.clone();
                steps_only
                    .replay_journal(&graph, &steps, |e| model.options(e))
                    .unwrap();
                assert!(!steps_only.same_outcome(&live), "{context}");
            }
        }
    }

    #[test]
    fn a_recalibration_with_a_journal_of_its_own_replays_after_the_calibration() {
        let graph = ProcessorGraph::square_lattice(12, 12);
        let model = engine_models(&graph).0;
        let options = |e: ElementId| model.options(e);
        for threads in [1, 4] {
            let dir = tempfile::tempdir().unwrap();
            let (live, saved, e1, e2) = recalibrated_run(&graph, threads, true, dir.path());
            let context = format!("{threads} threads");
            // The pass numbers continued: the second journal starts after the first.
            let last = |entries: &[JournalEntry]| {
                entries
                    .iter()
                    .map(|e| match e {
                        JournalEntry::Step(s) => s.pass,
                        JournalEntry::Invalidated { pass, .. } => *pass,
                    })
                    .max()
                    .unwrap()
            };
            let first_of_second = match &e2[0] {
                JournalEntry::Invalidated { pass, .. } => *pass,
                JournalEntry::Step(s) => s.pass,
            };
            assert_eq!(first_of_second, last(&e1) + 1, "{context}");
            let both: Vec<JournalEntry> = e1.iter().chain(&e2).cloned().collect();
            assert_entries_are_the_history(&both, &live);
            assert_engine_replay(
                &graph,
                &model,
                &both,
                &live,
                &[("before the recalibration", saved.clone())],
                &context,
            );
            // The second journal alone is enough on top of the saved calibration.
            let mut db = saved.clone();
            assert_eq!(
                db.replay_entries(&graph, &e2, options).unwrap(),
                e2.len(),
                "{context}"
            );
            assert!(db.same_outcome(&live), "{context}");
            // And a journal concatenated in the wrong order is the same journal.
            let swapped: Vec<JournalEntry> = e2.iter().chain(&e1).cloned().collect();
            let mut db = ParameterDatabase::<f64>::new(&graph);
            db.replay_entries(&graph, &swapped, options).unwrap();
            assert!(db.same_outcome(&live), "{context}");
        }
    }

    /// Runs `stitch` until the pass numbered `cut` finished, then cancels (a clean pass
    /// boundary); with `cut == None` it runs to the end.
    fn stitch_until(
        graph: &ProcessorGraph,
        db: &mut ParameterDatabase<f64>,
        threads: usize,
        journal: &Path,
        cut: Option<u32>,
    ) -> bool {
        let goal = goal_of(graph);
        let (model, _) = engine_models(graph);
        let token = CancellationToken::new();
        let hook = {
            let token = token.clone();
            move |event: &Event| {
                if matches!(event, Event::PassFinished { pass, .. } if Some(*pass) == cut) {
                    token.cancel();
                }
            }
        };
        // The journal is made when the run (re)starts, with `create`.
        let observer = Arc::new(CheckpointObserver::create(journal, graph).unwrap());
        let snake = Snake::new(graph, goal.clone(), stitch_config(threads))
            .cancellation(token)
            .observer(hook)
            .observer(observer.clone());
        let plan = snake.stitch_plan(quadrants(graph, &goal, 12)).unwrap();
        let result = snake.stitch(db, &plan, &model);
        assert!(observer.take_error().is_none());
        match result {
            Ok(_) => false,
            Err(FaultTuneError::Cancelled { .. }) => true,
            Err(other) => panic!("{other}"),
        }
    }

    #[test]
    fn a_crash_in_the_middle_of_a_parallel_pass_is_recovered() {
        // The journal of a 4-thread run is cut at an arbitrary byte, so that a pass is half
        // written (some subgoals have their steps in it, others lost theirs). The recovery
        // replays what is left onto an empty database and resumes with a journal of its own.
        let graph = ProcessorGraph::square_lattice(12, 12);
        let model = engine_models(&graph).0;
        let options = |e: ElementId| model.options(e);
        let dir = tempfile::tempdir().unwrap();
        let full = dir.path().join("full.jsonl");
        let (_, entries) = stitched_run(&graph, 4, &full);
        let bytes = fs::read(&full).unwrap();
        assert!(entries.len() > 20);
        for (k, tenth) in [1usize, 3, 5, 8].into_iter().enumerate() {
            let cut = bytes.len() * tenth / 10 + 7; // mid-line, most likely
            let (torn, second) = (
                dir.path().join(format!("torn{k}.jsonl")),
                dir.path().join(format!("resumed{k}.jsonl")),
            );
            fs::write(&torn, &bytes[..cut]).unwrap();
            let survived = read_journal_entries(&torn, &graph).unwrap();
            assert!(!survived.is_empty() && survived.len() < entries.len());
            // The survivors replay into a valid database.
            let mut db = ParameterDatabase::<f64>::new(&graph);
            assert_eq!(
                db.replay_entries(&graph, &survived, options).unwrap(),
                survived.len()
            );
            let recovered = db.clone();
            // Resume. The pass numbers continue after the highest pass that has a step.
            assert!(!stitch_until(&graph, &mut db, 4, &second, None));
            assert_eq!(db.calibrated(), &goal_of(&graph), "cut {k}");
            let resumed = read_journal_entries(&second, &graph).unwrap();
            assert_entries_are_the_history(
                &survived.iter().chain(&resumed).cloned().collect::<Vec<_>>(),
                &db,
            );
            // Both journals, or the second one alone on the recovered database, give the
            // final database; so do they in any other combination.
            let both: Vec<JournalEntry> = survived.iter().chain(&resumed).cloned().collect();
            assert_engine_replay(
                &graph,
                &model,
                &both,
                &db,
                &[("recovered", recovered.clone())],
                &format!("cut {k}"),
            );
            let mut again = recovered.clone();
            again.replay_entries(&graph, &resumed, options).unwrap();
            assert!(again.same_outcome(&db), "cut {k}");
        }
    }

    #[test]
    fn a_run_cancelled_on_a_pass_boundary_and_resumed_with_new_journals_replays_exactly() {
        let graph = ProcessorGraph::square_lattice(12, 12);
        let model = engine_models(&graph).0;
        let options = |e: ElementId| model.options(e);
        for threads in [1, 2, 4, 8] {
            let dir = tempfile::tempdir().unwrap();
            let context = format!("{threads} threads");
            let journals: Vec<_> = (0..3)
                .map(|k| dir.path().join(format!("{k}.jsonl")))
                .collect();
            let mut db = ParameterDatabase::<f64>::new(&graph);
            // Part one: cancelled after pass 0. Part two: resumed with a journal made after the
            // resume, cancelled after pass 3. Part three: resumed again, to the end.
            assert!(
                stitch_until(&graph, &mut db, threads, &journals[0], Some(0)),
                "{context}"
            );
            let after_one = db.clone();
            assert!(
                stitch_until(&graph, &mut db, threads, &journals[1], Some(3)),
                "{context}"
            );
            let after_two = db.clone();
            assert!(
                !stitch_until(&graph, &mut db, threads, &journals[2], None),
                "{context}"
            );
            let live = db;
            assert_eq!(live.calibrated(), &goal_of(&graph), "{context}");
            let parts: Vec<Vec<JournalEntry>> = journals
                .iter()
                .map(|j| read_journal_entries(j, &graph).unwrap())
                .collect();
            assert!(
                parts.iter().all(|p| !p.is_empty()),
                "{context}: every part ran"
            );
            // The pass numbers continued across the calls.
            let passes: Vec<u32> = live.steps().map(|s| s.pass).collect();
            assert!(passes.windows(2).all(|w| w[0] <= w[1]), "{context}");
            assert!(passes.iter().max().unwrap() >= &2, "{context}");

            let all: Vec<JournalEntry> = parts.iter().flatten().cloned().collect();
            assert_entries_are_the_history(&all, &live);
            // All the journals, or each journal on the save that precedes it, or the journals of
            // the later parts on the earlier saves.
            assert_engine_replay(
                &graph,
                &model,
                &all,
                &live,
                &[
                    ("after part one", after_one.clone()),
                    ("after part two", after_two.clone()),
                ],
                &context,
            );
            let mut db = after_one.clone();
            let applied = db.replay_entries(&graph, &parts[1], options).unwrap();
            assert_eq!(applied, parts[1].len(), "{context}");
            assert!(
                db.same_outcome(&after_two),
                "{context}: part two on the first save"
            );
            let mut db = after_two.clone();
            assert_eq!(
                db.replay_entries(&graph, &parts[2], options).unwrap(),
                parts[2].len(),
                "{context}"
            );
            assert!(
                db.same_outcome(&live),
                "{context}: part three on the second save"
            );
            // Part one's journal on a database that is further along changes nothing.
            let mut db = live.clone();
            assert_eq!(db.replay_entries(&graph, &parts[0], options).unwrap(), 0);
            // The journals of the later parts, replayed one after the other onto the first
            // save, give the final database, also when the middle one is replayed twice.
            let mut db = after_one.clone();
            for part in [&parts[1], &parts[1], &parts[2]] {
                db.replay_entries(&graph, part, options).unwrap();
            }
            assert!(db.same_outcome(&live), "{context}");
        }
    }

    #[test]
    fn long_file_names_can_be_saved() {
        let graph = lattice();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a".repeat(250));
        let db = sample_db(&graph);
        db.save(&graph, &path).unwrap();
        assert!(ParameterDatabase::<f64>::load(&graph, &path)
            .unwrap()
            .same_outcome(&db));
        // Nothing but the target is left in the directory.
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 1);
    }

    #[test]
    fn fingerprints_are_written_with_all_sixteen_digits() {
        assert_eq!(hex16(0xab), "00000000000000ab");
        assert_eq!(hex16(0), "0000000000000000");
        assert_eq!(hex16(u64::MAX), "ffffffffffffffff");
        for v in [0, 1, 0xab, 0x0123_4567_89ab_cdef, u64::MAX] {
            assert_eq!(parse_hex16(&hex16(v)).unwrap(), v);
        }
        assert!(parse_hex16("ab").is_err());
        assert!(parse_hex16("+0000000000000ab").is_err());
        // A graph whose fingerprint starts with zero digits saves and loads.
        let mut found = false;
        for n in 1..400 {
            let graph = ProcessorGraph::square_lattice(1, n);
            if graph.fingerprint() >> 56 == 0 {
                let db = ParameterDatabase::<f64>::new(&graph);
                let text = db.to_json_string(&graph).unwrap();
                assert!(text.contains(&hex16(graph.fingerprint())));
                ParameterDatabase::<f64>::from_json_str(&graph, &text).unwrap();
                found = true;
                break;
            }
        }
        assert!(
            found,
            "no lattice with a fingerprint of leading zeros in range"
        );
    }

    #[test]
    fn format_versions_must_be_plain_digits() {
        assert_eq!(parse_format_version("faulttune.parameters/1").unwrap(), 1);
        assert_eq!(parse_format_version("faulttune.parameters/12").unwrap(), 12);
        for bad in [
            "faulttune.parameters/+1",
            "faulttune.parameters/-1",
            "faulttune.parameters/",
            "faulttune.parameters/1.0",
            "faulttune.parameters/ 1",
            "faulttune.parameters/x",
        ] {
            assert!(parse_format_version(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn values_that_cannot_be_read_back_are_refused_when_saving() {
        let graph = lattice();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("p.json");
        let good = sample_db(&graph);
        good.save(&graph, &path).unwrap();
        let before = fs::read(&path).unwrap();
        for bad in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            let mut db = ParameterDatabase::<f64>::new(&graph);
            db.commit_step(step_record(0, &[0], &[0], &[], 0.1), vec![bad]);
            match db.to_json_string(&graph) {
                Err(FaultTuneError::Format(msg)) => assert!(msg.contains("q0_0"), "{msg}"),
                r => panic!("expected Format for {bad}, got {r:?}"),
            }
            assert!(db.save(&graph, &path).is_err());
            // The old file is untouched.
            assert_eq!(fs::read(&path).unwrap(), before);
        }
        // A value type for which null is legitimate still works.
        let mut db = ParameterDatabase::<Option<f64>>::new(&graph);
        db.commit_step(
            step_record(0, &[0, 1], &[0, 1], &[], 0.1),
            vec![None, Some(1.5)],
        );
        let text = db.to_json_string(&graph).unwrap();
        let back = ParameterDatabase::<Option<f64>>::from_json_str(&graph, &text).unwrap();
        assert!(db.same_outcome(&back));
    }

    #[test]
    fn values_nested_deeper_than_null_are_refused_when_saving_too() {
        let graph = lattice();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("p.json");
        let mut db = ParameterDatabase::<Vec<f64>>::new(&graph);
        db.commit_step(
            step_record(0, &[0, 1], &[0, 1], &[], 0.1),
            vec![vec![1.0, 2.0], vec![3.0, f64::NAN]],
        );
        match db.save(&graph, &path) {
            Err(FaultTuneError::Format(msg)) => assert!(msg.contains("q0_1"), "{msg}"),
            r => panic!("expected Format, got {r:?}"),
        }
        assert!(!path.exists());
        let mut ok = ParameterDatabase::<Vec<f64>>::new(&graph);
        ok.commit_step(step_record(0, &[0], &[0], &[], 0.1), vec![vec![1.0, 2.0]]);
        ok.save(&graph, &path).unwrap();
        assert!(ParameterDatabase::<Vec<f64>>::load(&graph, &path).is_ok());
    }
}
