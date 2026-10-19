//! The parameter database: learned values plus the ordered calibration history
//! (ARCHITECTURE §3.6).
//!
//! `P*` (the calibration status) is the set of elements with an entry, ordered by the step
//! that set them. The history is append-only, so it doubles as an audit trail: re-calibration
//! adds an [`HistoryRecord::Invalidated`] record instead of rewriting the past.
//!
//! During a calibration pass every subgoal writes only to its own [`Overlay`] on top of a
//! read-only snapshot of the database; overlays are merged afterwards, in subgoal order
//! (ARCHITECTURE §6). Global step indices are assigned at merge time, which is what makes
//! parallel and sequential runs identical.

use std::collections::HashMap;
use std::time::Duration;

use crate::error::FaultTuneError;
use crate::graph::{ElementId, ProcessorGraph};
use crate::sets::ElementSet;

/// The current value of one calibrated element.
#[derive(Clone, Debug, PartialEq)]
pub struct Entry<V> {
    /// The learned value.
    pub value: V,
    /// Its index among the element's options (`model.options(element)[option_index]`).
    pub option_index: u32,
    /// The index of the step that set it (a [`StepRecord::index`]).
    pub step: u64,
}

/// One committed calibration step.
#[derive(Clone, Debug, PartialEq)]
pub struct StepRecord {
    /// Global index in the database's history (0, 1, 2, …). Inside an [`Overlay`], and in
    /// events emitted while a pass runs, it is only the ordinal within the subgoal; the final
    /// value is assigned by [`ParameterDatabase::merge`].
    pub index: u64,
    /// The pass of the `calibrate_*` call that ran the step (0-based).
    pub pass: u32,
    /// The subgoal within the pass (0-based, ordered by smallest member ID).
    pub subgoal: u32,
    /// The element the step was centred on.
    pub centre: ElementId,
    /// The parameters that were optimized, in candidate-column order.
    pub parameters: Vec<ElementId>,
    /// The committed option index of each parameter (aligned with `parameters`).
    pub option_indices: Vec<u32>,
    /// The elements that constrained the step (ascending by ID).
    pub constraints: Vec<ElementId>,
    /// The error of the committed assignment (possibly `+∞` if nothing feasible was found).
    pub error: f64,
    /// Number of distinct candidates evaluated (after within-step de-duplication, before the
    /// cross-step cache; so it does not depend on scheduling).
    pub evaluations: u64,
    /// Time spent in `ErrorModel::prepare`.
    pub prepare_time: Duration,
    /// Time spent evaluating errors.
    pub error_time: Duration,
    /// Total time of the step.
    pub duration: Duration,
}

impl StepRecord {
    /// Whether two records describe the same outcome, ignoring the three timing fields.
    pub fn same_outcome(&self, other: &Self) -> bool {
        self.index == other.index
            && self.pass == other.pass
            && self.subgoal == other.subgoal
            && self.centre == other.centre
            && self.parameters == other.parameters
            && self.option_indices == other.option_indices
            && self.constraints == other.constraints
            && (self.error == other.error || (self.error.is_nan() && other.error.is_nan()))
            && self.evaluations == other.evaluations
    }
}

/// An entry of the append-only history.
#[derive(Clone, Debug, PartialEq)]
pub enum HistoryRecord {
    /// A committed step.
    Step(StepRecord),
    /// Elements removed from `P*` by re-calibration.
    Invalidated(Vec<ElementId>),
}

/// Learned values and the ordered calibration history for one graph.
#[derive(Clone, Debug)]
pub struct ParameterDatabase<V> {
    graph_fingerprint: u64,
    entries: Vec<Option<Entry<V>>>,
    calibrated: ElementSet,
    history: Vec<HistoryRecord>,
    steps: u64,
}

impl<V> ParameterDatabase<V> {
    /// An empty database for `graph`.
    pub fn new(graph: &ProcessorGraph) -> Self {
        let n = graph.len();
        Self {
            graph_fingerprint: graph.fingerprint(),
            entries: (0..n).map(|_| None).collect(),
            calibrated: ElementSet::new(n),
            history: Vec::new(),
            steps: 0,
        }
    }

    /// Rebuilds a database from loaded parts (used by the persistence layer).
    ///
    /// `calibrated` and the step counter are derived from `entries` and `history`; the
    /// caller is responsible for their consistency (see `persist`).
    #[cfg(feature = "serde")]
    pub(crate) fn from_parts(
        graph: &ProcessorGraph,
        entries: Vec<Option<Entry<V>>>,
        history: Vec<HistoryRecord>,
    ) -> Self {
        debug_assert_eq!(entries.len(), graph.len());
        let mut calibrated = ElementSet::new(entries.len());
        for (i, entry) in entries.iter().enumerate() {
            if entry.is_some() {
                calibrated.insert(ElementId::new(i as u32));
            }
        }
        let steps = history
            .iter()
            .filter(|h| matches!(h, HistoryRecord::Step(_)))
            .count() as u64;
        Self {
            graph_fingerprint: graph.fingerprint(),
            entries,
            calibrated,
            history,
            steps,
        }
    }

    /// Fingerprint of the graph this database belongs to.
    pub fn graph_fingerprint(&self) -> u64 {
        self.graph_fingerprint
    }

    /// Fails with [`FaultTuneError::GraphMismatch`] unless `graph` is the graph this database was made for.
    pub fn check_graph(&self, graph: &ProcessorGraph) -> Result<(), FaultTuneError> {
        if graph.fingerprint() == self.graph_fingerprint && graph.len() == self.entries.len() {
            Ok(())
        } else {
            Err(FaultTuneError::GraphMismatch {
                expected: graph.fingerprint(),
                found: self.graph_fingerprint,
            })
        }
    }

    /// Number of elements the database can hold (the graph's element count).
    pub fn capacity(&self) -> usize {
        self.entries.len()
    }

    /// The entry of an element, if calibrated.
    pub fn get(&self, id: ElementId) -> Option<&Entry<V>> {
        self.entries.get(id.index()).and_then(Option::as_ref)
    }

    /// The value of an element, if calibrated.
    pub fn value(&self, id: ElementId) -> Option<&V> {
        self.get(id).map(|e| &e.value)
    }

    /// Whether the element has a value.
    pub fn is_calibrated(&self, id: ElementId) -> bool {
        self.calibrated.contains(id)
    }

    /// `P*`: the set of calibrated elements.
    pub fn calibrated(&self) -> &ElementSet {
        &self.calibrated
    }

    /// Number of calibrated elements.
    pub fn len(&self) -> usize {
        self.calibrated.len()
    }

    /// Whether nothing is calibrated.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Iterates over `(element, entry)` of calibrated elements in ascending ID order.
    pub fn iter(&self) -> impl Iterator<Item = (ElementId, &Entry<V>)> + '_ {
        self.entries
            .iter()
            .enumerate()
            .filter_map(|(i, e)| e.as_ref().map(|e| (ElementId::new(i as u32), e)))
    }

    /// The append-only history.
    pub fn history(&self) -> &[HistoryRecord] {
        &self.history
    }

    /// The committed steps, in order (invalidation records skipped).
    pub fn steps(&self) -> impl Iterator<Item = &StepRecord> + '_ {
        self.history.iter().filter_map(|h| match h {
            HistoryRecord::Step(s) => Some(s),
            HistoryRecord::Invalidated(_) => None,
        })
    }

    /// Number of steps ever committed (the index the next step will get).
    pub fn step_count(&self) -> u64 {
        self.steps
    }
}

impl<V: Clone> ParameterDatabase<V> {
    /// Appends a step: sets `values[i]` (with `record.option_indices[i]`) for
    /// `record.parameters[i]`, overwriting any existing entry, and assigns and returns the next
    /// global step index (`record.index` is ignored).
    ///
    /// # Panics
    /// Panics if the lengths of `parameters`, `option_indices` and `values` differ, or if a
    /// parameter is not an element of the database's graph. Nothing is modified in that case.
    pub fn commit_step(&mut self, mut record: StepRecord, values: Vec<V>) -> u64 {
        assert_eq!(
            record.parameters.len(),
            values.len(),
            "commit_step: {} parameters but {} values",
            record.parameters.len(),
            values.len()
        );
        assert_eq!(
            record.parameters.len(),
            record.option_indices.len(),
            "commit_step: {} parameters but {} option indices",
            record.parameters.len(),
            record.option_indices.len()
        );
        let capacity = self.entries.len();
        for id in &record.parameters {
            assert!(
                id.index() < capacity,
                "commit_step: element {} out of range for a database of {capacity} elements",
                id.index()
            );
        }
        let index = self.steps;
        for ((&id, &option_index), value) in record
            .parameters
            .iter()
            .zip(&record.option_indices)
            .zip(values)
        {
            self.entries[id.index()] = Some(Entry {
                value,
                option_index,
                step: index,
            });
            self.calibrated.insert(id);
        }
        record.index = index;
        self.history.push(HistoryRecord::Step(record));
        self.steps += 1;
        index
    }

    /// Removes the entries of `elements` from `P*` and appends an
    /// [`HistoryRecord::Invalidated`] record (only if at least one was calibrated).
    ///
    /// The record lists the elements actually removed, in ascending order and without
    /// duplicates. Elements that are not calibrated (or not in the graph) are ignored.
    /// The step counter is unchanged.
    pub fn invalidate(&mut self, elements: &[ElementId]) {
        let mut removed = Vec::new();
        for &id in elements {
            if let Some(slot) = self.entries.get_mut(id.index()) {
                if slot.take().is_some() {
                    self.calibrated.remove(id);
                    removed.push(id);
                }
            }
        }
        if !removed.is_empty() {
            removed.sort_unstable();
            self.history.push(HistoryRecord::Invalidated(removed));
        }
    }

    /// Appends all steps of `overlay` in order, assigning global step indices.
    ///
    /// Each step behaves exactly like [`commit_step`](Self::commit_step), so merging the
    /// overlays of one pass in subgoal order gives the same database as committing their
    /// steps directly in that order.
    ///
    /// # Panics
    /// Panics if the overlay contains an element outside this database's graph. Every step is
    /// checked before the first one is applied, so the database is unchanged in that case.
    pub fn merge(&mut self, overlay: Overlay<V>) {
        let capacity = self.entries.len();
        for record in &overlay.records {
            for id in &record.parameters {
                assert!(
                    id.index() < capacity,
                    "merge: element {} out of range for a database of {capacity} elements",
                    id.index()
                );
            }
        }
        for (record, values) in overlay.records.into_iter().zip(overlay.step_values) {
            self.commit_step(record, values);
        }
    }
}

impl<V: PartialEq> ParameterDatabase<V> {
    /// Whether two databases hold the same values and the same history, ignoring timings.
    ///
    /// Compares the capacity, the graph fingerprint, every entry (value, option index and
    /// step) and the histories (steps through [`StepRecord::same_outcome`]).
    pub fn same_outcome(&self, other: &Self) -> bool {
        self.entries.len() == other.entries.len()
            && self.graph_fingerprint == other.graph_fingerprint
            && self.entries == other.entries
            && self.history.len() == other.history.len()
            && self
                .history
                .iter()
                .zip(&other.history)
                .all(|pair| match pair {
                    (HistoryRecord::Step(a), HistoryRecord::Step(b)) => a.same_outcome(b),
                    (HistoryRecord::Invalidated(a), HistoryRecord::Invalidated(b)) => a == b,
                    _ => false,
                })
    }
}

/// The writes of one subgoal during a pass, layered over a read-only snapshot.
///
/// Reads go through the overlay first and then the snapshot; the overlay's own
/// [`calibrated`](Overlay::calibrated) set already includes the snapshot's, and is updated
/// as steps are committed.
#[derive(Clone, Debug)]
pub struct Overlay<V> {
    calibrated: ElementSet,
    values: HashMap<ElementId, (V, u32)>,
    records: Vec<StepRecord>,
    /// `step_values[i][j]` is the value committed for `records[i].parameters[j]`.
    step_values: Vec<Vec<V>>,
}

impl<V: Clone> Overlay<V> {
    /// An empty overlay over `snapshot`.
    ///
    /// Only the calibrated set is copied (`O(|G|/64)` words); no values are.
    pub fn new(snapshot: &ParameterDatabase<V>) -> Self {
        Self {
            calibrated: snapshot.calibrated.clone(),
            values: HashMap::new(),
            records: Vec::new(),
            step_values: Vec::new(),
        }
    }

    /// `P*` as seen through the overlay: the snapshot's calibrated set plus this overlay's writes.
    pub fn calibrated(&self) -> &ElementSet {
        &self.calibrated
    }

    /// Whether `id` is calibrated in the overlay or the snapshot.
    pub fn is_calibrated(&self, id: ElementId) -> bool {
        self.calibrated.contains(id)
    }

    /// The value and option index of `id`, looking at the overlay first and then `snapshot`.
    pub fn value<'a>(
        &'a self,
        snapshot: &'a ParameterDatabase<V>,
        id: ElementId,
    ) -> Option<(&'a V, u32)> {
        match self.values.get(&id) {
            Some((value, option_index)) => Some((value, *option_index)),
            None => snapshot.get(id).map(|e| (&e.value, e.option_index)),
        }
    }

    /// Records a step: `values[i]` for `record.parameters[i]`. `record.index` is replaced by
    /// the step's ordinal within the overlay (0, 1, …). Existing values are overwritten.
    ///
    /// # Panics
    /// Panics if the lengths of `parameters`, `option_indices` and `values` differ, or if a
    /// parameter is outside the overlay's capacity. Nothing is modified in that case.
    pub fn commit(&mut self, mut record: StepRecord, values: Vec<V>) {
        assert_eq!(
            record.parameters.len(),
            values.len(),
            "Overlay::commit: {} parameters but {} values",
            record.parameters.len(),
            values.len()
        );
        assert_eq!(
            record.parameters.len(),
            record.option_indices.len(),
            "Overlay::commit: {} parameters but {} option indices",
            record.parameters.len(),
            record.option_indices.len()
        );
        let capacity = self.calibrated.capacity();
        for id in &record.parameters {
            assert!(
                id.index() < capacity,
                "Overlay::commit: element {} out of range for {capacity} elements",
                id.index()
            );
        }
        for ((&id, &option_index), value) in record
            .parameters
            .iter()
            .zip(&record.option_indices)
            .zip(&values)
        {
            self.values.insert(id, (value.clone(), option_index));
            self.calibrated.insert(id);
        }
        record.index = self.records.len() as u64;
        self.records.push(record);
        self.step_values.push(values);
    }

    /// The steps committed to this overlay, in order.
    pub fn steps(&self) -> &[StepRecord] {
        &self.records
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use std::panic::{catch_unwind, AssertUnwindSafe};

    fn id(i: u32) -> ElementId {
        ElementId::new(i)
    }

    fn ids(v: &[u32]) -> Vec<ElementId> {
        v.iter().map(|&i| id(i)).collect()
    }

    /// A record with the given parameters and option indices (everything else fixed).
    fn rec(centre: u32, params: &[u32], opts: &[u32]) -> StepRecord {
        StepRecord {
            index: 9999,
            pass: 1,
            subgoal: 2,
            centre: id(centre),
            parameters: ids(params),
            option_indices: opts.to_vec(),
            constraints: ids(&[0]),
            error: 0.5,
            evaluations: 7,
            prepare_time: Duration::from_millis(1),
            error_time: Duration::from_millis(2),
            duration: Duration::from_millis(3),
        }
    }

    fn lattice() -> ProcessorGraph {
        ProcessorGraph::square_lattice(2, 3)
    }

    fn commit(db: &mut ParameterDatabase<i64>, params: &[u32], opts: &[u32], vals: &[i64]) -> u64 {
        db.commit_step(rec(params[0], params, opts), vals.to_vec())
    }

    #[test]
    fn new_database_is_empty() {
        let graph = lattice();
        let db = ParameterDatabase::<i64>::new(&graph);
        assert_eq!(db.capacity(), graph.len());
        assert_eq!(db.graph_fingerprint(), graph.fingerprint());
        assert!(db.is_empty());
        assert_eq!(db.len(), 0);
        assert_eq!(db.step_count(), 0);
        assert!(db.history().is_empty());
        assert_eq!(db.steps().count(), 0);
        assert_eq!(db.iter().count(), 0);
        assert!(db.calibrated().is_empty());
        assert_eq!(db.calibrated().capacity(), graph.len());
        for i in graph.ids() {
            assert!(db.get(i).is_none());
            assert!(db.value(i).is_none());
            assert!(!db.is_calibrated(i));
        }
        // Out-of-range IDs are simply absent.
        assert!(db.get(id(10_000)).is_none());
        assert!(!db.is_calibrated(id(10_000)));
    }

    #[test]
    fn tiny_graphs() {
        for (rows, cols) in [(0, 0), (1, 1), (1, 2), (1, 5), (5, 1)] {
            let graph = ProcessorGraph::square_lattice(rows, cols);
            let mut db = ParameterDatabase::<i64>::new(&graph);
            assert_eq!(db.capacity(), graph.len());
            assert!(db.check_graph(&graph).is_ok());
            let overlay = Overlay::new(&db);
            assert!(overlay.calibrated().is_empty());
            db.invalidate(&[]);
            assert!(db.history().is_empty());
            if !graph.is_empty() {
                let last = graph.len() as u32 - 1;
                let step = commit(&mut db, &[0, last], &[3, 4], &[10, 20]);
                assert_eq!(step, 0);
                assert_eq!(db.len(), if last == 0 { 1 } else { 2 });
                db.invalidate(&[id(last)]);
                assert!(!db.is_calibrated(id(last)));
            }
        }
    }

    #[test]
    fn check_graph_accepts_same_and_rejects_other() {
        let graph = lattice();
        let db = ParameterDatabase::<i64>::new(&graph);
        assert!(db.check_graph(&graph).is_ok());
        assert!(db.check_graph(&lattice()).is_ok());
        let other = ProcessorGraph::square_lattice(3, 2);
        match db.check_graph(&other) {
            Err(FaultTuneError::GraphMismatch { expected, found }) => {
                assert_eq!(expected, other.fingerprint());
                assert_eq!(found, graph.fingerprint());
                assert_ne!(expected, found);
            }
            r => panic!("expected GraphMismatch, got {r:?}"),
        }
    }

    #[test]
    fn commit_step_sets_entries_and_history() {
        let graph = lattice();
        let mut db = ParameterDatabase::<i64>::new(&graph);
        let i0 = commit(&mut db, &[0, 6], &[1, 2], &[100, 200]);
        let i1 = commit(&mut db, &[1], &[5], &[300]);
        assert_eq!((i0, i1), (0, 1));
        assert_eq!(db.step_count(), 2);
        assert_eq!(db.len(), 3);
        assert_eq!(
            db.get(id(0)),
            Some(&Entry {
                value: 100,
                option_index: 1,
                step: 0
            })
        );
        assert_eq!(db.get(id(6)).unwrap().value, 200);
        assert_eq!(db.get(id(6)).unwrap().option_index, 2);
        assert_eq!(db.value(id(1)), Some(&300));
        assert_eq!(db.get(id(1)).unwrap().step, 1);
        assert!(db.calibrated().contains(id(6)));
        let order: Vec<u32> = db.iter().map(|(e, _)| e.raw()).collect();
        assert_eq!(order, vec![0, 1, 6]);
        let steps: Vec<&StepRecord> = db.steps().collect();
        assert_eq!(steps.len(), 2);
        // The caller's index (9999) was replaced by the global one.
        assert_eq!(steps[0].index, 0);
        assert_eq!(steps[1].index, 1);
        assert_eq!(steps[0].parameters, ids(&[0, 6]));
        assert_eq!(steps[0].option_indices, vec![1, 2]);
        assert_eq!(steps[0].error, 0.5);
        assert_eq!(steps[0].duration, Duration::from_millis(3));
    }

    #[test]
    fn commit_step_overwrites() {
        let graph = lattice();
        let mut db = ParameterDatabase::<i64>::new(&graph);
        commit(&mut db, &[0, 1], &[1, 1], &[10, 11]);
        commit(&mut db, &[1, 2], &[2, 2], &[21, 22]);
        assert_eq!(db.len(), 3);
        assert_eq!(db.value(id(0)), Some(&10));
        assert_eq!(db.get(id(0)).unwrap().step, 0);
        let e = db.get(id(1)).unwrap();
        assert_eq!((e.value, e.option_index, e.step), (21, 2, 1));
        // History keeps both steps: it is append-only.
        assert_eq!(db.steps().count(), 2);
    }

    #[test]
    fn commit_step_with_no_parameters_still_counts() {
        let graph = lattice();
        let mut db = ParameterDatabase::<i64>::new(&graph);
        assert_eq!(db.commit_step(rec(0, &[], &[]), vec![]), 0);
        assert_eq!(db.step_count(), 1);
        assert!(db.is_empty());
    }

    #[test]
    fn commit_step_panics_on_bad_input_and_leaves_database_unchanged() {
        let graph = lattice();
        let mut db = ParameterDatabase::<i64>::new(&graph);
        commit(&mut db, &[0], &[0], &[1]);
        let before = db.clone();
        let n = graph.len() as u32;
        let bad: Vec<(StepRecord, Vec<i64>)> = vec![
            (rec(0, &[1, 2], &[0, 0]), vec![1]),
            (rec(0, &[1], &[0]), vec![1, 2]),
            (rec(0, &[1, 2], &[0]), vec![1, 2]),
            (rec(0, &[1], &[0, 1]), vec![1]),
            (rec(0, &[1, n], &[0, 0]), vec![1, 2]),
            (rec(0, &[n + 100], &[0]), vec![1]),
        ];
        for (record, values) in bad {
            let result = catch_unwind(AssertUnwindSafe(|| db.commit_step(record, values)));
            assert!(result.is_err());
            assert!(
                db.same_outcome(&before),
                "a failed commit changed the database"
            );
            assert_eq!(db.step_count(), before.step_count());
        }
    }

    #[test]
    #[should_panic(expected = "parameters but")]
    fn commit_step_length_mismatch_panics() {
        let graph = lattice();
        let mut db = ParameterDatabase::<i64>::new(&graph);
        db.commit_step(rec(0, &[0, 1], &[0, 0]), vec![1]);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn commit_step_out_of_range_panics() {
        let graph = lattice();
        let mut db = ParameterDatabase::<i64>::new(&graph);
        db.commit_step(rec(0, &[graph.len() as u32], &[0]), vec![1]);
    }

    #[test]
    fn invalidate_removes_and_records() {
        let graph = lattice();
        let mut db = ParameterDatabase::<i64>::new(&graph);
        commit(&mut db, &[0, 1, 2, 3], &[0, 0, 0, 0], &[1, 2, 3, 4]);
        // Unsorted, with a duplicate, an uncalibrated element and an out-of-range ID.
        db.invalidate(&ids(&[3, 1, 3, 5, 9_999]));
        assert_eq!(db.len(), 2);
        assert!(db.is_calibrated(id(0)) && db.is_calibrated(id(2)));
        assert!(!db.is_calibrated(id(1)) && !db.is_calibrated(id(3)));
        assert!(db.get(id(1)).is_none());
        assert_eq!(db.step_count(), 1);
        assert_eq!(db.history().len(), 2);
        assert_eq!(db.history()[1], HistoryRecord::Invalidated(ids(&[1, 3])));
        // `steps()` skips the invalidation record.
        assert_eq!(db.steps().count(), 1);
        // The next step continues the step numbering.
        assert_eq!(commit(&mut db, &[1], &[0], &[7]), 1);
        assert_eq!(db.get(id(1)).unwrap().step, 1);
    }

    #[test]
    fn invalidate_nothing_adds_no_record() {
        let graph = lattice();
        let mut db = ParameterDatabase::<i64>::new(&graph);
        db.invalidate(&ids(&[0, 1]));
        db.invalidate(&[]);
        assert!(db.history().is_empty());
        commit(&mut db, &[0], &[0], &[1]);
        db.invalidate(&ids(&[1, 2]));
        assert_eq!(db.history().len(), 1);
        db.invalidate(&ids(&[0]));
        db.invalidate(&ids(&[0]));
        assert_eq!(db.history().len(), 2, "removing twice records once");
    }

    #[test]
    fn overlay_new_copies_calibrated_set_but_no_values() {
        let graph = lattice();
        let mut db = ParameterDatabase::<i64>::new(&graph);
        commit(&mut db, &[0, 1], &[4, 5], &[10, 11]);
        let overlay = Overlay::new(&db);
        assert_eq!(overlay.calibrated(), db.calibrated());
        assert!(overlay.is_calibrated(id(0)));
        assert!(!overlay.is_calibrated(id(2)));
        assert!(overlay.values.is_empty());
        assert!(overlay.steps().is_empty());
        // Reads fall through to the snapshot.
        assert_eq!(overlay.value(&db, id(1)), Some((&11, 5)));
        assert_eq!(overlay.value(&db, id(2)), None);
    }

    #[test]
    fn overlay_reads_own_writes_first() {
        let graph = lattice();
        let mut db = ParameterDatabase::<i64>::new(&graph);
        commit(&mut db, &[0, 1], &[4, 5], &[10, 11]);
        let mut overlay = Overlay::new(&db);
        overlay.commit(rec(0, &[1, 2], &[8, 9]), vec![111, 222]);
        assert_eq!(overlay.value(&db, id(0)), Some((&10, 4)));
        assert_eq!(overlay.value(&db, id(1)), Some((&111, 8)));
        assert_eq!(overlay.value(&db, id(2)), Some((&222, 9)));
        assert!(overlay.is_calibrated(id(2)));
        assert!(overlay.calibrated().contains(id(0)));
        // The snapshot is untouched.
        assert_eq!(db.value(id(1)), Some(&11));
        assert!(!db.is_calibrated(id(2)));
        // Overwrite inside the overlay.
        overlay.commit(rec(2, &[2], &[1]), vec![333]);
        assert_eq!(overlay.value(&db, id(2)), Some((&333, 1)));
    }

    #[test]
    fn overlay_commit_numbers_steps_by_ordinal() {
        let graph = lattice();
        let db = ParameterDatabase::<i64>::new(&graph);
        let mut overlay = Overlay::new(&db);
        for k in 0..4u32 {
            overlay.commit(rec(k, &[k], &[k]), vec![k as i64]);
        }
        let indices: Vec<u64> = overlay.steps().iter().map(|s| s.index).collect();
        assert_eq!(indices, vec![0, 1, 2, 3]);
        assert_eq!(overlay.steps()[2].centre, id(2));
    }

    #[test]
    fn overlay_commit_panics_on_bad_input() {
        let graph = lattice();
        let db = ParameterDatabase::<i64>::new(&graph);
        let mut overlay = Overlay::new(&db);
        let n = graph.len() as u32;
        let bad: Vec<(StepRecord, Vec<i64>)> = vec![
            (rec(0, &[1, 2], &[0, 0]), vec![1]),
            (rec(0, &[1], &[0, 0]), vec![1]),
            (rec(0, &[1, n], &[0, 0]), vec![1, 2]),
        ];
        for (record, values) in bad {
            let result = catch_unwind(AssertUnwindSafe(|| overlay.commit(record, values)));
            assert!(result.is_err());
            assert!(overlay.steps().is_empty());
            assert!(overlay.calibrated().is_empty());
            assert!(overlay.values.is_empty());
        }
    }

    #[test]
    fn merge_assigns_global_indices() {
        let graph = lattice();
        let mut db = ParameterDatabase::<i64>::new(&graph);
        commit(&mut db, &[0], &[0], &[1]);
        commit(&mut db, &[1], &[0], &[2]);
        let mut a = Overlay::new(&db);
        a.commit(rec(2, &[2, 3], &[1, 1]), vec![20, 30]);
        a.commit(rec(2, &[2], &[2]), vec![21]);
        let mut b = Overlay::new(&db);
        b.commit(rec(4, &[4], &[3]), vec![40]);
        db.merge(a);
        db.merge(b);
        assert_eq!(db.step_count(), 5);
        let indices: Vec<u64> = db.steps().map(|s| s.index).collect();
        assert_eq!(indices, vec![0, 1, 2, 3, 4]);
        assert_eq!(db.get(id(2)).unwrap().step, 3, "final write wins");
        assert_eq!(db.value(id(2)), Some(&21));
        assert_eq!(db.get(id(3)).unwrap().step, 2);
        assert_eq!(db.get(id(4)).unwrap().step, 4);
        assert_eq!(db.get(id(4)).unwrap().option_index, 3);
        assert_eq!(db.len(), 5);
    }

    #[test]
    fn merge_of_empty_overlay_is_a_no_op() {
        let graph = lattice();
        let mut db = ParameterDatabase::<i64>::new(&graph);
        commit(&mut db, &[0], &[0], &[1]);
        let before = db.clone();
        let overlay = Overlay::new(&db);
        db.merge(overlay);
        assert!(db.same_outcome(&before));
    }

    #[test]
    fn merge_of_a_foreign_overlay_panics_without_changing_the_database() {
        let big = ProcessorGraph::square_lattice(4, 4);
        let small = ProcessorGraph::square_lattice(1, 2);
        let source = ParameterDatabase::<i64>::new(&big);
        let mut overlay = Overlay::new(&source);
        overlay.commit(rec(0, &[0], &[1]), vec![10]);
        overlay.commit(rec(40, &[40], &[2]), vec![20]);
        let mut db = ParameterDatabase::<i64>::new(&small);
        commit(&mut db, &[1], &[5], &[50]);
        let before = db.clone();
        let outcome = catch_unwind(AssertUnwindSafe(|| db.merge(overlay)));
        assert!(outcome.is_err());
        assert!(
            db.same_outcome(&before),
            "a failed merge must not be partial"
        );
        assert_eq!(db.step_count(), 1);
        // The database is still usable.
        assert_eq!(commit(&mut db, &[0], &[0], &[1]), 1);
    }

    /// A step as plain data: parameters with (option, value) each.
    type Spec = Vec<(u32, u32, i64)>;

    fn make_record(centre: u32, spec: &Spec) -> (StepRecord, Vec<i64>) {
        let params: Vec<u32> = spec.iter().map(|s| s.0).collect();
        let opts: Vec<u32> = spec.iter().map(|s| s.1).collect();
        let vals: Vec<i64> = spec.iter().map(|s| s.2).collect();
        (rec(centre, &params, &opts), vals)
    }

    fn spec_strategy(n: u32) -> impl Strategy<Value = Spec> {
        prop::collection::vec((0..n, 0u32..50, -1000i64..1000), 0..5)
    }

    proptest! {
        #[test]
        fn merging_overlays_equals_direct_commits(
            initial in prop::collection::vec(spec_strategy(20), 0..6),
            invalidated in prop::collection::vec(0u32..20, 0..6),
            subgoals in prop::collection::vec(
                prop::collection::vec(spec_strategy(20), 0..5), 0..5),
        ) {
            // 4x4 lattice has 16 nodes + 24 engineered + 18 parasitic = 58 elements >= 20.
            let graph = ProcessorGraph::square_lattice(4, 4);
            let mut snapshot = ParameterDatabase::<i64>::new(&graph);
            for spec in &initial {
                let (r, v) = make_record(0, spec);
                snapshot.commit_step(r, v);
            }
            snapshot.invalidate(&ids(&invalidated));
            let frozen = snapshot.clone();

            // Overlays, all built from the same snapshot, merged in subgoal order.
            let mut overlays = Vec::new();
            for (k, steps) in subgoals.iter().enumerate() {
                let mut overlay = Overlay::new(&snapshot);
                for (j, spec) in steps.iter().enumerate() {
                    let (mut r, v) = make_record(k as u32, spec);
                    r.subgoal = k as u32;
                    r.pass = j as u32;
                    overlay.commit(r, v);
                }
                // Building and filling an overlay never alters the snapshot.
                prop_assert!(snapshot.same_outcome(&frozen));
                // The overlay sees its own writes.
                for spec in steps {
                    for &(e, _, _) in spec {
                        prop_assert!(overlay.is_calibrated(id(e)));
                        prop_assert!(overlay.value(&snapshot, id(e)).is_some());
                    }
                }
                overlays.push(overlay);
            }
            let mut merged = snapshot.clone();
            for overlay in overlays {
                merged.merge(overlay);
            }

            let mut direct = snapshot.clone();
            for (k, steps) in subgoals.iter().enumerate() {
                for (j, spec) in steps.iter().enumerate() {
                    let (mut r, v) = make_record(k as u32, spec);
                    r.subgoal = k as u32;
                    r.pass = j as u32;
                    direct.commit_step(r, v);
                }
            }
            prop_assert!(merged.same_outcome(&direct));
            prop_assert_eq!(merged.step_count(), direct.step_count());
            prop_assert_eq!(merged.calibrated(), direct.calibrated());
            // Global indices are 0, 1, 2, ... counting only steps.
            let indices: Vec<u64> = merged.steps().map(|s| s.index).collect();
            let expected: Vec<u64> = (0..merged.step_count()).collect();
            prop_assert_eq!(indices, expected);
        }

        #[test]
        fn calibrated_set_matches_entries(
            ops in prop::collection::vec(
                prop_oneof![
                    spec_strategy(12).prop_map(|s| (true, s)),
                    spec_strategy(12).prop_map(|s| (false, s)),
                ], 0..30),
        ) {
            let graph = ProcessorGraph::square_lattice(3, 3);
            let mut db = ParameterDatabase::<i64>::new(&graph);
            for (is_commit, spec) in &ops {
                if *is_commit {
                    let (r, v) = make_record(0, spec);
                    db.commit_step(r, v);
                } else {
                    let elements: Vec<ElementId> = spec.iter().map(|s| id(s.0)).collect();
                    db.invalidate(&elements);
                }
            }
            let from_entries: Vec<ElementId> = db.iter().map(|(e, _)| e).collect();
            prop_assert_eq!(db.calibrated().to_vec(), from_entries.clone());
            prop_assert_eq!(db.len(), from_entries.len());
            for (_, entry) in db.iter() {
                prop_assert!(entry.step < db.step_count());
            }
            // Each entry's step really wrote that element with that option.
            for (e, entry) in db.iter() {
                let step = db.steps().nth(entry.step as usize).unwrap();
                let pos = step.parameters.iter().rposition(|&p| p == e).unwrap();
                prop_assert_eq!(step.option_indices[pos], entry.option_index);
            }
        }
    }

    #[test]
    fn same_outcome_compares_values_options_steps_and_history() {
        let graph = lattice();
        let build = || {
            let mut db = ParameterDatabase::<i64>::new(&graph);
            commit(&mut db, &[0, 1], &[1, 2], &[10, 20]);
            commit(&mut db, &[2], &[3], &[30]);
            db.invalidate(&ids(&[1]));
            db
        };
        let a = build();
        assert!(a.same_outcome(&a));
        assert!(a.same_outcome(&build()));

        // Timings are ignored.
        let mut b = build();
        if let HistoryRecord::Step(s) = &mut b.history[0] {
            s.duration = Duration::from_secs(99);
            s.prepare_time = Duration::from_secs(1);
            s.error_time = Duration::from_secs(2);
        }
        assert!(a.same_outcome(&b));

        // Value, option index, step of an entry.
        let mut c = build();
        c.entries[0].as_mut().unwrap().value = 11;
        assert!(!a.same_outcome(&c));
        let mut c = build();
        c.entries[0].as_mut().unwrap().option_index = 9;
        assert!(!a.same_outcome(&c));
        let mut c = build();
        c.entries[0].as_mut().unwrap().step = 1;
        assert!(!a.same_outcome(&c));
        // Missing entry.
        let mut c = build();
        c.entries[2] = None;
        assert!(!a.same_outcome(&c));

        // History differences.
        let mut c = build();
        c.history.pop();
        assert!(!a.same_outcome(&c));
        let mut c = build();
        c.history[2] = HistoryRecord::Invalidated(ids(&[2]));
        assert!(!a.same_outcome(&c));
        let mut c = build();
        if let HistoryRecord::Step(s) = &mut c.history[1] {
            s.error = 0.25;
        }
        assert!(!a.same_outcome(&c));
        let mut c = build();
        c.history[2] = HistoryRecord::Step(rec(0, &[], &[]));
        assert!(!a.same_outcome(&c));

        // Different graph of the same size: fingerprint differs.
        let other = ParameterDatabase::<i64>::new(&ProcessorGraph::square_lattice(3, 2));
        let empty = ParameterDatabase::<i64>::new(&graph);
        assert_eq!(other.capacity(), empty.capacity());
        assert!(!other.same_outcome(&empty));
    }

    #[test]
    fn step_record_same_outcome_handles_nan() {
        let mut a = rec(0, &[1], &[1]);
        let mut b = a.clone();
        a.error = f64::NAN;
        b.error = f64::NAN;
        assert!(a.same_outcome(&b));
        b.error = f64::INFINITY;
        assert!(!a.same_outcome(&b));
        a.error = f64::INFINITY;
        assert!(a.same_outcome(&b));
    }
}
