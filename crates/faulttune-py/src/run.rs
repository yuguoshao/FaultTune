//! Running a calibration from Python (ARCHITECTURE §9.1).
//!
//! The engine runs on its own Rust thread without the GIL. The calling thread runs an *event
//! pump*: it sleeps (GIL released) until the engine commits a step or ~100 ms have passed,
//! checks for Ctrl-C, and calls `on_step` for committed steps, all on the calling thread.

use std::collections::VecDeque;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::time::Duration;

use faulttune::{
    CancellationToken, CheckpointObserver, ElementId, Event, FaultTuneError, Observer,
    ParameterDatabase, Report, Snake, StepRecord, StitchPlan,
};
use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::prelude::*;
use pyo3::PyErr;

use crate::database::PyDatabase;
use crate::errors::{self, FailureSlot};
use crate::graph::PyGraph;
use crate::model::PyErrorModel;
use crate::records::{OptionTables, PyReport, PyStepRecord};

/// How often the pump wakes up when nothing happens (to check for Ctrl-C).
const POLL: Duration = Duration::from_millis(100);

/// Stack size of the engine thread: user callbacks (NumPy models) run on it.
const ENGINE_STACK: usize = 16 * 1024 * 1024;

/// What the engine thread hands back when it is done.
pub struct Outcome {
    db: ParameterDatabase<f64>,
    result: Result<Report, FaultTuneError>,
    panic: Option<String>,
}

enum Msg {
    Step(StepRecord),
    Done(Box<Outcome>),
}

/// A queue from the engine thread to the pump.
pub struct Pump {
    queue: Mutex<VecDeque<Msg>>,
    wake: Condvar,
    wants_steps: bool,
}

impl Pump {
    fn new(wants_steps: bool) -> Self {
        Self {
            queue: Mutex::new(VecDeque::new()),
            wake: Condvar::new(),
            wants_steps,
        }
    }

    fn push(&self, msg: Msg) {
        self.queue
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push_back(msg);
        self.wake.notify_one();
    }

    /// Waits up to `timeout` for messages and takes all that are queued.
    fn wait(&self, timeout: Duration) -> Vec<Msg> {
        let mut queue = self.queue.lock().unwrap_or_else(PoisonError::into_inner);
        if queue.is_empty() {
            queue = self
                .wake
                .wait_timeout(queue, timeout)
                .unwrap_or_else(PoisonError::into_inner)
                .0;
        }
        queue.drain(..).collect()
    }
}

/// The observer installed in every Snake: forwards events to whatever the current call set up.
#[derive(Default)]
pub struct Dispatch {
    pump: Mutex<Option<Arc<Pump>>>,
    checkpoint: Mutex<Option<Arc<CheckpointObserver>>>,
}

impl Dispatch {
    fn install(&self, pump: Arc<Pump>, checkpoint: Option<Arc<CheckpointObserver>>) {
        *self.pump.lock().unwrap_or_else(PoisonError::into_inner) = Some(pump);
        *self
            .checkpoint
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = checkpoint;
    }

    fn clear(&self) {
        *self.pump.lock().unwrap_or_else(PoisonError::into_inner) = None;
        *self
            .checkpoint
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = None;
    }
}

impl Observer for Dispatch {
    fn on_event(&self, event: &Event) {
        if let Event::StepCommitted(record) = event {
            let pump = self
                .pump
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .clone();
            if let Some(pump) = pump.filter(|p| p.wants_steps) {
                pump.push(Msg::Step(record.clone()));
            }
        }
        let checkpoint = self
            .checkpoint
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        if let Some(checkpoint) = checkpoint {
            checkpoint.on_event(event);
        }
    }
}

/// What a call does.
pub enum Job {
    /// `calibrate`.
    Calibrate,
    /// `recalibrate` the neighbourhoods of these elements.
    Recalibrate(Vec<ElementId>),
    /// `stitch` a plan.
    Stitch(StitchPlan),
}

/// The arguments of one call.
pub struct Request<'a, 'py> {
    pub job: Job,
    pub database: Option<&'a Bound<'py, PyDatabase>>,
    pub model: &'a Bound<'py, PyAny>,
    pub on_step: Option<&'a Bound<'py, PyAny>>,
    pub checkpoint: Option<PathBuf>,
}

/// Everything a `Snake` needs to run calibrations from Python.
pub struct Engine {
    pub graph: Py<PyGraph>,
    pub snake: Arc<Snake<'static>>,
    pub dispatch: Arc<Dispatch>,
    pub token: CancellationToken,
    pub failures: Arc<FailureSlot>,
    /// Retries of a failing `options()` call (from `on_model_error`).
    option_retries: u32,
    running: AtomicBool,
    pub last_report: Mutex<Option<Py<PyReport>>>,
}

/// Marks the engine as running for the duration of a call.
struct RunGuard<'a>(&'a AtomicBool);

impl Drop for RunGuard<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::SeqCst);
    }
}

fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|s| (*s).to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "a panic with no message".into())
}

/// Body of the engine thread.
fn engine_main(
    snake: Arc<Snake<'static>>,
    model: Arc<PyErrorModel>,
    mut db: ParameterDatabase<f64>,
    job: Job,
    pump: Arc<Pump>,
) {
    let run = catch_unwind(AssertUnwindSafe(|| match &job {
        Job::Calibrate => snake.calibrate_into(&mut db, &*model),
        Job::Recalibrate(elements) => snake.recalibrate(&mut db, elements, &*model),
        Job::Stitch(plan) => snake.stitch(&mut db, plan, &*model),
    }));
    let outcome = match run {
        Ok(result) => Outcome {
            db,
            result,
            panic: None,
        },
        Err(payload) => Outcome {
            db,
            result: Err(FaultTuneError::NoProgress),
            panic: Some(panic_message(&*payload)),
        },
    };
    // Release the model here so that its Python references are dropped with the thread.
    drop(model);
    pump.push(Msg::Done(Box::new(outcome)));
}

/// Warns that a run that starts from an empty database appends to a journal that already
/// holds entries: a replay of the journal would then mix the two runs.
fn warn_journal_is_used(py: Python<'_>, path: &std::path::Path) -> PyResult<()> {
    let message = format!(
        "the checkpoint journal {} already holds entries and this run starts from an empty \
         database: the new run is appended to it, and replaying the journal would mix both \
         runs (use a new file, or pass the database that the journal belongs to)",
        path.display()
    );
    let message = std::ffi::CString::new(message).unwrap_or_default();
    PyErr::warn(
        py,
        &py.get_type::<pyo3::exceptions::PyRuntimeWarning>(),
        &message,
        1,
    )
}

/// Warns that writing the checkpoint journal failed.
fn warn_journal_failed(py: Python<'_>, io: &std::io::Error) -> PyResult<()> {
    let message = format!(
        "writing the checkpoint journal failed ({io}); the steps committed since are not in \
         the journal"
    );
    let message = std::ffi::CString::new(message).unwrap_or_default();
    PyErr::warn(
        py,
        &py.get_type::<pyo3::exceptions::PyRuntimeWarning>(),
        &message,
        1,
    )
}

/// Best effort: `exc.partial = database`.
fn attach_partial(py: Python<'_>, err: &PyErr, database: &Py<PyDatabase>) {
    let _ = err.value(py).setattr("partial", database.clone_ref(py));
}

impl Engine {
    /// Creates the engine state around a built Snake.
    pub fn new(
        graph: Py<PyGraph>,
        snake: Arc<Snake<'static>>,
        dispatch: Arc<Dispatch>,
        token: CancellationToken,
        failures: Arc<FailureSlot>,
        option_retries: u32,
    ) -> Self {
        Self {
            graph,
            snake,
            dispatch,
            token,
            failures,
            option_retries,
            running: AtomicBool::new(false),
            last_report: Mutex::new(None),
        }
    }

    /// Requests cancellation of the running call, if any.
    pub fn cancel(&self) {
        if self.running.load(Ordering::SeqCst) {
            self.token.cancel();
        }
    }

    /// Runs one call: starts the engine thread, pumps events, and maps the outcome to a
    /// result or an exception. Committed steps are never lost: they are in the caller's
    /// database, or attached to the exception as `.partial`.
    pub fn run(
        &self,
        py: Python<'_>,
        request: Request<'_, '_>,
    ) -> PyResult<(Py<PyDatabase>, Py<PyReport>)> {
        let graph = Arc::clone(&self.graph.get().inner);
        let Request {
            job,
            database,
            model,
            on_step,
            checkpoint,
        } = request;
        if let Some(callback) = on_step {
            if !callback.is_callable() {
                return Err(PyTypeError::new_err(format!(
                    "on_step must be callable, not {}",
                    callback.get_type().name()?
                )));
            }
        }
        if self
            .running
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return Err(PyRuntimeError::new_err(
                "this Snake is already running a calibration; wait for it or use another Snake",
            ));
        }
        let _running = RunGuard(&self.running);
        self.token.reset();
        *self
            .last_report
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = None;
        let _ = self.failures.take();

        let tables = Arc::new(OptionTables::new(graph.len()));
        let model = Arc::new(PyErrorModel::new(
            model,
            self.graph.clone_ref(py),
            Arc::clone(&tables),
            Arc::clone(&self.failures),
            self.option_retries,
        )?);
        let given: Option<Py<PyDatabase>> = database.map(|d| d.clone().unbind());
        if let Some(d) = &given {
            d.get().ensure_graph(&self.graph)?;
        }
        let mut used_journal = false;
        if let Some(path) = &checkpoint {
            let fresh = match &given {
                Some(d) => d.get().step_count_now()? == 0,
                None => true,
            };
            used_journal =
                fresh && std::fs::metadata(path).is_ok_and(|m| m.is_file() && m.len() > 0);
        }
        let journal_path = checkpoint.clone();
        let checkpoint = match checkpoint {
            Some(path) => Some(Arc::new(
                CheckpointObserver::append(&path, &graph).map_err(|e| errors::plain(py, e))?,
            )),
            None => None,
        };
        if used_journal {
            if let Some(path) = &journal_path {
                warn_journal_is_used(py, path)?;
            }
        }
        let db = match &given {
            Some(d) => d.get().lend()?,
            None => ParameterDatabase::new(&graph),
        };

        let pump = Arc::new(Pump::new(on_step.is_some()));
        self.dispatch.install(Arc::clone(&pump), checkpoint.clone());
        // The database travels in a slot so that it can be recovered if no thread starts.
        let slot = Arc::new(Mutex::new(Some(db)));
        let spawned = std::thread::Builder::new()
            .name("faulttune-engine".into())
            .stack_size(ENGINE_STACK)
            .spawn({
                let snake = Arc::clone(&self.snake);
                let pump = Arc::clone(&pump);
                let model = Arc::clone(&model);
                let slot = Arc::clone(&slot);
                move || {
                    let taken = slot.lock().unwrap_or_else(PoisonError::into_inner).take();
                    if let Some(db) = taken {
                        engine_main(snake, model, db, job, pump);
                    }
                }
            });
        drop(model);
        let handle = match spawned {
            Ok(handle) => handle,
            Err(err) => {
                self.dispatch.clear();
                let recovered = slot.lock().unwrap_or_else(PoisonError::into_inner).take();
                if let (Some(d), Some(db)) = (&given, recovered) {
                    d.get().restore(db);
                }
                return Err(PyRuntimeError::new_err(format!(
                    "could not start the engine thread: {err}"
                )));
            }
        };

        // The event pump.
        let mut outcome: Option<Box<Outcome>> = None;
        let mut signal: Option<PyErr> = None;
        let mut callback_error: Option<PyErr> = None;
        while outcome.is_none() {
            let messages = py.detach(|| pump.wait(POLL));
            // A journal that cannot be written is reported as soon as it is noticed, not only
            // when the run is over: a long run would otherwise go on without its safety net.
            if let Some(journal) = &checkpoint {
                if let Some(io) = journal.take_error() {
                    if let Err(err) = warn_journal_failed(py, &io) {
                        if callback_error.is_none() {
                            callback_error = Some(err);
                            self.token.cancel();
                        }
                    }
                }
            }
            if let Err(err) = py.check_signals() {
                if signal.is_none() {
                    signal = Some(err);
                    self.token.cancel();
                }
            }
            for message in messages {
                match message {
                    Msg::Step(record) => {
                        let Some(callback) = on_step else { continue };
                        if callback_error.is_some() || signal.is_some() {
                            continue;
                        }
                        let values = tables.values_of(&record);
                        let record = PyStepRecord::new(&graph, record, values);
                        let delivered = Bound::new(py, record)
                            .and_then(|record| callback.call1((record,)).map(|_| ()));
                        if let Err(err) = delivered {
                            callback_error = Some(err);
                            self.token.cancel();
                        }
                    }
                    Msg::Done(done) => outcome = Some(done),
                }
            }
        }
        let _ = py.detach(|| handle.join());
        self.dispatch.clear();
        let Some(outcome) = outcome else {
            return Err(PyRuntimeError::new_err("the engine ended without a result"));
        };
        let Outcome { db, result, panic } = *outcome;
        let database_object: Py<PyDatabase> = match given {
            Some(d) => {
                d.get().restore(db);
                d
            }
            None => Py::new(py, PyDatabase::from_core(self.graph.clone_ref(py), db))?,
        };
        if let Some(io) = checkpoint.as_ref().and_then(|c| c.take_error()) {
            // Not seen by the pump yet (the failure came with the last events).
            if let Err(err) = warn_journal_failed(py, &io) {
                callback_error.get_or_insert(err);
            }
        }

        let fail = |err: PyErr| {
            attach_partial(py, &err, &database_object);
            Err(err)
        };
        if let Some(message) = panic {
            return fail(PyRuntimeError::new_err(format!(
                "internal error in the FaultTune engine: {message}"
            )));
        }
        if let Some(err) = callback_error {
            return fail(err);
        }
        if let Some(err) = signal {
            return fail(err);
        }
        // A failure of user code that the engine could not carry (options(), a heuristic, a
        // seed policy or an optimizer) is raised in place of the error it provoked.
        let stopped_by_failure = matches!(
            &result,
            Ok(_) | Err(FaultTuneError::Cancelled { .. } | FaultTuneError::InvalidConfig(_))
        );
        if stopped_by_failure {
            if let Some(failure) = self.failures.take() {
                return fail(failure.into_pyerr(py, &graph));
            }
        }
        match result {
            Ok(report) => {
                let report = Py::new(py, PyReport::new(&graph, report, Some(&tables)))?;
                *self
                    .last_report
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner) = Some(report.clone_ref(py));
                Ok((database_object, report))
            }
            Err(FaultTuneError::Cancelled { report }) => {
                let report = Py::new(py, PyReport::new(&graph, *report, Some(&tables)))?;
                *self
                    .last_report
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner) = Some(report.clone_ref(py));
                let err = errors::convert(
                    py,
                    Some(&graph),
                    FaultTuneError::Cancelled {
                        report: Box::default(),
                    },
                    Some(report.into_any()),
                );
                fail(err)
            }
            Err(err) => fail(errors::convert(py, Some(&graph), err, None)),
        }
    }
}
