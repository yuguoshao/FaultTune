//! The worked example: gate frequencies of an 8x8 lattice, with the XEB algorithm subgraphs.
//!
//! The toy `GateFrequencyModel` stands in for the physics (relaxation defects, crosstalk).
//! The program walks through everything FaultTune offers around a calibration:
//!
//! 1. **Calibrate** the whole goal with the XEB setting of the paper (`d_P = 1`, `d_R = 2`,
//!    `d_T = 2`), printing progress through an `Observer`.
//! 2. **Compare** the calibrated database with a naive baseline and with other settings:
//!    `d_P = 0 / 1 / 2` and the three traversal heuristics, with wall time, steps, evaluations
//!    and mean error.
//! 3. **Re-calibrate locally** after a qubit drifted: only the region around it is forgotten
//!    and re-learned; the program shows which entries changed.
//! 4. **Save and load** the database as JSON.
//! 5. **Checkpoint and resume**: a journal records every committed step, a run is cancelled
//!    half-way (a `CancellationToken`), and a fresh process resumes from the journal.
//! 6. **Stitch** the four quadrants of the lattice: calibrate them in parallel, then calibrate
//!    the gap between them. The number of subgoals seen through an `Observer` shows the
//!    parallelism.
//!
//! Run it with `cargo run --release -p faulttune --example gate_frequency`. It takes a few
//! seconds. The numbers differ from machine to machine only in the timings.

use std::error::Error;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use faulttune::heuristics::{BreadthFirst, DepthFirst, Random};
use faulttune::optimizers::CoordinateDescent;
use faulttune::toy::GateFrequencyModel;
use faulttune::{
    read_journal, AlgorithmSubgraphs, CancellationToken, CandidateBatch, CheckpointObserver,
    EdgeType, ElementId, ElementKind, ElementSet, ErrorModel, Event, FaultTuneError, ModelError,
    Neighbourhoods, ParameterDatabase, ProcessorGraph, RecordingObserver, Snake, SnakeConfig, Step,
    TraversalHeuristic,
};

/// The size of the lattice: 64 qubits, 112 couplers, 98 parasitic diagonals.
const ROWS: usize = 8;
const COLS: usize = 8;

/// Everything that describes the device and the calibration problem.
struct Device {
    graph: ProcessorGraph,
    /// Nodes and engineered edges: the elements that have a frequency.
    goal: ElementSet,
    /// The five XEB subgraphs (Fig. 3 of the paper).
    xeb: AlgorithmSubgraphs,
    model: GateFrequencyModel,
}

impl Device {
    fn new() -> Result<Self, Box<dyn Error>> {
        let graph = ProcessorGraph::square_lattice(ROWS, COLS);
        let goal = graph.goal(|element| element.kind() != ElementKind::Edge(EdgeType::Parasitic));
        let xeb = AlgorithmSubgraphs::xeb_square_lattice(&graph, ROWS, COLS)?;
        let model = GateFrequencyModel::new(&graph, 53);
        Ok(Self {
            graph,
            goal,
            xeb,
            model,
        })
    }

    /// A Snake for this device with the given configuration and traversal heuristic.
    fn snake(&self, config: SnakeConfig, heuristic: Arc<dyn TraversalHeuristic>) -> Snake<'_> {
        Snake::new(&self.graph, self.goal.clone(), config)
            .algorithm(self.xeb.clone())
            .heuristic_arc(heuristic)
            .optimizer(CoordinateDescent::default())
    }

    /// The XEB setting with a different parameter distance `d_P`.
    fn config(parameter_distance: u32) -> SnakeConfig {
        SnakeConfig {
            parameter_distance,
            constraint_distance: 2,
            traversal_distance: 2,
            ..SnakeConfig::default()
        }
    }

    fn label(&self, id: ElementId) -> &str {
        self.graph.label(id)
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let device = Device::new()?;
    println!(
        "Device: {ROWS}x{COLS} lattice, {} elements, {} of them to calibrate, {} algorithm subgraphs",
        device.graph.len(),
        device.goal.len(),
        device.xeb.len()
    );

    let calibrated = calibrate_with_progress(&device)?;
    compare_settings(&device, &calibrated)?;
    recalibrate_after_drift(&device, calibrated.clone())?;
    let scratch = ScratchDir::new()?;
    save_and_load(&device, &calibrated, &scratch)?;
    checkpoint_and_resume(&device, &scratch)?;
    stitched_calibration(&device, &calibrated)?;
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// 1. Calibrate, with progress printed by an Observer
// ---------------------------------------------------------------------------------------------

/// Calibrates with the default (XEB) setting. An `Observer` prints the progress: observers see
/// every event of the run and may be closures.
fn calibrate_with_progress(device: &Device) -> Result<ParameterDatabase<f64>, Box<dyn Error>> {
    section("1. Calibrate (d_P = 1, d_R = 2, d_T = 2, breadth-first), with progress");

    // Observers outlive the call that creates them, so this one owns a copy of the labels
    // instead of borrowing the graph.
    let labels: Vec<String> = device
        .graph
        .ids()
        .map(|id| device.graph.label(id).to_owned())
        .collect();
    let progress = move |event: &Event| match event {
        Event::PassStarted {
            pass,
            subgoals,
            uncalibrated,
        } => {
            println!("  pass {pass}: {uncalibrated} uncalibrated elements in {subgoals} subgoal(s)")
        }
        // `index` is the ordinal of the step within its subgoal. Print every eighth step.
        Event::StepCommitted(step) if step.index % 8 == 0 => println!(
            "    step {:>3}: centre {:<10} {} parameters, {:>2} constraints, error {:.3}",
            step.index,
            labels[step.centre.index()],
            step.parameters.len(),
            step.constraints.len(),
            step.error
        ),
        Event::PassFinished { pass, calibrated } => {
            println!("  pass {pass} finished: {calibrated} elements calibrated")
        }
        _ => {}
    };

    let snake = device
        .snake(Device::config(1), Arc::new(BreadthFirst))
        .observer(progress);
    let (db, report) = snake.calibrate_with_report(&device.model)?;

    println!(
        "Done: {} steps, {} evaluations, {} wall time ({} of it in the error model)",
        report.steps,
        report.evaluations,
        millis(report.wall_time),
        millis(report.error_time)
    );
    println!("Worst steps (largest error), the natural candidates for re-optimization:");
    for step in report.worst_steps.iter().take(3) {
        println!(
            "  centre {:<10} error {:.3}",
            device.label(step.centre),
            step.error
        );
    }
    Ok(db)
}

// ---------------------------------------------------------------------------------------------
// 2. Compare with a baseline and with other settings
// ---------------------------------------------------------------------------------------------

/// Calibrates with several settings and prints a comparison table.
fn compare_settings(
    device: &Device,
    reference: &ParameterDatabase<f64>,
) -> Result<(), Box<dyn Error>> {
    section("2. Comparison of settings (mean error per calibrated gate, lower is better)");
    println!(
        "  {:<34} {:>10} {:>7} {:>12} {:>11}",
        "setting", "wall time", "steps", "evaluations", "mean error"
    );
    let baseline = device.model.baseline_mean_error(&device.graph, reference);
    println!(
        "  {:<34} {:>10} {:>7} {:>12} {:>11.4}",
        "naive: every gate at 6 GHz", "-", "-", "-", baseline
    );

    let settings: [(&str, u32, Arc<dyn TraversalHeuristic>); 5] = [
        ("d_P = 0, breadth-first", 0, Arc::new(BreadthFirst)),
        (
            "d_P = 1, breadth-first (default)",
            1,
            Arc::new(BreadthFirst),
        ),
        ("d_P = 2, breadth-first", 2, Arc::new(BreadthFirst)),
        ("d_P = 1, depth-first", 1, Arc::new(DepthFirst)),
        ("d_P = 1, random", 1, Arc::new(Random)),
    ];
    let mut means = Vec::new();
    for (name, parameter_distance, heuristic) in settings {
        let snake = device.snake(Device::config(parameter_distance), heuristic);
        let (db, report) = snake.calibrate_with_report(&device.model)?;
        let mean = device.model.mean_error(&device.graph, &db);
        means.push(mean);
        println!(
            "  {:<34} {:>10} {:>7} {:>12} {:>11.4}",
            name,
            millis(report.wall_time),
            report.steps,
            report.evaluations,
            mean
        );
    }
    println!(
        "A larger d_P optimizes more parameters together: fewer, bigger steps and a lower error, \
         at the price of more evaluations.\n  \
         Here d_P = 2 has {:.0}% lower error than d_P = 0 and every setting beats the naive baseline \
         by a factor of at least {:.0}.",
        100.0 * (1.0 - means[2] / means[0]),
        baseline / means.iter().copied().fold(f64::MIN, f64::max)
    );
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// 3. Local re-calibration of a drifted qubit
// ---------------------------------------------------------------------------------------------

/// The toy model plus one new defect (a two-level system) near a chosen qubit: the relaxation
/// error of that qubit, and of the gates it takes part in, gets a Lorentzian bump. It shows how
/// a model is wrapped to describe a device that changed.
struct DriftedModel<'a> {
    base: &'a GateFrequencyModel,
    qubit: ElementId,
    /// Centre frequency of the new defect (GHz).
    centre: f64,
}

impl DriftedModel<'_> {
    /// Half width of the defect (GHz).
    const WIDTH: f64 = 0.015;
    /// Height of the bump.
    const STRENGTH: f64 = 3.0;

    /// The extra error of `element` running at `frequency`: zero unless the element is the
    /// drifted qubit or a coupler attached to it.
    fn penalty(&self, graph: &ProcessorGraph, element: ElementId, frequency: f64) -> f64 {
        let touches = element == self.qubit
            || graph
                .endpoints(element)
                .is_some_and(|ends| ends.contains(&self.qubit));
        if !touches {
            return 0.0;
        }
        let d = frequency - self.centre;
        let w2 = Self::WIDTH * Self::WIDTH;
        Self::STRENGTH * w2 / (d * d + w2)
    }

    /// The summed extra error of every calibrated element of `db`.
    fn total_penalty(&self, graph: &ProcessorGraph, db: &ParameterDatabase<f64>) -> f64 {
        db.iter()
            .map(|(id, entry)| self.penalty(graph, id, entry.value))
            .sum()
    }
}

impl ErrorModel for DriftedModel<'_> {
    type Value = f64;
    type Data = ();

    fn options(&self, element: ElementId) -> Vec<f64> {
        self.base.options(element)
    }

    fn prepare(&self, step: &Step<'_, f64>) -> Result<(), ModelError> {
        self.base.prepare(step)
    }

    fn error(&self, step: &Step<'_, f64>, data: &(), candidate: &[f64]) -> f64 {
        let extra: f64 = step
            .parameters
            .iter()
            .zip(candidate)
            .map(|(&p, &f)| self.penalty(step.graph, p, f))
            .sum();
        self.base.error(step, data, candidate) + extra
    }

    fn error_batch(
        &self,
        step: &Step<'_, f64>,
        data: &(),
        candidates: &CandidateBatch<'_, f64>,
        out: &mut [f64],
    ) -> Result<(), ModelError> {
        self.base.error_batch(step, data, candidates, out)?;
        for (slot, row) in out.iter_mut().zip(candidates.rows()) {
            for (&p, &f) in step.parameters.iter().zip(row) {
                *slot += self.penalty(step.graph, p, f);
            }
        }
        Ok(())
    }
}

/// Qubit `q3_4` develops a defect right at its operating frequency. `recalibrate` forgets the
/// qubit and its couplers (radius `expiry_radius = 1`) and re-learns them subject to all the
/// surrounding gates, which stay untouched.
fn recalibrate_after_drift(
    device: &Device,
    mut db: ParameterDatabase<f64>,
) -> Result<(), Box<dyn Error>> {
    section("3. Local re-calibration of a drifted qubit");
    let qubit = device
        .graph
        .lattice_node(3, 4)
        .ok_or("the lattice has no qubit q3_4")?;
    let stale_frequency = *db.value(qubit).ok_or("q3_4 is not calibrated")?;
    let drifted = DriftedModel {
        base: &device.model,
        qubit,
        centre: stale_frequency,
    };
    println!(
        "{} runs at {stale_frequency:.3} GHz, and a new defect appears exactly there.",
        device.label(qubit)
    );

    let snake = device.snake(Device::config(1), Arc::new(BreadthFirst));
    let n = db.len() as f64;
    let stale_penalty = drifted.total_penalty(&device.graph, &db);
    let before = db.clone();
    let history_before = db.history().len();

    let report = snake.recalibrate(&mut db, &[qubit], &drifted)?;

    // What was forgotten: the neighbourhood of the qubit within the expiry radius.
    let radius = snake.config().expiry_radius;
    let neighbourhoods = Neighbourhoods::build(&device.graph, radius);
    let forgotten: Vec<ElementId> = neighbourhoods
        .within(qubit, radius)
        .iter()
        .copied()
        .filter(|&e| device.goal.contains(e))
        .collect();
    let changed: Vec<ElementId> = device
        .graph
        .ids()
        .filter(|&e| before.value(e) != db.value(e))
        .collect();
    let outside_changed = changed.iter().filter(|e| !forgotten.contains(e)).count();
    if outside_changed != 0 {
        return Err(format!("{outside_changed} entries outside the neighbourhood changed").into());
    }
    if !device.goal.is_subset(db.calibrated()) {
        return Err("the goal is not fully calibrated after the re-calibration".into());
    }

    println!(
        "Forgot {} elements (radius {radius}) and re-learned them in {} step(s), {} evaluations.",
        forgotten.len(),
        report.steps,
        report.evaluations
    );
    println!("Entries that changed value:");
    for &e in &changed {
        println!(
            "  {:<12} {:.3} GHz -> {:.3} GHz",
            device.label(e),
            before.value(e).copied().unwrap_or(f64::NAN),
            db.value(e).copied().unwrap_or(f64::NAN)
        );
    }
    println!(
        "{} of the {} calibrated entries are untouched; the history grew from {} to {} records \
         (one Invalidated record, then the new steps).",
        db.len() - changed.len(),
        db.len(),
        history_before,
        db.history().len()
    );
    let fresh_penalty = drifted.total_penalty(&device.graph, &db);
    println!(
        "Extra error caused by the defect, summed over all gates: {stale_penalty:.3} with the \
         stale calibration, {fresh_penalty:.3} after re-calibration (mean error per gate {:.4} -> {:.4}).",
        (device.model.mean_error(&device.graph, &before) * n + stale_penalty) / n,
        (device.model.mean_error(&device.graph, &db) * n + fresh_penalty) / n
    );
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// 4. Save to and load from JSON
// ---------------------------------------------------------------------------------------------

/// Saves the calibration, loads it back and shows that loading it onto another device fails.
fn save_and_load(
    device: &Device,
    db: &ParameterDatabase<f64>,
    scratch: &ScratchDir,
) -> Result<(), Box<dyn Error>> {
    section("4. Save and load (JSON)");
    let path = scratch.file("calibration.json");
    db.save(&device.graph, &path)?;
    let text = fs::read_to_string(&path)?;
    println!(
        "Saved {} entries and {} history records: {} bytes. The start of the file:",
        db.len(),
        db.history().len(),
        text.len()
    );
    for line in text.lines().take(12) {
        println!("  {line}");
    }
    println!("  ...");

    let loaded = ParameterDatabase::<f64>::load(&device.graph, &path)?;
    let same = loaded.same_outcome(db);
    println!("Loaded it back: same values and history (bit for bit): {same}");
    if !same {
        return Err("the loaded database differs from the saved one".into());
    }

    // The file records a fingerprint of the graph: another device is refused.
    let other = ProcessorGraph::square_lattice(4, 4);
    match ParameterDatabase::<f64>::load(&other, &path) {
        Err(err @ FaultTuneError::GraphMismatch { .. }) => {
            println!("Loading it onto a 4x4 lattice is refused: {err}")
        }
        Err(err) => return Err(err.into()),
        Ok(_) => return Err("a calibration was loaded onto the wrong graph".into()),
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// 5. Checkpoint journal and resume
// ---------------------------------------------------------------------------------------------

/// Cancels a run after 25 steps (standing in for a crash or Ctrl-C), then resumes it from the
/// journal that a `CheckpointObserver` wrote while the run was going.
fn checkpoint_and_resume(device: &Device, scratch: &ScratchDir) -> Result<(), Box<dyn Error>> {
    section("5. Checkpoint journal and resume after an interruption");
    const STOP_AFTER: usize = 25;
    let journal_path = scratch.file("run.journal");

    // First run: journal every committed step, and cancel after STOP_AFTER of them. The
    // journal observer is registered first, so it has written the last step before the
    // canceller sees it.
    let token = CancellationToken::new();
    let journal = Arc::new(CheckpointObserver::create(&journal_path, &device.graph)?);
    let committed = AtomicUsize::new(0);
    let canceller = {
        let token = token.clone();
        move |event: &Event| {
            if matches!(event, Event::StepCommitted(_))
                && committed.fetch_add(1, Ordering::SeqCst) + 1 == STOP_AFTER
            {
                token.cancel();
            }
        }
    };
    let snake = device
        .snake(Device::config(1), Arc::new(BreadthFirst))
        .cancellation(token)
        .observer_arc(journal.clone())
        .observer(canceller);
    let mut interrupted = ParameterDatabase::<f64>::new(&device.graph);
    match snake.calibrate_into(&mut interrupted, &device.model) {
        Err(FaultTuneError::Cancelled { report }) => println!(
            "Run cancelled after {} steps; the database kept all of them ({} of {} elements).",
            report.steps,
            interrupted.len(),
            device.goal.len()
        ),
        Ok(_) => return Err("the run finished before it was cancelled".into()),
        Err(err) => return Err(err.into()),
    }
    if let Some(err) = journal.take_error() {
        return Err(err.into());
    }

    // A crash in the middle of writing a line leaves a torn last line. Simulate one.
    let mut file = OpenOptions::new().append(true).open(&journal_path)?;
    file.write_all(br#"{"type":"step","pass":0,"subgoal":0,"centre":"q4"#)?;
    drop(file);

    // "New process": nothing is in memory but the journal. Replay it onto an empty database.
    let steps = read_journal(&journal_path, &device.graph)?;
    let mut restored = ParameterDatabase::<f64>::new(&device.graph);
    let applied = restored.replay_journal(&device.graph, &steps, |e| device.model.options(e))?;
    let identical = restored.same_outcome(&interrupted);
    println!(
        "Journal read (the torn last line is ignored): {} steps, {applied} replayed; the replayed \
         database equals the interrupted one: {identical}",
        steps.len()
    );
    if !identical {
        return Err("replaying the journal did not reproduce the interrupted database".into());
    }

    // Resume: the same call as a fresh calibration, on the partly filled database. The journal
    // is opened for appending (this also repairs the torn line), so it stays complete.
    let journal = Arc::new(CheckpointObserver::append(&journal_path, &device.graph)?);
    let snake = device
        .snake(Device::config(1), Arc::new(BreadthFirst))
        .observer_arc(journal.clone());
    let report = snake.calibrate_into(&mut restored, &device.model)?;
    if let Some(err) = journal.take_error() {
        return Err(err.into());
    }
    if !device.goal.is_subset(restored.calibrated()) {
        return Err("the resumed run did not calibrate the whole goal".into());
    }
    let (_, full_report) = device
        .snake(Device::config(1), Arc::new(BreadthFirst))
        .calibrate_with_report(&device.model)?;
    println!(
        "Resumed: {} more steps ({} in an uninterrupted run), all {} goal elements calibrated.",
        report.steps,
        full_report.steps,
        device.goal.len()
    );
    println!(
        "Mean error per gate after resuming: {:.4}. A resumed run starts a fresh traversal from \
         the remaining\n  elements, so it need not equal the uninterrupted run, only complete it.",
        device.model.mean_error(&device.graph, &restored)
    );
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// 6. Stitched parallel calibration
// ---------------------------------------------------------------------------------------------

/// The goal elements of each quadrant of the lattice, by the layout coordinates of the graph
/// (edges sit at the midpoint of their endpoints).
fn quadrants(device: &Device) -> Result<Vec<ElementSet>, Box<dyn Error>> {
    let coords = device.graph.coords().ok_or("the graph has no layout")?;
    let (mid_x, mid_y) = ((COLS as f32 - 1.0) / 2.0, (ROWS as f32 - 1.0) / 2.0);
    let mut regions = vec![ElementSet::new(device.graph.len()); 4];
    for e in &device.goal {
        let [x, y] = coords[e.index()];
        let quadrant = usize::from(x > mid_x) + 2 * usize::from(y > mid_y);
        regions[quadrant].insert(e);
    }
    Ok(regions)
}

/// Splits the lattice into quadrants, calibrates them in parallel and then the gap between
/// them. An observer records how many subgoals every pass of both stages had.
fn stitched_calibration(
    device: &Device,
    reference: &ParameterDatabase<f64>,
) -> Result<(), Box<dyn Error>> {
    section("6. Stitched calibration: four quadrants in parallel, then the gap");
    let regions = quadrants(device)?;
    let sizes: Vec<usize> = regions.iter().map(ElementSet::len).collect();

    let recorder = RecordingObserver::new();
    let parallel = device
        .snake(Device::config(1), Arc::new(BreadthFirst))
        .observer(recorder.clone());
    let plan = parallel.stitch_plan(regions)?;
    println!(
        "Quadrants hold {sizes:?} elements. Stage one calibrates {} of them (the cores, more than \
         max(d_P, d_R) = 2 apart),\n  stage two the {} elements of the gap.",
        plan.stage_one.len(),
        plan.gap.len()
    );

    let mut stitched = ParameterDatabase::<f64>::new(&device.graph);
    let report = parallel.stitch(&mut stitched, &plan, &device.model)?;

    // Pass numbers restart with every stage, so a pass 0 after the first one starts stage two.
    let mut stage = 0;
    for event in recorder.events() {
        if let Event::PassStarted {
            pass,
            subgoals,
            uncalibrated,
        } = event
        {
            if pass == 0 {
                stage += 1;
            }
            println!(
                "  stage {stage}, pass {pass}: {uncalibrated} uncalibrated elements -> {subgoals} subgoal(s)"
            );
        }
    }
    println!(
        "Stitched calibration: {} steps in {} passes, mean error per gate {:.4} (unstitched: {:.4}).",
        report.steps,
        report.passes,
        device.model.mean_error(&device.graph, &stitched),
        device.model.mean_error(&device.graph, reference)
    );

    // The result must not depend on the thread count: compare one thread with four.
    println!("Same stitched calibration with different thread counts:");
    let mut databases = Vec::new();
    for threads in [1, 4] {
        let config = SnakeConfig {
            threads: Some(threads),
            ..Device::config(1)
        };
        let snake = device.snake(config, Arc::new(BreadthFirst));
        let mut db = ParameterDatabase::<f64>::new(&device.graph);
        let report = snake.stitch(&mut db, &plan, &device.model)?;
        println!("  {threads} thread(s): {}", millis(report.wall_time));
        databases.push(db);
    }
    let identical =
        databases[0].same_outcome(&databases[1]) && databases[0].same_outcome(&stitched);
    println!("  identical databases, history included: {identical}");
    if !identical {
        return Err("the stitched calibration depends on the thread count".into());
    }
    println!(
        "(The toy model is so cheap that the timings above are mostly noise. With a model that \
         measures hardware, the\n  stage-one subgoals run side by side. On a lattice this small \
         the gap is most of the processor; on a big one the cores dominate.)"
    );
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------------

/// Prints a section heading.
fn section(title: &str) {
    println!("\n== {title} ==");
}

/// A duration in milliseconds with two decimals.
fn millis(d: Duration) -> String {
    format!("{:.2} ms", d.as_secs_f64() * 1e3)
}

/// A private directory in the system's temporary directory, removed when dropped.
struct ScratchDir(PathBuf);

impl ScratchDir {
    fn new() -> std::io::Result<Self> {
        let path = std::env::temp_dir().join(format!("faulttune-example-{}", std::process::id()));
        fs::create_dir_all(&path)?;
        Ok(Self(path))
    }

    fn file(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        // Best effort: a leftover temporary directory is harmless.
        let _ = fs::remove_dir_all(&self.0);
    }
}
