//! Toy error models for examples, tests and benchmarks (README: "Toy models for examples and
//! tests"). They are **pure**: the error of a candidate depends only on the step's parameters,
//! their values and the constraint values, so they are safe with `CachePolicy::AcrossSteps`.
//!
//! * [`SeparableModel`]: every element has a private target; the optimum is known exactly.
//! * [`CollisionModel`]: hard frequency-collision constraints expressed as `+∞` errors.
//! * [`GateFrequencyModel`]: a small physically flavoured model of gate frequencies on a
//!   lattice (relaxation defects, crosstalk), used by the worked example.
//!
//! # Determinism
//!
//! All randomness happens at construction time. Each random quantity is drawn from a
//! `StdRng` seeded from `(seed, purpose, element id)`, so the data of an element depends only on
//! the seed and its own ID: adding elements elsewhere in a graph does not change it. Evaluation
//! is a pure function of the step; only `+`, `-`, `*` and `/` are used on floating point
//! numbers (no libm calls), and sums are accumulated in a fixed order, so results are identical
//! from run to run and, for a fixed version of the `rand` crate (whose `StdRng` stream is not
//! guaranteed stable across releases; keep `Cargo.lock` under version control), from platform to
//! platform.
//!
//! # Deviations and notes
//!
//! * The constructors panic on invalid arguments (`k == 0`, `lo > hi`, non-finite bounds or a span `hi - lo` that overflows)
//!   rather than returning a `Result`: their signatures are fixed.
//! * [`CollisionModel::required_options`] returns `usize::MAX` when no number of options can
//!   guarantee a collision-free greedy assignment (see there).
//! * [`GateFrequencyModel`] builds its own [`Neighbourhoods`] (interaction distance 2) and
//!   never looks at [`Step::graph`]; element IDs must belong to the graph it was built for.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::database::ParameterDatabase;
use crate::error::ModelError;
use crate::graph::{ElementId, ElementKind, ProcessorGraph};
use crate::model::{CandidateBatch, ErrorModel, Step};
use crate::neighbourhoods::Neighbourhoods;

/// `k` evenly spaced values from `lo` to `hi` inclusive (`k == 1` gives `[lo]`, `k == 0` gives
/// an empty vector). The first value is exactly `lo` and, for `k >= 2`, the last exactly `hi`.
pub fn linspace(lo: f64, hi: f64, k: usize) -> Vec<f64> {
    match k {
        0 => Vec::new(),
        1 => vec![lo],
        _ => {
            let last = (k - 1) as f64;
            let mut values: Vec<f64> = (0..k).map(|i| lo + (hi - lo) * (i as f64 / last)).collect();
            values[k - 1] = hi;
            values
        }
    }
}

/// The finalizer of SplitMix64: a cheap, well-mixed bijection on `u64`.
#[inline]
fn mix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// A generator private to `(seed, purpose, element)`.
fn element_rng(seed: u64, purpose: u64, element: ElementId) -> StdRng {
    StdRng::seed_from_u64(mix64(mix64(seed ^ purpose) ^ u64::from(element.raw())))
}

/// A number in `[0, 1)` from the top 53 bits of a hash.
#[inline]
fn unit_interval(hash: u64) -> f64 {
    (hash >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
}

/// Purposes of the per-element generators (kept distinct so that streams never coincide).
const PURPOSE_TARGET: u64 = 0x7461_7267_6574_0001;
const PURPOSE_PREFERRED: u64 = 0x7072_6566_6572_0002;
const PURPOSE_DEFECT: u64 = 0x6465_6665_6374_0003;
const PURPOSE_COUPLING: u64 = 0x636f_7570_6c65_0004;

/// Checks the arguments shared by the constructors of the models with `k` options.
fn check_range(who: &str, k: usize, lo: f64, hi: f64) {
    assert!(
        k >= 1,
        "{who}: need at least one option per element (k = 0)"
    );
    assert!(
        lo.is_finite() && hi.is_finite() && lo <= hi,
        "{who}: the option range must be finite with lo <= hi (got lo = {lo}, hi = {hi})"
    );
    assert!(
        (hi - lo).is_finite(),
        "{who}: the option range is too wide: hi - lo must be finite (got lo = {lo}, hi = {hi})"
    );
}

/// `error = Σ_j (value_j − target(element_j))²` over the step's parameters; constraints are
/// ignored. Every element has `k` options `linspace(lo, hi, k)`, and its target is a
/// pseudo-random number in `[lo, hi]` derived from `(seed, element)`. Because the model is
/// separable, the Snake must find the global optimum for every `d_P`: each element should end up
/// with the option nearest to its target (first such option on ties).
#[derive(Clone, Debug)]
pub struct SeparableModel {
    options: Vec<f64>,
    targets: Vec<f64>,
}

impl SeparableModel {
    /// A model with `k` options in `[lo, hi]` for each element of `graph`.
    ///
    /// # Panics
    /// Panics if `k == 0`, or if `lo`/`hi` are not finite, `lo > hi`, or `hi - lo` overflows.
    pub fn new(graph: &ProcessorGraph, k: usize, lo: f64, hi: f64, seed: u64) -> Self {
        check_range("SeparableModel::new", k, lo, hi);
        let targets = graph
            .ids()
            .map(|id| element_rng(seed, PURPOSE_TARGET, id).random_range(lo..=hi))
            .collect();
        Self {
            options: linspace(lo, hi, k),
            targets,
        }
    }

    /// The target of an element.
    ///
    /// # Panics
    /// Panics if `element` is not in the graph the model was built for.
    pub fn target(&self, element: ElementId) -> f64 {
        self.targets[element.index()]
    }

    /// The best achievable value for an element: the option nearest to its target (the first
    /// one on ties).
    ///
    /// # Panics
    /// Panics if `element` is not in the graph the model was built for.
    pub fn optimum(&self, element: ElementId) -> f64 {
        let target = self.target(element);
        let mut best = self.options[0];
        let mut best_cost = (best - target) * (best - target);
        for &option in &self.options[1..] {
            let cost = (option - target) * (option - target);
            if cost < best_cost {
                best = option;
                best_cost = cost;
            }
        }
        best
    }

    /// Sum of squared distances to the targets over the calibrated elements of `db`
    /// (accumulated in ascending element order).
    pub fn total_error(&self, db: &ParameterDatabase<f64>) -> f64 {
        db.iter()
            .map(|(id, entry)| {
                let d = entry.value - self.target(id);
                d * d
            })
            .fold(0.0, |acc, x| acc + x)
    }
}

impl ErrorModel for SeparableModel {
    type Value = f64;
    type Data = ();

    fn options(&self, _element: ElementId) -> Vec<f64> {
        self.options.clone()
    }

    fn prepare(&self, _step: &Step<'_, f64>) -> Result<(), ModelError> {
        Ok(())
    }

    fn error(&self, step: &Step<'_, f64>, _data: &(), candidate: &[f64]) -> f64 {
        assert_eq!(
            candidate.len(),
            step.parameters.len(),
            "SeparableModel::error: candidate width differs from the number of parameters"
        );
        step.parameters
            .iter()
            .zip(candidate)
            .map(|(&id, &value)| {
                let d = value - self.target(id);
                d * d
            })
            .fold(0.0, |acc, x| acc + x)
    }
}

/// Frequency-collision constraints. Two elements *interact* if one is within
/// `interaction_distance` of the other (paper sense, `X_a^d`). A candidate is forbidden
/// (`+∞`) if any interacting pair among the step's parameters and constraints (with at least one
/// parameter in the pair) has `|f_a − f_b| < min_separation`. Otherwise the error is
/// `Σ_j (value_j − preferred(element_j))²`, so there is a smooth signal to optimize.
///
/// With enough options per element a collision-free assignment always exists (each interacting
/// neighbour blocks a bounded number of options), and the Snake must find one whenever
/// `interaction_distance <= d_R` and `interaction_distance <= d_P` or the interacting element is a
/// constraint of the step.
///
/// Every element has the options `linspace(lo, hi, k)`; its preferred frequency is a
/// pseudo-random number in `[lo, hi]` derived from `(seed, element)`.
#[derive(Clone, Debug)]
pub struct CollisionModel {
    options: Vec<f64>,
    preferred: Vec<f64>,
    min_separation: f64,
    interaction_distance: u32,
    neighbourhoods: Neighbourhoods,
    required: usize,
}

impl CollisionModel {
    /// A model with `k` options in `[lo, hi]` for each element of `graph`.
    ///
    /// # Panics
    /// Panics if `k == 0`, or if `lo`/`hi` are not finite, `lo > hi` or `hi - lo`
    /// overflows, or if `min_separation` is NaN.
    pub fn new(
        graph: &ProcessorGraph,
        k: usize,
        lo: f64,
        hi: f64,
        min_separation: f64,
        interaction_distance: u32,
        seed: u64,
    ) -> Self {
        check_range("CollisionModel::new", k, lo, hi);
        assert!(
            !min_separation.is_nan(),
            "CollisionModel::new: min_separation is NaN"
        );
        let preferred = graph
            .ids()
            .map(|id| element_rng(seed, PURPOSE_PREFERRED, id).random_range(lo..=hi))
            .collect();
        let neighbourhoods = Neighbourhoods::build(graph, interaction_distance);
        let max_neighbours = graph
            .ids()
            .map(|id| neighbourhoods.within(id, interaction_distance).len() - 1)
            .max()
            .unwrap_or(0);
        let required = options_bound(max_neighbours, min_separation, hi - lo);
        Self {
            options: linspace(lo, hi, k),
            preferred,
            min_separation,
            interaction_distance,
            neighbourhoods,
            required,
        }
    }

    /// The minimum number of options per element that guarantees a collision-free greedy
    /// assignment exists, for the given graph (worst case over elements).
    ///
    /// **The bound.** Let `Δ` be the largest number of other elements within
    /// `interaction_distance` of any element, `s = min_separation` and `W = hi − lo`. Calibrating
    /// elements one at a time in any order, in any way, an element has at most `Δ` calibrated
    /// interacting neighbours. Each one forbids the options in an open interval of length `2s`,
    /// which contains at most `⌈2s (k − 1) / W⌉ < 2s (k − 1) / W + 1` of `k` evenly spaced
    /// options. Some option is therefore free as soon as `Δ (2s (k − 1) / W + 1) ≤ k`, that is
    /// (with `ρ = 2s / W` and `Δρ < 1`) for every
    ///
    /// ```text
    /// k >= 1 + (Δ − 1) / (1 − Δρ).
    /// ```
    ///
    /// The method returns `1 + ⌈(Δ − 1) / (1 − Δρ)⌉ + 1` (the last `+ 1` absorbs floating
    /// point rounding at interval ends), and `1` if collisions are impossible (`Δ = 0` or
    /// `s <= 0`). The guarantee also holds for every larger `k`.
    ///
    /// If `Δ · 2s >= W` (or `W == 0` with `s > 0`), neighbours could forbid the whole range
    /// whatever `k` is and no number of options is enough: `usize::MAX` is returned.
    pub fn required_options(&self) -> usize {
        self.required
    }

    /// Every pair of calibrated elements of `db` that interact and are closer than
    /// `min_separation` (each pair once, ascending). Empty means the calibration is collision-free.
    pub fn collisions(&self, db: &ParameterDatabase<f64>) -> Vec<(ElementId, ElementId)> {
        let mut pairs = Vec::new();
        for (a, entry) in db.iter() {
            for &b in self.neighbourhoods.within(a, self.interaction_distance) {
                if b <= a {
                    continue;
                }
                if let Some(other) = db.value(b) {
                    if (entry.value - other).abs() < self.min_separation {
                        pairs.push((a, b));
                    }
                }
            }
        }
        pairs.sort_unstable();
        pairs
    }

    /// Whether `a` (with value `fa`) and `b` (with value `fb`) form a forbidden pair.
    #[inline]
    fn collide(&self, a: ElementId, fa: f64, b: ElementId, fb: f64) -> bool {
        // The cheap float test first: the neighbourhood lookup is only needed on a near miss.
        a != b
            && (fa - fb).abs() < self.min_separation
            && self
                .neighbourhoods
                .is_within(a, b, self.interaction_distance)
    }
}

/// The option-count bound of [`CollisionModel::required_options`].
fn options_bound(max_neighbours: usize, min_separation: f64, width: f64) -> usize {
    if max_neighbours == 0 || min_separation <= 0.0 {
        return 1;
    }
    let delta = max_neighbours as f64;
    let rho = 2.0 * min_separation / width;
    // `width == 0` gives `rho = inf` (or NaN for `min_separation == 0`, handled above).
    let load = delta * rho;
    if load < 1.0 {
        let steps = ((delta - 1.0) / (1.0 - load)).ceil();
        // Saturating float-to-int conversion: absurdly large bounds become `usize::MAX`.
        (steps as usize).saturating_add(2)
    } else {
        // No number of options is enough (or the load is NaN).
        usize::MAX
    }
}

impl ErrorModel for CollisionModel {
    type Value = f64;
    type Data = ();

    fn options(&self, _element: ElementId) -> Vec<f64> {
        self.options.clone()
    }

    fn prepare(&self, _step: &Step<'_, f64>) -> Result<(), ModelError> {
        Ok(())
    }

    fn error(&self, step: &Step<'_, f64>, _data: &(), candidate: &[f64]) -> f64 {
        assert_eq!(
            candidate.len(),
            step.parameters.len(),
            "CollisionModel::error: candidate width differs from the number of parameters"
        );
        for (j, (&a, &fa)) in step.parameters.iter().zip(candidate).enumerate() {
            for &(c, fc) in step.constraints {
                if self.collide(a, fa, c, fc) {
                    return f64::INFINITY;
                }
            }
            for (&b, &fb) in step.parameters[j + 1..].iter().zip(&candidate[j + 1..]) {
                if self.collide(a, fa, b, fb) {
                    return f64::INFINITY;
                }
            }
        }
        step.parameters
            .iter()
            .zip(candidate)
            .map(|(&id, &value)| {
                let d = value - self.preferred[id.index()];
                d * d
            })
            .fold(0.0, |acc, x| acc + x)
    }
}

/// One relaxation defect: a Lorentzian bump of the relaxation error of a qubit.
#[derive(Copy, Clone, Debug)]
struct Bump {
    /// Centre frequency (GHz).
    centre: f64,
    /// Squared half width at half maximum (GHz²).
    width_sq: f64,
    /// Height of the bump at its centre.
    strength: f64,
}

impl Bump {
    /// The contribution of the bump at frequency `f`.
    #[inline]
    fn at(&self, f: f64) -> f64 {
        let d = f - self.centre;
        self.strength * self.width_sq / (d * d + self.width_sq)
    }
}

/// A toy model of gate frequencies on a qubit lattice, used by the worked example.
///
/// Nodes are single-qubit gate frequencies, engineered edges are two-qubit gate frequencies;
/// other kinds are rejected by `options`. Each element chooses among 100 frequencies in
/// `[5.0, 7.0]` GHz. The error of an element is a smooth sum of: relaxation (a Lorentzian bump
/// for each of a few random defects per qubit; a two-qubit gate pays for both of its qubits),
/// a detuning cost (a two-qubit gate prefers a frequency near the mean of its two qubits) and
/// crosstalk with the other elements of the step and its constraints (a Lorentzian in the
/// frequency difference, scaled by a random coupling strength, for elements within the
/// interaction distance). It overrides `error_batch`.
///
/// # The physics
///
/// For a frequency `f` (GHz) of element `e`:
///
/// * **Relaxation.** Every qubit has [`DEFECTS_PER_QUBIT`](Self::DEFECTS_PER_QUBIT) defects, each
///   a Lorentzian `s w² / ((f − c)² + w²)` with a random centre `c ∈ [5, 7]`, half width
///   `w ∈ [5, 20]` MHz and strength `s ∈ [0.5, 2]`. A single-qubit gate pays the sum over the
///   defects of its qubit, a two-qubit gate the sum over both of its qubits, at the gate
///   frequency.
/// * **Detuning.** A two-qubit gate on qubits `a`, `b` pays
///   `DETUNING_WEIGHT · (f − f_ref)²` with `f_ref = (f_a + f_b) / 2`, where `f_a` and `f_b` are
///   the frequencies of the endpoint qubits *when known*, that is when they are constraints or
///   parameters of the step. If only one is known it serves as `f_ref`; if neither is, there is
///   no detuning cost.
/// * **Crosstalk.** Each pair of elements within [`INTERACTION_DISTANCE`](Self::INTERACTION_DISTANCE)
///   (by [`Neighbourhoods`]) pays `g_ab · w_x² / ((f_a − f_b)² + w_x²)` with `w_x =`
///   [`CROSSTALK_WIDTH`](Self::CROSSTALK_WIDTH) and a coupling strength `g_ab ∈ [0.2, 1]` that
///   is a fixed pseudo-random function of the seed and the two element IDs (symmetric).
///
/// The error of a candidate is the sum of the own terms (relaxation and detuning) of the step's
/// parameters, plus the crosstalk of every interacting pair among the parameters (each pair
/// once) and of every interacting (parameter, constraint) pair. Constraint-constraint pairs are
/// not part of a step's error (they are constant).
///
/// The model is pure, deterministic in `(graph, seed)` and `Sync`.
#[derive(Clone, Debug)]
pub struct GateFrequencyModel {
    seed: u64,
    options: Vec<f64>,
    kinds: Vec<ElementKind>,
    endpoints: Vec<Option<[ElementId; 2]>>,
    /// The defects relevant to each element, flat: element `i` owns
    /// `bumps[bump_start[i]..bump_start[i + 1]]` (its own for a node, those of both endpoints
    /// for an edge).
    bumps: Vec<Bump>,
    bump_start: Vec<u32>,
    neighbourhoods: Neighbourhoods,
}

/// Where the frequency of an endpoint qubit of a two-qubit gate comes from in one step.
#[derive(Copy, Clone, Debug)]
enum Source {
    /// Not a constraint nor a parameter of the step.
    Unknown,
    /// A constraint: a fixed value.
    Fixed(f64),
    /// A parameter: a column of the candidate.
    Column(usize),
}

impl Source {
    #[inline]
    fn resolve(self, row: &[f64]) -> Option<f64> {
        match self {
            Source::Unknown => None,
            Source::Fixed(f) => Some(f),
            Source::Column(j) => Some(row[j]),
        }
    }
}

/// What the evaluation of one parameter needs, resolved once per step.
struct ParamPlan<'m> {
    bumps: &'m [Bump],
    /// The endpoint qubits of a two-qubit gate.
    endpoints: Option<[Source; 2]>,
}

/// A step prepared for fast evaluation of many candidates: all lookups are resolved, so the
/// per-row work is a tight loop without allocation.
struct Plan<'m> {
    params: Vec<ParamPlan<'m>>,
    /// `(column, coupling, constraint frequency)`.
    with_constraints: Vec<(usize, f64, f64)>,
    /// `(column, column, coupling)` with the first column smaller.
    among_params: Vec<(usize, usize, f64)>,
}

impl Plan<'_> {
    /// The error of one candidate (a row of `params.len()` frequencies).
    #[inline]
    fn eval(&self, row: &[f64]) -> f64 {
        let mut total = 0.0;
        for (j, param) in self.params.iter().enumerate() {
            let f = row[j];
            total += relaxation(param.bumps, f);
            if let Some([a, b]) = param.endpoints {
                if let Some(reference) = reference_frequency(a.resolve(row), b.resolve(row)) {
                    total += detuning_cost(f, reference);
                }
            }
        }
        for &(j, coupling, fc) in &self.with_constraints {
            total += crosstalk(coupling, row[j] - fc);
        }
        for &(j, l, coupling) in &self.among_params {
            total += crosstalk(coupling, row[j] - row[l]);
        }
        total
    }
}

/// The relaxation error at frequency `f` from a set of defects.
#[inline]
fn relaxation(bumps: &[Bump], f: f64) -> f64 {
    let mut sum = 0.0;
    for bump in bumps {
        sum += bump.at(f);
    }
    sum
}

/// The frequency a two-qubit gate wants to sit at: the mean of its qubits' frequencies when both
/// are known, the known one if only one is, nothing if neither is.
#[inline]
fn reference_frequency(a: Option<f64>, b: Option<f64>) -> Option<f64> {
    match (a, b) {
        (Some(a), Some(b)) => Some(0.5 * (a + b)),
        (Some(f), None) | (None, Some(f)) => Some(f),
        (None, None) => None,
    }
}

/// The detuning cost of a two-qubit gate at `f` whose preferred frequency is `reference`.
#[inline]
fn detuning_cost(f: f64, reference: f64) -> f64 {
    let d = f - reference;
    GateFrequencyModel::DETUNING_WEIGHT * d * d
}

/// The crosstalk of a pair with the given coupling and frequency difference.
#[inline]
fn crosstalk(coupling: f64, difference: f64) -> f64 {
    let w2 = GateFrequencyModel::CROSSTALK_WIDTH * GateFrequencyModel::CROSSTALK_WIDTH;
    coupling * w2 / (difference * difference + w2)
}

impl GateFrequencyModel {
    /// Number of candidate frequencies of every element.
    pub const NUM_OPTIONS: usize = 100;
    /// Lowest candidate frequency (GHz).
    pub const FREQUENCY_MIN: f64 = 5.0;
    /// Highest candidate frequency (GHz).
    pub const FREQUENCY_MAX: f64 = 7.0;
    /// Relaxation defects per qubit.
    pub const DEFECTS_PER_QUBIT: usize = 3;
    /// Two elements within this distance (paper sense) have crosstalk.
    pub const INTERACTION_DISTANCE: u32 = 2;
    /// Half width at half maximum of the crosstalk Lorentzian (GHz).
    pub const CROSSTALK_WIDTH: f64 = 0.04;
    /// Weight of the (quadratic) detuning cost of a two-qubit gate (per GHz²).
    pub const DETUNING_WEIGHT: f64 = 1.0;
    /// The frequency of the naive baseline of
    /// [`baseline_mean_error`](Self::baseline_mean_error): the middle of the band.
    pub const BASELINE_FREQUENCY: f64 = 6.0;

    /// Random device data for `graph`, derived from `seed`.
    ///
    /// The defects of a qubit depend only on `(seed, qubit ID)`, and the coupling of a pair only
    /// on `(seed, the two IDs)`.
    pub fn new(graph: &ProcessorGraph, seed: u64) -> Self {
        let n = graph.len();
        let node_bumps: Vec<[Bump; Self::DEFECTS_PER_QUBIT]> = graph
            .ids()
            .map(|id| {
                let mut rng = element_rng(seed, PURPOSE_DEFECT, id);
                let mut defects = [Bump {
                    centre: 0.0,
                    width_sq: 1.0,
                    strength: 0.0,
                }; Self::DEFECTS_PER_QUBIT];
                if graph.kind(id).is_node() {
                    for defect in &mut defects {
                        let centre = rng.random_range(Self::FREQUENCY_MIN..=Self::FREQUENCY_MAX);
                        let width = rng.random_range(0.005..=0.020);
                        let strength = rng.random_range(0.5..=2.0);
                        *defect = Bump {
                            centre,
                            width_sq: width * width,
                            strength,
                        };
                    }
                }
                defects
            })
            .collect();
        let mut bumps = Vec::new();
        let mut bump_start = Vec::with_capacity(n + 1);
        bump_start.push(0u32);
        for id in graph.ids() {
            match graph.endpoints(id) {
                None => bumps.extend_from_slice(&node_bumps[id.index()]),
                Some([a, b]) => {
                    bumps.extend_from_slice(&node_bumps[a.index()]);
                    bumps.extend_from_slice(&node_bumps[b.index()]);
                }
            }
            bump_start.push(bumps.len() as u32);
        }
        Self {
            seed,
            options: linspace(Self::FREQUENCY_MIN, Self::FREQUENCY_MAX, Self::NUM_OPTIONS),
            kinds: graph.ids().map(|id| graph.kind(id)).collect(),
            endpoints: graph.ids().map(|id| graph.endpoints(id)).collect(),
            bumps,
            bump_start,
            neighbourhoods: Neighbourhoods::build(graph, Self::INTERACTION_DISTANCE),
        }
    }

    /// The coupling strength of the pair `{a, b}`, in `[0.2, 1]`: a symmetric, deterministic
    /// function of the seed and the two IDs.
    fn coupling(&self, a: ElementId, b: ElementId) -> f64 {
        let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
        let h = mix64(mix64(self.seed ^ PURPOSE_COUPLING) ^ u64::from(lo.raw()));
        let h = mix64(h ^ u64::from(hi.raw()).wrapping_mul(0xD6E8_FEB8_6659_FD93));
        0.2 + 0.8 * unit_interval(h)
    }

    /// The defects relevant to an element.
    fn bumps_of(&self, id: ElementId) -> &[Bump] {
        let i = id.index();
        &self.bumps[self.bump_start[i] as usize..self.bump_start[i + 1] as usize]
    }

    /// Resolves a step into a [`Plan`].
    fn plan<'m>(&'m self, step: &Step<'_, f64>) -> Plan<'m> {
        // Columns by element ID, for the lookups of interacting parameters.
        let mut columns: Vec<(ElementId, usize)> = step
            .parameters
            .iter()
            .enumerate()
            .map(|(j, &id)| (id, j))
            .collect();
        columns.sort_unstable();
        let column_of = |id: ElementId| {
            columns
                .binary_search_by_key(&id, |&(e, _)| e)
                .ok()
                .map(|i| columns[i].1)
        };
        let source = |id: ElementId| match step.constraint(id) {
            Some(&f) => Source::Fixed(f),
            None => column_of(id).map_or(Source::Unknown, Source::Column),
        };

        let mut params = Vec::with_capacity(step.parameters.len());
        let mut with_constraints = Vec::new();
        let mut among_params = Vec::new();
        for (j, &p) in step.parameters.iter().enumerate() {
            params.push(ParamPlan {
                bumps: self.bumps_of(p),
                endpoints: self.endpoints[p.index()].map(|[a, b]| [source(a), source(b)]),
            });
            for &q in self.neighbourhoods.within(p, Self::INTERACTION_DISTANCE) {
                if q == p {
                    continue;
                }
                if let Some(&fc) = step.constraint(q) {
                    with_constraints.push((j, self.coupling(p, q), fc));
                } else if let Some(l) = column_of(q) {
                    if l > j {
                        among_params.push((j, l, self.coupling(p, q)));
                    }
                }
            }
        }
        Plan {
            params,
            with_constraints,
            among_params,
        }
    }

    /// The error of the assignment `values[i]` for `ids[i]` (ascending IDs) with no constraints.
    fn assignment_error(&self, graph: &ProcessorGraph, ids: &[ElementId], values: &[f64]) -> f64 {
        let step = Step {
            index: 0,
            centre: ids[0],
            parameters: ids,
            constraints: &[],
            graph,
        };
        self.plan(&step).eval(values)
    }

    /// Mean error over the calibrated elements of `db`, evaluated with the full model (every
    /// pair of calibrated elements that interact).
    ///
    /// This is the total error of the assignment (the own terms of every calibrated element,
    /// plus the crosstalk of every interacting pair of calibrated elements, each pair once),
    /// divided by the number of calibrated elements; detuning uses the calibrated endpoint
    /// qubits of each two-qubit gate. It is `0.0` for an empty database.
    pub fn mean_error(&self, graph: &ProcessorGraph, db: &ParameterDatabase<f64>) -> f64 {
        let (ids, values): (Vec<ElementId>, Vec<f64>) =
            db.iter().map(|(id, entry)| (id, entry.value)).unzip();
        if ids.is_empty() {
            return 0.0;
        }
        self.assignment_error(graph, &ids, &values) / ids.len() as f64
    }

    /// The mean error of an assignment where every element uses the same fixed frequency
    /// (a naive baseline for comparisons), over the elements in `db`.
    ///
    /// The frequency is [`BASELINE_FREQUENCY`](Self::BASELINE_FREQUENCY) = 6 GHz, the middle of
    /// the band; the values stored in `db` are ignored, only which elements are calibrated
    /// matters. Every interacting pair then sits at its full crosstalk peak. `0.0` for an empty
    /// database.
    pub fn baseline_mean_error(&self, graph: &ProcessorGraph, db: &ParameterDatabase<f64>) -> f64 {
        let ids = db.calibrated().to_vec();
        if ids.is_empty() {
            return 0.0;
        }
        let values = vec![Self::BASELINE_FREQUENCY; ids.len()];
        self.assignment_error(graph, &ids, &values) / ids.len() as f64
    }
}

impl ErrorModel for GateFrequencyModel {
    type Value = f64;
    type Data = ();

    /// The 100 frequencies of `[5.0, 7.0]` GHz.
    ///
    /// # Panics
    /// Panics if `element` is not a node or an engineered edge (parasitic and custom edges have
    /// no frequency: exclude them from the goal), or is not in the graph of the model.
    fn options(&self, element: ElementId) -> Vec<f64> {
        match self.kinds[element.index()] {
            ElementKind::Node | ElementKind::Edge(crate::graph::EdgeType::Engineered) => {
                self.options.clone()
            }
            kind => panic!(
                "GateFrequencyModel::options: element {element} is {kind:?}; only nodes and \
                 engineered edges have a gate frequency (exclude other edges from the goal)"
            ),
        }
    }

    fn prepare(&self, _step: &Step<'_, f64>) -> Result<(), ModelError> {
        Ok(())
    }

    fn error(&self, step: &Step<'_, f64>, _data: &(), candidate: &[f64]) -> f64 {
        assert_eq!(
            candidate.len(),
            step.parameters.len(),
            "GateFrequencyModel::error: candidate width differs from the number of parameters"
        );
        self.plan(step).eval(candidate)
    }

    /// Resolves the step once (interactions, couplings, endpoint frequencies) and then runs a
    /// tight loop over the rows; each result equals [`error`](Self::error) bit for bit.
    fn error_batch(
        &self,
        step: &Step<'_, f64>,
        _data: &(),
        candidates: &CandidateBatch<'_, f64>,
        out: &mut [f64],
    ) -> Result<(), ModelError> {
        if out.len() != candidates.len() {
            return Err(ModelError::new(format!(
                "GateFrequencyModel::error_batch: {} candidates but {} output slots",
                candidates.len(),
                out.len()
            )));
        }
        if candidates.width() != step.parameters.len() {
            return Err(ModelError::new(format!(
                "GateFrequencyModel::error_batch: candidates have {} columns but the step has {} \
                 parameters",
                candidates.width(),
                step.parameters.len()
            )));
        }
        let plan = self.plan(step);
        if candidates.width() == 0 {
            return Ok(());
        }
        for (slot, row) in out
            .iter_mut()
            .zip(candidates.as_flat().chunks_exact(candidates.width()))
        {
            *slot = plan.eval(row);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use proptest::prelude::*;
    use rand::Rng;

    use super::*;
    use crate::database::StepRecord;
    use crate::graph::EdgeType;

    // ----- helpers -------------------------------------------------------------------------

    fn id(i: u32) -> ElementId {
        ElementId::new(i)
    }

    fn step<'a>(
        graph: &'a ProcessorGraph,
        parameters: &'a [ElementId],
        constraints: &'a [(ElementId, f64)],
    ) -> Step<'a, f64> {
        Step {
            index: 0,
            centre: parameters.first().copied().unwrap_or(id(0)),
            parameters,
            constraints,
            graph,
        }
    }

    fn record(parameters: &[ElementId], option_indices: &[u32]) -> StepRecord {
        StepRecord {
            index: 0,
            pass: 0,
            subgoal: 0,
            centre: parameters[0],
            parameters: parameters.to_vec(),
            option_indices: option_indices.to_vec(),
            constraints: Vec::new(),
            error: 0.0,
            evaluations: 0,
            prepare_time: Duration::ZERO,
            error_time: Duration::ZERO,
            duration: Duration::ZERO,
        }
    }

    /// A database holding the given values, committed one element per step.
    fn database(graph: &ProcessorGraph, values: &[(ElementId, f64)]) -> ParameterDatabase<f64> {
        let mut db = ParameterDatabase::new(graph);
        for &(e, v) in values {
            db.commit_step(record(&[e], &[0]), vec![v]);
        }
        db
    }

    /// The goal of the worked example: nodes and engineered edges.
    fn goal_ids(graph: &ProcessorGraph) -> Vec<ElementId> {
        graph
            .ids()
            .filter(|&e| graph.kind(e) != ElementKind::Edge(EdgeType::Parasitic))
            .collect()
    }

    /// Sequential greedy assignment: in ascending ID order, each element takes the first
    /// option of least error given everything calibrated before it.
    fn greedy_database<M: ErrorModel<Value = f64, Data = ()>>(
        graph: &ProcessorGraph,
        model: &M,
        order: &[ElementId],
    ) -> ParameterDatabase<f64> {
        let mut db = ParameterDatabase::new(graph);
        for &e in order {
            let constraints: Vec<(ElementId, f64)> =
                db.iter().map(|(c, entry)| (c, entry.value)).collect();
            let options = model.options(e);
            let params = [e];
            let st = step(graph, &params, &constraints);
            let flat = options.clone();
            let mut out = vec![0.0; options.len()];
            model
                .error_batch(&st, &(), &CandidateBatch::new(&flat, 1), &mut out)
                .unwrap();
            let mut best = 0;
            for (i, &err) in out.iter().enumerate() {
                if err < out[best] {
                    best = i;
                }
            }
            db.commit_step(record(&params, &[best as u32]), vec![options[best]]);
        }
        db
    }

    // ----- linspace ------------------------------------------------------------------------

    #[test]
    fn linspace_edge_cases() {
        assert!(linspace(0.0, 1.0, 0).is_empty());
        assert_eq!(linspace(3.0, 9.0, 1), vec![3.0]);
        assert_eq!(linspace(3.0, 9.0, 2), vec![3.0, 9.0]);
        assert_eq!(linspace(2.0, 2.0, 4), vec![2.0; 4]);
        assert_eq!(linspace(0.0, 1.0, 3), vec![0.0, 0.5, 1.0]);
        assert_eq!(linspace(1.0, -1.0, 3), vec![1.0, 0.0, -1.0]);
    }

    proptest! {
        #[test]
        fn linspace_properties(lo in -100.0f64..100.0, span in 0.0f64..50.0, k in 1usize..200) {
            let hi = lo + span;
            let v = linspace(lo, hi, k);
            prop_assert_eq!(v.len(), k);
            prop_assert_eq!(v[0], lo);
            if k >= 2 {
                prop_assert_eq!(v[k - 1], hi);
            }
            for w in v.windows(2) {
                prop_assert!(w[0] <= w[1]);
            }
            for &x in &v {
                prop_assert!(x >= lo && x <= hi);
            }
        }
    }

    // ----- SeparableModel ------------------------------------------------------------------

    #[test]
    fn separable_options_and_targets_are_in_range() {
        let graph = ProcessorGraph::square_lattice(3, 4);
        let model = SeparableModel::new(&graph, 11, -2.0, 3.0, 5);
        for e in graph.ids() {
            let o = model.options(e);
            assert_eq!(o.len(), 11);
            assert_eq!(o[0], -2.0);
            assert_eq!(o[10], 3.0);
            let t = model.target(e);
            assert!((-2.0..=3.0).contains(&t));
        }
        // Targets are spread out, not all equal.
        let first = model.target(id(0));
        assert!(graph.ids().any(|e| model.target(e) != first));
    }

    #[test]
    fn separable_degenerate_range_and_single_option() {
        let graph = ProcessorGraph::square_lattice(2, 2);
        let model = SeparableModel::new(&graph, 1, 4.0, 4.0, 0);
        for e in graph.ids() {
            assert_eq!(model.target(e), 4.0);
            assert_eq!(model.optimum(e), 4.0);
            assert_eq!(model.options(e), vec![4.0]);
        }
    }

    #[test]
    #[should_panic(expected = "at least one option")]
    fn separable_rejects_zero_options() {
        SeparableModel::new(&ProcessorGraph::square_lattice(2, 2), 0, 0.0, 1.0, 0);
    }

    #[test]
    #[should_panic(expected = "lo <= hi")]
    fn separable_rejects_inverted_range() {
        SeparableModel::new(&ProcessorGraph::square_lattice(2, 2), 3, 1.0, 0.0, 0);
    }

    #[test]
    fn separable_is_deterministic_and_independent_of_the_graph() {
        let small = ProcessorGraph::square_lattice(3, 3);
        let large = ProcessorGraph::square_lattice(6, 6);
        let a = SeparableModel::new(&small, 8, 0.0, 1.0, 17);
        let b = SeparableModel::new(&small, 8, 0.0, 1.0, 17);
        let c = SeparableModel::new(&large, 8, 0.0, 1.0, 17);
        let d = SeparableModel::new(&small, 8, 0.0, 1.0, 18);
        let mut differs = false;
        for e in small.ids() {
            assert_eq!(a.target(e).to_bits(), b.target(e).to_bits());
            // Node IDs 0..9 exist in both graphs; edges of `small` are other elements in `large`
            // but the data depends on the ID only.
            assert_eq!(a.target(e).to_bits(), c.target(e).to_bits());
            differs |= a.target(e) != d.target(e);
        }
        assert!(differs, "a different seed must change the targets");
    }

    #[test]
    fn separable_optimum_is_the_nearest_option_first_on_ties() {
        let graph = ProcessorGraph::square_lattice(4, 4);
        let model = SeparableModel::new(&graph, 7, 0.0, 1.0, 3);
        for e in graph.ids() {
            let options = model.options(e);
            let t = model.target(e);
            let mut best = options[0];
            for &o in &options {
                if (o - t).abs() < (best - t).abs() {
                    best = o;
                }
            }
            assert_eq!(model.optimum(e), best);
            // The optimum minimises the single-parameter error.
            let params = [e];
            let st = step(&graph, &params, &[]);
            let best_err = model.error(&st, &(), &[model.optimum(e)]);
            for &o in &options {
                assert!(model.error(&st, &(), &[o]) >= best_err);
            }
        }
        // A target exactly between two options resolves to the lower one: with options
        // {0, 1} and a target of 0.5 (found by search) the first option wins.
        let tie = SeparableModel {
            options: vec![0.0, 1.0],
            targets: vec![0.5],
        };
        assert_eq!(tie.optimum(id(0)), 0.0);
    }

    #[test]
    fn separable_error_sums_squares_and_ignores_constraints() {
        let graph = ProcessorGraph::square_lattice(2, 2);
        let model = SeparableModel::new(&graph, 5, 0.0, 1.0, 9);
        let params = [id(1), id(3)];
        let none: [(ElementId, f64); 0] = [];
        let some = [(id(0), 0.25), (id(2), 0.75)];
        let cand = [0.3, 0.8];
        let expected = (0.3 - model.target(id(1))).powi(2) + (0.8 - model.target(id(3))).powi(2);
        let e1 = model.error(&step(&graph, &params, &none), &(), &cand);
        let e2 = model.error(&step(&graph, &params, &some), &(), &cand);
        assert!((e1 - expected).abs() < 1e-15);
        assert_eq!(e1.to_bits(), e2.to_bits());
        // Purity.
        assert_eq!(
            e1.to_bits(),
            model
                .error(&step(&graph, &params, &none), &(), &cand)
                .to_bits()
        );
        // The empty step has zero error.
        assert_eq!(model.error(&step(&graph, &[], &none), &(), &[]), 0.0);
    }

    #[test]
    fn separable_batch_agrees_with_error() {
        let graph = ProcessorGraph::square_lattice(2, 3);
        let model = SeparableModel::new(&graph, 5, 0.0, 1.0, 9);
        let params = [id(0), id(2), id(4)];
        let st = step(&graph, &params, &[]);
        let flat = [0.0, 0.5, 1.0, 0.25, 0.25, 0.75];
        let batch = CandidateBatch::new(&flat, 3);
        let mut out = [0.0; 2];
        model.error_batch(&st, &(), &batch, &mut out).unwrap();
        for (i, row) in batch.rows().enumerate() {
            assert_eq!(out[i].to_bits(), model.error(&st, &(), row).to_bits());
        }
    }

    #[test]
    fn separable_total_error_over_the_calibrated_elements() {
        let graph = ProcessorGraph::square_lattice(3, 3);
        let model = SeparableModel::new(&graph, 6, 0.0, 1.0, 21);
        let empty = ParameterDatabase::<f64>::new(&graph);
        assert_eq!(model.total_error(&empty), 0.0);
        let values: Vec<(ElementId, f64)> = [0u32, 4, 7, 12]
            .iter()
            .map(|&i| (id(i), 0.4 + 0.1 * f64::from(i % 3)))
            .collect();
        let db = database(&graph, &values);
        let expected: f64 = values
            .iter()
            .map(|&(e, v)| (v - model.target(e)).powi(2))
            .sum();
        assert!((model.total_error(&db) - expected).abs() < 1e-15);
        // The optimum assignment attains the least possible total.
        let best: Vec<(ElementId, f64)> = graph.ids().map(|e| (e, model.optimum(e))).collect();
        let best_db = database(&graph, &best);
        let other: Vec<(ElementId, f64)> = graph.ids().map(|e| (e, 0.5)).collect();
        assert!(model.total_error(&best_db) <= model.total_error(&database(&graph, &other)));
    }

    #[test]
    #[should_panic(expected = "finite")]
    fn separable_rejects_a_range_whose_span_overflows() {
        SeparableModel::new(&ProcessorGraph::square_lattice(2, 2), 5, -1e308, 1e308, 1);
    }

    #[test]
    #[should_panic(expected = "finite")]
    fn collision_rejects_a_range_whose_span_overflows() {
        CollisionModel::new(
            &ProcessorGraph::square_lattice(2, 2),
            5,
            -1e308,
            1e308,
            0.1,
            1,
            1,
        );
    }

    #[test]
    fn empty_sums_are_positive_zero() {
        let graph = ProcessorGraph::square_lattice(2, 2);
        let none: [(ElementId, f64); 0] = [];
        let st = step(&graph, &[], &none);
        let sep = SeparableModel::new(&graph, 4, 0.0, 1.0, 1);
        assert_eq!(sep.error(&st, &(), &[]).to_bits(), 0.0f64.to_bits());
        let empty = ParameterDatabase::<f64>::new(&graph);
        assert_eq!(sep.total_error(&empty).to_bits(), 0.0f64.to_bits());
        let col = CollisionModel::new(&graph, 4, 0.0, 1.0, 0.1, 1, 1);
        assert_eq!(col.error(&st, &(), &[]).to_bits(), 0.0f64.to_bits());
        let gate = GateFrequencyModel::new(&graph, 1);
        assert_eq!(gate.error(&st, &(), &[]).to_bits(), 0.0f64.to_bits());
    }

    // ----- CollisionModel ------------------------------------------------------------------

    #[test]
    fn collision_options_have_length_k_and_span_the_range() {
        let graph = ProcessorGraph::square_lattice(3, 3);
        for k in [1usize, 2, 7, 40] {
            let model = CollisionModel::new(&graph, k, -2.0, 3.0, 0.1, 1, 5);
            for e in graph.ids() {
                let o = model.options(e);
                assert_eq!(o.len(), k);
                assert_eq!(o[0].to_bits(), (-2.0f64).to_bits());
                if k >= 2 {
                    assert_eq!(o[k - 1].to_bits(), 3.0f64.to_bits());
                }
                assert!(o.windows(2).all(|w| w[0] < w[1]));
                assert!(o.iter().all(|v| (-2.0..=3.0).contains(v)));
            }
        }
    }

    #[test]
    fn collision_preferred_is_deterministic_and_independent_of_the_graph() {
        let small = ProcessorGraph::square_lattice(3, 3);
        let large = ProcessorGraph::square_lattice(6, 6);
        let a = CollisionModel::new(&small, 8, 5.0, 7.0, 0.1, 1, 17);
        let b = CollisionModel::new(&small, 8, 5.0, 7.0, 0.1, 1, 17);
        let c = CollisionModel::new(&large, 8, 5.0, 7.0, 0.2, 2, 17);
        let d = CollisionModel::new(&small, 8, 5.0, 7.0, 0.1, 1, 18);
        let mut differs_by_seed = false;
        for e in small.ids() {
            let p = a.preferred[e.index()];
            assert!((5.0..=7.0).contains(&p));
            assert_eq!(p.to_bits(), b.preferred[e.index()].to_bits());
            assert_eq!(p.to_bits(), c.preferred[e.index()].to_bits());
            differs_by_seed |= p != d.preferred[e.index()];
        }
        assert!(
            differs_by_seed,
            "a different seed must change the preferences"
        );
        // The elements do not all prefer the same frequency.
        let first = a.preferred[0];
        assert!(a.preferred.iter().any(|&p| p != first));
        // The preference is the documented function of (seed, purpose, id).
        for e in small.ids() {
            let expected: f64 = element_rng(17, PURPOSE_PREFERRED, e).random_range(5.0..=7.0);
            assert_eq!(a.preferred[e.index()].to_bits(), expected.to_bits());
        }
    }

    #[test]
    fn collision_error_is_pure_and_batch_agrees_bit_for_bit() {
        let graph = ProcessorGraph::square_lattice(3, 3);
        let model = CollisionModel::new(&graph, 9, 0.0, 1.0, 0.1, 2, 3);
        let params = [id(0), id(1), id(4)];
        let constraints = [(id(3), 0.5), (id(9), 0.8)];
        let st = step(&graph, &params, &constraints);
        // Rows with finite and infinite errors.
        let flat = [
            0.0, 0.25, 0.75, //
            0.5, 0.5, 0.5, //
            0.125, 0.875, 0.25, //
            0.52, 0.0, 1.0, //
            1.0, 0.0, 0.375,
        ];
        let batch = CandidateBatch::new(&flat, 3);
        let mut out = [0.0; 5];
        model.error_batch(&st, &(), &batch, &mut out).unwrap();
        for (i, row) in batch.rows().enumerate() {
            let e = model.error(&st, &(), row);
            assert_eq!(out[i].to_bits(), e.to_bits());
            assert_eq!(model.error(&st, &(), row).to_bits(), e.to_bits());
        }
        assert!(out.iter().any(|e| e.is_infinite()));
        assert!(out.iter().any(|e| e.is_finite()));
    }

    #[test]
    fn collision_forbids_close_interacting_pairs() {
        let graph = ProcessorGraph::square_lattice(3, 3);
        // Distance 1: node 4 interacts with its incident edges, nodes do not interact.
        let model = CollisionModel::new(&graph, 11, 0.0, 1.0, 0.1, 1, 0);
        let q4 = graph.lattice_node(1, 1).unwrap();
        let q5 = graph.lattice_node(1, 2).unwrap();
        let e45 = graph
            .edge_between(q4, q5, Some(EdgeType::Engineered))
            .unwrap();
        let params = [q4];
        let constraints = [(e45, 0.5)];
        let st = step(&graph, &params, &constraints);
        assert_eq!(model.error(&st, &(), &[0.5]), f64::INFINITY);
        assert_eq!(model.error(&st, &(), &[0.55]), f64::INFINITY);
        assert_eq!(model.error(&st, &(), &[0.45]), f64::INFINITY);
        assert!(model.error(&st, &(), &[0.7]).is_finite());
        assert!(model.error(&st, &(), &[0.3]).is_finite());
        // The boundary |df| == min_separation is allowed (strict inequality).
        // (Binary fractions, so the difference is exactly the separation.)
        let exact = CollisionModel::new(&graph, 9, 0.0, 1.0, 0.125, 1, 0);
        let tight = [(e45, 0.5)];
        let st = step(&graph, &params, &tight);
        assert!(exact.error(&st, &(), &[0.625]).is_finite());
        assert!(exact.error(&st, &(), &[0.375]).is_finite());
        assert_eq!(exact.error(&st, &(), &[0.5 + 0.0625]), f64::INFINITY);
        // Two nodes do not interact at distance 1: no collision even at equal values.
        let q3 = graph.lattice_node(1, 0).unwrap();
        let far = [(q3, 0.5)];
        assert!(model
            .error(&step(&graph, &params, &far), &(), &[0.5])
            .is_finite());
    }

    #[test]
    fn collision_checks_pairs_among_parameters() {
        let graph = ProcessorGraph::square_lattice(2, 2);
        let model = CollisionModel::new(&graph, 11, 0.0, 1.0, 0.1, 2, 0);
        let params = [id(0), id(1), id(2)];
        let none: [(ElementId, f64); 0] = [];
        let st = step(&graph, &params, &none);
        // All of a 2x2 lattice's nodes are within distance 2 of each other.
        assert_eq!(model.error(&st, &(), &[0.1, 0.5, 0.52]), f64::INFINITY);
        assert_eq!(model.error(&st, &(), &[0.5, 0.1, 0.52]), f64::INFINITY);
        assert_eq!(model.error(&st, &(), &[0.52, 0.5, 0.1]), f64::INFINITY);
        assert!(model.error(&st, &(), &[0.1, 0.5, 0.9]).is_finite());
    }

    #[test]
    fn collision_smooth_part_is_squared_distance_to_preferred() {
        let graph = ProcessorGraph::square_lattice(2, 2);
        let model = CollisionModel::new(&graph, 5, 0.0, 1.0, 0.0, 2, 4);
        let params = [id(0), id(3)];
        let none: [(ElementId, f64); 0] = [];
        let st = step(&graph, &params, &none);
        // With min_separation 0 nothing is ever forbidden, even at equal values.
        let e = model.error(&st, &(), &[0.5, 0.5]);
        let expected = (0.5 - model.preferred[0]).powi(2) + (0.5 - model.preferred[3]).powi(2);
        assert!((e - expected).abs() < 1e-15);
        assert!(model.preferred.iter().all(|p| (0.0..=1.0).contains(p)));
    }

    #[test]
    fn collision_collisions_on_hand_built_databases() {
        let graph = ProcessorGraph::square_lattice(3, 3);
        let model = CollisionModel::new(&graph, 11, 0.0, 1.0, 0.1, 1, 0);
        let q4 = graph.lattice_node(1, 1).unwrap();
        let q5 = graph.lattice_node(1, 2).unwrap();
        let q3 = graph.lattice_node(1, 0).unwrap();
        let e45 = graph
            .edge_between(q4, q5, Some(EdgeType::Engineered))
            .unwrap();
        let e34 = graph
            .edge_between(q3, q4, Some(EdgeType::Engineered))
            .unwrap();
        assert!(model.collisions(&ParameterDatabase::new(&graph)).is_empty());
        // q4 and e45 collide; q4 and e34 do not; q3 and q5 are far apart.
        let db = database(
            &graph,
            &[(q4, 0.30), (e45, 0.35), (e34, 0.9), (q5, 0.30), (q3, 0.3)],
        );
        let mut expected = vec![(q4, e45), (q5, e45)];
        expected.sort();
        assert_eq!(model.collisions(&db), expected);
        // Each pair once, ascending, whatever the order of calibration.
        let db2 = database(
            &graph,
            &[(q5, 0.30), (q3, 0.3), (e34, 0.9), (e45, 0.35), (q4, 0.30)],
        );
        assert_eq!(model.collisions(&db2), expected);
        // Uncalibrated elements do not collide with anything.
        let db3 = database(&graph, &[(q4, 0.30)]);
        assert!(model.collisions(&db3).is_empty());
        // Collision-free.
        let ok = database(&graph, &[(q4, 0.30), (e45, 0.6), (e34, 0.9)]);
        assert!(model.collisions(&ok).is_empty());
    }

    #[test]
    fn collision_error_and_collisions_agree() {
        // Committing values one at a time with `error` finite never creates a collision;
        // forcing a forbidden value does.
        let graph = ProcessorGraph::square_lattice(3, 3);
        let model = CollisionModel::new(&graph, 21, 0.0, 1.0, 0.08, 2, 6);
        let mut db = ParameterDatabase::new(&graph);
        let mut rng = StdRng::seed_from_u64(1);
        for e in graph.ids() {
            let constraints: Vec<(ElementId, f64)> =
                db.iter().map(|(c, entry)| (c, entry.value)).collect();
            let v = model.options(e)[rng.random_range(0..21)];
            let params = [e];
            let finite = model
                .error(&step(&graph, &params, &constraints), &(), &[v])
                .is_finite();
            db.commit_step(record(&params, &[0]), vec![v]);
            let involved = model.collisions(&db).iter().any(|&(a, b)| a == e || b == e);
            // `error` is finite exactly when the new element collides with nothing calibrated.
            assert_eq!(finite, !involved);
        }
    }

    #[test]
    fn required_options_formula() {
        let graph = ProcessorGraph::square_lattice(5, 5);
        // Interaction distance 0: nothing interacts.
        assert_eq!(
            CollisionModel::new(&graph, 3, 0.0, 1.0, 0.5, 0, 0).required_options(),
            1
        );
        // No separation required: nothing can collide.
        assert_eq!(
            CollisionModel::new(&graph, 3, 0.0, 1.0, 0.0, 2, 0).required_options(),
            1
        );
        // A separation so large that neighbours can forbid the whole range.
        assert_eq!(
            CollisionModel::new(&graph, 3, 0.0, 1.0, 0.5, 1, 0).required_options(),
            usize::MAX
        );
        assert_eq!(
            CollisionModel::new(&graph, 3, 1.0, 1.0, 0.1, 1, 0).required_options(),
            usize::MAX
        );
        // Δ = 16 at distance 1 on a lattice with an interior engineered edge, ρ = 0.02.
        let m = CollisionModel::new(&graph, 3, 0.0, 1.0, 0.01, 1, 0);
        let delta = graph
            .ids()
            .map(|e| m.neighbourhoods.within(e, 1).len() - 1)
            .max()
            .unwrap();
        assert_eq!(delta, 16);
        let expected = ((delta as f64 - 1.0) / (1.0 - delta as f64 * 0.02)).ceil() as usize + 2;
        assert_eq!(m.required_options(), expected);
        // The bound does not depend on `k`.
        assert_eq!(
            CollisionModel::new(&graph, 500, 0.0, 1.0, 0.01, 1, 0).required_options(),
            expected
        );
    }

    /// How the greedy test picks among the free options.
    type Pick = Box<dyn Fn(&[usize], &mut StdRng) -> usize>;

    /// Greedy assignment in a random order on `graph` with exactly `required_options()` options;
    /// asserts that a free option always exists, choosing among the free ones with `pick`.
    fn greedy_never_forced(
        graph: &ProcessorGraph,
        fill: f64,
        d: u32,
        seed: u64,
        pick: &dyn Fn(&[usize], &mut StdRng) -> usize,
    ) {
        // Choose the separation so that the worst-case neighbours cover the fraction `fill` of
        // the range: Δ · 2 · sep = fill.
        let balls = Neighbourhoods::build(graph, d);
        let delta = graph
            .ids()
            .map(|e| balls.within(e, d).len() - 1)
            .max()
            .unwrap_or(0);
        let sep = fill / (2.0 * delta as f64);
        let probe = CollisionModel::new(graph, 2, 0.0, 1.0, sep, d, seed);
        let k = probe.required_options();
        assert!(k < 5_000, "bound too large for this test: {k}");
        let model = CollisionModel::new(graph, k, 0.0, 1.0, sep, d, seed);
        assert_eq!(model.required_options(), k);
        let mut rng = StdRng::seed_from_u64(seed ^ 0x5eed);
        let mut order: Vec<ElementId> = graph.ids().collect();
        // Fisher-Yates.
        for i in (1..order.len()).rev() {
            order.swap(i, rng.random_range(0..=i));
        }
        let mut db = ParameterDatabase::new(graph);
        for e in order {
            let constraints: Vec<(ElementId, f64)> =
                db.iter().map(|(c, entry)| (c, entry.value)).collect();
            let options = model.options(e);
            let params = [e];
            let st = step(graph, &params, &constraints);
            let free: Vec<usize> = (0..k)
                .filter(|&i| model.error(&st, &(), &[options[i]]).is_finite())
                .collect();
            assert!(
                !free.is_empty(),
                "a collision was forced at {e} with k = {k}"
            );
            let chosen = free[pick(&free, &mut rng)];
            db.commit_step(record(&params, &[chosen as u32]), vec![options[chosen]]);
        }
        assert!(model.collisions(&db).is_empty());
        assert_eq!(db.len(), graph.len());
    }

    #[test]
    fn required_options_guarantee_a_collision_free_greedy_assignment() {
        let lattices = [(3, 3), (5, 5), (4, 7), (6, 6)];
        let picks: [Pick; 3] = [
            Box::new(|_, _| 0),
            Box::new(|free, _| free.len() - 1),
            Box::new(|free, rng| rng.random_range(0..free.len())),
        ];
        let mut cases = 0;
        for (rows, cols) in lattices {
            let graph = ProcessorGraph::square_lattice(rows, cols);
            for (fill, d) in [(0.05, 1), (0.5, 1), (0.9, 1), (0.5, 2), (0.9, 2)] {
                for seed in 0..3 {
                    for pick in &picks {
                        greedy_never_forced(&graph, fill, d, seed, pick.as_ref());
                        cases += 1;
                    }
                }
            }
        }
        assert_eq!(cases, 4 * 5 * 3 * 3);
    }

    #[test]
    fn too_few_options_can_force_a_collision() {
        // The bound is not vacuous: with two options and two neighbours sitting on them, nothing
        // is left for the new element.
        let graph = ProcessorGraph::square_lattice(3, 3);
        let model = CollisionModel::new(&graph, 2, 0.0, 1.0, 0.9, 1, 0);
        let p = graph.lattice_node(0, 0).unwrap();
        let incident = graph.incident(p).to_vec();
        let blockers = [(incident[0], 0.0), (incident[1], 1.0)];
        let params = [p];
        let st = step(&graph, &params, &blockers);
        for o in model.options(p) {
            assert_eq!(model.error(&st, &(), &[o]), f64::INFINITY);
        }
        // ... and the model reports that no number of options would be enough here.
        assert_eq!(model.required_options(), usize::MAX);
    }

    // ----- GateFrequencyModel --------------------------------------------------------------

    /// An independent, naive implementation of the model's definition, working on a map of known
    /// frequencies (parameters and constraints together) and the list of parameters whose own
    /// terms and pair terms are counted.
    fn naive_error(
        model: &GateFrequencyModel,
        graph: &ProcessorGraph,
        parameters: &[ElementId],
        constraints: &[(ElementId, f64)],
        row: &[f64],
    ) -> f64 {
        let known = |e: ElementId| -> Option<f64> {
            if let Some(j) = parameters.iter().position(|&p| p == e) {
                Some(row[j])
            } else {
                constraints.iter().find(|&&(c, _)| c == e).map(|&(_, f)| f)
            }
        };
        let w2 = GateFrequencyModel::CROSSTALK_WIDTH.powi(2);
        let mut total = 0.0;
        for (j, &p) in parameters.iter().enumerate() {
            let f = row[j];
            let qubits: Vec<ElementId> = match graph.endpoints(p) {
                None => vec![p],
                Some([a, b]) => vec![a, b],
            };
            for q in &qubits {
                for bump in model.bumps_of(*q) {
                    let d = f - bump.centre;
                    total += bump.strength * bump.width_sq / (d * d + bump.width_sq);
                }
            }
            if graph.endpoints(p).is_some() {
                let freqs: Vec<f64> = qubits.iter().filter_map(|&q| known(q)).collect();
                if !freqs.is_empty() {
                    let reference = freqs.iter().sum::<f64>() / freqs.len() as f64;
                    total += GateFrequencyModel::DETUNING_WEIGHT * (f - reference).powi(2);
                }
            }
        }
        // Pairs: (parameter, parameter) once, (parameter, constraint).
        let balls = Neighbourhoods::build(graph, 2);
        for (j, &p) in parameters.iter().enumerate() {
            for (l, &q) in parameters.iter().enumerate() {
                if l > j && balls.is_within(p, q, 2) {
                    let d = row[j] - row[l];
                    total += model.coupling(p, q) * w2 / (d * d + w2);
                }
            }
            for &(c, fc) in constraints {
                if balls.is_within(p, c, 2) {
                    let d = row[j] - fc;
                    total += model.coupling(p, c) * w2 / (d * d + w2);
                }
            }
        }
        total
    }

    fn random_gate_step(
        graph: &ProcessorGraph,
        rng: &mut StdRng,
    ) -> (Vec<ElementId>, Vec<(ElementId, f64)>) {
        let goal = goal_ids(graph);
        let mut parameters = Vec::new();
        let mut constraints = Vec::new();
        for &e in &goal {
            match rng.random_range(0..6) {
                0 => parameters.push(e),
                1 | 2 => constraints.push((e, rng.random_range(5.0..=7.0))),
                _ => {}
            }
        }
        // Parameters in a scrambled order: the model must not rely on any ordering.
        for i in (1..parameters.len()).rev() {
            parameters.swap(i, rng.random_range(0..=i));
        }
        parameters.truncate(6);
        (parameters, constraints)
    }

    #[test]
    fn gate_options_are_100_frequencies_for_nodes_and_engineered_edges() {
        let graph = ProcessorGraph::square_lattice(3, 3);
        let model = GateFrequencyModel::new(&graph, 1);
        for e in goal_ids(&graph) {
            let o = model.options(e);
            assert_eq!(o.len(), 100);
            assert_eq!(o[0], 5.0);
            assert_eq!(o[99], 7.0);
            assert!(o.windows(2).all(|w| w[0] < w[1]));
        }
    }

    #[test]
    #[should_panic(expected = "only nodes and engineered edges")]
    fn gate_options_reject_parasitic_edges() {
        let graph = ProcessorGraph::square_lattice(3, 3);
        let model = GateFrequencyModel::new(&graph, 1);
        let parasitic = graph
            .edges()
            .find(|&e| graph.kind(e) == ElementKind::Edge(EdgeType::Parasitic))
            .unwrap();
        model.options(parasitic);
    }

    #[test]
    #[should_panic(expected = "only nodes and engineered edges")]
    fn gate_options_reject_custom_edges() {
        let mut builder = ProcessorGraph::builder();
        let a = builder.add_node("a").unwrap();
        let b = builder.add_node("b").unwrap();
        let e = builder.add_edge(a, b, EdgeType::Custom(7)).unwrap();
        let graph = builder.build().unwrap();
        GateFrequencyModel::new(&graph, 0).options(e);
    }

    #[test]
    fn gate_data_is_deterministic_and_independent_of_the_graph() {
        let small = ProcessorGraph::square_lattice(3, 3);
        let large = ProcessorGraph::square_lattice(6, 6);
        let a = GateFrequencyModel::new(&small, 99);
        let b = GateFrequencyModel::new(&small, 99);
        let c = GateFrequencyModel::new(&large, 99);
        let d = GateFrequencyModel::new(&small, 100);
        for n in small.nodes() {
            for (x, y) in a.bumps_of(n).iter().zip(b.bumps_of(n)) {
                assert_eq!(x.centre.to_bits(), y.centre.to_bits());
                assert_eq!(x.strength.to_bits(), y.strength.to_bits());
            }
            // Node IDs 0..9 are nodes in both graphs.
            for (x, y) in a.bumps_of(n).iter().zip(c.bumps_of(n)) {
                assert_eq!(x.centre.to_bits(), y.centre.to_bits());
                assert_eq!(x.width_sq.to_bits(), y.width_sq.to_bits());
                assert_eq!(x.strength.to_bits(), y.strength.to_bits());
            }
        }
        assert_ne!(a.bumps_of(id(0))[0].centre, d.bumps_of(id(0))[0].centre);
        for x in 0..20 {
            for y in 0..20 {
                if x != y {
                    assert_eq!(
                        a.coupling(id(x), id(y)).to_bits(),
                        c.coupling(id(x), id(y)).to_bits()
                    );
                    assert_eq!(
                        a.coupling(id(x), id(y)).to_bits(),
                        a.coupling(id(y), id(x)).to_bits()
                    );
                    assert!((0.2..=1.0).contains(&a.coupling(id(x), id(y))));
                }
            }
        }
        assert_ne!(a.coupling(id(0), id(1)), d.coupling(id(0), id(1)));
    }

    #[test]
    fn gate_defects_have_the_documented_ranges() {
        let graph = ProcessorGraph::square_lattice(5, 5);
        let model = GateFrequencyModel::new(&graph, 4);
        for n in graph.nodes() {
            let bumps = model.bumps_of(n);
            assert_eq!(bumps.len(), 3);
            for b in bumps {
                assert!((5.0..=7.0).contains(&b.centre));
                let w = b.width_sq.sqrt();
                assert!((0.005 - 1e-12..=0.020 + 1e-12).contains(&w));
                assert!((0.5..=2.0).contains(&b.strength));
            }
        }
        for e in graph.edges() {
            let [a, b] = graph.endpoints(e).unwrap();
            assert_eq!(model.bumps_of(e).len(), 6);
            assert_eq!(
                model.bumps_of(e)[..3]
                    .iter()
                    .map(|x| x.centre)
                    .collect::<Vec<_>>(),
                model
                    .bumps_of(a)
                    .iter()
                    .map(|x| x.centre)
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                model.bumps_of(e)[3..]
                    .iter()
                    .map(|x| x.centre)
                    .collect::<Vec<_>>(),
                model
                    .bumps_of(b)
                    .iter()
                    .map(|x| x.centre)
                    .collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn gate_error_matches_the_naive_definition() {
        let graph = ProcessorGraph::square_lattice(4, 4);
        let model = GateFrequencyModel::new(&graph, 8);
        let mut rng = StdRng::seed_from_u64(77);
        let mut nonzero = 0;
        for _ in 0..60 {
            let (parameters, constraints) = random_gate_step(&graph, &mut rng);
            let mut constraints = constraints;
            constraints.sort_by_key(|&(e, _)| e);
            let st = step(&graph, &parameters, &constraints);
            for _ in 0..4 {
                let row: Vec<f64> = parameters
                    .iter()
                    .map(|_| rng.random_range(5.0..=7.0))
                    .collect();
                let got = model.error(&st, &(), &row);
                let want = naive_error(&model, &graph, &parameters, &constraints, &row);
                assert!(
                    (got - want).abs() <= 1e-9 * (1.0 + want.abs()),
                    "got {got}, want {want}"
                );
                nonzero += usize::from(got > 0.0);
            }
        }
        assert!(nonzero > 100);
    }

    #[test]
    fn gate_error_and_batch_agree_bit_for_bit() {
        let graph = ProcessorGraph::square_lattice(5, 5);
        let model = GateFrequencyModel::new(&graph, 3);
        let mut rng = StdRng::seed_from_u64(5);
        for _ in 0..40 {
            let (parameters, mut constraints) = random_gate_step(&graph, &mut rng);
            constraints.sort_by_key(|&(e, _)| e);
            if parameters.is_empty() {
                continue;
            }
            let st = step(&graph, &parameters, &constraints);
            let rows = rng.random_range(1..40);
            let width = parameters.len();
            let flat: Vec<f64> = (0..rows * width)
                .map(|_| rng.random_range(5.0..=7.0))
                .collect();
            let batch = CandidateBatch::new(&flat, width);
            let mut out = vec![f64::NAN; rows];
            model.error_batch(&st, &(), &batch, &mut out).unwrap();
            for (i, row) in batch.rows().enumerate() {
                assert_eq!(out[i].to_bits(), model.error(&st, &(), row).to_bits());
                // Purity: again the same bits.
                assert_eq!(out[i].to_bits(), model.error(&st, &(), row).to_bits());
            }
            let mut again = vec![0.0; rows];
            model.error_batch(&st, &(), &batch, &mut again).unwrap();
            assert_eq!(
                out.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                again.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn gate_error_does_not_depend_on_the_order_of_parameters() {
        let graph = ProcessorGraph::square_lattice(4, 4);
        let model = GateFrequencyModel::new(&graph, 12);
        let params = [id(5), id(16 + 3), id(2), id(9)];
        let permuted = [id(9), id(2), id(5), id(16 + 3)];
        let row = [5.5, 6.1, 6.9, 5.2];
        let permuted_row = [5.2, 6.9, 5.5, 6.1];
        let constraints = [(id(0), 5.8), (id(6), 6.4), (id(16 + 8), 5.1)];
        let a = model.error(&step(&graph, &params, &constraints), &(), &row);
        let b = model.error(&step(&graph, &permuted, &constraints), &(), &permuted_row);
        assert!((a - b).abs() < 1e-12, "{a} vs {b}");
    }

    #[test]
    fn gate_batch_validates_its_shapes() {
        let graph = ProcessorGraph::square_lattice(2, 2);
        let model = GateFrequencyModel::new(&graph, 1);
        let params = [id(0), id(1)];
        let st = step(&graph, &params, &[]);
        let flat = [5.0, 6.0, 5.5, 6.5];
        let mut short = [0.0; 1];
        assert!(model
            .error_batch(&st, &(), &CandidateBatch::new(&flat, 2), &mut short)
            .is_err());
        let mut out = [0.0; 4];
        assert!(model
            .error_batch(&st, &(), &CandidateBatch::new(&flat, 1), &mut out)
            .is_err());
        // No parameters: an empty batch is fine.
        let none = step(&graph, &[], &[]);
        model
            .error_batch(&none, &(), &CandidateBatch::new(&[], 0), &mut [])
            .unwrap();
        assert_eq!(model.error(&none, &(), &[]), 0.0);
    }

    #[test]
    fn gate_exact_collision_is_penalised_more_than_a_far_frequency() {
        let graph = ProcessorGraph::square_lattice(3, 3);
        let model = GateFrequencyModel::new(&graph, 6);
        let a = graph.lattice_node(1, 1).unwrap();
        let b = graph.lattice_node(1, 2).unwrap();
        let params = [a];
        let fb = 6.2;
        let constraints = [(b, fb)];
        let without = step(&graph, &params, &[]);
        let with = step(&graph, &params, &constraints);
        let excess = |f: f64| model.error(&with, &(), &[f]) - model.error(&without, &(), &[f]);
        // Coupling in [0.2, 1]: the crosstalk at an exact collision is the coupling itself.
        let g = model.coupling(a, b);
        assert!((excess(fb) - g).abs() < 1e-12);
        assert!(excess(fb) > 10.0 * excess(fb + 0.8));
        assert!(excess(fb) > 10.0 * excess(fb - 0.6));
        assert!(excess(fb) > excess(fb + 0.04));
        assert!(excess(fb + 0.04) > excess(fb + 0.2));
    }

    #[test]
    fn gate_crosstalk_has_a_finite_range() {
        // On a chain of qubits, one far beyond the interaction distance adds nothing.
        let graph = ProcessorGraph::square_lattice(1, 8);
        let model = GateFrequencyModel::new(&graph, 6);
        let a = graph.lattice_node(0, 0).unwrap();
        let near = graph.lattice_node(0, 1).unwrap();
        let far = graph.lattice_node(0, 6).unwrap();
        let balls = Neighbourhoods::build(&graph, 2);
        assert!(balls.is_within(a, near, 2));
        assert!(!balls.is_within(a, far, 2));
        let params = [a];
        let f = 6.0;
        let none = model.error(&step(&graph, &params, &[]), &(), &[f]);
        let with_far = [(far, f)];
        let with_near = [(near, f)];
        assert_eq!(
            model
                .error(&step(&graph, &params, &with_far), &(), &[f])
                .to_bits(),
            none.to_bits()
        );
        assert!(model.error(&step(&graph, &params, &with_near), &(), &[f]) > none);
    }

    #[test]
    fn gate_crosstalk_among_parameters_is_counted_once() {
        let graph = ProcessorGraph::square_lattice(3, 3);
        let model = GateFrequencyModel::new(&graph, 6);
        let a = graph.lattice_node(1, 1).unwrap();
        let b = graph.lattice_node(1, 2).unwrap();
        let pair = [a, b];
        let single_a = [a];
        let single_b = [b];
        let (fa, fb) = (6.0, 6.01);
        let both = model.error(&step(&graph, &pair, &[]), &(), &[fa, fb]);
        let alone = model.error(&step(&graph, &single_a, &[]), &(), &[fa])
            + model.error(&step(&graph, &single_b, &[]), &(), &[fb]);
        let w2 = GateFrequencyModel::CROSSTALK_WIDTH.powi(2);
        let x = model.coupling(a, b) * w2 / ((fa - fb).powi(2) + w2);
        assert!((both - alone - x).abs() < 1e-12);
    }

    #[test]
    fn gate_relaxation_defects_raise_the_error_near_their_centres() {
        let graph = ProcessorGraph::square_lattice(4, 4);
        let model = GateFrequencyModel::new(&graph, 2);
        for n in graph.nodes() {
            let params = [n];
            let st = step(&graph, &params, &[]);
            for bump in model.bumps_of(n) {
                let at_centre = model.error(&st, &(), &[bump.centre]);
                assert!(at_centre >= bump.strength - 1e-12);
                // Half a width away the bump has lost at most half its height, and a hundred
                // widths away almost all of it.
                let w = bump.width_sq.sqrt();
                let near = model.error(&st, &(), &[bump.centre + w]);
                assert!(near < at_centre);
                let own_far = bump.at(bump.centre + 100.0 * w);
                assert!(own_far < 1e-3 * bump.strength);
            }
            // The error landscape over the options is not flat and is never negative.
            let options = model.options(n);
            let errs: Vec<f64> = options
                .iter()
                .map(|&f| model.error(&st, &(), &[f]))
                .collect();
            assert!(errs.iter().all(|&e| e >= 0.0));
            let (lo, hi) = errs
                .iter()
                .fold((f64::MAX, f64::MIN), |(l, h), &e| (l.min(e), h.max(e)));
            assert!(hi > lo);
        }
        // A two-qubit gate pays for the defects of both qubits.
        let e = graph.edges().next().unwrap();
        let [a, b] = graph.endpoints(e).unwrap();
        let params = [e];
        let st = step(&graph, &params, &[]);
        let f = 6.3;
        let expected = relaxation(model.bumps_of(a), f) + relaxation(model.bumps_of(b), f);
        assert!((model.error(&st, &(), &[f]) - expected).abs() < 1e-12);
    }

    #[test]
    fn gate_detuning_grows_with_distance_from_the_mean_of_the_qubits() {
        for (r, near, far) in [(6.0, 6.1, 6.7), (5.5, 5.4, 5.0)] {
            assert!(detuning_cost(r, r) == 0.0);
            assert!(detuning_cost(near, r) > 0.0);
            assert!(detuning_cost(far, r) > detuning_cost(near, r));
        }
        assert_eq!(detuning_cost(6.2, 6.0), detuning_cost(5.8, 6.0));
        assert_eq!(reference_frequency(Some(5.0), Some(7.0)), Some(6.0));
        assert_eq!(reference_frequency(Some(5.0), None), Some(5.0));
        assert_eq!(reference_frequency(None, Some(7.0)), Some(7.0));
        assert_eq!(reference_frequency(None, None), None);

        // Through the model: with both endpoints known, moving the gate away from their mean
        // costs more once the (tiny, far-away) other terms are removed by differencing.
        let graph = ProcessorGraph::square_lattice(2, 2);
        let model = GateFrequencyModel::new(&graph, 3);
        let e = graph.edges().next().unwrap();
        let [a, b] = graph.endpoints(e).unwrap();
        let params = [e];
        let known = [(a, 5.5), (b, 6.5)];
        let st = step(&graph, &params, &known);
        let unknown = step(&graph, &params, &[]);
        // Difference between the known-endpoint and unknown-endpoint steps: detuning plus the
        // crosstalk with the endpoints.
        let w2 = GateFrequencyModel::CROSSTALK_WIDTH.powi(2);
        let extra = |f: f64| {
            let xt = |q: ElementId, fq: f64| model.coupling(e, q) * w2 / ((f - fq).powi(2) + w2);
            model.error(&st, &(), &[f]) - model.error(&unknown, &(), &[f]) - xt(a, 5.5) - xt(b, 6.5)
        };
        for f in [5.2f64, 5.9, 6.0, 6.3, 6.8] {
            let want = GateFrequencyModel::DETUNING_WEIGHT * (f - 6.0).powi(2);
            assert!((extra(f) - want).abs() < 1e-9, "f = {f}");
        }
        // Only one endpoint known: the reference is that endpoint.
        let one = [(a, 5.5)];
        let st1 = step(&graph, &params, &one);
        let xt = |f: f64| model.coupling(e, a) * w2 / ((f - 5.5).powi(2) + w2);
        let f = 6.4;
        let d = model.error(&st1, &(), &[f]) - model.error(&unknown, &(), &[f]) - xt(f);
        assert!((d - GateFrequencyModel::DETUNING_WEIGHT * (f - 5.5).powi(2)).abs() < 1e-9);
    }

    #[test]
    fn gate_mean_error_is_the_naive_total_over_the_calibrated_elements() {
        let graph = ProcessorGraph::square_lattice(4, 4);
        let model = GateFrequencyModel::new(&graph, 15);
        let empty = ParameterDatabase::<f64>::new(&graph);
        assert_eq!(model.mean_error(&graph, &empty), 0.0);
        assert_eq!(model.baseline_mean_error(&graph, &empty), 0.0);

        let mut rng = StdRng::seed_from_u64(3);
        let mut values: Vec<(ElementId, f64)> = Vec::new();
        for e in goal_ids(&graph) {
            if rng.random_range(0..3) != 0 {
                values.push((e, rng.random_range(5.0..=7.0)));
            }
        }
        let db = database(&graph, &values);
        let ids: Vec<ElementId> = values.iter().map(|&(e, _)| e).collect();
        let row: Vec<f64> = values.iter().map(|&(_, v)| v).collect();
        let want = naive_error(&model, &graph, &ids, &[], &row) / ids.len() as f64;
        let got = model.mean_error(&graph, &db);
        assert!((got - want).abs() <= 1e-9 * (1.0 + want), "{got} vs {want}");

        let base_row = vec![GateFrequencyModel::BASELINE_FREQUENCY; ids.len()];
        let want = naive_error(&model, &graph, &ids, &[], &base_row) / ids.len() as f64;
        let got = model.baseline_mean_error(&graph, &db);
        assert!((got - want).abs() <= 1e-9 * (1.0 + want), "{got} vs {want}");

        // The baseline ignores the values stored in the database.
        let other: Vec<(ElementId, f64)> = values.iter().map(|&(e, _)| (e, 5.0)).collect();
        assert_eq!(
            model
                .baseline_mean_error(&graph, &database(&graph, &other))
                .to_bits(),
            model.baseline_mean_error(&graph, &db).to_bits()
        );
    }

    #[test]
    fn gate_greedy_assignment_beats_the_baseline_clearly() {
        let graph = ProcessorGraph::square_lattice(6, 6);
        for seed in 0..6 {
            let model = GateFrequencyModel::new(&graph, seed);
            let db = greedy_database(&graph, &model, &goal_ids(&graph));
            let greedy = model.mean_error(&graph, &db);
            let baseline = model.baseline_mean_error(&graph, &db);
            assert_eq!(db.len(), 36 + 60);
            assert!(
                greedy < 0.25 * baseline,
                "seed {seed}: greedy {greedy} vs baseline {baseline}"
            );
            assert!(greedy > 0.0);
        }
    }

    #[test]
    fn gate_greedy_assignment_is_deterministic() {
        let graph = ProcessorGraph::square_lattice(4, 4);
        let run = || {
            let model = GateFrequencyModel::new(&graph, 31);
            let db = greedy_database(&graph, &model, &goal_ids(&graph));
            db.iter()
                .map(|(_, e)| e.value.to_bits())
                .collect::<Vec<_>>()
        };
        assert_eq!(run(), run());
    }

    proptest! {
        #[test]
        fn gate_batch_matches_error_on_random_rows(
            seed in 0u64..1000,
            rows in 1usize..20,
            width in 1usize..5,
            values in proptest::collection::vec(5.0f64..7.0, 100),
        ) {
            let graph = ProcessorGraph::square_lattice(4, 4);
            let model = GateFrequencyModel::new(&graph, seed);
            let parameters: Vec<ElementId> = (0..width as u32).map(|i| id(i * 3 % 16)).collect();
            let mut unique = parameters.clone();
            unique.sort();
            unique.dedup();
            prop_assume!(unique.len() == parameters.len());
            let constraints = [(id(1), 6.1), (id(16 + 2), 5.3)];
            let constraints: Vec<(ElementId, f64)> = constraints
                .into_iter()
                .filter(|(e, _)| !parameters.contains(e))
                .collect();
            let st = step(&graph, &parameters, &constraints);
            let flat: Vec<f64> = values.iter().copied().cycle().take(rows * width).collect();
            let batch = CandidateBatch::new(&flat, width);
            let mut out = vec![0.0; rows];
            model.error_batch(&st, &(), &batch, &mut out).unwrap();
            for (i, row) in batch.rows().enumerate() {
                prop_assert_eq!(out[i].to_bits(), model.error(&st, &(), row).to_bits());
            }
        }
    }
}
