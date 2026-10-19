//! Evaluation caches (ARCHITECTURE §7).
//!
//! | Cache | Scope | Default |
//! |---|---|---|
//! | Option table | one run | always on (engine) |
//! | Within-step de-duplication ([`StepDedup`]) | one step | always on |
//! | Cross-step cache ([`CrossStepCache`]) | one run | **off**: [`CachePolicy::AcrossSteps`] is the user's assertion that the model's error is a pure function of the parameters, their values and the constraint values |
//!
//! Cache hits never change results, only how many model evaluations are run, so cached and
//! uncached runs give identical databases.

use std::borrow::Borrow;
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use crate::graph::ElementId;

/// Which caches the engine uses.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum CachePolicy {
    /// Only the always-on within-step de-duplication.
    #[default]
    StepOnly,
    /// Also keep a bounded cross-step LRU cache of at most `capacity` entries.
    AcrossSteps {
        /// Maximum number of cached candidate errors.
        capacity: usize,
    },
}

impl CachePolicy {
    /// A reasonable cross-step capacity (`2^20` entries).
    ///
    /// Each entry costs roughly 150 bytes for short candidates (the key is held both in the
    /// LRU list and in the index map, plus one small allocation for the candidate), so the
    /// default may take on the order of 150 MB when full. This is an estimate, not a
    /// measurement.
    pub const DEFAULT_CAPACITY: usize = 1 << 20;

    /// `AcrossSteps` with [`DEFAULT_CAPACITY`](Self::DEFAULT_CAPACITY).
    pub fn across_steps() -> Self {
        Self::AcrossSteps {
            capacity: Self::DEFAULT_CAPACITY,
        }
    }
}

/// A fast, deterministic hasher for maps keyed by short index vectors.
///
/// It is used for in-memory maps only (nothing it computes is persisted or compared across
/// runs), so it does not have to resist adversarial input.
#[derive(Default)]
struct FastHasher(u64);

type FastBuild = BuildHasherDefault<FastHasher>;

impl FastHasher {
    const K: u64 = 0x517c_c1b7_2722_0a95;

    #[inline]
    fn add(&mut self, word: u64) {
        self.0 = (self.0.rotate_left(5) ^ word).wrapping_mul(Self::K);
    }
}

impl Hasher for FastHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut chunks = bytes.chunks_exact(8);
        for chunk in &mut chunks {
            let mut word = [0u8; 8];
            word.copy_from_slice(chunk);
            self.add(u64::from_le_bytes(word));
        }
        let rest = chunks.remainder();
        if !rest.is_empty() {
            let mut word = [0u8; 8];
            word[..rest.len()].copy_from_slice(rest);
            self.add(u64::from_le_bytes(word) ^ ((rest.len() as u64) << 56));
        }
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.add(u64::from(i));
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.add(i);
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.add(i as u64);
    }

    #[inline]
    fn finish(&self) -> u64 {
        mix64(self.0)
    }
}

/// The splitmix64 finalizer: a bijection of `u64` with full avalanche.
#[inline]
const fn mix64(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// Within-step de-duplication: the same candidate within one step, with the same data, has
/// the same error. Coordinate descent revisits its current point every sweep.
///
/// The map is exact (candidates are compared in full) and works for any width, including 0.
#[derive(Debug)]
pub struct StepDedup {
    width: usize,
    map: HashMap<Box<[u32]>, f64, FastBuild>,
}

impl StepDedup {
    /// A de-duplicator for candidates of `width` indices.
    pub fn new(width: usize) -> Self {
        Self {
            width,
            map: HashMap::default(),
        }
    }

    /// Indices per candidate.
    pub fn width(&self) -> usize {
        self.width
    }

    /// The remembered error of `candidate`, if any.
    pub fn lookup(&self, candidate: &[u32]) -> Option<f64> {
        self.map.get(candidate).copied()
    }

    /// Remembers an error. An existing entry for the same candidate is overwritten.
    pub fn insert(&mut self, candidate: &[u32], error: f64) {
        debug_assert_eq!(candidate.len(), self.width, "candidate width mismatch");
        match self.map.get_mut(candidate) {
            Some(slot) => *slot = error,
            None => {
                self.map.insert(candidate.into(), error);
            }
        }
    }

    /// Number of remembered candidates.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Whether nothing is remembered.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Identifies the context of a step for the cross-step cache: the parameter IDs and the
/// constraint IDs with their option indices. Two steps with equal keys and equal candidates
/// have equal errors if the model is pure.
///
/// The key is a 128-bit hash made of two independent 64-bit streams with fixed constants;
/// it is identical on every run and platform (it does not use `std`'s randomized hashers).
/// The final combination of the two streams is a bijection of their 128-bit state, so no
/// state bits are lost there; distinct contexts collide with probability about `2^-128`
/// per pair, assuming the streams behave like random functions (this is a fast
/// non-cryptographic hash, so that assumption is not proven).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct StepKey(pub u128);

impl StepKey {
    /// Hashes a step context.
    ///
    /// Both lists are length-prefixed, so `([1], [])` differs from `([], [(1, 0)])`. The
    /// order of `constraints` matters: the engine passes them sorted by ID.
    pub fn new(parameters: &[ElementId], constraints: &[(ElementId, u32)]) -> Self {
        let mut h = KeyHasher::new();
        h.absorb(parameters.len() as u64);
        for id in parameters {
            h.absorb(u64::from(id.index() as u32));
        }
        h.absorb(constraints.len() as u64);
        for &(id, option) in constraints {
            h.absorb((u64::from(id.index() as u32) << 32) | u64::from(option));
        }
        h.finish()
    }
}

/// Constant folded into the second half of the key, so that it differs from the first.
const KEY_FINISH_SALT: u64 = 0x2545_f491_4f6c_dd1d;

/// Two independent 64-bit mixing streams. Each `absorb` step is a bijection in both the
/// state and the word, so contexts that differ in a single word never collide in a stream.
struct KeyHasher {
    a: u64,
    b: u64,
}

impl KeyHasher {
    fn new() -> Self {
        Self {
            a: 0x243f_6a88_85a3_08d3,
            b: 0x1319_8a2e_0370_7344,
        }
    }

    fn absorb(&mut self, word: u64) {
        // Stream A: multiply, xor-shift, multiply.
        let mut x = (self.a ^ word).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        x ^= x >> 29;
        self.a = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
        // Stream B: add, multiply, rotate, xor-shift, multiply (different constants).
        let mut y = self
            .b
            .wrapping_add(word)
            .wrapping_mul(0xd6e8_feb8_6659_fd93);
        y = y.rotate_left(31);
        y ^= y >> 17;
        self.b = y.wrapping_mul(0xa076_1d64_78bd_642f);
    }

    /// Combines the two streams. `hi` is a bijection of `a`, and `lo` a bijection of `b` for
    /// a given `hi`, so the whole map is a bijection of the 128-bit state: it loses no bits.
    fn finish(self) -> StepKey {
        let hi = mix64(self.a);
        let lo = mix64(self.b ^ hi ^ KEY_FINISH_SALT);
        StepKey((u128::from(hi) << 64) | u128::from(lo))
    }
}

/// Counters of a [`CrossStepCache`].
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct CacheStats {
    /// Lookups that found an entry.
    pub hits: u64,
    /// Lookups that did not.
    pub misses: u64,
    /// Entries dropped to stay within capacity.
    pub evictions: u64,
    /// Entries currently stored.
    pub len: usize,
}

/// The key of a cache entry: the step context and the candidate's option indices.
#[derive(Clone)]
struct Key {
    step: StepKey,
    candidate: Arc<[u32]>,
}

/// Borrowed view of a key, so that lookups need no allocation.
trait KeyView {
    fn step(&self) -> StepKey;
    fn candidate(&self) -> &[u32];
}

impl KeyView for Key {
    fn step(&self) -> StepKey {
        self.step
    }
    fn candidate(&self) -> &[u32] {
        &self.candidate
    }
}

struct Query<'a> {
    step: StepKey,
    candidate: &'a [u32],
}

impl KeyView for Query<'_> {
    fn step(&self) -> StepKey {
        self.step
    }
    fn candidate(&self) -> &[u32] {
        self.candidate
    }
}

impl Hash for dyn KeyView + '_ {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.step().0 as u64);
        state.write_u64((self.step().0 >> 64) as u64);
        self.candidate().hash(state);
    }
}

impl PartialEq for dyn KeyView + '_ {
    fn eq(&self, other: &Self) -> bool {
        self.step() == other.step() && self.candidate() == other.candidate()
    }
}

impl Eq for dyn KeyView + '_ {}

impl Hash for Key {
    fn hash<H: Hasher>(&self, state: &mut H) {
        <dyn KeyView as Hash>::hash(self, state);
    }
}

impl PartialEq for Key {
    fn eq(&self, other: &Self) -> bool {
        <dyn KeyView as PartialEq>::eq(self, other)
    }
}

impl Eq for Key {}

impl<'a> Borrow<dyn KeyView + 'a> for Key {
    fn borrow(&self) -> &(dyn KeyView + 'a) {
        self
    }
}

/// Marks "no neighbour" in the intrusive list.
const NIL: usize = usize::MAX;

struct Slot {
    key: Key,
    value: f64,
    /// Neighbour towards the most recently used end.
    prev: usize,
    /// Neighbour towards the least recently used end.
    next: usize,
}

/// A single-threaded LRU map: a hash map from key to slot index plus a doubly linked list
/// of slots threaded through a `Vec`. Every operation is O(1).
struct Lru {
    capacity: usize,
    map: HashMap<Key, usize, FastBuild>,
    slots: Vec<Slot>,
    /// Most recently used slot.
    head: usize,
    /// Least recently used slot.
    tail: usize,
}

impl Lru {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            map: HashMap::default(),
            slots: Vec::new(),
            head: NIL,
            tail: NIL,
        }
    }

    fn len(&self) -> usize {
        self.slots.len()
    }

    fn unlink(&mut self, i: usize) {
        let (prev, next) = (self.slots[i].prev, self.slots[i].next);
        if prev == NIL {
            self.head = next;
        } else {
            self.slots[prev].next = next;
        }
        if next == NIL {
            self.tail = prev;
        } else {
            self.slots[next].prev = prev;
        }
    }

    fn push_front(&mut self, i: usize) {
        self.slots[i].prev = NIL;
        self.slots[i].next = self.head;
        if self.head != NIL {
            self.slots[self.head].prev = i;
        }
        self.head = i;
        if self.tail == NIL {
            self.tail = i;
        }
    }

    fn touch(&mut self, i: usize) {
        if self.head != i {
            self.unlink(i);
            self.push_front(i);
        }
    }

    fn get(&mut self, step: StepKey, candidate: &[u32]) -> Option<f64> {
        let query = Query { step, candidate };
        let i = *self.map.get(&query as &dyn KeyView)?;
        self.touch(i);
        Some(self.slots[i].value)
    }

    /// Stores a value; returns whether an entry was evicted.
    fn insert(&mut self, step: StepKey, candidate: &[u32], value: f64) -> bool {
        debug_assert!(self.capacity > 0);
        let query = Query { step, candidate };
        if let Some(&i) = self.map.get(&query as &dyn KeyView) {
            self.slots[i].value = value;
            self.touch(i);
            return false;
        }
        let key = Key {
            step,
            candidate: Arc::from(candidate),
        };
        if self.slots.len() < self.capacity {
            let i = self.slots.len();
            self.slots.push(Slot {
                key: key.clone(),
                value,
                prev: NIL,
                next: NIL,
            });
            self.map.insert(key, i);
            self.push_front(i);
            false
        } else {
            let i = self.tail;
            self.unlink(i);
            self.map.remove(&self.slots[i].key);
            self.slots[i].key = key.clone();
            self.slots[i].value = value;
            self.map.insert(key, i);
            self.push_front(i);
            true
        }
    }

    /// Drops every entry and releases the memory held for them.
    fn clear(&mut self) {
        *self = Self::new(self.capacity);
    }
}

/// A bounded, thread-safe LRU map from `(StepKey, candidate option indices)` to error.
///
/// `get` and `insert` run in O(1) under one mutex; the counters are atomics. Share it
/// between threads by reference or inside an `Arc`.
pub struct CrossStepCache {
    capacity: usize,
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
    inner: Mutex<Lru>,
}

impl std::fmt::Debug for CrossStepCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CrossStepCache")
            .field("capacity", &self.capacity)
            .field("stats", &self.stats())
            .finish()
    }
}

impl CrossStepCache {
    /// A cache holding at most `capacity` entries (`capacity == 0` stores nothing).
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            inner: Mutex::new(Lru::new(capacity)),
        }
    }

    /// The maximum number of entries.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    fn lock(&self) -> MutexGuard<'_, Lru> {
        // The list is only ever left half-updated by a panic inside `Lru`, which does not
        // call user code; recover the guard rather than propagating poison.
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Looks up a candidate's error, counting a hit or a miss and refreshing recency.
    pub fn get(&self, step: StepKey, candidate: &[u32]) -> Option<f64> {
        let found = if self.capacity == 0 {
            None
        } else {
            self.lock().get(step, candidate)
        };
        match found {
            Some(_) => self.hits.fetch_add(1, Ordering::Relaxed),
            None => self.misses.fetch_add(1, Ordering::Relaxed),
        };
        found
    }

    /// Stores a candidate's error, evicting the least recently used entry if full. An
    /// existing entry for the same key is overwritten and becomes the most recent one.
    pub fn insert(&self, step: StepKey, candidate: &[u32], error: f64) {
        if self.capacity == 0 {
            return;
        }
        if self.lock().insert(step, candidate, error) {
            self.evictions.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Current counters.
    pub fn stats(&self) -> CacheStats {
        let len = self.lock().len();
        CacheStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            len,
        }
    }

    /// Forgets every entry and releases the memory held for them (counters are kept).
    pub fn clear(&self) {
        self.lock().clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use std::collections::HashSet;

    fn id(i: u32) -> ElementId {
        ElementId::new(i)
    }

    fn key(n: u128) -> StepKey {
        StepKey(n)
    }

    // ---- StepDedup ----

    #[test]
    fn dedup_basic() {
        let mut d = StepDedup::new(3);
        assert_eq!(d.width(), 3);
        assert!(d.is_empty());
        assert_eq!(d.lookup(&[1, 2, 3]), None);
        d.insert(&[1, 2, 3], 0.5);
        d.insert(&[1, 2, 4], 1.5);
        d.insert(&[3, 2, 1], f64::INFINITY);
        assert_eq!(d.len(), 3);
        assert!(!d.is_empty());
        assert_eq!(d.lookup(&[1, 2, 3]), Some(0.5));
        assert_eq!(d.lookup(&[1, 2, 4]), Some(1.5));
        assert_eq!(d.lookup(&[3, 2, 1]), Some(f64::INFINITY));
        assert_eq!(d.lookup(&[2, 2, 3]), None);
        d.insert(&[1, 2, 3], 9.0);
        assert_eq!(d.len(), 3, "overwrite does not grow");
        assert_eq!(d.lookup(&[1, 2, 3]), Some(9.0));
    }

    #[test]
    fn dedup_stores_nan_and_distinguishes_neighbours() {
        let mut d = StepDedup::new(2);
        d.insert(&[0, 1], f64::NAN);
        assert!(d.lookup(&[0, 1]).unwrap().is_nan());
        assert_eq!(d.lookup(&[1, 0]), None);
        assert_eq!(d.lookup(&[0, 0]), None);
    }

    #[test]
    fn dedup_width_zero() {
        let mut d = StepDedup::new(0);
        assert_eq!(d.lookup(&[]), None);
        d.insert(&[], 2.0);
        assert_eq!(d.lookup(&[]), Some(2.0));
        assert_eq!(d.len(), 1);
        d.insert(&[], 3.0);
        assert_eq!(d.len(), 1);
        assert_eq!(d.lookup(&[]), Some(3.0));
    }

    #[test]
    fn dedup_all_small_candidates_stay_distinct() {
        let mut d = StepDedup::new(4);
        let mut n = 0.0;
        for a in 0..6 {
            for b in 0..6 {
                for c in 0..6 {
                    for e in 0..6 {
                        d.insert(&[a, b, c, e], n);
                        n += 1.0;
                    }
                }
            }
        }
        assert_eq!(d.len(), 6 * 6 * 6 * 6);
        let mut n = 0.0;
        for a in 0..6 {
            for b in 0..6 {
                for c in 0..6 {
                    for e in 0..6 {
                        assert_eq!(d.lookup(&[a, b, c, e]), Some(n));
                        n += 1.0;
                    }
                }
            }
        }
    }

    proptest! {
        #[test]
        fn dedup_matches_a_reference_map(
            ops in proptest::collection::vec(
                (proptest::collection::vec(0u32..4, 3), -5i32..5, any::<bool>()), 0..200)
        ) {
            let mut d = StepDedup::new(3);
            let mut reference: std::collections::BTreeMap<Vec<u32>, f64> = Default::default();
            for (cand, value, is_insert) in ops {
                if is_insert {
                    d.insert(&cand, f64::from(value));
                    reference.insert(cand, f64::from(value));
                } else {
                    prop_assert_eq!(d.lookup(&cand), reference.get(&cand).copied());
                }
                prop_assert_eq!(d.len(), reference.len());
            }
        }
    }

    #[test]
    fn clear_releases_memory_and_keeps_counters() {
        let cache = CrossStepCache::new(1000);
        for i in 0..1000u32 {
            cache.insert(key(1), &[i], f64::from(i));
        }
        assert!(cache.get(key(1), &[3]).is_some());
        let before = cache.stats();
        cache.clear();
        {
            let lru = cache.lock();
            assert_eq!(lru.slots.capacity(), 0, "the slot storage is released");
            assert_eq!(lru.map.capacity(), 0, "the index map is released");
        }
        let after = cache.stats();
        assert_eq!((after.len, after.hits), (0, before.hits));
        assert_eq!(cache.get(key(1), &[3]), None);
        // The cache works normally after a clear, at full capacity.
        for i in 0..1000u32 {
            cache.insert(key(1), &[i], f64::from(i) + 0.5);
        }
        assert_eq!(cache.stats().len, 1000);
        assert_eq!(cache.get(key(1), &[999]), Some(999.5));
    }

    // ---- StepKey ----

    #[test]
    fn step_key_is_deterministic_and_stable() {
        let a = StepKey::new(&[id(1), id(2)], &[(id(5), 3)]);
        let b = StepKey::new(&[id(1), id(2)], &[(id(5), 3)]);
        assert_eq!(a, b);
        // Golden values: the key must not change between runs, platforms or releases, since
        // saved caches may rely on it.
        assert_eq!(StepKey::new(&[], &[]), StepKey(GOLDEN_EMPTY));
        assert_eq!(a, StepKey(GOLDEN_A));
    }

    /// Modular inverse of an odd `u64` (Newton iteration).
    fn inverse(m: u64) -> u64 {
        let mut x = m;
        for _ in 0..6 {
            x = x.wrapping_mul(2u64.wrapping_sub(m.wrapping_mul(x)));
        }
        x
    }

    /// The inverse of `mix64`.
    fn unmix64(mut x: u64) -> u64 {
        x ^= x >> 31 ^ x >> 62;
        x = x.wrapping_mul(inverse(0x94d0_49bb_1331_11eb));
        x ^= x >> 27 ^ x >> 54;
        x = x.wrapping_mul(inverse(0xbf58_476d_1ce4_e5b9));
        x ^ x >> 30 ^ x >> 60
    }

    #[test]
    fn unmix64_inverts_mix64() {
        let mut rng = StdRng::seed_from_u64(5);
        for _ in 0..1000 {
            let x: u64 = rng.random();
            assert_eq!(unmix64(mix64(x)), x);
        }
    }

    #[test]
    fn key_finish_is_a_bijection_of_the_128_bit_state() {
        // `finish` must not lose any of the 128 state bits: from the key one can recover the
        // exact state of both streams. (A previous linear pre-mixing step had GF(2) rank 112
        // and so could reach at most 2^112 distinct keys.)
        let mut rng = StdRng::seed_from_u64(9);
        for _ in 0..1000 {
            let (a, b): (u64, u64) = (rng.random(), rng.random());
            let StepKey(key) = KeyHasher { a, b }.finish();
            let (hi, lo) = ((key >> 64) as u64, key as u64);
            let a_back = unmix64(hi);
            let b_back = unmix64(lo) ^ hi ^ KEY_FINISH_SALT;
            assert_eq!((a_back, b_back), (a, b));
        }
    }

    const GOLDEN_EMPTY: u128 = 278170581013246836383412916741485836953;
    const GOLDEN_A: u128 = 282915086910633528439949585255433434496;

    #[test]
    fn step_key_separates_lists_and_fields() {
        let p = |v: &[u32]| v.iter().map(|&i| id(i)).collect::<Vec<_>>();
        let keys = [
            StepKey::new(&[], &[]),
            StepKey::new(&p(&[1]), &[]),
            StepKey::new(&[], &[(id(1), 0)]),
            StepKey::new(&p(&[0]), &[]),
            StepKey::new(&[], &[(id(0), 0)]),
            StepKey::new(&p(&[1, 2]), &[]),
            StepKey::new(&p(&[2, 1]), &[]),
            StepKey::new(&p(&[1]), &[(id(2), 0)]),
            StepKey::new(&p(&[1, 2]), &[(id(0), 0)]),
            StepKey::new(&p(&[1]), &[(id(2), 0), (id(0), 0)]),
            StepKey::new(&p(&[1]), &[(id(2), 1)]),
            StepKey::new(&p(&[1]), &[(id(3), 0)]),
            StepKey::new(&[], &[(id(1), 2)]),
            StepKey::new(&[], &[(id(2), 1)]),
            StepKey::new(&[], &[(id(1), 0), (id(2), 0)]),
            StepKey::new(&[], &[(id(2), 0), (id(1), 0)]),
        ];
        let distinct: HashSet<_> = keys.iter().collect();
        assert_eq!(distinct.len(), keys.len());
    }

    #[test]
    fn step_key_exhaustive_small_contexts_are_distinct() {
        let mut seen = HashSet::new();
        let mut count = 0;
        // Parameter lists over ids 0..4 of length 0..=3; constraint lists over
        // (id 0..3, option 0..3) of length 0..=2.
        let mut params: Vec<Vec<u32>> = vec![vec![]];
        for len in 1..=3 {
            let mut n = 4usize.pow(len);
            while n > 0 {
                n -= 1;
                let mut v = Vec::new();
                let mut r = n;
                for _ in 0..len {
                    v.push((r % 4) as u32);
                    r /= 4;
                }
                params.push(v);
            }
        }
        let pairs: Vec<(u32, u32)> = (0..3).flat_map(|i| (0..3).map(move |o| (i, o))).collect();
        let mut cons: Vec<Vec<(u32, u32)>> = vec![vec![]];
        for &a in &pairs {
            cons.push(vec![a]);
            for &b in &pairs {
                cons.push(vec![a, b]);
            }
        }
        for p in &params {
            for c in &cons {
                let ids: Vec<ElementId> = p.iter().map(|&i| id(i)).collect();
                let cs: Vec<(ElementId, u32)> = c.iter().map(|&(i, o)| (id(i), o)).collect();
                seen.insert(StepKey::new(&ids, &cs));
                count += 1;
            }
        }
        assert_eq!(seen.len(), count);
    }

    #[test]
    fn step_key_has_no_collisions_on_100k_random_contexts() {
        let mut rng = StdRng::seed_from_u64(42);
        type Context = (Vec<u32>, Vec<(u32, u32)>);
        let mut inputs: HashSet<Context> = HashSet::new();
        let mut keys: HashSet<u128> = HashSet::new();
        while inputs.len() < 100_000 {
            let np = rng.random_range(0..6);
            let nc = rng.random_range(0..9);
            let params: Vec<u32> = (0..np).map(|_| rng.random_range(0..2000)).collect();
            let mut cons: Vec<(u32, u32)> = (0..nc)
                .map(|_| (rng.random_range(0..2000), rng.random_range(0..200)))
                .collect();
            cons.sort_unstable();
            if inputs.insert((params.clone(), cons.clone())) {
                let ids: Vec<ElementId> = params.iter().map(|&i| id(i)).collect();
                let cs: Vec<(ElementId, u32)> = cons.iter().map(|&(i, o)| (id(i), o)).collect();
                keys.insert(StepKey::new(&ids, &cs).0);
            }
        }
        assert_eq!(keys.len(), inputs.len());
        // Both halves carry entropy: the high and low 64 bits are each nearly unique.
        let hi: HashSet<u64> = keys.iter().map(|&k| (k >> 64) as u64).collect();
        let lo: HashSet<u64> = keys.iter().map(|&k| k as u64).collect();
        assert_eq!(hi.len(), keys.len());
        assert_eq!(lo.len(), keys.len());
    }

    #[test]
    fn step_key_bits_are_balanced() {
        let mut ones = [0u32; 128];
        let n = 4000u32;
        for i in 0..n {
            let k = StepKey::new(&[id(i)], &[(id(i / 3), i % 5)]).0;
            for (b, count) in ones.iter_mut().enumerate() {
                *count += ((k >> b) & 1) as u32;
            }
        }
        for (b, &count) in ones.iter().enumerate() {
            assert!(
                (1700..2300).contains(&count),
                "bit {b} is set {count} times of {n}"
            );
        }
    }

    // ---- CrossStepCache ----

    #[test]
    fn cache_hit_miss_counters() {
        let c = CrossStepCache::new(10);
        assert_eq!(c.get(key(1), &[1, 2]), None);
        c.insert(key(1), &[1, 2], 0.25);
        assert_eq!(c.get(key(1), &[1, 2]), Some(0.25));
        assert_eq!(c.get(key(1), &[1, 3]), None);
        assert_eq!(c.get(key(2), &[1, 2]), None);
        assert_eq!(c.get(key(1), &[1, 2]), Some(0.25));
        assert_eq!(
            c.stats(),
            CacheStats {
                hits: 2,
                misses: 3,
                evictions: 0,
                len: 1
            }
        );
        assert_eq!(c.capacity(), 10);
    }

    #[test]
    fn cache_evicts_least_recently_used() {
        let c = CrossStepCache::new(3);
        for i in 0..3u32 {
            c.insert(key(0), &[i], f64::from(i));
        }
        // Touch 0 so that 1 is now the oldest.
        assert_eq!(c.get(key(0), &[0]), Some(0.0));
        c.insert(key(0), &[3], 3.0);
        assert_eq!(c.stats().evictions, 1);
        assert_eq!(c.get(key(0), &[1]), None, "1 was least recently used");
        assert_eq!(c.get(key(0), &[0]), Some(0.0));
        assert_eq!(c.get(key(0), &[2]), Some(2.0));
        assert_eq!(c.get(key(0), &[3]), Some(3.0));
        // Order of recency now (old to new): 0, 2, 3. Inserting evicts 0.
        c.insert(key(0), &[4], 4.0);
        assert_eq!(c.get(key(0), &[0]), None);
        assert_eq!(c.stats().evictions, 2);
        assert_eq!(c.stats().len, 3);
    }

    #[test]
    fn cache_without_reads_evicts_in_insertion_order() {
        let c = CrossStepCache::new(4);
        for i in 0..20u32 {
            c.insert(key(7), &[i, i], f64::from(i));
        }
        let s = c.stats();
        assert_eq!((s.len, s.evictions), (4, 16));
        for i in 0..16u32 {
            assert_eq!(c.get(key(7), &[i, i]), None);
        }
        for i in 16..20u32 {
            assert_eq!(c.get(key(7), &[i, i]), Some(f64::from(i)));
        }
    }

    #[test]
    fn cache_overwrite_updates_and_refreshes() {
        let c = CrossStepCache::new(2);
        c.insert(key(1), &[0], 1.0);
        c.insert(key(1), &[1], 2.0);
        c.insert(key(1), &[0], 10.0); // overwrite: [0] becomes the most recent
        let s = c.stats();
        assert_eq!((s.len, s.evictions), (2, 0));
        c.insert(key(1), &[2], 3.0); // evicts [1]
        assert_eq!(c.get(key(1), &[1]), None);
        assert_eq!(c.get(key(1), &[0]), Some(10.0));
        assert_eq!(c.get(key(1), &[2]), Some(3.0));
    }

    #[test]
    fn cache_capacity_zero_stores_nothing_but_counts_misses() {
        let c = CrossStepCache::new(0);
        c.insert(key(1), &[1], 1.0);
        assert_eq!(c.get(key(1), &[1]), None);
        assert_eq!(c.get(key(1), &[1]), None);
        assert_eq!(
            c.stats(),
            CacheStats {
                hits: 0,
                misses: 2,
                evictions: 0,
                len: 0
            }
        );
        c.clear();
    }

    #[test]
    fn cache_capacity_one() {
        let c = CrossStepCache::new(1);
        c.insert(key(1), &[1], 1.0);
        c.insert(key(1), &[1], 1.5);
        assert_eq!(c.stats().evictions, 0);
        c.insert(key(1), &[2], 2.0);
        assert_eq!(c.stats().evictions, 1);
        assert_eq!(c.get(key(1), &[1]), None);
        assert_eq!(c.get(key(1), &[2]), Some(2.0));
    }

    #[test]
    fn cache_clear_keeps_counters_and_works_afterwards() {
        let c = CrossStepCache::new(2);
        c.insert(key(1), &[1], 1.0);
        c.insert(key(1), &[2], 2.0);
        c.insert(key(1), &[3], 3.0);
        assert_eq!(c.get(key(1), &[3]), Some(3.0));
        assert_eq!(c.get(key(1), &[1]), None);
        c.clear();
        let s = c.stats();
        assert_eq!((s.hits, s.misses, s.evictions, s.len), (1, 1, 1, 0));
        assert_eq!(c.get(key(1), &[3]), None);
        c.insert(key(1), &[5], 5.0);
        c.insert(key(1), &[6], 6.0);
        c.insert(key(1), &[7], 7.0);
        assert_eq!(c.get(key(1), &[5]), None);
        assert_eq!(c.get(key(1), &[7]), Some(7.0));
        assert_eq!(c.stats().len, 2);
    }

    #[test]
    fn cache_keys_include_step_and_empty_candidates() {
        let c = CrossStepCache::new(8);
        c.insert(key(1), &[], 1.0);
        c.insert(key(2), &[], 2.0);
        assert_eq!(c.get(key(1), &[]), Some(1.0));
        assert_eq!(c.get(key(2), &[]), Some(2.0));
        assert_eq!(c.get(key(3), &[]), None);
        // Same high/low halves swapped are different steps.
        c.insert(StepKey(1 << 64), &[0], 3.0);
        assert_eq!(c.get(StepKey(1), &[0]), None);
        assert_eq!(c.get(StepKey(1 << 64), &[0]), Some(3.0));
    }

    #[test]
    fn cache_stores_special_floats() {
        let c = CrossStepCache::new(4);
        c.insert(key(1), &[0], f64::INFINITY);
        c.insert(key(1), &[1], f64::NAN);
        c.insert(key(1), &[2], -0.0);
        assert_eq!(c.get(key(1), &[0]), Some(f64::INFINITY));
        assert!(c.get(key(1), &[1]).unwrap().is_nan());
        assert!(c.get(key(1), &[2]).unwrap().is_sign_negative());
    }

    /// Reference LRU: a vector ordered from least to most recently used.
    struct ModelLru {
        capacity: usize,
        entries: Vec<((u128, Vec<u32>), f64)>,
        evictions: u64,
        hits: u64,
        misses: u64,
    }

    impl ModelLru {
        fn get(&mut self, k: &(u128, Vec<u32>)) -> Option<f64> {
            if self.capacity == 0 {
                self.misses += 1;
                return None;
            }
            match self.entries.iter().position(|(e, _)| e == k) {
                Some(i) => {
                    let entry = self.entries.remove(i);
                    let v = entry.1;
                    self.entries.push(entry);
                    self.hits += 1;
                    Some(v)
                }
                None => {
                    self.misses += 1;
                    None
                }
            }
        }

        fn insert(&mut self, k: (u128, Vec<u32>), v: f64) {
            if self.capacity == 0 {
                return;
            }
            if let Some(i) = self.entries.iter().position(|(e, _)| *e == k) {
                self.entries.remove(i);
            } else if self.entries.len() == self.capacity {
                self.entries.remove(0);
                self.evictions += 1;
            }
            self.entries.push((k, v));
        }
    }

    proptest! {
        #[test]
        fn cache_matches_a_reference_lru(
            capacity in 0usize..6,
            ops in proptest::collection::vec(
                (0u128..3, proptest::collection::vec(0u32..3, 0..3), 0i32..100, any::<bool>()),
                0..300)
        ) {
            let cache = CrossStepCache::new(capacity);
            let mut model = ModelLru {
                capacity, entries: Vec::new(), evictions: 0, hits: 0, misses: 0,
            };
            for (step, cand, value, is_insert) in ops {
                let k = (step, cand.clone());
                if is_insert {
                    cache.insert(StepKey(step), &cand, f64::from(value));
                    model.insert(k, f64::from(value));
                } else {
                    prop_assert_eq!(cache.get(StepKey(step), &cand), model.get(&k));
                }
                let s = cache.stats();
                prop_assert_eq!(s.len, model.entries.len());
                prop_assert_eq!(s.evictions, model.evictions);
                prop_assert_eq!(s.hits, model.hits);
                prop_assert_eq!(s.misses, model.misses);
            }
        }
    }

    /// The value every thread must observe for a key, whatever the interleaving.
    fn value_of(step: u128, cand: &[u32]) -> f64 {
        let mut x = step as u64 ^ 0x9e37_79b9;
        for &c in cand {
            x = x.rotate_left(9) ^ u64::from(c).wrapping_mul(0x1000_0000_01b3);
        }
        (x % 1_000_003) as f64
    }

    #[test]
    fn cache_multithreaded_hammer() {
        let cache = CrossStepCache::new(64);
        let threads = 8usize;
        let ops = 20_000usize;
        let total_gets: u64 = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..threads)
                .map(|t| {
                    let cache = &cache;
                    scope.spawn(move || {
                        let mut rng = StdRng::seed_from_u64(t as u64);
                        let mut gets = 0u64;
                        for _ in 0..ops {
                            // A small key space with many collisions on purpose.
                            let step = u128::from(rng.random_range(0..4u32));
                            let n = rng.random_range(0..3usize);
                            let cand: Vec<u32> = (0..n).map(|_| rng.random_range(0..6)).collect();
                            if rng.random_range(0..3) == 0 {
                                cache.insert(StepKey(step), &cand, value_of(step, &cand));
                            } else {
                                gets += 1;
                                if let Some(v) = cache.get(StepKey(step), &cand) {
                                    assert_eq!(v, value_of(step, &cand), "wrong value for key");
                                }
                            }
                        }
                        gets
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).sum()
        });
        let s = cache.stats();
        assert_eq!(s.hits + s.misses, total_gets);
        assert!(s.len <= 64);
        assert!(s.evictions > 0, "the key space is larger than the capacity");
        assert!(s.hits > 0);
    }

    #[test]
    fn cache_is_send_and_sync() {
        fn check<T: Send + Sync>() {}
        check::<CrossStepCache>();
        check::<StepDedup>();
        check::<StepKey>();
    }

    #[test]
    fn cache_survives_a_large_working_set() {
        let c = CrossStepCache::new(1000);
        for i in 0..50_000u32 {
            c.insert(key(u128::from(i % 7)), &[i, i + 1, i + 2], f64::from(i));
        }
        let s = c.stats();
        assert_eq!((s.len, s.evictions), (1000, 49_000));
        for i in 49_000..50_000u32 {
            assert_eq!(
                c.get(key(u128::from(i % 7)), &[i, i + 1, i + 2]),
                Some(f64::from(i))
            );
        }
    }
}
