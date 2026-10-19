//! [`ElementSet`]: a fixed-capacity bitset over [`ElementId`]s.
//!
//! It is used for the calibration goal `G*`, the uncalibrated set, subgoal
//! membership and calibration status. Membership tests are O(1).

use std::fmt;

use crate::graph::ElementId;

/// A set of elements, stored as a bitset over the dense element IDs `0..capacity`.
///
/// Binary operations require both operands to have the same capacity (they must
/// come from the same graph); mixing capacities is a programming error and panics.
#[derive(Clone, PartialEq, Eq, Hash, Default)]
pub struct ElementSet {
    words: Vec<u64>,
    capacity: usize,
}

impl ElementSet {
    /// An empty set over `capacity` elements.
    pub fn new(capacity: usize) -> Self {
        Self {
            words: vec![0; capacity.div_ceil(64)],
            capacity,
        }
    }

    /// The set of all `capacity` elements.
    pub fn full(capacity: usize) -> Self {
        let mut set = Self::new(capacity);
        for word in &mut set.words {
            *word = u64::MAX;
        }
        set.mask_tail();
        set
    }

    /// A set containing `ids`.
    ///
    /// # Panics
    /// Panics if an ID is `>= capacity`.
    pub fn from_ids(capacity: usize, ids: impl IntoIterator<Item = ElementId>) -> Self {
        let mut set = Self::new(capacity);
        set.extend(ids);
        set
    }

    fn mask_tail(&mut self) {
        let rem = self.capacity % 64;
        if rem != 0 {
            if let Some(last) = self.words.last_mut() {
                *last &= (1u64 << rem) - 1;
            }
        }
    }

    /// Number of elements the set can hold (the graph's element count).
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Number of elements in the set.
    pub fn len(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Whether the set is empty.
    pub fn is_empty(&self) -> bool {
        self.words.iter().all(|&w| w == 0)
    }

    /// Whether `id` is in the set. IDs outside the capacity are never members.
    #[inline]
    pub fn contains(&self, id: ElementId) -> bool {
        let i = id.index();
        i < self.capacity && (self.words[i >> 6] >> (i & 63)) & 1 == 1
    }

    /// Adds `id`; returns `true` if it was not already present.
    ///
    /// # Panics
    /// Panics if `id >= capacity`.
    #[inline]
    pub fn insert(&mut self, id: ElementId) -> bool {
        let i = id.index();
        assert!(
            i < self.capacity,
            "element {i} out of range for a set of capacity {}",
            self.capacity
        );
        let word = &mut self.words[i >> 6];
        let bit = 1u64 << (i & 63);
        let was_absent = *word & bit == 0;
        *word |= bit;
        was_absent
    }

    /// Removes `id`; returns `true` if it was present.
    #[inline]
    pub fn remove(&mut self, id: ElementId) -> bool {
        let i = id.index();
        if i >= self.capacity {
            return false;
        }
        let word = &mut self.words[i >> 6];
        let bit = 1u64 << (i & 63);
        let was_present = *word & bit != 0;
        *word &= !bit;
        was_present
    }

    /// Removes every element.
    pub fn clear(&mut self) {
        self.words.iter_mut().for_each(|w| *w = 0);
    }

    /// The smallest element, if any.
    pub fn first(&self) -> Option<ElementId> {
        self.iter().next()
    }

    /// Iterates over the members in ascending ID order.
    pub fn iter(&self) -> Iter<'_> {
        Iter {
            words: &self.words,
            word_index: 0,
            current: self.words.first().copied().unwrap_or(0),
        }
    }

    /// The members as a sorted vector.
    pub fn to_vec(&self) -> Vec<ElementId> {
        self.iter().collect()
    }

    fn check_same_capacity(&self, other: &Self) {
        assert_eq!(
            self.capacity, other.capacity,
            "element sets from different graphs (capacity {} vs {})",
            self.capacity, other.capacity
        );
    }

    /// `self ∪= other`.
    pub fn union_with(&mut self, other: &Self) {
        self.check_same_capacity(other);
        for (a, b) in self.words.iter_mut().zip(&other.words) {
            *a |= *b;
        }
    }

    /// `self ∩= other`.
    pub fn intersect_with(&mut self, other: &Self) {
        self.check_same_capacity(other);
        for (a, b) in self.words.iter_mut().zip(&other.words) {
            *a &= *b;
        }
    }

    /// `self ∖= other`.
    pub fn difference_with(&mut self, other: &Self) {
        self.check_same_capacity(other);
        for (a, b) in self.words.iter_mut().zip(&other.words) {
            *a &= !*b;
        }
    }

    /// `self ∪ other`.
    pub fn union(&self, other: &Self) -> Self {
        let mut out = self.clone();
        out.union_with(other);
        out
    }

    /// `self ∩ other`.
    pub fn intersection(&self, other: &Self) -> Self {
        let mut out = self.clone();
        out.intersect_with(other);
        out
    }

    /// `self ∖ other`.
    pub fn difference(&self, other: &Self) -> Self {
        let mut out = self.clone();
        out.difference_with(other);
        out
    }

    /// Whether every member of `self` is in `other`.
    pub fn is_subset(&self, other: &Self) -> bool {
        self.check_same_capacity(other);
        self.words
            .iter()
            .zip(&other.words)
            .all(|(a, b)| a & !b == 0)
    }

    /// Whether the two sets share no member.
    pub fn is_disjoint(&self, other: &Self) -> bool {
        self.check_same_capacity(other);
        self.words.iter().zip(&other.words).all(|(a, b)| a & b == 0)
    }
}

impl fmt::Debug for ElementSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set()
            .entries(self.iter().map(|id| id.raw()))
            .finish()
    }
}

impl Extend<ElementId> for ElementSet {
    fn extend<T: IntoIterator<Item = ElementId>>(&mut self, iter: T) {
        for id in iter {
            self.insert(id);
        }
    }
}

impl<'a> IntoIterator for &'a ElementSet {
    type Item = ElementId;
    type IntoIter = Iter<'a>;
    fn into_iter(self) -> Iter<'a> {
        self.iter()
    }
}

/// Ascending iterator over an [`ElementSet`].
pub struct Iter<'a> {
    words: &'a [u64],
    word_index: usize,
    current: u64,
}

impl Iterator for Iter<'_> {
    type Item = ElementId;

    fn next(&mut self) -> Option<ElementId> {
        loop {
            if self.current != 0 {
                let bit = self.current.trailing_zeros() as usize;
                self.current &= self.current - 1;
                return Some(ElementId::new((self.word_index * 64 + bit) as u32));
            }
            self.word_index += 1;
            if self.word_index >= self.words.len() {
                return None;
            }
            self.current = self.words[self.word_index];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(i: u32) -> ElementId {
        ElementId::new(i)
    }

    #[test]
    fn insert_remove_contains() {
        let mut s = ElementSet::new(130);
        assert!(s.is_empty());
        assert!(s.insert(id(0)));
        assert!(!s.insert(id(0)));
        assert!(s.insert(id(64)));
        assert!(s.insert(id(129)));
        assert_eq!(s.len(), 3);
        assert!(s.contains(id(64)));
        assert!(!s.contains(id(65)));
        assert!(!s.contains(id(500)));
        assert!(s.remove(id(64)));
        assert!(!s.remove(id(64)));
        assert_eq!(s.to_vec(), vec![id(0), id(129)]);
    }

    #[test]
    fn full_masks_tail() {
        let s = ElementSet::full(70);
        assert_eq!(s.len(), 70);
        assert_eq!(s.iter().last(), Some(id(69)));
        assert_eq!(ElementSet::full(64).len(), 64);
        assert_eq!(ElementSet::full(0).len(), 0);
    }

    #[test]
    fn set_algebra() {
        let a = ElementSet::from_ids(100, [1, 2, 3, 70].map(id));
        let b = ElementSet::from_ids(100, [3, 4, 70, 99].map(id));
        assert_eq!(a.union(&b).to_vec(), [1, 2, 3, 4, 70, 99].map(id).to_vec());
        assert_eq!(a.intersection(&b).to_vec(), [3, 70].map(id).to_vec());
        assert_eq!(a.difference(&b).to_vec(), [1, 2].map(id).to_vec());
        assert!(a.intersection(&b).is_subset(&a));
        assert!(!a.is_subset(&b));
        assert!(a.difference(&b).is_disjoint(&b));
    }

    #[test]
    #[should_panic]
    fn insert_out_of_range_panics() {
        ElementSet::new(10).insert(id(10));
    }
}
