//! Borrowed values that Python callbacks may touch for the duration of one call.
//!
//! The engine hands some values to user callbacks by reference: the traversal context to a
//! heuristic, the objective to an optimizer. A `#[pyclass]` must be `'static`, so the Python
//! objects wrapping these values hold a type-erased pointer that is valid only while the
//! callback runs. This module is the only place of the crate that needs `unsafe`.
//!
//! Soundness argument, for both types below:
//!
//! * the pointer is installed by `scope`, which borrows the value for the whole call of the
//!   closure and clears the pointer again before it returns (also on unwind, through a drop
//!   guard), so the pointer is never dereferenced after the borrow has ended;
//! * the pointer is dereferenced only on the thread that called `scope` (checked against the
//!   recorded thread id), so values that are not `Send` (the engine's objective) are never
//!   touched from another thread; a Python object that outlives the call or is passed to
//!   another thread can only be asked, and then fails with a `None` from `with`. Moving the
//!   holder itself between threads is harmless: it is a pointer that is not dereferenced there;
//! * the slot is behind a mutex that is held while the value is used, so two uses never
//!   overlap and a re-entrant use (the user's code calling the objective from inside the error
//!   model that the objective is calling) is detected with `try_lock` instead of aliasing.

use std::ptr::NonNull;
use std::sync::{Mutex, TryLockError};
use std::thread::{self, ThreadId};

use faulttune::{Objective, TraversalContext};

/// A pointer that may be moved to other threads because it is only dereferenced on `owner`.
struct Erased<T: ?Sized> {
    ptr: NonNull<T>,
    owner: ThreadId,
}

// SAFETY: see the module documentation. The pointer is never dereferenced off `owner`.
unsafe impl<T: ?Sized> Send for Erased<T> {}

/// The slot shared by [`ScopedObjective`] and [`ScopedContext`].
struct Slot<T: ?Sized>(Mutex<Option<Erased<T>>>);

impl<T: ?Sized> Default for Slot<T> {
    fn default() -> Self {
        Self(Mutex::new(None))
    }
}

impl<T: ?Sized> Slot<T> {
    fn set(&self, value: Option<Erased<T>>) {
        *self.0.lock().unwrap_or_else(|e| e.into_inner()) = value;
    }

    /// Runs `f` with the pointer if it is set, on its owner thread, and not in use.
    fn with<R>(&self, f: impl FnOnce(NonNull<T>) -> R) -> Option<R> {
        let guard = match self.0.try_lock() {
            Ok(guard) => guard,
            Err(TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
            Err(TryLockError::WouldBlock) => return None,
        };
        let erased = guard.as_ref()?;
        if erased.owner != thread::current().id() {
            return None;
        }
        Some(f(erased.ptr))
    }
}

/// Clears a slot when dropped.
struct Clear<'a, T: ?Sized>(&'a Slot<T>);

impl<T: ?Sized> Drop for Clear<'_, T> {
    fn drop(&mut self) {
        self.0.set(None);
    }
}

/// A `&mut dyn Objective` that can be used through a `'static` Python object while `scope` runs.
#[derive(Default)]
pub struct ScopedObjective {
    slot: Slot<dyn Objective>,
}

impl ScopedObjective {
    /// An empty slot: `with` fails until `scope` is running.
    pub fn new() -> Self {
        Self::default()
    }

    /// Runs `f` while `objective` is reachable through [`with`](Self::with).
    pub fn scope<R>(&self, objective: &mut dyn Objective, f: impl FnOnce() -> R) -> R {
        let raw: *mut (dyn Objective + '_) = objective;
        // SAFETY: only the (erased) lifetime of the trait object changes. The pointer is
        // cleared by `Clear` before `objective`'s borrow ends, and it is only dereferenced in
        // `with`, which requires the pointer to be set. See the module documentation.
        let erased: *mut (dyn Objective + 'static) = unsafe { std::mem::transmute(raw) };
        self.slot.set(NonNull::new(erased).map(|ptr| Erased {
            ptr,
            owner: thread::current().id(),
        }));
        let _clear = Clear(&self.slot);
        f()
    }

    /// Uses the objective. `None` if no `scope` is running, if called from another thread than
    /// the one that runs it, or if the objective is in use already.
    pub fn with<R>(&self, f: impl FnOnce(&mut dyn Objective) -> R) -> Option<R> {
        // SAFETY: the pointer is set, so `scope` is still running and the objective is alive;
        // the slot's mutex is held, so this is the only live reference to it, and `Slot::with`
        // checked that we are on the thread that owns it.
        self.slot.with(|mut ptr| f(unsafe { ptr.as_mut() }))
    }
}

/// A `&TraversalContext` that can be used through a `'static` Python object while `scope` runs.
#[derive(Default)]
pub struct ScopedContext {
    slot: Slot<TraversalContext<'static>>,
}

impl ScopedContext {
    /// An empty slot: `with` fails until `scope` is running.
    pub fn new() -> Self {
        Self::default()
    }

    /// Runs `f` while `ctx` is reachable through [`with`](Self::with).
    pub fn scope<R>(&self, ctx: &TraversalContext<'_>, f: impl FnOnce() -> R) -> R {
        let raw: *const TraversalContext<'_> = ctx;
        // SAFETY: only the lifetime parameter changes; see `ScopedObjective::scope`.
        let erased: *const TraversalContext<'static> = unsafe { std::mem::transmute(raw) };
        self.slot
            .set(NonNull::new(erased.cast_mut()).map(|ptr| Erased {
                ptr,
                owner: thread::current().id(),
            }));
        let _clear = Clear(&self.slot);
        f()
    }

    /// Uses the context. `None` if no `scope` is running or on another thread.
    pub fn with<R>(&self, f: impl FnOnce(&TraversalContext<'_>) -> R) -> Option<R> {
        // SAFETY: the pointer is set, so `scope` is still running and the context is alive.
        // Only shared references are created.
        self.slot.with(|ptr| f(unsafe { ptr.as_ref() }))
    }
}
