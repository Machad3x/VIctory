//! Query heaps. Nothing runs on a GPU yet, so no query ever produces a
//! result; the heap only remembers its description.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use crate::object::{self, E_INVALIDARG, HRESULT, Iid, Object, QUERY_HEAP, Table, This, guid};
use crate::resource::PAGEABLE_IIDS;

const HEAP_IIDS: [Iid; 4] = [
    guid(
        0x0d9658ae,
        0xed45,
        0x469e,
        [0xa6, 0x1d, 0x97, 0x0e, 0xc5, 0x83, 0xca, 0xb4],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct Desc {
    kind: u32,
    count: u32,
    nodes: u32,
}

#[repr(C)]
struct Heap {
    object: Object,
    desc: Desc,
}

/// Device method.
pub(crate) extern "system" fn create(
    _: This,
    desc: *const Desc,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    tracing::debug!(?desc, "query heap");
    let heap = Box::new(Heap {
        object: Object::new(&FUNCTIONS),
        desc,
    });
    object::hand_out(QUERY_HEAP, heap, iid, &HEAP_IIDS, out)
}

extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(
        QUERY_HEAP,
        iid,
        object::is(iid, &HEAP_IIDS).then_some(this),
        out,
    )
}
extern "system" fn release(this: This) -> u32 {
    object::release_boxed::<Heap>(this)
}

/// IUnknown 3, object 4, device child 1.
static FUNCTIONS: Table<24> = {
    let mut f = object::d3d12_object(object::missing_table::<QUERY_HEAP, 24>());
    f[0] = query as *const ();
    f[2] = release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    Table(f)
};

/// How the title uses queries, measured before real host queries replace the
/// made-up results.
#[cfg(feature = "diagnostics")]
pub(crate) mod counts {
    use std::cell::Cell;
    use std::collections::HashMap;
    use std::sync::{LazyLock, Mutex};

    const KINDS: [&str; 5] = ["occlusion", "binary occlusion", "timestamp", "statistics", "other"];

    #[derive(Default)]
    struct Mark {
        /// The list that began it and [`DRAWS_DROPPED`] then, until it ends.
        open: Option<(usize, u64)>,
    }

    #[derive(Default)]
    struct Counts {
        /// Per kind: begins, ends, resolve calls, queries resolved.
        kinds: [[u64; 4]; 5],
        spans_with_drops: u64,
        cross_list: u64,
        early_resolves: u64,
        destinations: Vec<usize>,
    }

    #[derive(Default)]
    struct All {
        marks: HashMap<(usize, u32), Mark>,
        counts: Counts,
    }

    static ALL: LazyLock<Mutex<All>> = LazyLock::new(Default::default);

    // shortcut: per thread, not per list (a list is recorded by one thread at
    // a time); a thread that switches lists or a list that switches threads
    // inside a span skews the count.
    thread_local! {
        static DRAWS_DROPPED: Cell<u64> = const { Cell::new(0) };
    }

    fn all() -> std::sync::MutexGuard<'static, All> {
        ALL.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn slot(kind: u32) -> usize {
        (kind as usize).min(KINDS.len() - 1)
    }

    pub(crate) fn draw_dropped() {
        DRAWS_DROPPED.set(DRAWS_DROPPED.get() + 1);
    }

    pub(crate) fn begin(list: usize, heap: usize, kind: u32, index: u32) {
        let mut all = all();
        all.counts.kinds[slot(kind)][0] += 1;
        let open = Some((list, DRAWS_DROPPED.get()));
        all.marks.insert((heap, index), Mark { open });
    }

    pub(crate) fn end(list: usize, heap: usize, kind: u32, index: u32) {
        let mut all = all();
        let All { marks, counts } = &mut *all;
        counts.kinds[slot(kind)][1] += 1;
        // A timestamp is only ended: nothing to remember.
        if let Some((began, drops)) = marks.get_mut(&(heap, index)).and_then(|m| m.open.take()) {
            if began != list {
                counts.cross_list += 1;
            } else if kind < 2 && drops != DRAWS_DROPPED.get() {
                counts.spans_with_drops += 1;
            }
        }
    }

    pub(crate) fn resolve(heap: usize, kind: u32, first: u32, count: u32, to: Option<usize>) {
        let mut all = all();
        let All { marks, counts } = &mut *all;
        counts.kinds[slot(kind)][2] += 1;
        counts.kinds[slot(kind)][3] += u64::from(count);
        let mut early = false;
        // Bounded: the count is the title's and unchecked here. A slot never
        // begun is not early, and a resolved one is forgotten.
        for index in (first..first.saturating_add(count)).take(1 << 16) {
            match marks.get(&(heap, index)) {
                Some(mark) if mark.open.is_some() => early = true,
                _ => {
                    marks.remove(&(heap, index));
                }
            }
        }
        counts.early_resolves += u64::from(early);
        if let Some(to) = to
            && counts.destinations.len() < 8
            && !counts.destinations.contains(&to)
        {
            counts.destinations.push(to);
        }
    }

    /// Logs and clears the counts of the last `frames` frames.
    pub(crate) fn log(frames: f64) {
        let c = std::mem::take(&mut all().counts);
        for (name, n) in KINDS.iter().zip(c.kinds) {
            if n == [0; 4] {
                continue;
            }
            let [begins, ends, resolves, queries] = n.map(|n| n as f64 / frames);
            tracing::info!(
                "{name} queries/frame: {begins:.1} begins, {ends:.1} ends, {resolves:.1} resolves of {queries:.1} queries"
            );
        }
        if c.kinds != [[0; 4]; 5] {
            tracing::info!(
                "query spans/frame: {:.1} occlusion with dropped draws, {:.1} ended in another list, \
                 {:.1} resolves before an end (all in recording order); resolve destinations {:x?}",
                c.spans_with_drops as f64 / frames,
                c.cross_list as f64 / frames,
                c.early_resolves as f64 / frames,
                c.destinations
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::{E_NOINTERFACE, E_POINTER, S_FALSE, S_OK, slot};

    #[test]
    fn null_descriptions_are_rejected() {
        let mut heap = std::ptr::null();
        assert_eq!(
            create(std::ptr::null(), std::ptr::null(), &HEAP_IIDS[0], &mut heap),
            E_INVALIDARG
        );
        assert!(heap.is_null());
        // The description is checked first: a probe with no room for the
        // heap fails on it rather than answering S_FALSE.
        assert_eq!(
            create(
                std::ptr::null(),
                std::ptr::null(),
                &HEAP_IIDS[0],
                std::ptr::null_mut()
            ),
            E_INVALIDARG
        );
    }

    #[test]
    fn creation_without_an_out_pointer_only_probes() {
        let desc = Desc {
            kind: 1, // timestamps
            count: 64,
            nodes: 0,
        };
        // A known interface with no out pointer asks whether creation would
        // succeed; the heap is dropped instead of handed out.
        assert_eq!(
            create(std::ptr::null(), &desc, &HEAP_IIDS[0], std::ptr::null_mut()),
            S_FALSE
        );
        // A null interface id is tolerated: no interface, no heap.
        let mut heap = std::ptr::null();
        assert_eq!(
            create(std::ptr::null(), &desc, std::ptr::null(), &mut heap),
            E_NOINTERFACE
        );
        assert!(heap.is_null());
    }

    #[test]
    fn heaps_answer_their_interfaces_and_count_references() {
        let desc = Desc {
            kind: 0, // occlusion
            count: 8,
            nodes: 0,
        };
        let null = std::ptr::null();
        let mut heap = null;
        assert_eq!(create(null, &desc, &HEAP_IIDS[0], &mut heap), S_OK);
        assert!(!heap.is_null());
        let query = slot::<extern "system" fn(This, *const Iid, *mut This) -> HRESULT>(heap, 0x0);
        // Every interface the heap answers to hands out the same object.
        for iid in &HEAP_IIDS {
            let mut out = null;
            assert_eq!(query(heap, iid, &mut out), S_OK);
            assert_eq!(out as usize, heap as usize);
        }
        // An unknown interface writes null behind and reports it missing.
        let unknown = [0u8; 16];
        let mut out = heap;
        assert_eq!(query(heap, &unknown, &mut out), E_NOINTERFACE);
        assert!(out.is_null());
        // A null interface id is tolerated rather than matched.
        let mut out = heap;
        assert_eq!(query(heap, std::ptr::null(), &mut out), E_NOINTERFACE);
        assert!(out.is_null());
        // No room for the answer is a pointer error.
        assert_eq!(query(heap, &unknown, std::ptr::null_mut()), E_POINTER);
        // References: one from creation, four from the queries, one added;
        // each release hands back what is left, the last frees the heap.
        assert_eq!(slot::<extern "system" fn(This) -> u32>(heap, 0x8)(heap), 6);
        let release = slot::<extern "system" fn(This) -> u32>(heap, 0x10);
        for left in [5, 4, 3, 2, 1, 0] {
            assert_eq!(release(heap), left);
        }
    }
}
