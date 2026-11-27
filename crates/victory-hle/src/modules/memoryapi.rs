//! The memory functions the title shares with PC that change memory
//! (`VirtualFree`, `VirtualProtect`, unmapping views): the host's, after the
//! GPU side lets go of title memory it holds in place there
//! (`victory_gpu::title_memory_change`). `VirtualAlloc` at an address the
//! title names is the host's first, and lets go only if refused as locked
//! (`victory_gpu::title_memory_retry`). A GPU heap opened over memory
//! makes each of them fail on the host (`ERROR_INVALID_PARAMETER`) while it
//! lives; on the console the GPU reads title memory in place and these
//! always work. Everything else in these DLLs is the host's directly.
//!
//! Cost while the GPU side holds nothing in place: one atomic load a call,
//! and a lookup more when the host refuses a change as locked.

// Named as the imports; the title passes the pointers to the host as is.
#![allow(non_snake_case)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::c_void;

use windows_sys::Win32::Foundation::{GetLastError, HANDLE, SetLastError};
use windows_sys::Win32::System::Memory::{
    self as mem, MEM_RELEASE, MEMORY_BASIC_INFORMATION, MEMORY_MAPPED_VIEW_ADDRESS,
};
use windows_sys::Win32::System::Threading::{GetCurrentProcess, GetCurrentProcessId, GetProcessId};
use windows_sys::core::BOOL;

use crate::{HleModule, hle_export};

/// The bytes from `at` to the end of the allocation holding it (0 if it is
/// not allocated).
fn rest_of_allocation(at: usize) -> usize {
    let mut info: MEMORY_BASIC_INFORMATION = unsafe_zeroed();
    let size = size_of::<MEMORY_BASIC_INFORMATION>();
    // SAFETY: a valid out pointer of `size` bytes.
    if unsafe { mem::VirtualQuery(at as *const c_void, &mut info, size) } == 0
        || info.State == mem::MEM_FREE
    {
        return 0;
    }
    let base = info.AllocationBase;
    let mut end = info.BaseAddress as usize + info.RegionSize;
    loop {
        // SAFETY: as above.
        let n = unsafe { mem::VirtualQuery(end as *const c_void, &mut info, size) };
        if n == 0 || info.AllocationBase != base || info.State == mem::MEM_FREE {
            break;
        }
        end = info.BaseAddress as usize + info.RegionSize;
    }
    end - at
}

fn unsafe_zeroed() -> MEMORY_BASIC_INFORMATION {
    // SAFETY: plain data; zero is a valid value.
    unsafe { std::mem::zeroed() }
}

/// Whether `process` is this one (the pseudo handle or a handle to it).
fn this_process(process: HANDLE) -> bool {
    // SAFETY: plain process queries; an invalid handle gives 0.
    unsafe { process == GetCurrentProcess() || GetProcessId(process) == GetCurrentProcessId() }
}

/// Changes title memory at `[at, at + len)` with `change` (a host call,
/// nonzero when it worked) after the GPU side lets go of what it holds in
/// place there (`victory_gpu::title_memory_change`); `len` 0 means to the
/// end of the allocation (`VirtualFree`'s whole release or decommit, a
/// view's unmapping).
/// A change over memory a page table still names may be queued and done
/// later (`victory_gpu::title_memory_deferred`): `change` holds what it needs
/// by value.
pub(crate) fn change(at: *const c_void, len: usize, change: impl FnMut() -> BOOL + Send + 'static) -> BOOL {
    let mut change = change;
    if at.is_null() {
        return change();
    }
    let at = at as usize;
    // The range is worked out only when something may hold it: with nothing
    // held, a change refused as locked (a holder registered since the look,
    // or let go just before) is tried again there too.
    let range = || if len == 0 { rest_of_allocation(at) } else { len };
    victory_gpu::title_memory_change_from(at, range, change)
}

/// The bytes `VirtualFree` changes: a release is of the whole allocation,
/// whatever the size.
fn freed(size: usize, kind: u32) -> usize {
    if kind & MEM_RELEASE != 0 { 0 } else { size }
}

pub extern "system" fn VirtualFree(at: *mut c_void, size: usize, kind: u32) -> BOOL {
    let a = at as usize;
    // SAFETY: the title's arguments.
    change(at, freed(size, kind), move || unsafe { mem::VirtualFree(a as *mut c_void, size, kind) })
}

pub extern "system" fn VirtualFreeEx(
    process: HANDLE,
    at: *mut c_void,
    size: usize,
    kind: u32,
) -> BOOL {
    if this_process(process) {
        let a = at as usize;
        // SAFETY: the title's arguments, in this process.
        change(at, freed(size, kind), move || unsafe { mem::VirtualFree(a as *mut c_void, size, kind) })
    } else {
        // SAFETY: the title's arguments.
        unsafe { mem::VirtualFreeEx(process, at, size, kind) }
    }
}

pub extern "system" fn VirtualProtect(
    at: *const c_void,
    size: usize,
    protect: u32,
    old: *mut u32,
) -> BOOL {
    if size != 0 {
        protect_title(at, size, protect, old)
    } else {
        // SAFETY: the title's arguments.
        unsafe { mem::VirtualProtect(at, size, protect, old) }
    }
}

/// `VirtualProtect` of title memory through [`change`]. The old protection
/// goes to `old` now, also when the change is queued (it is then the one
/// the memory has, which the queued change finds as well).
fn protect_title(at: *const c_void, size: usize, protect: u32, old: *mut u32) -> BOOL {
    use std::sync::atomic::{AtomicU32, Ordering};
    let was = std::sync::Arc::new(AtomicU32::new(0));
    let (a, seen) = (at as usize, was.clone());
    let done = change(at, size, move || {
        let mut o = 0;
        // SAFETY: the title's arguments; the old protection kept here.
        let worked = unsafe { mem::VirtualProtect(a as *const c_void, size, protect, &mut o) };
        seen.store(o, Ordering::Relaxed);
        worked
    });
    let was = if victory_gpu::title_memory_deferred() {
        // SAFETY: plain data filled in by the query.
        let mut info: mem::MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        unsafe { mem::VirtualQuery(at, &mut info, size_of_val(&info)) };
        info.Protect
    } else {
        was.load(Ordering::Relaxed)
    };
    if done != 0 && !old.is_null() {
        // SAFETY: the title's out pointer.
        unsafe { old.write(was) };
    }
    done
}

pub extern "system" fn VirtualProtectEx(
    process: HANDLE,
    at: *const c_void,
    size: usize,
    protect: u32,
    old: *mut u32,
) -> BOOL {
    if this_process(process) && size != 0 {
        protect_title(at, size, protect, old)
    } else {
        // SAFETY: the title's arguments.
        unsafe { mem::VirtualProtectEx(process, at, size, protect, old) }
    }
}

/// Allocates at `at` with `allocate` (a host call): over memory the GPU
/// side holds in place, a commit with another protection is refused as
/// locked, so the GPU side lets go and it is tried again
/// (`victory_gpu::title_memory_retry`). Elsewhere, the host's answer. A
/// commit is told to the GPU side (`victory_gpu::title_memory_committed`),
/// which may open there what it could not before.
fn allocate_at(
    at: *const c_void,
    size: usize,
    kind: u32,
    mut allocate: impl FnMut() -> *mut c_void,
) -> *mut c_void {
    let out = if at.is_null() || size == 0 {
        allocate()
    } else {
        let mut out = std::ptr::null_mut();
        victory_gpu::title_memory_retry(at as usize, size, || {
            out = allocate();
            BOOL::from(!out.is_null())
        });
        out
    };
    if !out.is_null() && kind & mem::MEM_COMMIT != 0 {
        // SAFETY: a plain read and restore of this thread's last error,
        // which the title may read after a call that worked.
        let error = unsafe { GetLastError() };
        victory_gpu::title_memory_committed(out as usize, size);
        // SAFETY: as above.
        unsafe { SetLastError(error) };
    }
    out
}

pub extern "system" fn VirtualAlloc(
    at: *const c_void,
    size: usize,
    kind: u32,
    protect: u32,
) -> *mut c_void {
    // SAFETY: the title's arguments.
    allocate_at(at, size, kind, || unsafe {
        mem::VirtualAlloc(at, size, kind, protect)
    })
}

pub extern "system" fn VirtualAllocEx(
    process: HANDLE,
    at: *const c_void,
    size: usize,
    kind: u32,
    protect: u32,
) -> *mut c_void {
    // SAFETY: the title's arguments.
    let allocate = || unsafe { mem::VirtualAllocEx(process, at, size, kind, protect) };
    if this_process(process) {
        allocate_at(at, size, kind, allocate)
    } else {
        allocate()
    }
}

pub extern "system" fn UnmapViewOfFile(at: MEMORY_MAPPED_VIEW_ADDRESS) -> BOOL {
    let a = at.Value as usize;
    // SAFETY: the title's argument.
    change(at.Value, 0, move || unsafe { mem::UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: a as *mut c_void }) })
}

pub extern "system" fn UnmapViewOfFileEx(at: MEMORY_MAPPED_VIEW_ADDRESS, flags: u32) -> BOOL {
    let a = at.Value as usize;
    // SAFETY: the title's arguments.
    change(at.Value, 0, move || unsafe { mem::UnmapViewOfFileEx(MEMORY_MAPPED_VIEW_ADDRESS { Value: a as *mut c_void }, flags) })
}

pub extern "system" fn UnmapViewOfFile2(
    process: HANDLE,
    at: MEMORY_MAPPED_VIEW_ADDRESS,
    flags: u32,
) -> BOOL {
    if this_process(process) {
        let a = at.Value as usize;
        // SAFETY: the title's arguments, in this process.
        change(at.Value, 0, move || unsafe {
            mem::UnmapViewOfFile2(GetCurrentProcess(), MEMORY_MAPPED_VIEW_ADDRESS { Value: a as *mut c_void }, flags)
        })
    } else {
        // SAFETY: the title's arguments.
        unsafe { mem::UnmapViewOfFile2(process, at, flags) }
    }
}

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "memoryapi",
        dlls: &["api-ms-win-core-memory-*", "kernel32.dll", "kernelbase.dll"],
        exports: vec![
            hle_export!(VirtualAlloc),
            hle_export!(VirtualAllocEx),
            hle_export!(VirtualFree),
            hle_export!(VirtualFreeEx),
            hle_export!(VirtualProtect),
            hle_export!(VirtualProtectEx),
            hle_export!(UnmapViewOfFile),
            hle_export!(UnmapViewOfFileEx),
            hle_export!(UnmapViewOfFile2),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_rest_of_an_allocation_spans_its_regions() {
        const PAGE: usize = 4096;
        // SAFETY: plain allocation; the middle page gets other protection,
        // so the allocation is three regions.
        let at = unsafe {
            let at = mem::VirtualAlloc(
                std::ptr::null(),
                3 * PAGE,
                mem::MEM_RESERVE | mem::MEM_COMMIT,
                mem::PAGE_READWRITE,
            ) as usize;
            assert_ne!(at, 0);
            let mut old = 0;
            assert_ne!(
                mem::VirtualProtect(
                    (at + PAGE) as *const c_void,
                    PAGE,
                    mem::PAGE_READONLY,
                    &mut old
                ),
                0
            );
            at
        };
        assert_eq!(rest_of_allocation(at), 3 * PAGE);
        assert_eq!(rest_of_allocation(at + PAGE + 16), 2 * PAGE - 16);
        // The wrappers do what the host's functions do.
        let mut old = 0;
        assert_ne!(
            VirtualProtect(
                (at + PAGE) as *const c_void,
                PAGE,
                mem::PAGE_READWRITE,
                &mut old
            ),
            0
        );
        assert_eq!(old, mem::PAGE_READONLY);
        // A commit at an address with another protection, nothing held.
        let again = VirtualAlloc(at as *const c_void, PAGE, mem::MEM_COMMIT, mem::PAGE_READONLY);
        assert_eq!(again as usize, at);
        assert!(VirtualAlloc(0x1000 as *const c_void, PAGE, mem::MEM_COMMIT, 0).is_null(), "the host's refusal");
        assert_ne!(VirtualFree(at as *mut c_void, 0, MEM_RELEASE), 0);
        assert_eq!(rest_of_allocation(at), 0, "freed");
    }

    /// The entry goes through the GPU side's change even with nothing held
    /// (`victory_gpu::title_memory_change_from`): a change refused as locked
    /// with nothing held or let go there is the host's answer with its last
    /// error, tried at most twice; one that works is done once.
    #[test]
    fn the_entry_keeps_the_host_s_answer_with_nothing_held() {
        use windows_sys::Win32::Foundation::ERROR_INVALID_PARAMETER;
        use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
        // Held by changes that may be queued: 'static.
        static CALLS: AtomicUsize = AtomicUsize::new(0);
        let calls = &CALLS;
        let refused = move || {
            calls.fetch_add(1, Relaxed);
            // SAFETY: plain write of this thread's last error.
            unsafe { SetLastError(ERROR_INVALID_PARAMETER) };
            0
        };
        // An address nothing in this process holds in place.
        let at = 0x7f00_0000_0000usize as *const c_void;
        assert_eq!(change(at, 0x1000, refused), 0);
        // SAFETY: a plain read of this thread's last error.
        assert_eq!(unsafe { GetLastError() }, ERROR_INVALID_PARAMETER);
        assert!((1..=2).contains(&calls.load(Relaxed)), "tried {} times", calls.load(Relaxed));
        calls.store(0, Relaxed);
        assert_eq!(change(at, 0, move || { calls.fetch_add(1, Relaxed); 1 }), 1);
        assert_eq!(calls.load(Relaxed), 1);
    }

    #[test]
    fn the_title_s_memory_functions_are_served_here_and_the_rest_by_the_host() {
        let r = crate::HleResolver::default();
        let name = |n: &str| victory_pe::ImportSymbol::Name {
            hint: 0,
            name: n.into(),
        };
        for dll in [
            "api-ms-win-core-memory-l1-1-0.dll",
            "kernel32.dll",
            "kernelbase.dll",
        ] {
            assert_eq!(
                r.classify(dll, &name("VirtualFree")),
                Some(victory_loader::Provenance::Hle)
            );
        }
        assert_eq!(
            r.classify("api-ms-win-core-memory-l1-1-0.dll", &name("VirtualAlloc")),
            Some(victory_loader::Provenance::Hle)
        );
        assert_eq!(
            r.classify("api-ms-win-core-memory-l1-1-0.dll", &name("VirtualQuery")),
            Some(victory_loader::Provenance::Host)
        );
    }
}
