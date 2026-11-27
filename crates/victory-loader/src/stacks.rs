//! Where every thread is: `VICTORY_STACKS=<seconds>[,<seconds>...]` prints
//! each thread's unwound stack as `module+rva` that many seconds after
//! [`dump_stacks_on_schedule`], to find what a title waits on. An entry
//! `crash` prints them once more at the first unhandled exception (and to
//! the crash log): a title watchdog that ends a hung run with a write to
//! address 0 then shows who waited on what.

use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use windows_sys::Wdk::System::Threading::{
    NtQueryInformationThread, ThreadBasicInformation, ThreadQuerySetWin32StartAddress,
};
use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE, LocalFree};
use windows_sys::Win32::System::Diagnostics::Debug::{
    CONTEXT, CONTEXT_FULL_AMD64, GetThreadContext, ReadProcessMemory,
};
use windows_sys::Win32::System::Diagnostics::ToolHelp::{
    CreateToolhelp32Snapshot, TH32CS_SNAPTHREAD, THREADENTRY32, Thread32First, Thread32Next,
};
use windows_sys::Win32::System::Threading::{
    GetCurrentProcess, GetCurrentProcessId, GetCurrentThreadId, GetThreadDescription, OpenThread, ResumeThread,
    SuspendThread, THREAD_GET_CONTEXT, THREAD_QUERY_INFORMATION, THREAD_SUSPEND_RESUME,
};

use crate::crash::{MAX_FRAMES, locate, unwind_into};

/// Reads `VICTORY_STACKS` and, if set, starts a thread that prints every
/// thread's stack at each listed time (seconds from now).
pub fn dump_stacks_on_schedule() {
    let Ok(list) = std::env::var("VICTORY_STACKS") else {
        return;
    };
    let times = schedule(&list);
    ON_CRASH.store(asks_crash(&list), Ordering::Relaxed);
    let start = std::time::Instant::now();
    std::thread::spawn(move || {
        for at in times {
            std::thread::sleep(at.saturating_sub(start.elapsed()));
            dump_stacks();
        }
    });
}

/// The times in a `VICTORY_STACKS` list (seconds, comma-separated), in
/// order. Negative times are now; times no `Duration` holds (`inf`, `NaN`,
/// `1e20`) are left out instead of panicking the dump thread.
fn schedule(list: &str) -> Vec<std::time::Duration> {
    let mut times: Vec<_> = list
        .split(',')
        .filter_map(|t| t.trim().parse::<f64>().ok())
        .filter(|t| !t.is_nan())
        .filter_map(|t| std::time::Duration::try_from_secs_f64(t.max(0.0)).ok())
        .collect();
    times.sort();
    times
}

/// Whether a `VICTORY_STACKS` list asks for the stacks at a crash too.
fn asks_crash(list: &str) -> bool {
    list.split(',').any(|t| t.trim() == "crash")
}

/// `VICTORY_STACKS` named `crash` and no crash has printed them yet.
static ON_CRASH: AtomicBool = AtomicBool::new(false);

/// How long the crash filter waits for the stacks.
const CRASH_DEADLINE: std::time::Duration = std::time::Duration::from_secs(10);

/// At an unhandled exception: every other thread's stack, once, if asked;
/// the text, for the crash log. Taken on a thread of its own and waited for
/// [`CRASH_DEADLINE`] at most: a thread suspended while it holds a lock the
/// unwinder or the new thread's start needs (the loader's) would hang the
/// filter, and the run would never end. Past it no thread is suspended any
/// more and the one that is, is resumed: the rest of the crash path may
/// need a lock it holds (the heap's, stdio's).
pub(crate) fn dump_stacks_on_crash() -> Option<String> {
    if !ON_CRASH.swap(false, Ordering::Relaxed) {
        return None;
    }
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::Builder::new()
        .name("victory stacks".into())
        .spawn(move || {
            let _ = tx.send(dump_stacks_text());
        })
        .ok()?;
    match rx.recv_timeout(CRASH_DEADLINE) {
        Ok(text) => Some(text),
        Err(_) => {
            GIVE_UP.store(true, Ordering::SeqCst);
            resume_suspended();
            let late = format!("VIctory: the stacks took longer than {CRASH_DEADLINE:?}; left out\n");
            eprint!("{late}");
            Some(late)
        }
    }
}

/// The thread the dump has suspended (its handle), if any, for the crash
/// filter to resume when it stops waiting; and whether it did.
static SUSPENDED: AtomicUsize = AtomicUsize::new(0);
static GIVE_UP: AtomicBool = AtomicBool::new(false);

/// Suspends `thread` for the dump, published in [`SUSPENDED`]: false (and
/// not suspended) once the filter gave up, or took it back meanwhile.
fn suspend(thread: HANDLE) -> bool {
    if GIVE_UP.load(Ordering::SeqCst) {
        return false;
    }
    // One suspended at a time, whichever dump (a scheduled one, a crash).
    while SUSPENDED.compare_exchange(0, thread as usize, Ordering::SeqCst, Ordering::SeqCst).is_err() {
        if GIVE_UP.load(Ordering::SeqCst) {
            return false;
        }
        std::thread::yield_now();
    }
    // SAFETY: a thread of this process that is not this one.
    if unsafe { SuspendThread(thread) } == u32::MAX {
        SUSPENDED.store(0, Ordering::SeqCst);
        return false;
    }
    if SUSPENDED.load(Ordering::SeqCst) != thread as usize {
        // The filter resumed it, maybe before the suspend: once more.
        // SAFETY: suspended above.
        unsafe { ResumeThread(thread) };
        return false;
    }
    true
}

/// Resumes `thread` unless the filter already did.
fn resume(thread: HANDLE) {
    if SUSPENDED.compare_exchange(thread as usize, 0, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
        // SAFETY: suspended by `suspend`.
        unsafe { ResumeThread(thread) };
    }
}

/// The filter's side: resumes the thread the dump holds suspended.
fn resume_suspended() {
    let thread = SUSPENDED.swap(0, Ordering::SeqCst);
    if thread != 0 {
        // SAFETY: a thread handle the dump holds open while it is published.
        unsafe { ResumeThread(thread as HANDLE) };
    }
}

#[link(name = "ntdll")]
unsafe extern "system" {
    fn NtQuerySystemInformation(class: u32, info: *mut c_void, len: u32, returned: *mut u32) -> i32;
}

/// `SYSTEM_HANDLE_TABLE_ENTRY_INFO_EX` (ntddk).
#[repr(C)]
struct HandleEntry {
    object: usize,
    process: usize,
    handle: usize,
    access: u32,
    trace: u16,
    kind: u16,
    attributes: u32,
    reserved: u32,
}

/// This process's open handles, sorted: listed by the system, so no
/// value is probed (an invalid one raises under strict handle checks).
fn open_handles() -> Vec<u64> {
    const EXTENDED_HANDLE_INFORMATION: u32 = 64;
    const LENGTH_MISMATCH: i32 = 0xc000_0004_u32 as i32;
    // SAFETY: a plain query.
    let me = unsafe { GetCurrentProcessId() } as usize;
    // Words: a count, a reserved word, then the entries.
    let mut buffer: Vec<u64> = vec![0; 1 << 17];
    loop {
        let mut needed = 0u32;
        let bytes = u32::try_from(buffer.len() * 8).unwrap_or(u32::MAX);
        // SAFETY: writes at most `bytes` into `buffer`.
        let status = unsafe { NtQuerySystemInformation(EXTENDED_HANDLE_INFORMATION, buffer.as_mut_ptr().cast(), bytes, &mut needed) };
        if status == LENGTH_MISMATCH && buffer.len() < 1 << 25 {
            let more = (needed as usize / 8 + 4096).max(buffer.len() * 2);
            buffer.resize(more, 0);
            continue;
        }
        if status != 0 {
            return Vec::new();
        }
        break;
    }
    let fit = (buffer.len() - 2) * 8 / size_of::<HandleEntry>();
    let count = (buffer[0] as usize).min(fit);
    // SAFETY: the system wrote `count` entries after the two header words.
    let entries = unsafe { std::slice::from_raw_parts(buffer.as_ptr().add(2).cast::<HandleEntry>(), count) };
    let mut open: Vec<u64> = entries.iter().filter(|e| e.process == me).map(|e| e.handle as u64).collect();
    open.sort_unstable();
    open
}

/// `THREAD_BASIC_INFORMATION` (ntddk).
#[repr(C)]
struct BasicInfo {
    exit_status: i32,
    teb: *const usize,
    client_id: [usize; 2],
    affinity: usize,
    priority: i32,
    base_priority: i32,
}

/// `GetThreadContext` wants a 16-byte aligned context.
#[repr(C, align(16))]
struct Aligned(CONTEXT);

/// Prints every other thread's stack to stderr. Each thread is suspended
/// only while its context is read and unwound into a buffer allocated
/// beforehand (it may hold the heap lock).
pub fn dump_stacks() {
    dump_stacks_text();
}

/// [`dump_stacks`], also returning what it printed.
fn dump_stacks_text() -> String {
    use std::fmt::Write;
    let mut text = String::new();
    let mut line = |s: String| {
        eprintln!("{s}");
        let _ = writeln!(text, "{s}");
    };
    // SAFETY: plain queries and a snapshot of this process's threads.
    let (me, process, snapshot) = unsafe {
        (
            GetCurrentThreadId(),
            GetCurrentProcessId(),
            CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0),
        )
    };
    if snapshot == INVALID_HANDLE_VALUE {
        return text;
    }
    let mut ids = Vec::new();
    // SAFETY: zeroed is a valid THREADENTRY32; its size is set before use.
    let mut entry: THREADENTRY32 = unsafe { std::mem::zeroed() };
    entry.dwSize = size_of::<THREADENTRY32>() as u32;
    // SAFETY: `snapshot` is open; `entry` is sized.
    let mut more = unsafe { Thread32First(snapshot, &mut entry) } != 0;
    while more {
        if entry.th32OwnerProcessID == process && entry.th32ThreadID != me {
            ids.push(entry.th32ThreadID);
        }
        // SAFETY: as above.
        more = unsafe { Thread32Next(snapshot, &mut entry) } != 0;
    }
    // SAFETY: we opened it.
    unsafe { CloseHandle(snapshot) };
    let open = open_handles();
    line(format!("VIctory: stacks of {} threads", ids.len()));
    for id in ids {
        // SAFETY: opens a thread of our own process.
        let thread = unsafe {
            OpenThread(
                THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION,
                0,
                id,
            )
        };
        if thread.is_null() {
            continue;
        }
        let (frames, (priority, base), at) = frames_of(thread);
        line(format!(
            "  thread {id} \"{}\" start {} priority {priority} (base {base}) rip {} rsp {:#x} handles near rsp {:x?}",
            name_of(thread),
            locate(start_of(thread)),
            locate(at.rip as usize),
            at.rsp,
            at.handles(&open)
        ));
        for (i, pc) in frames.into_iter().enumerate() {
            line(format!("    #{i:<2} {}", locate(pc as usize)));
        }
        // SAFETY: we opened it.
        unsafe { CloseHandle(thread) };
    }
    text
}

/// Words read from the top of a stopped thread's stack.
const STACK_WORDS: usize = 64;

/// Where a thread stopped, and the top of its stack. A thread in a system
/// call keeps no argument register: `syscall` overwrites `rcx` with the
/// return address and the kernel does not keep `r10`. The handle a wait
/// waits on is still among its callers' spills and locals near `rsp`.
#[derive(Clone, Copy)]
struct Stopped {
    rip: u64,
    rsp: u64,
    stack: [u64; STACK_WORDS],
}

impl Default for Stopped {
    fn default() -> Self {
        Stopped { rip: 0, rsp: 0, stack: [0; STACK_WORDS] }
    }
}

impl Stopped {
    /// Words near `rsp` that are in `open` (sorted), each once: what a
    /// thread waiting in the kernel may wait on.
    fn handles(&self, open: &[u64]) -> Vec<u64> {
        let mut found: Vec<u64> = Vec::new();
        for &w in &self.stack {
            if w != 0 && !found.contains(&w) && open.binary_search(&w).is_ok() {
                found.push(w);
            }
        }
        found
    }
}

/// The thread's frames, its current and base scheduling priority (0-31),
/// and where it stopped.
fn frames_of(thread: HANDLE) -> (Vec<u64>, (i32, i32), Stopped) {
    let mut frames = Vec::with_capacity(MAX_FRAMES);
    let mut info: BasicInfo = BasicInfo {
        exit_status: 0,
        teb: std::ptr::null(),
        client_id: [0; 2],
        affinity: 0,
        priority: 0,
        base_priority: 0,
    };
    // Suspended first: a running thread may exit and free its TEB between
    // the query and the read below.
    if !suspend(thread) {
        return (frames, (0, 0), Stopped::default());
    }
    // SAFETY: writes `info`, whose layout and size are the system's.
    let status = unsafe {
        NtQueryInformationThread(
            thread,
            ThreadBasicInformation,
            (&raw mut info).cast(),
            size_of::<BasicInfo>() as u32,
            std::ptr::null_mut(),
        )
    };
    if status != 0 || info.teb.is_null() {
        resume(thread);
        return (frames, (0, 0), Stopped::default());
    }
    // SAFETY: the TEB of a suspended thread of this process: its NT_TIB
    // holds the stack base at +8 and the stack limit at +0x10.
    let (high, low) = unsafe { (info.teb.add(1).read(), info.teb.add(2).read()) };
    let mut stopped = Stopped::default();
    // SAFETY: zeroed is a valid CONTEXT.
    let mut context = Aligned(unsafe { std::mem::zeroed() });
    context.0.ContextFlags = CONTEXT_FULL_AMD64;
    // SAFETY: reads and resumes the thread suspended above; nothing until
    // the resume allocates or takes a lock the thread could hold besides
    // the unwinder's own.
    unsafe {
        if GetThreadContext(thread, &mut context.0) != 0 {
            unwind_into(context.0, low, high, &mut frames);
            stopped.rip = context.0.Rip;
            stopped.rsp = context.0.Rsp;
            // Within the stack only; a short read keeps what it got.
            let len = (high as u64).saturating_sub(stopped.rsp).min(size_of_val(&stopped.stack) as u64) as usize;
            let mut done = 0;
            ReadProcessMemory(
                GetCurrentProcess(),
                stopped.rsp as *const c_void,
                stopped.stack.as_mut_ptr().cast(),
                len,
                &mut done,
            );
        }
    }
    resume(thread);
    (frames, (info.priority, info.base_priority), stopped)
}

fn start_of(thread: HANDLE) -> usize {
    let mut start = 0usize;
    // SAFETY: writes one pointer-sized start address.
    unsafe {
        NtQueryInformationThread(
            thread,
            ThreadQuerySetWin32StartAddress,
            (&raw mut start).cast(),
            size_of::<usize>() as u32,
            std::ptr::null_mut(),
        )
    };
    start
}

fn name_of(thread: HANDLE) -> String {
    let mut name = std::ptr::null_mut();
    // SAFETY: the system allocates the string; we free it with LocalFree.
    unsafe {
        if GetThreadDescription(thread, &mut name) < 0 || name.is_null() {
            return String::new();
        }
        let len = (0..).take_while(|&i| *name.add(i) != 0).count();
        let text = String::from_utf16_lossy(std::slice::from_raw_parts(name, len));
        LocalFree(name.cast::<c_void>());
        text
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_takes_times_and_crash() {
        assert!(!asks_crash("60, 20"));
        assert!(asks_crash("crash") && schedule("crash").is_empty());
        assert!(asks_crash("30, crash") && schedule("30,crash").len() == 1);
    }

    #[test]
    fn another_threads_stack_unwinds_to_its_start() {
        let (tx, rx) = std::sync::mpsc::channel();
        let (stop_tx, stop_rx) = std::sync::mpsc::channel::<()>();
        let parked = std::thread::spawn(move || {
            // SAFETY: plain query.
            tx.send(unsafe { GetCurrentThreadId() }).unwrap();
            stop_rx.recv().ok();
        });
        let id = rx.recv().unwrap();
        // SAFETY: a thread of this process.
        let thread = unsafe {
            OpenThread(
                THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION,
                0,
                id,
            )
        };
        assert!(!thread.is_null());
        let (frames, _, _) = frames_of(thread);
        // Waiting in the kernel, several frames down from the thread start.
        assert!(frames.len() >= 3, "{frames:?}");
        assert!(dump_stacks_text().contains(&format!("thread {id} ")));
        assert!(start_of(thread) != 0);
        // SAFETY: opened above.
        unsafe { CloseHandle(thread) };
        dump_stacks();
        stop_tx.send(()).unwrap();
        parked.join().unwrap();
    }

    /// A thread waiting on an event in the kernel: the event is among the
    /// handles near its `rsp`, and `rip` is in ntdll's stub.
    #[test]
    fn a_waiting_thread_shows_the_handle_it_waits_on() {
        use windows_sys::Win32::System::Threading::{CreateEventW, INFINITE, SetEvent, WaitForSingleObject};
        // SAFETY: an unnamed manual-reset event, closed below.
        let event = unsafe { CreateEventW(std::ptr::null(), 1, 0, std::ptr::null()) };
        assert!(!event.is_null());
        let (tx, rx) = std::sync::mpsc::channel();
        let at = event as usize;
        let waiter = std::thread::spawn(move || {
            // SAFETY: plain query.
            tx.send(unsafe { GetCurrentThreadId() }).unwrap();
            // SAFETY: the event lives until this thread is joined.
            unsafe { WaitForSingleObject(at as HANDLE, INFINITE) };
        });
        let id = rx.recv().unwrap();
        // SAFETY: a thread of this process.
        let thread = unsafe { OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, 0, id) };
        assert!(!thread.is_null());
        // In the wait once its stub is where it stays.
        let open = open_handles();
        assert!(open.binary_search(&(at as u64)).is_ok(), "the system lists the event");
        let mut stopped = Stopped::default();
        for _ in 0..200 {
            stopped = frames_of(thread).2;
            if stopped.handles(&open).contains(&(at as u64)) {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(stopped.handles(&open).contains(&(at as u64)), "{:x?} has {at:#x}", stopped.handles(&open));
        // The filter giving up resumes the thread the dump holds.
        assert!(suspend(thread));
        resume_suspended();
        // SAFETY: as above; the previous count is 0 once resumed.
        assert_eq!(unsafe { SuspendThread(thread) }, 0, "resumed");
        // SAFETY: suspended just above.
        unsafe { ResumeThread(thread) };
        resume(thread);
        assert!(locate(stopped.rip as usize).starts_with("ntdll"), "{}", locate(stopped.rip as usize));
        // SAFETY: ours; the waiter returns, then both handles go.
        unsafe {
            SetEvent(event);
            waiter.join().unwrap();
            CloseHandle(thread);
            CloseHandle(event);
        }
    }

    /// Times no `Duration` holds are left out instead of panicking the dump
    /// thread (`from_secs_f64` panicked on `inf`); the rest come in order.
    #[test]
    fn schedules_skip_times_out_of_range() {
        let secs = |list| {
            schedule(list)
                .iter()
                .map(|d| d.as_secs_f64())
                .collect::<Vec<_>>()
        };
        assert_eq!(secs("30, 5,x,-2,inf,NaN,1e20"), [0.0, 5.0, 30.0]);
        assert!(schedule("").is_empty());
    }
}
