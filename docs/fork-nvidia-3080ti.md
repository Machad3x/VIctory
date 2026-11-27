# Fork notes: NVIDIA RTX 3080 Ti

This fork stacks open upstream pull requests on `main` (940d32f, then merged
with `main` at 67e8ae1) and a few local changes, aimed at one machine: Ryzen 7 5700X, RTX 3080 Ti 12 GB
(driver 621.40 since 26 November, 621.24 before), 32 GB RAM, Windows 11.
Nothing here is tested elsewhere.

## How the console and this host differ

| | Xbox Series | RTX 3080 Ti |
|---|---|---|
| Memory | One GDDR6 pool; CPU and GPU use the same addresses | Separate video memory; the host keeps copies of title memory and writes GPU results back |
| Waves | 64 and 32 lanes | 32 lanes only: wave64 programs need translation to two host waves |
| Rectangle lists | A hardware primitive | Emulated with a geometry shader |
| View descriptors | The title lays out 1,245,184 | Shader-visible heaps are capped at 1,000,000 |

## Changes and why

| Problem seen | Cause | Change |
|---|---|---|
| Black world, device removed | Raw loads through structured views read 0; view-address tables did not cover tables made afresh | #1762, #1906, #2098 |
| Over-exposed, unlit picture | Wave64 lighting passes refused | #439 (replaces #1960 here) |
| Dynamic descriptor slots dropped | Capped view heap left too little room | #1751, #1548 and a local change keeping room for view pairs |
| Saw-tooth triangles on a tile grid | The fourth corner of a rectangle was added across the wrong diagonal | #2366 |
| Stretched triangles and missing heads | The ray-tracing builder ran only in part: 5 of its 52 programs (wave64 in groups of more than 64 threads) did not translate, and the others wrote results computed from data never produced | b3efa1c from #2214 (console flow for groups of several console waves); offline all 52 programs translate and validate |
| About 240 dispatches a frame dropped without a message, up to 2,171 while the world loads: the ray-tracing builder never finished its structures | The host took any user-data dword numbered 4 for a root view offset, so root constants with five or more dwords found no view | Local: only a root descriptor's entry takes a view offset |
| Compute passes with implicit-derivative samples refused in the console flow | Such a sample is not valid in a split wave64 | Local: a gradient sample with the quad's gradients from lane reads |
| Descriptors chosen by control flow refused | The choice was not followed through nested joins and selects | #2342 |
| One-off passes lost at the world's first frame (lighting tables) | Their pipelines waited behind more than a thousand others, past the half-second budget | Local: passes over the screen go first and may be waited for up to 20 s (idea from Belobug's fork) |

Other upstream pull requests in the stack: #688, #1065, #1763, #1800, #1883,
#1989, #1993, #1750, #2062, #2088, #2136, #2175, #2242, #2289, #2324, #2325,
#2229, #2332, #2342, #2345, #2346, #2348, #2355, #2377, #2382, #2402.

## Last measured run

6 minutes with the root view offset fix, against the same stack without it
(two runs, 15 and 8 minutes), steady state in the world:

| Mean of the last 20 reports | Without (2 runs) | With (1 run) |
|---|---|---|
| Draws and dispatches dropped per frame | 252-253 | 18 |
| Copy-back per frame | 231-272 ms | 156 ms |
| Time on the GPU per frame | 488-531 ms | 551 ms |
| Frame time | 904-1000 ms | 901 ms |

A later 10-minute run after a cache warm-up, with #2229 and #2402 added,
kept 18 drops and about 0.9 s a frame; its screenshots show the walls around
the cell window drawn (unlit), where earlier runs showed sky, and the yard
with people outside.

No device removal in either. Picture quality varies between moments of a run:
runs with and without the fix both have dark and broken frames, and it is not
the intended picture yet (interior walls, hair and window glass are missing).
The screen-pass priority has not been exercised by a run without a pipeline
cache.

## Merge of upstream `main` (82 commits)

Upstream's newer mechanisms replace some stacked changes (rectangle corners of
#2366, two memory and copy changes). Offline, refused containers went from 260
to 223; all 52 builder programs translate. In the game, the cell window shows
its wired glass and the yard behind it, far closer to the intended picture.

| Steady state, no pipeline being made | Before | Merged |
|---|---|---|
| Frame time | about 0.9 s | 0.9 to 1.1 s |
| Title memory copied per frame | about 98 MB | about 400 MB |
| Video memory used | not recorded | 10.9 of 11.8 GB (textures are given up past 90%) |

The copies and the memory use are open problems. The runs so far had
`VICTORY_SHADER_LOOP_LIMIT=4096` set by `autotest.ps1`: translated loops stop
after 4,096 iterations instead of 2^20.

## After the merge

| Change | What it addresses |
|---|---|
| #2626 | The 16-bit packed colour formats: targets this driver cannot write are RGBA8 on the host, with texels converted. Replaces the stand-in that read four-bit colours as five-bit ones |
| #2601 | Textures with host writes still pending are not given up |
| #2598 | Giving up a texture leaves the dynamic slots of recorded lists alone |
| #2576 | Read marks of forgotten slots are cleared before the submission runs |
| #2611 | A buffered piece of a command stream runs before its memory wait is evaluated |

A 10-minute run with these reached play inside the cell (about 7,000 draws a
frame, 1.0 to 1.2 s, 29 dropped from seven programs that do not translate), with
no device removal. The cutscenes looked better and the lighting followed the
character; colour banding remains.

Of the view heap, the title wrote 1,002,400 distinct slots in a 10-minute run,
in 15,677 of its 32,768 blocks of 64: more than the host heap holds.

### Frame time

A frame is three steps in series: preparing (about 200 ms, title memory copies),
the GPU (about 550 ms, each submission waited for) and the copy-back. At the
window scene, copy-back was 181 ms in a run without #2280 and #2409 and 135 ms
in one with them (means of 20 reports, one run each). The two runs also differ
by #2626, #2601, #2598, #2576 and #2611, so the gain is not isolated.

Title memory copied per frame stays near 400 MB. Some list reads memory with no
host copy every frame, so what lists using title memory wrote (about 280 MB of
views) is withheld and then copied again from title memory. The diagnostics
build shows misses both on committed, mapped title memory, which the console
would read, and on addresses with nothing there.

Copies made for the missed pages (local branch `claude/demand-copies`) did not
lower the frame time: unbounded they reached 11 GB and 18 s of copy-back a
frame, and bounded to 512 MiB the budget filled while the world loaded, with
the window scene still missing 70 to 90 pages a frame.

## Copies in video memory

Host copies of title buffers are in system memory, so shaders read them over
PCIe. `autotest.ps1` sets `VICTORY_GPU_UPLOAD=1` (#2120: buffers the title makes
without unordered access) and `VICTORY_GPU_UPLOAD_WRITES=1` (#2130: the others,
with a mirror in system memory that the CPU reads). Both need resizable BAR.

Window scene, means of 20 reports, one run each:

| | Frame | Preparing | GPU | Copy-back |
|---|---|---|---|---|
| Neither | 919 ms | 203 | 577 | 135 |
| #2120 | 881 ms | 189 | 557 | 131 |
| #2130, mirror always written | 869 ms | 288 | 296 | 282 |
| #2130, lazy mirror | 792 ms | 270 | 334 | 183 |

The GPU step is much shorter; the CPU pays part of it back. A copy shaders
write must reach title memory through the mirror, so a missed update there
corrupts memory silently: set `VICTORY_GPU_UPLOAD_WRITES=0` first when
something new looks broken. `victory-gpu` tests pass with both switches on.

## Copy-back on several threads

Local change: the marked pages of a submission go back to title memory on up to
six threads, each page on one thread (about 2,000 pages of 64 KiB a frame).
Same scene and switches, medians of 30 and 35 reports, one run each:

| | Frame | Copy-back | Marked pages |
|---|---|---|---|
| One thread | 777 ms | 184 ms | 166 ms |
| Several | 690 ms | 124 ms | 111 ms |

Two reports of the second run fell in a burst of streaming (1.4 to 2.0 GB
copied a frame), which its means include and its medians do not.

## Refresh threads and smaller per-draw costs

| Change | What it addresses |
|---|---|
| #2051 | A submission's refreshes copy on several threads. Ported: mirrored copies and ranges with bytes saved for a copy-back still copy at once |
| #1978 | No texture layouts or name copies per draw for unchanged answers |
| #2102 | Per-list and per-submission sets hash with FxHash |
| #2724 | Texture reads at a computed table entry keep their read mark |

One run with the four (`-Warm`), the character moved during it, so the scene
numbers are indicative: median window frame 717 ms (690 in the run before),
copies 84 ms for 414 MB (95 ms), copy-back 122 ms. No device removal, the same
seven programs untranslated, and rooms of up to 13,600 draws a frame were
walked through.

## Descriptors read as data

#2363, with the record count and format derived from the view (the pull
request's head copies host-layout words there). The program that failed with
"a descriptor read only as data" translates, leaving six untranslated. One run:
median window frame 733 ms (717 before, within the spread between runs), and
the window scene showed the lit block wall, the poster and the shirt lettering.

## Permlane and half sums

#1737 (`v_permlane16_b32`, `v_permlanex16_b32`) and a local change: whole-wave
16-bit float sums in a wave64 program split in halves. One 7-minute run each,
window scene medians:

| | Before | After |
|---|---|---|
| Programs not translated in the run | 6 | 4 |
| Draws and dispatches dropped per frame | 15 | 1 |
| Shader containers refused offline | 223 | 218 |
| Frame time | 717-733 ms (two runs) | 669 ms |
| GPU step | 327-347 ms | 310 ms |

No device removal. The cell walls were as dark in this run as in some earlier
ones; the lighting at the window still differs from run to run.

## Geometry, depth twins and texture decoding

| Change | What it addresses |
|---|---|
| #2729 | One rectangle corner order across vertex streams |
| #2737 | Depth textures and their writable twins copy directly; D32 twins exist |
| #2736, #2735 | Faster tiled texture copies and BC7 BCPack decoding |
| #2786 | Texture refresh workers are kept between submissions |
| #2365, #2382 | Whole-wave sections: a scalar's halves kept apart, better refusal reasons |

Window scene, median of one run: 659 ms a frame, GPU step 308 ms (669 and
310 before), no device removal.

## Whole-wave sections in partial groups

Local change: wave64 compute programs with a whole-wave section run in groups
that are not whole console waves (1, 32 and 36 threads here, among them the
exposure pass's average luminance and a deferred lighting pass). Offline,
refused containers went from 218 to 214. In one run, three programs stay
untranslated and the window scene's typical frame dropped none. The GPU step
settled at 304 to 320 ms, after a first part of the run at up to 460 ms
(median 331 ms, against 308 before).

## Partial image stores, restored masks, first-lane reads

| Change | What it addresses |
|---|---|
| #332 | `image_store` with partial channel masks translates |
| #2816 | A wave64 mask restored dword by dword is recognised as that mask; a test that expected a refusal now runs on the GPU and takes the right lanes |
| #2831 | Dwords read from the first lane outside a waterfall are one lane's |

Offline, refused containers went from 214 to 213. One 7-minute run each,
window scene medians:

| | Before | After |
|---|---|---|
| Frame time | 694 ms | 670 ms |
| GPU step | 331 ms | 307 ms |
| Copy-back | 120 ms | 112 ms |

No device removal, the same three programs untranslated, and the window,
yard and character drawn as intended; the cell walls were black in this run.

## Series S

One 7-minute run with `-Console series-s`, before the three changes above:
the title reached play (3,800 to 5,800 draws a frame, 0.5 to 1.9 s) with no
device removal, but the picture turned grey and then black after the opening
card. Seen in its log, none confirmed as the cause: four compute programs
whose whole-wave sections do not translate besides the two water ones, one
refused for a partial-channel image store (#332 addresses that), and a
1920x1080 target in format 67 with no host texture (#2832 addresses that;
not ported, it overlaps #2626 here).

The format 67 target is now handled by a stand-in; see "Series S profile" below.

## Thread-count dispatches and placed texture writes

| Change | What it addresses |
|---|---|
| #2841 | `DispatchX` with the thread-dimensions flag (0x20) counts threads, not groups: the ray-tracing builder dispatches this way and ran many times its groups |
| Local | The bounded form's entry guard numbers console lanes in its own block (with the console flow, #2841 alone panicked at world load) |
| #1273 | Unordered views of textures the title placed bind the texture instead of a null view |
| Local, as #2851 | A null shader view has no aliased copy (#1273 alone ended the process at world load) |

One 7-minute run, window scene medians:

| | Before | After |
|---|---|---|
| Frame time | 670 ms | 688 ms |
| GPU step | 307 ms | 299 ms |
| Copy-back | 112 ms | 138 ms |

No device removal, the same three programs untranslated. The picture is the
closest to the intended one so far: the cell walls, the poster, the shirt
lettering and the hair are lit. Interior lighting has differed between runs
before, and this is one run. The longer copy-back is not explained yet, and
blocky patches remain along the bottom.

## Wide indirect dispatches

| Change | What it addresses |
|---|---|
| Local, after avatarmanz 84eae352 | Indirect dispatches of more than 65,535 groups run whole: the title's tile lists are a group per tile, and the cut tiles were the lowest of the picture. This driver runs them without spreading them over z (upstream issue 2256) |
| #2850 | Constant buffers of a reused descriptor table are refreshed |

The blocky patches along the bottom of the window scene are gone in the
screenshots of one 7-minute run (they were in the run before). Window scene
medians: 692 ms a frame, GPU step 297 ms (675 and 314 the run before). The
cell walls were black in both of these runs.

## Rebased on upstream 63bcca43

The fork's changes now sit on upstream main as commits by area (formats;
loader, runtime and config; shaders and GPU; docs and tools) instead of a
history with merges. Where both had a version of the same change, upstream's
was kept, except:

- Title-memory page bitmaps stay in video memory (not the pooled bitmaps of
  #1865), and copy-backs keep this fork's comparison (not #2054's).
- #2172 is joined with this fork's queued and mirrored refreshes: tracked
  memory takes a refreshed range in one call.
- Integer render targets keep this fork's blend change; the depth bounds
  test of upstream is in.

Also new here: barycentrics read directly come from `SV_Barycentrics`,
rounding modes set from constant SGPRs and kept through whole-wave sections,
and a side effect inside a whole-wave section refused (the last two from
Yumaste's changes).

Runs on the cell window, one each, 4,900 to 5,100 draws:

| | before the rebase | after |
|---|---|---|
| Frame | 708 ms | 671 ms |
| Preparing | 261 ms | 225 ms |
| GPU | 305 ms | 318 ms |
| Copy-back | 136 ms | 121 ms |
| Draws dropped | 0 | 0 |

The run after the rebase showed lit walls and the poster; interior lighting
has differed between runs of one build before, so one run does not settle it.
The two programs still untranslated in it are whole-wave sections of the
water shading. Unmodified upstream main stopped on this machine 39 s in with
an access violation inside the title's image (not looked into).

The first run after a translator change is black in the world for minutes
while the host pipelines are made (about 2,300 queued here, done during the
second run).

## Rendering ports on the rebase

On top of the rebase, taken from open upstream pull requests: #2928, #2923,
#2915, #2917 (shader semantics), #2882 with #2896, #2782, #2771, #2884,
#2926, #2802, #2830, #2843, #2873, #2879 (64 KiB render-target XOR tiling),
#2895 (descriptor table reuse) and #2810 (a deferred copy of memory the
submission's shaders write runs after the copy-backs). The wait for a host
submission also checks the fence value (as #2899). From other forks:
interpolation modes from the console container's `PSV0`, and the textures of
aliasing and unordered access barriers held until the list is reset.

Last run (warm shader cache, cell window, 4,900 to 5,100 draws): median
637 ms a frame, 298 ms on the GPU, no draws dropped, no device removal; the
block walls and the poster were lit. Interior walls still come out black in
some runs of the same build (open). #2810 did not change that; the cutscene
looked better with it.

## Marked pages: comparison and guard changes

Putting a marked page back cost about 165 µs summed over the threads: 55 to
62 µs comparing it byte by byte, and about 130 µs in the three steps that take
the guard lock (opening the page, marking it, taking it again), most of that
waiting while another thread changed a page's protection. Now:

- A block that differs is compared 8 bytes at once (same runs as before).
- Up to 4 marked pages next to each other in one copy are compared, then
  opened once and taken once. Pages with nothing to write, or that keep the
  title's bytes, are not opened.

Cell window, 4,900 to 5,100 draws, one run each, medians:

| | Before | 8-byte comparison | And runs of pages |
|---|---|---|---|
| Marked pages | 115 ms | 102 ms | 91 ms |
| Copy-back | 128 ms | 118 ms | 104 ms |
| Frame | 644 ms | 644 ms | 635 ms |

The frame difference is within the noise between runs (about 5 %). No draws
dropped and no device removal.

## One mark per wave

From upstream pull request #2958: lanes whose write or read mark is the
leading lane's leave the load and the atomic on the bitmap, which is in system
memory, to that lane. With it #3013 (`Tracked::taken` leaves pages the
tracker does not hold writable).

Same scene, warm cache, medians of two runs against the one before:

| | Before | Run 1 | Run 2 |
|---|---|---|---|
| Frame | 635 ms | 488 ms | 519 ms |
| GPU step | 311 ms | 168 ms | 167 ms |
| Copy-back | 104 ms | 111 ms | 119 ms |

No draws dropped and no device removal. The last screenshot of run 1 shows
rectangular blocks on the character's hair; they were not seen again in the
17 screenshots of run 2, whose screenshot during the transition into the cell
has one hard-edged lit rectangle. Whether either comes from this change is not
established.

## Root UAVs only read

From upstream pull request #2948: the translator records which root UAVs a
shader may write, and the host declares writes for those only. The snapshots
and the batch copies of GPU-written memory they caused are gone.

| | Before (2 runs) | Run 1 | Run 2 |
|---|---|---|---|
| Frame | 488, 519 ms | 496 ms | 453 ms |
| Preparing | 205, 224 ms | 186 ms | 181 ms |
| Snapshots | 24, 30 ms | 0 | 0 |
| Marked pages | 101, 100 ms | 128 ms | 92 ms |

The frame times overlap the noise between runs; preparing is lower in both.
The higher marked-pages time of run 1 did not repeat. No draws dropped and no
device removal.

Also taken: #3031's fix (a second, overlapping copy-back of a submission took
the title bytes the first had put into the copy for the shaders'). Its test
failed here before it. One run after: 448 ms median, nothing dropped.

## Copy-back through a view of our own

The title's physical pages are slices of pagefile sections, so `xmem` maps
each section once more, writable, for the layer alone. Marked pages go back
through that view: the title's views stay read-only for the guards, nothing is
opened or taken again, and a title write still faults. The trackers other than
the writing copy's are marked changed before and after the write, and so are
those at the other addresses showing the pages. The idea is fxceless's fork's
(`VICTORY_BACK_OWN_VIEW`); the code here is this tree's.
`VICTORY_BACK_OWN_VIEW=0` writes through the title's views as before.

Cell window, one run each, medians:

| | Before | Run 1 | Run 2 |
|---|---|---|---|
| Frame | 448 ms | 418 ms | 430 ms |
| Copy-back | 102 ms | 76 ms | 84 ms |
| Marked pages | 102 ms | 71 ms | 78 ms |

Run 2 has the fixes from a review: the pages stay busy while written (a take
meanwhile could copy part of a write), the other addresses showing them at
the write's start are marked at its end, and a section's view is unmapped
only with no write under way. No draws dropped and no device removal.

## Copies of title memory in user mode

Upstream pull request #1010 (open there): copies from title memory into host
memory run in a small routine of the layer instead of `ReadProcessMemory`; a
vectored handler turns a fault of that routine into a failed copy, and a
guard page it met stays a guard page. Copies into title memory still go
through the kernel.

Cell window, one run each, medians (before: the two runs of the previous
section's second build and its review):

| | Before | After |
|---|---|---|
| Frame | 418 and 430 ms | 407 ms |
| Preparing | 169 and 173 ms | 151 ms |
| Copies | 93 and 95 ms | 77 ms |

No draws dropped and no device removal. One known gap, not changed: the
handler puts a title's guard page back without the write watch's lock, so a
guard page the title set inside watched memory could race the watch.

## Released buffers, fills and integer targets

Fixes after a review of the fork's own changes.

| Change | What was wrong |
|---|---|
| Copies go quiet when their memory is freed (from upstream pull request #3036, extended) | A pending copy-back wrote into addresses the title had taken back. Upstream's version stops at the release of the resource object and covers one of the write-back paths: here the marked pages, the copies of views and released placed buffers whose heap is freed later are covered, and releasing an object whose memory stays (placed, title-mapped) still writes back |
| A deferred fill never waits for the copy-backs | Held back, it ran after the lists that read its destination: they read memory never filled |
| Integer render targets | The format list held 37 and 39 for 36 and 38, and `IndependentBlendEnable` off was ignored: pipelines refused, their draws dropped |
| Heap textures walked once a submission, copy-back threads kept, page table hashing | Repeated work and thread creation per submission |
| Copy counters | The per-buffer line summed bytes asked for, not copied; the copy time held some flushes twice. Older logs do not compare in these two numbers |

Not covered: memory the title unmaps itself is not told to the copies, and a
copy-back already under way when memory is freed can still write.

Cell window, one run each, medians:

| | Before (#1010) | After |
|---|---|---|
| Frame | 407 ms | 406 ms |
| Preparing | 151 ms | 154 ms |
| Copy-back | 81 ms | 78 ms |
| Marked pages | 73 ms | 73 ms |

No gain measured at the frame. One run of an intermediate build measured 457 ms
with marked pages at 101 ms, and one binary has given 68 to 119 ms there across
runs: marked pages need more than one run to compare. No draws dropped and no
device removal.

## Marked pages into video memory

The marked copy-back pushed every compared page from its mirror into video
memory, though the pull before it had made the two equal. Now only a page
whose mirror took title bytes is pushed (`VICTORY_PUSH_ALL=1` pushes all).

Cell window, one run, medians:

| | Before (`gpu184`) | After |
|---|---|---|
| Frame | 406 ms | 390 ms |
| Copy-back | 78 ms | 67 ms |
| Marked pages | 73 ms | 61 ms |

One run: marked pages have varied that much between runs of one binary, and an
intermediate build without this change gave 64 ms. No draws dropped and no
device removal.

## Withheld ranges give up marked pages only

A range withheld because a list read memory with no copy left the host copies
whole, so all of it was copied from title memory again, about 148 MB a frame in
the cell scene. A counting run showed every such range marking the pages its
shaders write, with about 59 % of the pages unmarked. Only the marked pages are
dropped now (`VICTORY_RETAKE_ALL=1` drops whole ranges). Review found that the
marks in the last word of the bitmap were cleared unread; no copy gets bits
there any more.

Cell window, one run each, medians:

| | Before (`gpu189`) | After (`gpu190`) |
|---|---|---|
| Copies | 397 MB, 59 ms | 232 MB, 43 ms |
| Preparing | 147 ms | 129 ms |
| Marked pages | 62 ms | 80 ms |
| Copy-back | 70 ms | 89 ms |
| Frame | 384 ms | 377 ms |

The copies shrink as counted. The frame gain is within the noise: marked pages
rose in this run, as they have between runs of one binary (61 to 119 ms), and
whether the change adds to them is not settled. Image unchanged at the window,
no draws dropped and no device removal. Reported upstream as
[#3089](https://github.com/Marvisak/VIctory/issues/3089); the bitmap fix is
also open there as [#2269](https://github.com/Marvisak/VIctory/pull/2269).

## Miss check, host copy wait and mark tests

The miss check after a list using title memory now walks only the set bits of
its bitmap. The GPU copy before a copy-back waits as submissions do, reporting
a removed or stalled device, instead of forever; review found that a removed
device completes its fences and passed that wait as a finished copy, which it
no longer does. [#3085](https://github.com/Marvisak/VIctory/pull/3085)'s first
commit is in (its author's), and the other mark tests hold off submissions too.

One run (`gpu191`): frame 376 ms (377 before), copy-back 77 ms (89), marked
pages 74 ms (80), no draws dropped and no device removal. The people in the
yard rendered as dark silhouettes in it, as in a run before these changes
(`gpu189`): the cause is not established.

## Title memory page table rewrites

The step costs about 19 ms a frame, nearly all of it rewriting the page table
(about 9 ms a rewrite, two a frame). Timed by phase, about 4.8 ms went in
replacing the old layout, dropped in place with some 15,000 copy references,
and 3.5 ms in building the table. The old layout now goes to the release
thread. Review found that `copy_back_marks` could then drop a copy while
holding the marks lock that dropping it takes; copies are now looked at after
the lock. [#3096](https://github.com/Marvisak/VIctory/pull/3096)'s one-pass
split of changed and unmapped pages is in too (its author's; its change to
the step timer does not apply here).

Cell window, one run each, with temporary timers, medians:

| | Before (`gpu194`) | After (`gpu195`) |
|---|---|---|
| Rewrite, layout phase | about 4.8 ms | 0.15 ms |
| Title memory step | 19 ms | 7 ms |
| Preparing | 140 ms | 114 ms |
| Copy-back | 89 ms | 66 ms |
| Frame | 398 ms | 359 ms |

Part of the difference is run to run variation (the earlier run was uneven, and
marked pages have varied this much between runs of one binary); the title
memory step is the change's own. Image unchanged, no draws dropped.

A run before these changes (`gpu193`, the build of the section above plus CPU
timers) ended in a device removal while the world loaded. The two runs after it
completed. Its cause is not established; the PC has also logged processor
machine checks. Reported upstream as
[#3137](https://github.com/Marvisak/VIctory/issues/3137).

## Held pages drop after the marked copy-backs

Pages held back for shaders using title memory left the copies before the
marked copy-backs, so after a miss another copy of the same memory skipped the
page and its ordinary write was lost. Review found it, and a test reproduced
it; they leave after the copy-backs now. Three end-to-end tests of withheld
ranges from [#3099](https://github.com/Marvisak/VIctory/pull/3099) (its
author's) came with it.

One run (`gpu196`): frame 385 ms, preparing 125 ms, marked pages 84 ms, within
what runs of one binary have given here (marked pages 63 to 86 ms over the last
runs without changes to them). The yard people rendered dark, as in some runs
since `gpu189`, before these changes.

## Bound targets kept and partial 64 KiB pages

Two commits taken with their author's name (fuzeofficial):

- [#3128](https://github.com/Marvisak/VIctory/pull/3128): a bound render or
  depth target stays alive with the command list that recorded it, also when no
  draw follows and the title releases it before submission.
- [#3125](https://github.com/Marvisak/VIctory/pull/3125): a 64 KiB page counts
  as unmapped only after all of its regions were scanned, so a partly mapped
  page is not zeroed.

Three runs completed with them (`gpu199` to `gpu201`). In the same window the
guard fallbacks stayed at about 7 a frame; protect calls went from about 2,070
to 2,290 and protecting from about 9 to 11 ms (one run, and the driver changed
too).

## Driver 621.40 and a streamed session

On 26 November the driver changed from 621.24 to 621.40 and the machine is
driven through Sunshine, so earlier timings are not a baseline. The new driver
rejected the pipeline library, and two runs went into making pipelines again.

| Median, cell window | `gpu196` (621.24) | `gpu201` (621.40, Sunshine) |
|---|---|---|
| Frame | 385 ms | 369 ms |
| Preparing | 125 ms | 121 ms |
| GPU | 163 ms | 171 ms |
| Copy-back | 90 ms | 68 ms |
| Marked pages | 84 ms | 63 ms |
| Process commit, peak | 36.9 GB | 30.5 GB |
| Rust heap, peak | 10.5 GB | 3.6 GB |

One run each, within the spread of repeated runs except for memory. The lower
memory goes with a pipeline library of a few hundred MB instead of several GB
(every save serialises the whole library into one buffer); that link is not
measured on its own. No dark yard people in the last screenshot of `gpu201`.

Three of six runs on 621.24 (`gpu193` to `gpu198`) ended in a device removal
while the world loaded; none of the three on 621.40 did. The cause was never
established.

## CPU clock and core parking

On a Ryzen 7 5700X the busy logical processors ran at 4.59 GHz in the cell
scene (Windows reports the fixed base clock, 3401 MHz, in
`ProcessorFrequency`). The process used 2.5 of 16, its busiest thread 39 % of
one, with a median of 6 parked. With parking off for one run preparing took
141 ms (151 and 162 before) and copy-back 99 ms (81 and 78): the frame did not
change, and the setting was put back.

## Texture uploads and marked page comparison

Found by reading our upload path; no visible effect has been tied to any of
them yet:

- A detile that fails for lack of staging space is retried; the written pages
  had already been taken from the tracker. A subresource the layout cannot
  detile no longer drops the others of the same refresh.
- A copy into part of a texture uploads title memory first, so the rest is not
  left zeroed. A boxless copy that replaces a one-subresource texture skips
  it; the two-plane R32G8X24 and R24G8 formats do not count as replaced.
- A rewritten heap slot of a placed texture already made, and read lately, is
  viewed at once instead of staying null for a submission.
- BC7 channel rotation applies to modes 4 and 5 only
  ([#3111](https://github.com/Marvisak/VIctory/pull/3111)).

Marked page comparison now takes 64 bytes at a time and skips the earlier
bytes where nothing differs. It is faster in a test and made no measurable
difference in the game (`gpu203`). Reading a whole run of title pages with one
call was tried and reverted: in `gpu204` the marked pages took 92 ms a frame
against 63 and page faults went from about 6,500 to 9,200. That is one run, and
two later runs of nearly the same build gave 54 (`gpu207`) and 76 ms
(`gpu208`), so the loss is not established; no gain was measured either.

## Water sums, host-written textures and shader fixes

- Two compute programs of the water that did not translate (a sum of 64-bit
  values over a console wave of 64) translate now. In `gpu206` both ran from
  frame 1763 to the end. `gpu205` ended in a device removal 48 s after their
  first use; `gpu206`, the same build with removal tracking on, did not, and
  the cause is not established.
- A texture the host GPU wrote was preferred over title memory for good
  ([#758](https://github.com/Marvisak/VIctory/issues/758)). The preference now
  ends when title writes reached every page of a page-aligned texture of two
  pages or more. Copy-backs, read backs and remaps do not count.
- VOP3 `fmac` and `mac` apply the accumulator's abs and neg
  ([#3158](https://github.com/Marvisak/VIctory/pull/3158),
  [#3192](https://github.com/Marvisak/VIctory/pull/3192)); no document of the
  hardware was checked for it.
- `image_get_resinfo` answers six faces for a cube
  ([#3010](https://github.com/Marvisak/VIctory/pull/3010); cube arrays still
  answer six), and a `saveexec` without a restore no longer hides the sections
  after it ([#3187](https://github.com/Marvisak/VIctory/pull/3187)).
- 32-bit integer typed UAVs accessed as floats gave the same bits on the
  RTX 3080 Ti and removed the WARP device; left as it is.
- `gpu207` and `gpu208` ran the whole five minutes with every program of the
  scene translated. The yard buildings were black in one screenshot of
  `gpu207` and lit in the same moment of `gpu208`.
- Queries, counted in `gpu208`: 59 occlusion queries a frame, resolved with one
  call, each ended in the list that began it, none with a dropped draw inside.
  Their results were made up then; occlusion and binary occlusion queries
  now run on the host GPU (ported from
  [#159](https://github.com/Marvisak/VIctory/pull/159), see the next
  section).

## Host occlusion queries and fixes from upstream reports

Unit tests cover these; none is verified in the game yet.

| Change | What was wrong |
|---|---|
| Occlusion and binary occlusion queries begin, end and resolve in a host query heap, and the results reach title memory after the submission ran (ported from [#159](https://github.com/Marvisak/VIctory/pull/159)) | Every query answered 1 << 20, so nothing tested with the GPU could be culled. A query keeps that "visible" answer when a draw in it was not run, it is indirect, or its result cannot be had. `VICTORY_NO_HOST_QUERIES=1` restores the made-up results |
| Each place of a list submitted twice at once has its own prelude ([#3162](https://github.com/Marvisak/VIctory/pull/3162) reports it) | Both places recorded their barriers into one command list, so the first ran the second's |
| `v_mac_f32`, `v_mad_f32`, `madak` and `madmk` flush a denormal product, keeping its sign, before the add | The instructions never keep denormals. [#3212](https://github.com/Marvisak/VIctory/pull/3212) flushes to +0; this follows the tree's own flush, and no hardware was measured. Legacy forms and denormal inputs or sums are unchanged |
| Floating-point `wavePrefixOp` sums pass the signed kind ([#3151](https://github.com/Marvisak/VIctory/pull/3151)) | They passed the unsigned kind, which DXIL allows for integers only |
| Untracked pages of a partly tracked texture are hashed on every refresh ([#3041](https://github.com/Marvisak/VIctory/pull/3041), issue [#2638](https://github.com/Marvisak/VIctory/issues/2638)) | Only dirty ranges were asked, so a write to an untracked page was never staged. No such texture appears in this title's logs |
| A barrier batch is submitted before a transition on a resource with a depth twin ([#3129](https://github.com/Marvisak/VIctory/pull/3129)) | The twin copy records at once, so it could come before the transition it depends on |
| Texture states are masked to texture states, with one write state alone ([#3140](https://github.com/Marvisak/VIctory/pull/3140)) | A state could keep buffer or reserved bits, or join a write state with others, which D3D12 rejects |
| View heaps with title slots make the view address table when they are created ([#3175](https://github.com/Marvisak/VIctory/pull/3175)) | Made on first use, it could lose addresses noted before, or zero entries |

Timestamps are still made up by the CPU; their use is not yet measured.

## Copy-backs, EXEC, pinned views and pipeline order

None of these is verified in the game.

| Change | What was wrong |
|---|---|
| A declared copy-back after a marked run of the same page keeps newer title bytes (upstream [#2436](https://github.com/Marvisak/VIctory/issues/2436)) | It restored old title bytes over a title write made after the run |
| Partial flush ranges are joined only where that covers no new whole page (upstream [#2473](https://github.com/Marvisak/VIctory/issues/2473)) | Parts of a page could make it count as current with bytes nothing had copied. The check looks only at the pages holding the ends of the two ranges, as no other page can be newly covered |
| A remap marks only the remapped pages; writes in flight still mark aliases (upstream [#2414](https://github.com/Marvisak/VIctory/issues/2414)) | A remap also marked the other mappings, which had not changed, and their copies lost shaders' writes |
| Interpolation writes obey EXEC (upstream [#3153](https://github.com/Marvisak/VIctory/issues/3153)) | Inactive lanes lost their destination. A lane whose EXEC differs between `p1` and `p2` still does not get the console's partial value. Exception: f16 packed interpolation whose destination holds its own first half is not guarded |
| Title resources are pinned while views of them are read (upstream [#3213](https://github.com/Marvisak/VIctory/issues/3213)) | A release on another thread could free a resource still being read. Shader and unordered views only: render target and depth lookups are not pinned |
| Render targets are resolved when set and replayed on a cut (upstream [#3217](https://github.com/Marvisak/VIctory/issues/3217)) | A cut read the title's descriptors again, which may have changed since |
| Early pipelines are made in first-use order. Waiting for compute pipelines like screen passes is off unless `VICTORY_COMPUTE_WAIT=1`: on by default at first, it made frames take 15 to 26 s while the cache was cold (`gpu218`, against about 1 s in `gpu209` and `gpu210`) and did not explain the dark scene | A one-off dispatch dropped while its pipeline was made is lost |

Observation: the scene was dark in runs `gpu196`, `gpu200`, `gpu210` and
`gpu211` (which dropped passes while pipelines were being made), but also in
`gpu213`, which dropped none, so dropped passes do not explain it. It was lit
in `gpu206` to `gpu208` and in `gpu212` (previous build). The cause is not
established.

- Builds at `c24dd958` (`gpu214`) and `8f3b3193` (`gpu215`) rendered lit;
  `d7d2ba0c` rendered dark in `gpu210`, `gpu211` and `gpu213`. Each midpoint
  had one run only.
- `d7d2ba0c` with the texture state change taken out rendered lit (`gpu216`),
  and its added warning never fired: the title asked no write state together
  with shader reads in that run, so the change is unlikely to be the cause.
- The build with all changes above rendered lit (`gpu217`). A race between runs
  is the likeliest reading, not shown.
- An audit of its view address table change found nothing.
- Dropped first-use computes do not explain it: `gpu213` dropped none and was
  dark.

## SDWA sign extension into the low word

The shader crate tests pass but one that already failed; one game run
(`gpu217`) showed no visible change.

| Change | What was wrong |
|---|---|
| A 16-bit result written to `WORD_0` with a sign-extending SDWA destination fills the high word with its sign (the `SEXT` arm only of upstream [#3124](https://github.com/Marvisak/VIctory/pull/3124)) | The high word was zeroed |

## Measurements without a change of behaviour

- Diagnostics builds log once each wave32 compute program that keeps a plain
  group barrier in per-lane control flow, with whether the console control
  flow could lower it.
- Diagnostics builds log each pair of pipeline and bound render target formats
  that differ (invalid in D3D12) and count those draws and the unresolved
  slots.
- Diagnostics builds count end-of-pipe clock writes per frame (GPU timestamp
  and raw system clock).
- The heap view census of diagnostics builds counts the unordered 3D slots
  among those left null.
- Diagnostics builds count the host-written texture marks cleared.
- Diagnostics builds warn once when a partly tracked texture is sampled, and
  once when a texture state drops shader read states beside a write state.
- Test code behind the diagnostics feature was repaired to match current
  signatures (`pipeline/placed_uav_tests.rs`, tests in `pipeline/make.rs`).
- Diagnostics builds log each distinct thread affinity, ideal processor,
  priority and thread creation value the title uses, and call totals.
  `VICTORY_DSTORAGE_WORKERS` (1 to 64) replaces the default of one
  DirectStorage worker per logical CPU.
- `VICTORY_MODE_SCAN` makes `victory translate` count the changes of the MODE
  register and the initial float modes. One offline scan found only
  `s_setreg_imm32_b32` of the f16/f64 rounding field with literals 0 and 3,
  which `emit/lower.rs` already models: no fix. The initial mode was readable
  for 29 compute programs only (0xc0), none of the pixel programs. The
  translator assumes 32-bit denormals flushed and does not read the denormal
  modes.

## View heap limit

Upstream [#1777](https://github.com/Marvisak/VIctory/pull/1777) with the title
index redirect of [#2628](https://github.com/Marvisak/VIctory/pull/2628) and
review fixes. `gpu218` (compile run) and `gpu219` (warm, 5 minutes) ran with
it: no errors, `gpu219` rendered lit at about 365 to 405 ms per frame, inside
the spread of `gpu214` to `gpu217`. `gpu219` warned of title slots past the host
heap (from 721,469, none for a draw) and of one texture read past the heap. No
effect on the picture is established.

| Change | What was wrong |
|---|---|
| The device is probed for the view heap size (2^21, else 1,000,000), and shaders clamp indices and place the null buffer views below that size | This host takes 1,000,000 descriptors in a shader-visible heap, but shaders used null views and a clamp at 2^21, outside the heap. The long-failing test `emit::stride_tests::a_heap_buffer_read_where_no_buffer_view_is_reads_zeros`, failing until now, was this |
| On a capped heap, a title index taken from the heap itself past the title's slots reads the heap's last, null view; title, dynamic and pair slots end before three null views | Such an index could reach the host's dynamic slots and view pairs |
| The shader and pipeline cache keys include the view heap size | Translations made for another size would be reused. The shader cache is invalidated |

Limits on capped hosts:

- Title slots at or above 721,469 read null when a shader indexes the heap
  itself. Their content is not restored.
- A redirected texture read gets a 2D null view, whatever its dimension.

Diagnostics builds warn when texture reads land on the null views.

## Lighting findings

From logs and code; no fix yet and no proven cause of the dark interiors.

- Every `title memory this frame` line of runs `gpu215` to `gpu217` shows
  `0 pages copied back`: persistent title-memory misses hold back the
  copy-back of what shaders wrote.
- The title's pipeline sync (`sync` in `list.rs`) now records a global
  barrier. The title rarely calls it: one report each in `gpu218` and
  `gpu219` showed 0.1 calls per frame, the others none.
- Computes `5e5b5c98afaa1753`, `f0afa89ecfb6944e` and `8d5d3f97369050bd` do
  not tell lit runs from dark ones, as first thought: `gpu220` was lit without
  binding them and `gpu200` bound them and was dark.
- `gpu220` and `gpu221` ran the same build with the same settings; one was
  lit, the other dark. The dark interior varies between runs. Dark runs tend
  to reach the scene cut later (a threshold picked afterwards, with lit
  exceptions); nothing in the logs separates the two groups cleanly.

`VICTORY_DEMAND_COPIES=1` is an experiment against these misses, off by
default. With the first rule, a copy for a page missed 8 frames running
(`gpu222`, lit): 49 MiB of copies were made, steady frames still missed a
median of 3 pages, and copy-back ran in 5 of 582 frames, at the same frame
time as `gpu221`. The remaining pages recur every 8 frames, so a page now
counts as persistent when missed in 2 of 16 frames. With that rule
(`gpu223`, lit) the 256 MiB budget ran out, 53 to 59 pages were still missed
in every steady frame, no frame copied back, copy-back rose to about 107 to
119 ms (62 in `gpu222`) and the frame to about 415 to 455 ms. 256 KiB copies
per page do not scale to what the title misses; the switch stays off.

`gpu223` also gave the first copy-back and barrier counters (with demand
copies on, so its totals are inflated): of 95 ms of marked-page wall time the
threads spent about 306 ms comparing, 145 ms writing and 20 ms reading;
5118 indirect argument clamps per frame, 2236 with no shader write since the
one before; no null UAV barriers for title buffer barriers.

`gpu224` (lit, demand copies off, with the clamp barrier, staging and
writer-cache changes): no errors, frames 351 to 436 ms, noisy; copy-back 71
to 112 ms, above `gpu221` (54) and `gpu222` (62), cause not established (the
new counters time every page). Of 55 to 68 ms of marked-page wall time the
threads spent about 150 ms comparing, 90 to 117 ms writing and 20 ms reading.
Pull and copy-back overlap could save at most about 13 ms per frame, so the
first overlap stage is not worth building.

`gpu225` (dark interior, with runs joined while comparing): no errors. Median
of the last 20 reports against `gpu224`: comparing 20 ms of thread time
against 133, marked-page wall time 33 ms against 50, copy-back 49 ms against
92, frame 379 ms against 436 (one run each).

Demand copies now go in 64 KiB windows asked for at the frame's end, and
`VICTORY_CLAMP_WINDOW=1` joins indirect draws that do not store into one
argument clamp pass (up to 64 records from one source buffer); both are off
by default. Diagnostics builds also time each GPU command (`VICTORY_GPU_TIMING`)
and report how long the GPU was busy around title submissions. `gpu228`
(both switches off, lit): no errors, frames 338 to 469 ms (about 370 typical),
GPU busy 172.7 ms per frame in timestamps around 31 submissions (177.9 ms of
wall time), preparing 111 to 214 ms, copy-back 47 to 71 ms; no regression
against `gpu225`.

The windowed clamp is now on by default (`VICTORY_CLAMP_WINDOW=0` turns it
off), and an indirect draw whose shaders may store no longer ends the window
and gets a clamp of its own: it joins as the window's last member, since the
pass reads every record before any member runs. Steady frames (about 5000
draws), median of the last reports:

| Run | Clamp | Frame ms | GPU ms | GPU busy ms |
|---|---|---|---|---|
| `gpu228` | single clamps | 378 | 184 | 175 |
| `gpu230` | windows, storing draws alone (63 windows, 1916 draws alone) | 335 | 153 | 150 |
| `gpu231` | windows, storing draws closing them (1974 windows, none alone) | 304 | 144 | 138 |

In timing runs (`gpu227` against `gpu229`, windows without the storing
change) the commands' GPU time fell from about 177 to 135 ms per frame:
`host-clamp` 27.9 to 14.9 ms, `host-setup` 10.2 to 5.5 ms. Pictures did not
change and no draw was dropped. `VICTORY_CLAMP_WINDOW_CHECK=1` (diagnostics,
adapted from fxceless's window check) clamps each windowed draw's records
again and compares them on the GPU: `gpu232` checked 1.24 million records,
none differed. The timing report now also sums GPU time by command kind,
since its 25 costliest entries left about 45% of the time unlisted.

### Upper bound of dirty tracking (`gpu233`, `gpu234`)

`VICTORY_MARKS_OFF=15` (diagnostics, measurement only: picture and
copy-backs are wrong) translates shaders without view write and read
marks and without title memory write and miss marks. After a compile run
(`gpu233`), `gpu234` ran at 289 to 300 ms per frame with 88 to 101 ms on
the GPU, against 304 and 144 ms in `gpu231`. The marks cost about 45 to
55 ms of GPU time per frame, roughly a third; the frame barely moves
because about 110 ms of preparation on the CPU then bounds it. Cutting
marks pays only together with cheaper preparation. The switch is kept as
a patch, not committed.

### Read marks no longer close clamp windows (`gpu236` to `gpu240`)

Counters of why clamp windows end (diagnostics) showed about 1,900 of
1,950 windows a frame closed by an indirect draw "that may store", 1,779
of them only because its shaders bind the write-mark bitmap. Read marks
bind the same bitmap, so `view_marks` was true for shaders that only read.
The translator now reports `view_stores`, set where a write through a
buffer view is lowered, and the host's store test uses it. Writes through
images, heap views, root views and title memory are still caught by the
other clauses.

| Run | Windows a frame | Frame ms | GPU ms |
|---|---|---|---|
| `gpu237` (before) | 1,988 | 358-360 | 143-144 |
| `gpu239` (window check on) | 220 | 352-382 | 168-171 |
| `gpu240` | 216 | 309-321 | 98-105 |

`gpu239` checked 1.86 million windowed records against fresh clamps: none
differed, none cut. The GPU time fell by about 40 ms a frame; the frame
did not, as preparing (125-144 ms) and copy-back (60-67 ms) on the CPU now
bound it.

### Cheaper write faults (`gpu242`)

Upstream PR #3343 (danielibabet): a fault on a page with no other mapping
skips the alias scan and its allocations, and contiguous guarded pages
open with one `VirtualProtect`. Against `gpu240`: frame 296-298 ms
against 309-321, time in the fault handler 93 ms of thread time a frame
against 120, the interior lit.

### Guard lock timing (`gpu246`)

The fault handler waits 57 to 76 ms a frame for the guard lock, summed
over threads (longest single wait 6 to 67 ms), while all holders together
hold it about 33 ms: the handler 11 ms, with single holds up to 3.4 to
3.7 ms, and `take_split` 10.4 ms, with holds of 0.4 ms or less. Faults
queue behind faults; the handler's own long holds are the target, not the
take.

## Series S profile

The scene colour target is DXGI format 67 (`R9G9B9E5_SHAREDEXP`), which the
host can neither render to nor store to: creating the target failed
(`E_INVALIDARG`) and the picture was black (`gpu243`). `R11G11B10_FLOAT`
(26) now stands in for format 67 targets through the existing packed
stand-in path (as 86 and 115 map to 28):

- texels are converted on upload and read-back (RGB9E5 to and from
  R11G11B10);
- read-only placed views of the same memory share the stand-in, also when
  made before the target;
- a format 67 target view of the stand-in is made in 26;
- GPU copies from the stand-in to a native format 67 texture are converted
  by a compute pass (2D, one slice; other copies are skipped).

The test is ported from Biskwit/VIctory-reloaded (shared-exponent target
host copy).

| Run | Picture | Frame ms | GPU ms | Copy-back ms |
|---|---|---|---|---|
| `gpu243` | Black, wall luma 0.4 | 340-395 | | |
| `gpu245` | Back, wall luma 45.2, floor 25.3 | 304-337 | 87-95 | 73-84 |
| `gpu246` (review fixes, guard lock timing) | Back | 286-348 | 90-93 | 66.6 |

`gpu246` dropped 3 draws: a whole-wave `v_or_b32` step does not translate.
From now on runs use the Series S profile.

## Rectangles, 16-bit and typed loads, root UAV pages

Runs `gpu220` to `gpu222` had these: no errors, `gpu221` at 345 to 354 ms per
frame with copy-back at 54 ms. `gpu220` and `gpu222` were lit, `gpu221` dark
(the intermittent dark interior, seen before these changes). Its final frame
differs from the lit one in indirect light only: sky and sunlit surfaces
match, shadowed surfaces are near black. No change here is verified to alter
the picture.

An alias copy now also returns its source to the unordered or shader view
state the same draw or dispatch bound it in, instead of leaving it in
`COPY_SOURCE`. Not yet run in the game.

| Change | What was wrong |
|---|---|
| The range a root UAV can write is declared in whole 64 KiB pages (first part of [#1963](https://github.com/Marvisak/VIctory/pull/1963)) | A marked page partly in the range was left out of the page copy-back, losing what shaders wrote in its other part. A page the table reads from another copy is still not covered |
| Both triangles of a rectangle start with v0, on the geometry shader and CPU paths ([#2970](https://github.com/Marvisak/VIctory/pull/2970)) | The second triangle took flat attributes from another vertex |
| `buffer_load_format_d16_x` and `_xyz` keep the high half of their last register ([#1441](https://github.com/Marvisak/VIctory/pull/1441)) | It was zeroed. Evidence: Mesa and LLVM treat it as preserved; AMD's ISA text does not say |
| Typed buffer loads read 0 in every missing channel ([#1516](https://github.com/Marvisak/VIctory/pull/1516), issue [#1060](https://github.com/Marvisak/VIctory/issues/1060)) | W read 1. Evidence: the ISA's identity channel selection and a measurement on RDNA2 desktop hardware |

Rejected: [#2297](https://github.com/Marvisak/VIctory/pull/2297) (textures
upload once, before all lists, so forgetting the uploaded set changes
nothing here).

`lower.rs` had passed the asset policy's 512 KiB limit; the lane spill
helpers moved to `lower/spill.rs`.

## Tried and left out

| Change | Result here |
|---|---|
| Copies and read backs taking a newer alias first (local, committed and reverted) | Review: it copied a whole alias on every dispatch binding the source as unordered access and the alias as a view, left the source in a state it cannot be copied from, skipped copies across lists recorded in parallel, and could overwrite texels filled from title memory. No safe small version was found; the stale read ([#3170](https://github.com/Marvisak/VIctory/pull/3170)) stays, noted at `alias_source` |
| Shader heaps behind a read-write lock (as in fxceless's and uiliannn's forks, upstream #3320), `gpu241` | Title threads no longer waited to write views (209 ms to 0), but the heap sync on the submitting thread, now contending with them for each heap's slot lock, rose from 43 to 160 ms a frame: preparing 245-277 ms against 125-144, frame 408-445 against 309-321 |
| A second heap slot sync during the GPU wait | Preparing 161 ms against 154, frame 445 against 406, title threads waited longer to write views (one run). A slot's view copy now retired with its resource stayed from it |
| Refreshes of copies in video memory through the copy pool | Refresh batches 69 ms against 64 (one run) |
| Copy-back reads of title memory in user mode instead of `ReadProcessMemory` | Marked pages 81 ms against 61, copy-back 92 against 67, frame 439 against 390 (one run, a noisy one: preparing varied twice as much as before) |
| #2849 (guarded pages opened in runs) | One `VirtualProtect` cannot span two views: the tracker test ended in an access violation until such runs open page by page; no measured gain (the guard lock swings from 114 to 330 ms between runs) |
| #2848 (frame-path costs) | With its shared heap lock, a descriptor test failed in 3 of 20 runs of the suite |
| #2755 (title-page dirty marks coalesced within a wave) | GPU step 337 ms with it, 308 ms in the same build without it (one run each) |
| Without the memory performance commits (#1993, #2136, #2175 and related) | Crash at world entry, 2 of 2 runs |
| #2121 (paged view heap) | Black picture: the host pages ran out, indirect draws dropped |
| #2388 | Refuses 19 builder programs that translate today: their descriptors are two pointers read from memory, taken for a loaded descriptor |
| #2341, #2374, #2380 | Dark pictures in runs that had them; the cause was not isolated |
| Hot pages (leave pages written every frame unguarded), `gpu244` against `gpu242` (Series X profile) | Frame 361-365 ms against 296-298, moves 49-53 against 24 ms, unmapped 20-21 against 1 MB a frame, watches 5,430-5,500 (27-37 ms) against 2,500-2,650 (13-16 ms), guard faults about 680 against 922, fault handler 110-146 against 93 ms. Open pages stay in `changed` (dirty), so their copies are taken again at each of about 31 submissions, and unmapped hot pages fall into the slow mapped-copy fallback. Kept on branch `dropped/hot-pages` |

## November 26, night: Kelvaroq ideas and reviews, `gpu250` to `gpu256`

Ideas from Kelvaroq (Stellarr Studio, GPL-3.0-or-later, on fxceless's
victory-fork) are ported one at a time, each checked against the picture
before it stays. Rule for these ports: nothing may disable, refuse, stub or
drop shaders, pipelines, draws or dispatches. Kelvaroq's refused pipelines,
wave64 and ray tracing cuts and FSR refusal are therefore left out.

### Guard faults opened outside the lock (reverted)

`gpu250` (Series X profile, commits 4fa39898, 0a9a7850, b07f661b) opened the
faulting page outside the guard lock. `gpu251` ran the same build with
`VICTORY_FAULT_OPEN_UNLOCKED=0`.

| Run | Picture (wall / floor luma) | Draws a frame | Frame ms | Handler lock wait ms a frame |
|---|---|---|---|---|
| `gpu242` (reference) | Lit, 45.8 / 20.1 | 5,283 | 296-298 | |
| `gpu250` (unlocked) | Dark, 2.3 / 9.5, HUD prompt missing | 4,914 | 270-320 at the end, spikes of 400-1,541 mid-run | 82-153 |
| `gpu251` (switch off) | Lit, 41.8 / 25.3 | | 259-483 | 74-154 |

Opening the page outside the lock loses writes or lets stale data through,
and gains no frame time. Reverted (aa4bf8d1, bf7939b8, d3099a6a). The lock
wait did not move, so it is not caused by the time `VirtualProtect` holds
the lock alone.

### Descriptors translated once (audit: no-go as designed)

Kelvaroq's "observed key" skips a heap slot whose descriptor was already
seen. Here `made` already acts as that key; a skip saves only 5.5 to 8.9 ms
of comparison a frame, while the 52 to 75 ms is spent making slots. Counters
now split stale slots by reason (unmade, recycled, rewritten, released, API
view writes, forgets). If rewritten slots dominate, the next step is cheaper
making per slot: one lock of the view table per sync, and host descriptors
cached by (index, generation) and copied with `CopyDescriptorsSimple`.

### Submissions in flight (audit)

From Kelvaroq's late flights and relay: the first step (P1) is one
submission in flight with a fixed barrier and no relay, estimated at 40 to
60 ms a frame and not additive with the descriptor work. Drains stay
mandatory for declared written ranges, title fence and value waits, our own
sync executes, present and frame end, an idle card and device removal;
present stays a drain. Flight study counters (preparation, heap sync, GPU
wait, settle, drain reasons, fence polls) measure whether it would pay.

### Further counters

`VICTORY_DIAG_FAULT_KINDS=1` counts guard faults by kind (pool, armed, view,
fixed, other), and dropped indirect mesh dispatches are counted.

`gpu252` (lit, 43.3 / 22.6, 259-315 ms a frame) answered all three:

- Heap sync: 250k-309k stale slots a frame, about 95% "unmade" (record
  current, nothing made), against about 20k API view writes. Cause (audit):
  the title's view heap has 1,245,184 slots, but the NVIDIA host heap is
  capped at 1,000,000 (the 2^21 probe is refused), which leaves 721,469 for
  the title; every sync visited the slots above the cap, which `mirror`
  passes by.
- Flight study: before about 25 of 29 submissions the title waits on a
  fence; only 2 or 3 would be free to stay in flight. One submission in
  flight gains little until those waits are met sooner.
- Guard faults: about 200 a frame in write-watched pools against 600-680 in
  mapped views, below the 50% that would make write-watch on pools worth it.
- No indirect mesh dispatch was dropped in this scene.

### Per-command GPU time (`gpu253`, timing run)

Lists take 110-127 ms of wall time, their commands 83-98 ms. By kind:
indirect mesh draws 39.2 ms (5,104 a frame), dispatches 24.5 ms (2,512),
direct mesh 8.6 ms, the indirect argument clamp 4.7 ms (1,136). No single
command passes 5 ms: the time is a long tail, so the wins are in what every
command pays (translation, barriers, memory). Video memory is at 11,143 of
11,779 MB with 6.3 GB in system memory.

### Changes measured one at a time

| Run | Change | Picture (wall / floor) | Frame ms | Result |
|---|---|---|---|---|
| `gpu254` | Heap sync stops at the host heap's mirrored slots | Lit, 40.2 / 29.6 | 266-307 | Unmade slots 250k to about 2k; sync 58-63 to 55-57 ms. Kept |
| `gpu255` | Tables every shader reads (page table, view strides, marks, addresses) in GPU upload heaps (`VICTORY_TABLES_GPU_UPLOAD`, capped by `VICTORY_TABLES_GPU_UPLOAD_MAX_MB`, default 512) | Lit, 44.4 / 21.0 | 250-303 (GPU 79-92) | About 10 ms off the frame. Kept |
| `gpu256` | Heap sync takes the heaps' lock one page at a time | Lit, 43.6 / 23.6 | 262-300 | Title threads still wait 200-220 ms a frame to write views: the wait is not this lock. Reverted |

A draw still reads title slot 999,999, past the host heap: indexed heap
reads past 721,469 read a null view. A sparse remap of title slots to host
slots is being designed.

### Shader review fixes, `gpu257` to `gpu260`

The shader review's fixes: title-memory pages are marked once per wave bit
after a load, with the lead lanes found by one `WaveMatch`; sub-dword loads
read the next dword only at byte 3 and take their dword and byte from the
raw view's offset; `buffer_store_short` at byte 3 writes its high byte. A
uniform page handle for scalar loads was tried and left out (scalar registers
are not wave-uniform in the per-lane form).

The changed programs give about 1,700 new pipelines, which took three runs to
build: `gpu257` and `gpu258` still dropped 500 to 940 draws a frame waiting
for them, and `gpu259` (393 still being made) was dark, 6.2 / 5.2.

| Run | Pipelines being made | Picture (wall / floor) | Frame ms | GPU ms |
|---|---|---|---|---|
| `gpu255` (before) | 0 | 44.4 / 21.0 | 250-303 | 79-92 |
| `gpu260` | 0 | 34.8 / 28.1 | 281-299 | 92-97 |

Lit, no draws dropped, no device removal; the frame is in the same range.

## Remap counters and heap sync diff, `gpu261` to `gpu278`

`gpu261` measured a sparse title-to-host slot remap (counters only, on a
diagnostics branch not merged): title view pages holding a current view peak
at 14,587 against a host pool of 11,272 pages of 64 slots, so the remap does
not fit as designed; dynamic slots peak at 135,873; no heap-direct reads past
the cap. The picture was dark.

Heap syncs now compare each slot's record with what the slot was made from
and its view entry's generation, read under one views lock per range, and
send only changed slots to `mirror` (a superset of the old ones, never
fewer). View generations moved to their own column.

Medians of the last 15 frames:

| Run | Build | Prep / sync heaps / GPU / settle ms | Frame ms | Page faults | Picture |
|---|---|---|---|---|---|
| `gpu273` | main before the change | 136.3 / 57.1 / 89.4 / 50.9 | 297 | ~8k | Lit |
| `gpu275` | first sync diff | 190.3 / 106.2 / 114.3 / 55.9 | 364 | ~77k | Dark |
| `gpu278` | sync diff, generations read in place | 131.4 / 51.7 / 94.2 / 49.8 | 290 | ~8.7k | Lit |

`gpu274` and `gpu275` copied the whole generation column (~9 MB) at each
sync: fresh pages and soft faults every time, under the heaps' lock. That
copy was dropped; `gpu278` is kept.

Picture notes: the dark interior is still intermittent across runs and
branches. Bleached, near-white hair showed in `gpu277` and `gpu278` on
different branches, cause not established; `gpu278` also showed blocky
rectangles on a poster.

## Audit fixes and dark runs, `gpu276` to `gpu280`

Six fixes from the code audit:

- Root views the title leaves unbound read a zeroed scratch buffer
  (`VICTORY_UNBOUND_SCRATCH=0` leaves them as they are, with a warning
  naming where they lie).
- The pipeline library is saved every 20 s, not only at exit.
- A riding pull's list is freed only after its fence.
- A riding pull holds the locks of the pages it found stale until its wait,
  and marks them stale again if it did not bring them back.
- The save pool is allocated outside its lock.

Medians of the last 15 frames:

| Run | Build | Prep / sync heaps / GPU / settle ms | Frame ms | Picture |
|---|---|---|---|---|
| `gpu276` | audit fixes | 134.5 / 55.4 / 97.4 / 52.3 | 302 | Lit |
| `gpu277` | audit fixes | 128.3 / 53.9 / 92.0 / 49.2 | 279 | Dark, bleached hair |
| `gpu279` | audit fixes | 134.6 / 52.6 / 96.2 / 53.1 | 281 | Dark; no unbound root view warnings (the scratch was never bound) |
| `gpu280` | main, control | 136.7 / 55.4 / 95.9 / 51.6 | 287 | Dark |

Waits on stale page locks: 0 to 0.04 ms. The fixes neither cost nor save
time. The dark interior is intermittent on both builds: 2 of the last 5 main
runs and 2 of 3 audit runs.

### Dark interior: what separates the runs

In dark runs the title never binds 7 lighting compute pipelines, which are
all translated and created. In runs with bleached hair it never binds 3 hair
compute pipelines and 1 mesh pipeline. Under study, as what could make the
title skip them: GPU writes the copy-back withholds that the title reads
back, occlusion queries measured on a black first frame, and timestamps that
report CPU time.

## Compute waits and a descriptor mirror race, `gpu281` to `gpu285`

`gpu281` measured two branches, not merged yet:

- A native pipeline stub for device methods `0x1c8` and `0x218`. It ran
  without trouble.
- A texture freshness audit (diagnostics only). Of about 165 textures audited
  a frame, none changed unseen, and no write-watch guard was lost.

`gpu282` and `gpu283` ran with `VICTORY_COMPUTE_WAIT=1`: compute pipelines
are waited for like passes over the screen. Both were lit. `gpu284` ran with
that wait on by default (`02fc024c`) and came out half lit, with no compute
pipeline waited for. The wait stays as policy, since a dispatch should never
be dropped, but it is not the fix for the dark interior.

`gpu285` (`9ef5cdde`) fixes a race: a late mirror of a descriptor slot could
null a newer view there until the title wrote that page again. A test
reproduces it. The run was lit, with no change in frame time.

Medians of the last 15 frames:

| Run | Build | Prep / sync heaps / GPU / settle ms | Frame ms | Picture |
|---|---|---|---|---|
| `gpu282` | main + gpu281 branches, `VICTORY_COMPUTE_WAIT=1` | 131.2 / 49.9 / 96.3 / 59.4 | 271 | Lit |
| `gpu283` | main + gpu281 branches, `VICTORY_COMPUTE_WAIT=1` | 132.1 / 51.2 / 100.8 / 59.5 | 304 | Lit |
| `gpu284` | compute wait by default | 126.2 / 49.6 / 93.8 / 51.1 | 290 | Half lit; no compute pipeline waited for |
| `gpu285` | descriptor mirror race fix | 128.2 / 51.4 / 93.3 / 52.5 | 302 | Lit |

The dark interior is still intermittent. Lit runs bind the 7 lighting
compute pipelines; dark runs do not.

## Pipelined submission, memory system and queue signals

Merged from the `pipeline`, `memsys` and `queue-signals` work, all behind
switches that are off by default except where noted:

- Submissions split into submit and retire with an owned ticket; a retire
  thread behind `VICTORY_PIPELINE=1`, predicted label values behind
  `VICTORY_PREDICT_LABELS=1`. Stale-page pins replace mutex guards.
- Memory system rewrite behind `VICTORY_MEMSYS=1` (`VICTORY_MEMSYS_POOL=0`
  in the runs below); `VICTORY_MEMSYS_ALIAS=1` shares one copy per physical
  page. A buffer now finds every copy holding its physical pages.
- Lost lists (submitted again after their descriptor slots were taken back)
  keep their waits and label writes in queue order; a fence Signal goes
  through the held path while work is held. The safety valve logs the word's
  value and its last writer, and queues stuck on a fence Wait are reported.
- Diagnostics: `VICTORY_STACKS=crash` prints every thread's stack at the
  first unhandled exception (diagnostics builds).

Runs (Series X profile, 250 s, streaming client attached): serial memory
system 278 to 316 ms a frame (prep/sync/GPU/settle medians 151/45/88/60 ms)
against 414 ms before; `VICTORY_PIPELINE=1` 312 to 320 ms, with predicted
labels 285 to 301 ms: neutral, because 27 of about 29 submissions still
overlap copies the one in flight uses. The queue-signal build ran 280 to
328 ms with no regression. An intermittent hang (the title's own watchdog
ends the process after 30 s without a frame) follows a compute queue stuck on
a label value; it is still open.

Follow-ups on the same base: lost lists' label writes ride a host list so the
memory-system gate and predicted labels see them; queues keep a reference to
the fences they wait on; the crash stack dump matches stack words against the
process's handle list and keeps its deadline; the long-copy reach shrinks when
the longest copy goes. Run gpu311: medians prep/sync/GPU/settle
174/33/75/53 ms, no hang, no regression.

## Title memory in place (off by default)

Shaders that reach title memory through the page table read zeros where no
host copy holds it, and their writes there are dropped. Two switches name
such memory in place instead: a host heap is opened over the pages
(`OpenExistingHeapFromAddress1`), one buffer is placed over it, and the
page-table entry names a raw view of that buffer. Nothing is copied.

| Switch | What is named in place |
|---|---|
| `VICTORY_INPLACE_BACKSTOP=1` | Pages no copy holds that show physical pages, through a view of our own of the section. |
| `VICTORY_RT_MEMORY=1` | The above, and committed read-write private memory no copy and no tracker holds, at its own address, in runs of at most 8 MiB. Nothing is withheld. |
| `VICTORY_INPLACE_MB` | Most megabytes open at once: 1024, or 8192 with `VICTORY_RT_MEMORY`. |

`VICTORY_RT_MEMORY` is for the title's acceleration-structure builder. Its
inputs, scratch and output are private memory it reaches by computed
addresses (`gpu287`, `gpu288`: about 12,000 pages in whole 256 MiB regions).
Every builder list read zeros there and had its writes dropped, so the ray
pass walked structures no shader built.

A heap over process memory locks its pages: `VirtualProtect`, `VirtualFree`,
unmapping, and a commit with another protection fail with
`ERROR_INVALID_PARAMETER` while it lives. So:

- Each private run registers how to let it go (`victory_gpu::pinned`). The
  title's `VirtualFree`, `VirtualProtect` and unmapping let it go first and
  retry for up to 100 ms. `VirtualAlloc` and `XMemVirtualAlloc` at an address
  the title names run first and let go only if refused as locked.
- Letting go drops the run from the page tables and from the layouts lists
  last ran with, once no submission runs (waited for up to 2 s).
- The backstop opens only views of ours. Private runs are the one exception:
  asked as such, wholly in private memory the runtime reported, committed
  read-write throughout, within one allocation (a range across two was
  refused by the host with `E_INVALIDARG`, `gpu308`). Anything else is
  refused. The check runs again once the run is registered and once its heap
  is open.
- The builder's dispatches are never held back. A builder dispatch first
  waits for every pipeline the builder has created, queued ahead of all
  others, up to 20 s a frame. Holding them back while pipelines were still
  being made left a cross-queue wait after them unmet, and the title's 30 s
  watchdog ended the run (`gpu307`, `gpu308`).
- A run is locked and registered before its heap is opened. A change of the
  title's that comes during the open waits for it to end, then lets the run
  go (a test stops an open there and frees the memory every time). A change
  refused as locked with nothing released looks again: a run registered
  after the first look, or one let go within 100 ms, is let go or waited
  out, and the change is tried again.
- A tracker made over such memory lets the runs there go without waiting. A
  page a live run still locks gets no guard and stays changed. The run stays
  registered until its heap is gone, so the title's `VirtualFree` there
  still drops it from the tables first.
- Runs are opened on a worker thread when the runtime reports the memory
  (`XMemVirtualAlloc`, or a miss in committed private memory not reported
  yet). A page table that still finds one pending waits, at most 100 ms a
  frame for all tables, woken when the worker is done (no polling).
- A miss in memory reported already costs one lookup. It rewrites the page
  tables at most once a frame. What the host refused stays refused until the
  title commits there again (`XMemVirtualAlloc` or its own `VirtualAlloc`).
- Debug builds assert the lock order: the guard lock is never taken while
  holding the locked ranges, and the opening thread never takes the page
  tables' lock.

The frame line `ray tracing memory (VICTORY_RT_MEMORY)` counts pages named,
runs open, opened and let go, page-table waits, and for builder lists the
pages written in place, copied back, withheld and missed.

`VICTORY_RT_MEMORY` has not been run with the game yet: it passes the unit
tests only.

## Open problems

- About 1 frame per second.
- Three programs of the world scene do not translate: two whole-wave sections
  of split wave64 programs and barycentrics read directly.
- Interior lighting has differed from run to run (walls black in many).
- The multi-wave barriers of #2214 are called undefined by D3D12 in its own
  review when reached from different places in one group; the pull request is
  a draft. Diagnostics builds now report wave32 programs whose lanes may skip
  a group barrier; those are not changed. Run `gpu209` logged 10 such compute
  programs, `gpu210` and later runs 11 (9 console wave64, 2 wave32), in groups
  of 64 to 1024 threads, all lowerable with the console flow; a correct fix
  needs one barrier site shared by the waves.
- A copy or read back of a texture can miss what the GPU wrote through another
  title texture at its memory (see Tried and left out).

## Testing

`cargo test -p victory-shader -p victory-gpu --lib`, then `autotest.ps1 -Name
<run> -Game <dir>` (5 minutes; `-Warm` after translator changes).

Changes are made with AI assistance (Claude Code).
