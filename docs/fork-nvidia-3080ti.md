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
  Their results are still made up.

## Tried and left out

| Change | Result here |
|---|---|
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

## Open problems

- About 1 frame per second.
- Three programs of the world scene do not translate: two whole-wave sections
  of split wave64 programs and barycentrics read directly.
- Interior lighting has differed from run to run (walls black in many).
- Tests failing on this GPU:
  `emit::stride_tests::a_heap_buffer_read_where_no_buffer_view_is_reads_zeros`
  (view heap cap).
- The multi-wave barriers of #2214 are called undefined by D3D12 in its own
  review when reached from different places in one group; the pull request is
  a draft.

## Testing

`cargo test -p victory-shader -p victory-gpu --lib`, then `autotest.ps1 -Name
<run> -Game <dir>` (5 minutes; `-Warm` after translator changes).

Changes are made with AI assistance (Claude Code).
