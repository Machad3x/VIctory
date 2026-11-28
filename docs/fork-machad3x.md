# Machad3x/VIctory-nvidia-RTX: fork notes (RTX 3080 Ti, Ryzen 7 5700X)

This fork is based on [trustno1-coder/VIctory-NVIDIA](https://github.com/trustno1-coder/VIctory-NVIDIA) main,
which builds on [Marvisak/VIctory](https://github.com/Marvisak/VIctory). Every run here has the title's ray tracing
on (`VICTORY_BVH_BUILDER=1`) and leaves no pipeline out (no `VICTORY_LEAVE_OUT`). Runs use `autotest.ps1`: diag
build, the launcher's NVIDIA switches, DLSS Quality at 1440p, frame generation off; 300 s in batch 1, 240 s with a
shot every 120 s from batch 2 on.

## Current state

| | |
|---|---|
| latest batch | 9 (run t9c, with `VICTORY_STORE_BESIDE=1`) |
| ms/frame, average of the last 120 s | 67.0 (max 75) |
| card (GPU) ms | 14.0 |
| prep ms | 41.7 |
| dropped draws | 0 |
| refused shaders or pipelines | 0 |

Open issues filed with the base:

- [trustno1-coder/VIctory-NVIDIA#688](https://github.com/trustno1-coder/VIctory-NVIDIA/issues/688): pages faulted in by BVH shaders stop after 2048 for the whole run, rays then read holes.
- [#689](https://github.com/trustno1-coder/VIctory-NVIDIA/issues/689): a compacted structure size is read back before the builder writes it.
- [#690](https://github.com/trustno1-coder/VIctory-NVIDIA/issues/690): root-view buffer accesses ignore OOB_SELECT 1, out-of-range stores land in memory.
- [#691](https://github.com/trustno1-coder/VIctory-NVIDIA/issues/691): VRAM at 99.9% of budget, about 1.9 GB free in partly used texture blocks.
- [#692](https://github.com/trustno1-coder/VIctory-NVIDIA/issues/692): an intermittent full white frame around 120 s into the opening cutscene.
- [#817](https://github.com/trustno1-coder/VIctory-NVIDIA/issues/817): a late lock-free write fault can undo a guard taken since (access violation); fixed here in 3175c5dc.
- [#846](https://github.com/trustno1-coder/VIctory-NVIDIA/issues/846): the title's watchdog ends the process after a stall with a held value wait never released; fixed here in batch 11.

## Runtime switches added by this fork

| switch | default | what |
|---|---|---|
| `VICTORY_BVH_SIZE_ANSWER` | on | The compacted size of an acceleration structure is answered at the title's call with the builder's own maximum. `=0` turns it off. |
| `VICTORY_LATE_WAIT=0` | not set | The pacing option: turns the late wait off, so submissions are not held back. Slower here (batch 8), it can feel steadier on some setups. |
| `VICTORY_DXR` | off | Watches the title's builder calls for DXR: `1` on, `check` compares both ways. |
| `VICTORY_REFRESH_ONCE` | on | A submission refreshes only the pages its lists read. `=0` turns it off. |
| `VICTORY_ASK_READ_ONLY` | on | A submission asks the write watch only about the pages its lists read. `=0` turns it off. |
| `VICTORY_RESIDENCY_PRIORITY` | on | High residency priority for card copies, render targets and UAV memory. `=0` turns it off (run t9b: 106.1 ms/frame against 85.2). |
| `VICTORY_TARGET_ROOM_SMALL` | on | From the world's first frame only a margin above the render targets is kept for them (the larger of 256 MiB and 15 %). `=0` keeps the whole reserve. |
| `VICTORY_PRESSURE_EXIT_GB` | 3.5 | Host memory pressure ends once this many GiB are free. |
| `VICTORY_VIEW_REMAP=1` | off | Places title view slots past NVIDIA's capped descriptor heap through a table (trustno1-coder/VIctory-NVIDIA#638). Costs about 47 ms/frame here with no visible change (runs lt1b-2 and lt1c-2). |
| `VICTORY_HELD_WAIT_MS` | 1500 | A value wait nothing meets is passed after this many ms when no present is queued, and passed at once when the title waits on a fence whose signal is held behind it. The held work still runs. `=0` turns it off. |
| `VICTORY_STORE_BESIDE=1` | off | Stores beside the guards instead of opening them. Used in the batch 9 runs: protections 29 to 22 ms in run t8s. |

`autotest.ps1` sets `VICTORY_BVH_BUILDER=1` itself; `-More @{VICTORY_BVH_BUILDER=''}` turns ray tracing off for a run.

## Batch 1 (2026-11-27)

- ad351371: the structure builder's pipelines are made by the workers from creation, and its dispatch waits a bounded time. Before this, with ray tracing on, the title's thread compiled 53 pipelines in a row (~26 s), and the title's own watchdog ended the process.
- d3bc4297: copy-back runs are found 64 bytes at a time (SSE2) instead of byte by byte.
- 7c72f843, a613ff5e: the pipeline library save is flushed and summed. On load, a library that does not match its sum is set aside. A library newer than its sum (saved by a build without sums) loads unchecked.
- autotest.ps1: the launcher's switches (`play-foxhound.ps1 -EnvOnly`), ray tracing on, nothing left out, and a count of refused or untranslated lines.
- The console pipeline-state deserializers (device 0x218/0x1c8, after Marvisak/VIctory#3586) were tried and taken back out. The compute pipelines they made did not translate (840 shaders), while without them the title creates those pipelines the normal way.

| run | ray tracing | ms/frame | GPU ms | prep ms | copy-back ms | dropped draws/frame | result |
|---|---|---|---|---|---|---|---|
| tn1a (base) | off | 40.5 | ~1 | 30-33 | 10-11 | 330 + 637 indirect (left out) | alive |
| tn1b (base) | on | - | - | - | - | - | watchdog crash at 84 s |
| t1b (batch 1) | on | 90-97 | 32-42 | 46-47 | 13-15 | 1 | alive 300 s |

## Batch 2 (2026-11-27)

Merged from the base: #516, #501, #498, #497, #532, #521. Fixes on top:

- watch: re-read the books when title memory is made real (race in #516);
- lazy: `hot_within` joins a piece only onto a range it overlaps (#501);
- ambient: `raise` scrubs the probe result channel by channel, so one bad channel no longer zeroes the whole probe (#521; the right side of the cell went dark).

Also: a BVH node past a buffer's 4 GiB reads as a miss, and the title's builder calls are watched for DXR (`VICTORY_DXR`, off by default). Autotest runs 240 s with shots every 120 s.

| run | ray tracing | ms/frame | GPU ms | prep ms | copy-back ms | dropped draws/frame | result |
|---|---|---|---|---|---|---|---|
| t2a (batch 2, before the probe fix) | on | 63.7-67.2 | 5-9 | 42-48 | 9-10 | 1 | alive 300 s, right side darker |
| t2b (batch 2) | on | 61.5-63.9 | 2-3 | 46-49 | 12 | 1 | alive 240 s, picture back |
| t2c (batch 2, ambient estimate switches unset) | on | 65.3-68.4 | 11-14 | 40-44 | 9-11 | 1 | alive 240 s, picture as t2b |

With ray tracing on, the ambient estimate now stays out for the whole run (it was only kept out in frames where the builder ran). t2c runs the same thing through the switches.

## Batch 3 (2026-11-27)

Merged from the base: #552, #548 (its counting-allocator tests now run in diagnostics builds only), #547, #544. #550 (the code table starts after the instructions) is left out: with it the card time per frame went from 11-19 ms to 76-224 ms on the Series X profile, with the same picture.

| run | profile | ms/frame | GPU ms | prep ms | copy-back ms | dropped draws/frame | result |
|---|---|---|---|---|---|---|---|
| t3b (batch 3 with #550) | Series X | 153-308 | 76-224 | 62-71 | 12 | 1 | alive 240 s, card time up |
| t3s (batch 3 with #550) | Series S | 50.4-53.2 | 1-5 | 39-41 | 7-10 | 1 | alive 240 s |
| t3r (batch 3 without #550) | Series X | 67.3-70.5 | 18-19 | 39-42 | 9-10 | 1 | alive 240 s, picture as t2b |

## Batch 4 (2026-11-27)

The defragmenting compute shader (02384a60fda6f2d0) is no longer left out:

- a loop shape is accepted only when it is narrower than the host wave or fits it;
- the second host wave does not poll in meet;
- the indirect tests follow `VICTORY_INDIRECT_REST`'s default;
- an old menu settings file that still leaves the shader out is rewritten when read.

| run | profile | ms/frame | GPU ms | prep ms | copy-back ms | dropped draws/frame | result |
|---|---|---|---|---|---|---|---|
| t4b (batch 4) | Series X | 71.7-72.7 | 17-18 | 41-45 | 10-11 | 0 | alive 240 s, picture as t3r |

Nothing is dropped or left out any more. The shader now running costs about 3 ms per frame.

## Batches 5 and 6 (2026-11-27)

Integration branches that were not taken to main. Batch 5 (copy pool, global loads into LDS, the write watch's refused blocks) stays on its branch. What batch 6 carried (the BVH walk, the precompiled-pipeline lookups, the root-view range check) came to main in batches 7 and 8.

## Batch 7 (2026-11-27)

- The emulated BVH walk is faster, with the same results:
  - the second half of a box node is read from the same page lookup unless it crosses a page;
  - a triangle node's 16 dwords are loaded up front;
  - node addresses are computed in 32-bit pairs.
- The precompiled-pipeline lookups (device methods 0x1c8 and 0x218) decline quietly. The title then builds the same pipelines through its normal path, so nothing was ever missing.

From now on, runs are compared on the average of the last 120 s (24 frame lines), because single lines swing between 70 and 110 ms.

| run | ms/frame (120 s avg, max) | GPU ms | prep ms | copy-back ms | dropped | result |
|---|---|---|---|---|---|---|
| t4b (batch 4) | 77.7 (110) | 24.3 | 41.5 | 10.0 | 0 | - |
| t7a-2 (batch 7) | 75.2 (100) | 19.4 | 43.4 | 10.5 | 0 | alive 240 s, picture as t4b |

## Batch 8 (2026-11-27)

- Root-view buffer accesses in OOB_SELECT 1 (index only) mode drop lanes whose index is at or past `num_records`: loads read 0 and stores are dropped, as on the console. Offsets stay signed.
- A null constant buffer view description writes a null view (port of trustno1-coder/VIctory-NVIDIA#646).
- Heap samplers keep point mips when anisotropic, as static samplers do (port of trustno1-coder/VIctory-NVIDIA#656).
- The compacted size of an acceleration structure is answered at the title's call with the builder's own maximum, and the builder's later size store is kept off the answered slot (`VICTORY_BVH_SIZE_ANSWER`, on by default, `=0` turns it off).

| run | ms/frame (120 s avg, max) | GPU ms | prep ms | copy-back ms | dropped | result |
|---|---|---|---|---|---|---|
| t7a-2 (batch 7) | 75.2 (100) | 19.4 | 43.4 | 10.5 | 0 | - |
| t8a-2 (batch 8) | 66.8 (85) | 6.1 | 47.1 | 11.4 | 0 | alive 240 s, picture as t7a-2; the 120 s shot is white (seen before in t4b, not attributed; #692) |

No shader or pipeline is refused.

## Batch 9 (2026-11-28)

Preparing time:
- 57093e3d, 99930de6: the write watch remembers blocks it refused; few refreshes are spread one per claim again.
- 7a84d45a: lock-free no-op refreshes, thin-worker refresh coalescing and an AVX2 copy diff (port of trustno1-coder/VIctory-NVIDIA#614 by wike1701).
- f1ab9cd8, 7a7bb526: a submission refreshes and asks about only the pages its lists read (`VICTORY_REFRESH_ONCE`, `VICTORY_ASK_READ_ONLY`, both on).

VRAM and early pop-in:
- 7e483ba5: textures go best-fit into the fullest pool block, so sparse blocks drain.
- de148876: residency priorities (`VICTORY_RESIDENCY_PRIORITY`, on).
- 5f2c4844, 115983e8, 8aae1c34: the render-target reserve is given back once the targets settle, and only a margin above them is kept from the world's first frame (`VICTORY_TARGET_ROOM_SMALL`, on). The settle clock now starts at the world's first frame. Before 8aae1c34, menus could release the reserve early, and run t9a went to 85.2 ms/frame.
- b9073aff: host memory pressure ends at 3.5 GiB free.
- aae614e6: the log says why texture slots wait in the queue.

Descriptors:
- cf28538c, 9bb19b95: view places past a capped host heap (port of trustno1-coder/VIctory-NVIDIA#638 by chemist6913), behind `VICTORY_VIEW_REMAP=1`, off.

Fixes:
- 3175c5dc: a late lock-free write fault no longer undoes a guard taken since. It was a real race that could crash on a title write (#817).
- df65d657: two tests no longer release a view while a mapping change is still in progress, which made a test hang.

| run | ms/frame (120 s avg, max) | GPU ms | prep ms | copy-back ms | dropped | result |
|---|---|---|---|---|---|---|
| lt1b-2 (base for this batch) | 66.6 (81) | 8.3 | 46.8 | 10.6 | 0 | - |
| t9a (before 8aae1c34) | 85.2 (100) | 30.9 | 43.0 | 10.0 | 0 | reserve given up in the menus |
| t9b (`RESIDENCY_PRIORITY=0`) | 106.1 (118) | 62.6 | 35.0 | 9.1 | 0 | priorities kept on |
| t9c (batch 9) | 67.0 (75) | 14.0 | 41.7 | 9.8 | 0 | alive 240 s, picture as lt1b-2 |

Over the first 80 s after the world starts, t9c averages 148.5 ms/frame (max 615) against 261.3 (max 714) for lt1b-2, so early loading is faster. No shader or pipeline is refused. The dark interior is unchanged.

## Batch 11 (2026-11-28)

- 10fcc252, plus the review fixes from d4c31913: held value waits that nothing meets no longer stall the line until the title's watchdog ends the process. That crash happened in about 1 run in 10 during the load into the world (trustno1-coder/VIctory-NVIDIA#846). A wait is passed after `VICTORY_HELD_WAIT_MS` (1500) with no present queued. It is passed at once when the title waits on a fence whose signal is held behind it in the same queue. The work it held still runs.

In t11a and t11b (240 s each) the release never fired and nothing crashed. GPU time in these runs (110 and 136 ms) shows the run-to-run spread described below, not this change: the change did not run.

### GPU time spread between runs

The same build gives 8 to 450 ms of GPU time per frame from run to run, and it often grows during a run. The work sent is the same: draws, indirect groups, structure builds and lists per frame. The card is compute-bound in slow runs (SM 100 %, memory bandwidth 2-5 %, clocks steady at 1950 MHz, little PCIe traffic). Slow runs mostly show the cell lit by bounce light (median 66 ms against about 25 ms for dark ones). So whether the title's ray-traced lighting walks real structure data varies between runs, and when it does the compute emulation of the rays is expensive. Moving those rays to the RT cores (DXR) is the planned fix. Runs that start less than 30 s after another one also read slower, so the runs now start 2 min apart.

### Pacing option

`VICTORY_LATE_WAIT=0` turns the late wait off: submissions are not held back. It is not set by default. On this machine it costs a lot (run t7w: 124.3 ms/frame against 75.2), but it can feel steadier on some setups.
