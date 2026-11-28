# Machad3x branch notes (RTX 3080 Ti, Ryzen 7 5700X)

This branch is based on trustno1-coder/VIctory-NVIDIA main. Every run here has the title's ray tracing on (`VICTORY_BVH_BUILDER=1`) and leaves no pipeline out (no `VICTORY_LEAVE_OUT`). Runs use `autotest.ps1`: 300 s, diag build, the launcher's NVIDIA switches, DLSS Quality at 1440p, frame generation off.

## Batch 1 (2026-11-27)

- ad351371: the structure builder's pipelines are made by the workers from creation, and its dispatch waits a bounded time. Before this, with ray tracing on, the title's thread compiled 53 pipelines in a row (~26 s), and the title's own watchdog ended the process.
- d3bc4297: copy-back runs are found 64 bytes at a time (SSE2) instead of byte by byte.
- 7c72f843, a613ff5e: the pipeline library save is flushed and summed. On load, a library that does not match its sum is set aside. A library newer than its sum (saved by a build without sums) loads unchecked.
- autotest.ps1: the launcher's switches (`play-foxhound.ps1 -EnvOnly`), ray tracing on, nothing left out, and a count of refused or untranslated lines.
- The console pipeline-state deserializers (device 0x218/0x1c8, after upstream #3586) were tried and taken back out. The compute pipelines they made did not translate (840 shaders), while without them the title creates those pipelines the normal way.

| run | ray tracing | ms/frame | GPU ms | prep ms | copy-back ms | dropped draws/frame | result |
|---|---|---|---|---|---|---|---|
| tn1a (base) | off | 40.5 | ~1 | 30-33 | 10-11 | 330 + 637 indirect (left out) | alive |
| tn1b (base) | on | — | — | — | — | — | watchdog crash at 84 s |
| t1b (batch 1) | on | 90-97 | 32-42 | 46-47 | 13-15 | 1 | alive 300 s |

## Batch 2 (2026-11-27)

Merged from tn1: #516, #501, #498, #497, #532, #521. Fixes on top:
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
