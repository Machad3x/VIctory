# Progress scale

This fork rates itself on the scale that [PortWatch VI](https://portwatch-vi.xyz/#method)
uses for every public port of the title, so its state can be compared with
the other projects on one number. The method below is PortWatch's, in our
words; the score at the end is our own estimate, not one PortWatch has given.

## The method

One scale from 0 to 100%, in two halves.

### Getting into the game: a ladder to 50

Each step needs the one before it. A step reached only in part (a menu that
shows but is broken, say) counts half.

| Points | Step | Reached when |
|---|---|---|
| 0-10 | Preparation | Done before the executable exists; weighted below |
| 10 | Executable running | The title's executable is mapped, its imports resolved, and its own code runs |
| 20 | First frame | The title draws real frames: logos, a loading screen or the intro |
| 30 | Main menu | The title screen or main menu shows and answers input |
| 40 | In the world | Story mode loads the world and draws it, even broken |
| 50 | Playable | A character is controlled in the world: walking around, with the HUD or pause menu working |

Preparation, weighted:

| Points | Part | What it covers |
|---|---|---|
| 2 | Loader and system libraries | PE loading and stand-ins for the console-only DLLs (kernelx, xgameruntime, xmem and others) |
| 2.8 | Shader translation | Share of the title's ~26,500 precompiled RDNA2 shaders translated to a PC format (SPIR-V or DXIL) |
| 3.2 | Graphics layer | Xbox D3D12.X to D3D12 or Vulkan: devices, command lists, pipelines, draws, presentation |
| 2 | Proven on another game | The same layer runs another commercial Xbox Series game |

### Playing well: four bars to 100

The last 50 points count only once a build is playable.

| Points | Bar | Levels |
|---|---|---|
| 17.5 | Story: how far it can be played | Stuck at the start, prologue done, first missions, first act, half the story, main story to the credits, 100% complete |
| 12.5 | Graphics: how close the picture is to the console's | Black or broken, shapes drawn but nearly unlit or black, lit but missing major effects, nearly the console's with small faults, the console's |
| 10 | Performance | Average fps divided by 30, on a stated PC |
| 10 | Stability | From crashing all the time to stable over long sessions |

PortWatch scores public projects from their README, commits and issues, and
marks claims it has not seen itself as self-reported. A README that lags
behind the build is called out, so this file and the README's status are
kept in step with each measured run.

## How this fork measures

| Bar | Evidence we use |
|---|---|
| Steps | `autotest.ps1` screenshots and log: first frame, menus, world entry, HUD |
| Story | Furthest mission reached in a logged run |
| Graphics | Screenshots against the reference frame of the prison cell window, and wall and floor luma from `target/patches/bright.py` (lit 35-46, dark 2-4) |
| Performance | Median frame time over the steady part of a run, in the world, on the machine below; fps = 1000 / frame ms |
| Stability | Device removals and `crashes.log` entries across recent runs, and run length |

Machine: Ryzen 7 5700X, RTX 3080 Ti 12 GB (driver 621.40), 32 GB RAM,
Windows 11, resizable BAR on, Series X profile.

## Our estimate (2026-11-26)

Self-assessed, not verified by PortWatch.

| Part | Level | Points |
|---|---|---|
| Ladder | Playable in part (counts half): play reached in the prison cell, HUD prompt shown. A walking session has not been logged yet | 45 |
| Story | Stuck at the start (the prison cell of the first mission) | 0 |
| Graphics | Lit, but missing major effects (no bounce light indoors against the reference; colour banding; lighting dark in some runs) | about 6 |
| Performance | 250-307 ms a frame, about 3.5 fps (`gpu254` to `gpu256`) | about 1.2 |
| Stability | No device removal and no crash in the recent runs, each 5 to 10 minutes; nothing longer measured | about 5 |
| **Total** | | **about 57%** |

PortWatch's own numbers per level are not published past the method above,
so the graphics and stability points are our reading of it. What would move
the score most: a logged walking session with the HUD (the last 5 ladder
points), getting past the cell (story), steady interior lighting (graphics),
and a logged session of 30 minutes or more (stability).

## Getting listed or re-scored

PortWatch's Scout finds public repositories on its own every few hours, and a
tracked README is re-scored when it changes. To ask for tracking or a
re-score, open an issue on
[4lwayz/portwatch-vi](https://github.com/4lwayz/portwatch-vi/issues) titled
`Tracking request: <owner>/<repo> (...)`, with the repository link, the
rubric table above, and evidence: screenshots or a video, figures on a named
PC, and where in the code they can be checked.
