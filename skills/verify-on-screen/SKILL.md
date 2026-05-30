---
name: verify-on-screen
description: The verification discipline for mathwave rooms — how to confirm a room ACTUALLY renders before claiming it works, using agent-browser plus window.__err / window.__diag canaries. Critically, node --check and a green fps reading are NOT sufficient. Use before committing any room.
---

# Verify a room on screen (don't trust node --check)

The hardest-won lesson in this repo: **a room can pass `node --check` and still be
broken on screen.** Several failure modes are invisible to syntax checks and even to a
naive "is fps a number" probe. Always do a real visual pass before claiming success.

## The canaries (put these in every room)

In the HTML, **before** the module script:
```html
<script>window.__err=null;addEventListener("error",e=>{window.__err=String((e&&e.message)||e.error||e)})</script>
```
In the JS: `window.__diag = () => JSON.stringify({ /* live, meaningful state */ })`.
Make `__diag` report something that proves the sim is *running* (step count, energy,
edge counts) — not just static config.

## The sweep (agent-browser)

```bash
AB=/Users/vanities/.vite-plus/bin/agent-browser
$AB close --all; sleep 1
$AB open "http://127.0.0.1:8124/pieces/<room>.html?cb=$(date +%s)"   # cache-bust!
$AB wait --load networkidle; sleep 5
$AB get text '#fps'                 # a number, not '–'
$AB eval 'String(window.__err)'     # must be "null"
$AB eval 'window.__diag && window.__diag()'
$AB screenshot /tmp/<room>.png      # then LOOK at it
```
Then actually **read the screenshot** — colors, structure, not-blank, not-one-solid-color.

## Failure signatures (what each broken state looks like)

| Symptom | Meaning |
|---|---|
| `#fps` = `–`, veil text still showing, `__err` = null | **GLSL shader failed to compile.** THREE logs it to `console.error`, which does NOT trip `window.__err`. (Caught truchet, hyperbolic this way.) |
| `__err` has a `SyntaxError` string, fps `–` | **JS failed to parse** (e.g. `-x**y` is illegal in JS; use `Math.pow`). The module never ran. |
| solid single color (e.g. yellow) at 60fps | a **simulation blew up** (unstable `dt`) or **display saturates** — see gpu-pingpong-sim-room. |
| washed white / flat | palette saturating to a white end-stop, or double-gamma — see raymarched-shader-room / ps1-pipeline. |
| black screen, fps fine | nothing in frame (camera framing, or a shader that compiled to black). |

## Hard rules

- **Cache-bust every reload**: `?cb=$(date +%s)` on the URL, and bump the `?v=` tag on
  the module `<script src>` when you edit JS — browsers cache aggressively and you will
  otherwise "verify" a stale build and falsely declare success.
- **Don't claim "verified" on a cached or blank render.** If you didn't see the new
  pixels, you didn't verify.
- A green `fps` is necessary, not sufficient. Pair it with `__err`, `__diag`, AND eyes
  on the screenshot.
- GLSL compile errors are the sneakiest — when fps is `–` but `__err` is null, open the
  browser console (`agent-browser console`) to read the shader log.

## Parallel-build note

When many rooms are built by subagents at once, they share one browser instance — only
the **orchestrator** drives agent-browser, after wiring each room into `ROOMS`. Subagents
build + `node --check` only.
