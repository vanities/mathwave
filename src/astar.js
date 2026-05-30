// astar.js — best-first graph search on a 2D grid (経路 / A* Search).
//
// A* finds a shortest path from START to GOAL over a grid of cells, some of
// which are WALLS. Every node carries three numbers:
//   g(n) = cost of the cheapest known path from START to n,
//   h(n) = a HEURISTIC estimate of the remaining cost from n to GOAL,
//   f(n) = g(n) + h(n)  — the estimated total cost of a path through n.
// We keep an OPEN set (a priority queue ordered by f) of discovered-but-not-yet-
// expanded nodes, and a CLOSED set of already-expanded nodes. Each step pops the
// lowest-f node, marks it CLOSED, and RELAXES its neighbors: for each neighbor we
// compute a tentative g; if that beats the neighbor's recorded g we update it,
// set its parent pointer, and push/decrease-key it in OPEN. We stop the instant
// GOAL is popped, then walk parent pointers back to reconstruct the PATH.
//
// If h never overestimates the true remaining cost (it is "admissible") A* is
// guaranteed to return an optimal path; if h is also consistent it expands each
// node at most once. The heuristic is what makes the frontier lean TOWARD the
// goal instead of ballooning in every direction — that pull is the whole point of
// this room, and you can watch it.
//
// This room runs three searches on the SAME maze so the difference is visible
// (↑/↓ cycles them):
//   • A*      f = g + h        — informed; explores a narrow corridor to the goal.
//   • Dijkstra f = g  (h = 0)  — uninformed; expands uniformly in all directions.
//   • Greedy  f = h            — chases the heuristic; fast but not optimal.
// We report cells-explored in __diag, so A*'s efficiency over Dijkstra is
// quantified, not just asserted.
//
// Ref: Hart, Nilsson & Raphael (1968), "A Formal Basis for the Heuristic
// Determination of Minimum Cost Paths", IEEE Trans. SSC 4(2), 100–107.
// Dijkstra (1959), "A note on two problems in connexion with graphs".
//
// Rendering: a flat 2D grid uploaded to a DataTexture (UnsignedByte red channel
// holds a cell STATE id) + a fullscreen display shader that maps each state to a
// vivid color on near-black — the same safe path as sandpile.js (no float-texture
// extension, needsUpdate every frame). The search advances a few expansions per
// frame so the frontier visibly grows.

import * as THREE from "three";
import { makeRenderer, onResize, loop, fpsMeter, liftVeil, bindRange, reducedMotion, setVariantCycler } from "./common.js";

const canvas = document.getElementById("scene");
const renderer = makeRenderer(canvas);
const scene = new THREE.Scene();
// Fullscreen quad in clip space; the grid lives entirely in a texture.
const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

// Grid presets (kept modest so a full search animates smoothly and never hangs).
const SIZES = [81, 121, 161];
let sizeIdx = 1;
let N = SIZES[sizeIdx];

// ---- cell STATE ids (ride in the texture red channel) ----
const EMPTY = 0;   // unvisited open space
const WALL  = 1;   // obstacle
const OPENF = 2;   // in the OPEN set (the frontier)
const CLOSD = 3;   // expanded (CLOSED)
const PATH  = 4;   // final reconstructed path
const START = 5;
const GOAL  = 6;
// A soft "f-heat" band (7..15) shades CLOSED cells by how good their f was, so
// the heuristic's pull is legible. The shader lerps explored→path color across it.
const HEAT0 = 7;
const HEAT_BANDS = 9;   // ids 7..15

// ---- buffers ----
let wall = new Uint8Array(N * N);   // 1 = obstacle
let tex = makeTex(N);

let startIdx = 0, goalIdx = 0;

function makeTex(n) {
  const t = new THREE.DataTexture(new Uint8Array(n * n * 4), n, n, THREE.RGBAFormat);
  t.minFilter = THREE.NearestFilter;
  t.magFilter = THREE.NearestFilter;
  t.needsUpdate = true;
  return t;
}

// ---- color schemes (vivid, NOT purple-dominant) ----
// Each scheme supplies: wall, frontier(open), explored(closed-cool),
// explored-hot(near goal), path, start, goal.
const SCHEMES = [
  ["neon", {
    wall:    [0.05, 0.06, 0.09],
    open:    [0.16, 0.92, 0.98],   // cyan frontier
    cool:    [0.06, 0.20, 0.34],   // dim teal/blue explored (far f)
    hot:     [0.10, 0.42, 0.58],   // brighter teal explored (near goal)
    path:    [1.00, 0.74, 0.16],   // bright amber path
    start:   [0.20, 0.95, 0.45],   // green
    goal:    [0.97, 0.99, 1.00],   // white
    bg:      [0.01, 0.02, 0.03],
  }],
  ["magma", {
    wall:    [0.06, 0.04, 0.05],
    open:    [0.20, 0.85, 0.92],
    cool:    [0.10, 0.10, 0.26],
    hot:     [0.36, 0.16, 0.40],
    path:    [1.00, 0.30, 0.62],   // magenta path
    start:   [0.30, 0.95, 0.55],
    goal:    [1.00, 0.98, 0.94],
    bg:      [0.02, 0.01, 0.03],
  }],
  ["ice", {
    wall:    [0.05, 0.07, 0.10],
    open:    [0.30, 0.98, 0.90],
    cool:    [0.05, 0.16, 0.30],
    hot:     [0.12, 0.34, 0.62],
    path:    [1.00, 0.84, 0.30],   // gold path
    start:   [0.25, 1.00, 0.60],
    goal:    [0.96, 1.00, 1.00],
    bg:      [0.01, 0.02, 0.04],
  }],
];
let scheme = 0;

const material = new THREE.ShaderMaterial({
  uniforms: {
    uTex:   { value: tex },
    uAspect:{ value: 1 },
    uWall:  { value: new THREE.Color() },
    uOpen:  { value: new THREE.Color() },
    uCool:  { value: new THREE.Color() },
    uHot:   { value: new THREE.Color() },
    uPath:  { value: new THREE.Color() },
    uStart: { value: new THREE.Color() },
    uGoal:  { value: new THREE.Color() },
    uBg:    { value: new THREE.Color() },
    uHeat0: { value: HEAT0 },
    uHeatN: { value: HEAT_BANDS },
  },
  vertexShader: `
    varying vec2 vUv;
    void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }
  `,
  fragmentShader: `
    precision highp float;
    varying vec2 vUv;
    uniform sampler2D uTex;
    uniform float uAspect;
    uniform vec3 uWall, uOpen, uCool, uHot, uPath, uStart, uGoal, uBg;
    uniform float uHeat0, uHeatN;
    void main(){
      // Fit the square grid centered in the viewport, preserving aspect ratio.
      vec2 p = vUv - 0.5;
      if (uAspect >= 1.0) p.x *= uAspect; else p.y /= uAspect;
      vec2 uv = p + 0.5;
      if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        gl_FragColor = vec4(uBg, 1.0);
        return;
      }
      float s = floor(texture2D(uTex, uv).r * 255.0 + 0.5);
      vec3 col = uBg;
      if (s < 0.5)        col = uBg;          // EMPTY
      else if (s < 1.5)   col = uWall;        // WALL
      else if (s < 2.5)   col = uOpen;        // OPEN / frontier
      else if (s < 3.5)   col = uCool;        // CLOSED (no heat info)
      else if (s < 4.5)   col = uPath;        // PATH
      else if (s < 5.5)   col = uStart;       // START
      else if (s < 6.5)   col = uGoal;        // GOAL
      else {
        // HEAT band: shade explored cells cool->hot by f-rank toward the goal.
        float t = (s - uHeat0) / max(uHeatN - 1.0, 1.0);
        t = clamp(t, 0.0, 1.0);
        col = mix(uCool, uHot, t);
      }
      gl_FragColor = vec4(col, 1.0);
    }
  `,
});

scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material));

function applyScheme() {
  const s = SCHEMES[scheme][1];
  material.uniforms.uWall.value.setRGB(...s.wall);
  material.uniforms.uOpen.value.setRGB(...s.open);
  material.uniforms.uCool.value.setRGB(...s.cool);
  material.uniforms.uHot.value.setRGB(...s.hot);
  material.uniforms.uPath.value.setRGB(...s.path);
  material.uniforms.uStart.value.setRGB(...s.start);
  material.uniforms.uGoal.value.setRGB(...s.goal);
  material.uniforms.uBg.value.setRGB(...s.bg);
}
applyScheme();

// ============================================================
// MAZE GENERATION
// ============================================================
let mazeKind = 0;   // 0 random obstacles, 1 recursive-division maze, 2 rooms
const MAZES = ["scatter", "maze", "rooms"];

function idx(x, y) { return y * N + x; }

// scatter: random obstacle field (~28% walls) with start/goal at opposite corners.
function genScatter() {
  wall.fill(0);
  for (let i = 0; i < N * N; i++) wall[i] = Math.random() < 0.28 ? 1 : 0;
  placeEndpoints();
  carveAround(startIdx); carveAround(goalIdx);
}

// recursive division: start solid-empty, add walls with gaps — classic maze look.
function genMaze() {
  wall.fill(0);
  // border
  for (let x = 0; x < N; x++) { wall[idx(x, 0)] = 1; wall[idx(x, N - 1)] = 1; }
  for (let y = 0; y < N; y++) { wall[idx(0, y)] = 1; wall[idx(N - 1, y)] = 1; }
  divide(1, 1, N - 2, N - 2, 0);
  placeEndpoints();
  carveAround(startIdx); carveAround(goalIdx);
}

function divide(x0, y0, x1, y1, depth) {
  const w = x1 - x0 + 1, h = y1 - y0 + 1;
  if (w < 3 || h < 3 || depth > 40) return;
  const horizontal = w < h ? true : (h < w ? false : Math.random() < 0.5);
  if (horizontal) {
    // pick an even row for the wall, odd col for the gap
    const rows = [];
    for (let y = y0 + 1; y < y1; y += 2) rows.push(y);
    if (!rows.length) return;
    const wy = rows[(Math.random() * rows.length) | 0];
    const gaps = [];
    for (let x = x0; x <= x1; x += 2) gaps.push(x);
    const gx = gaps[(Math.random() * gaps.length) | 0];
    for (let x = x0; x <= x1; x++) if (x !== gx) wall[idx(x, wy)] = 1;
    divide(x0, y0, x1, wy - 1, depth + 1);
    divide(x0, wy + 1, x1, y1, depth + 1);
  } else {
    const cols = [];
    for (let x = x0 + 1; x < x1; x += 2) cols.push(x);
    if (!cols.length) return;
    const wx = cols[(Math.random() * cols.length) | 0];
    const gaps = [];
    for (let y = y0; y <= y1; y += 2) gaps.push(y);
    const gy = gaps[(Math.random() * gaps.length) | 0];
    for (let y = y0; y <= y1; y++) if (y !== gy) wall[idx(wx, y)] = 1;
    divide(x0, y0, wx - 1, y1, depth + 1);
    divide(wx + 1, y0, x1, y1, depth + 1);
  }
}

// rooms: a handful of rectangular rooms joined by doorways — big open spaces so
// the heuristic's directed sweep is dramatic.
function genRooms() {
  wall.fill(1);
  const rooms = 7 + ((Math.random() * 5) | 0);
  for (let r = 0; r < rooms; r++) {
    const rw = 6 + ((Math.random() * (N / 4)) | 0);
    const rh = 6 + ((Math.random() * (N / 4)) | 0);
    const rx = 1 + ((Math.random() * (N - rw - 2)) | 0);
    const ry = 1 + ((Math.random() * (N - rh - 2)) | 0);
    for (let y = ry; y < ry + rh; y++)
      for (let x = rx; x < rx + rw; x++) wall[idx(x, y)] = 0;
  }
  // connect with random straight corridors
  for (let c = 0; c < rooms + 4; c++) {
    const ax = 1 + ((Math.random() * (N - 2)) | 0);
    const ay = 1 + ((Math.random() * (N - 2)) | 0);
    const bx = 1 + ((Math.random() * (N - 2)) | 0);
    for (let x = Math.min(ax, bx); x <= Math.max(ax, bx); x++) wall[idx(x, ay)] = 0;
    for (let y = Math.min(ay, ay); y <= Math.max(ay, ay); y++) wall[idx(ax, y)] = 0;
    const by = 1 + ((Math.random() * (N - 2)) | 0);
    for (let y = Math.min(ay, by); y <= Math.max(ay, by); y++) wall[idx(bx, y)] = 0;
  }
  placeEndpoints();
  carveAround(startIdx); carveAround(goalIdx);
}

// Put start near one corner, goal near the opposite, both on empty cells.
function placeEndpoints() {
  startIdx = nearestEmpty(2, 2, 1, 1);
  goalIdx  = nearestEmpty(N - 3, N - 3, -1, -1);
}

// spiral-ish search outward from (x,y) for an empty cell (clears one if needed).
function nearestEmpty(x, y, dx, dy) {
  for (let r = 0; r < N; r++) {
    for (let oy = 0; oy <= r; oy++) {
      for (let ox = 0; ox <= r; ox++) {
        const cx = x + ox * dx, cy = y + oy * dy;
        if (cx > 0 && cy > 0 && cx < N - 1 && cy < N - 1 && !wall[idx(cx, cy)]) return idx(cx, cy);
      }
    }
  }
  // fallback: force-clear the requested cell
  const cx = Math.max(1, Math.min(N - 2, x)), cy = Math.max(1, Math.min(N - 2, y));
  wall[idx(cx, cy)] = 0;
  return idx(cx, cy);
}

// guarantee a little breathing room around an endpoint so it never starts boxed in
function carveAround(i) {
  const x = i % N, y = (i / N) | 0;
  for (let oy = -1; oy <= 1; oy++)
    for (let ox = -1; ox <= 1; ox++) {
      const cx = x + ox, cy = y + oy;
      if (cx > 0 && cy > 0 && cx < N - 1 && cy < N - 1) wall[idx(cx, cy)] = 0;
    }
}

function genMazeBy(kind) {
  if (kind === 1) genMaze();
  else if (kind === 2) genRooms();
  else genScatter();
}

// ============================================================
// A* / DIJKSTRA / GREEDY  (shared search, differ only in f)
// ============================================================
let algo = 0;   // 0 A*, 1 Dijkstra, 2 Greedy
const ALGOS = ["A*", "Dijkstra", "Greedy"];

let diag8 = false;          // 4-connected by default; octile when 8-connected
const MOVE_STRAIGHT = 10;   // integer costs avoid float drift in the open-list
const MOVE_DIAG = 14;       // ~10*sqrt(2)

// search state
let gScore, parent, inOpen, closedFlag, heap;
let explored = 0, pathLen = 0, found = false, finished = false;
let pathCells = [];
let pathTraceI = 0;         // index for the bright path reveal
let bestF = 1, worstF = 1;  // f-range over closed nodes, for the heat shading
let goalX = 0, goalY = 0;

// ---- binary min-heap keyed by f (ties broken by h, so A* prefers nearer goal) ----
// Stores {i, f, h}. A plain array heap — small, correct, no deps.
function makeHeap() { return []; }
function hpush(h, node) {
  h.push(node);
  let c = h.length - 1;
  while (c > 0) {
    const p = (c - 1) >> 1;
    if (less(h[c], h[p])) { const t = h[c]; h[c] = h[p]; h[p] = t; c = p; }
    else break;
  }
}
function hpop(h) {
  const top = h[0], last = h.pop();
  if (h.length) {
    h[0] = last;
    let p = 0;
    const n = h.length;
    for (;;) {
      let l = 2 * p + 1, r = l + 1, m = p;
      if (l < n && less(h[l], h[m])) m = l;
      if (r < n && less(h[r], h[m])) m = r;
      if (m === p) break;
      const t = h[p]; h[p] = h[m]; h[m] = t; p = m;
    }
  }
  return top;
}
function less(a, b) { return a.f < b.f || (a.f === b.f && a.h < b.h); }

// heuristic from cell i to goal — scaled to match integer move costs.
function heuristic(x, y) {
  const dx = Math.abs(x - goalX), dy = Math.abs(y - goalY);
  if (algo === 1) return 0;                 // Dijkstra: no heuristic
  if (diag8) {
    // octile distance (admissible for 8-connected grids)
    return MOVE_STRAIGHT * (dx + dy) + (MOVE_DIAG - 2 * MOVE_STRAIGHT) * Math.min(dx, dy);
  }
  return MOVE_STRAIGHT * (dx + dy);         // Manhattan (admissible for 4-connected)
}

function resetSearch() {
  const n = N * N;
  gScore = new Int32Array(n).fill(0x7fffffff);
  parent = new Int32Array(n).fill(-1);
  inOpen = new Uint8Array(n);
  closedFlag = new Uint8Array(n);
  heap = makeHeap();
  explored = 0; pathLen = 0; found = false; finished = false;
  pathCells = []; pathTraceI = 0; bestF = 1; worstF = 1;

  goalX = goalIdx % N; goalY = (goalIdx / N) | 0;
  const sx = startIdx % N, sy = (startIdx / N) | 0;
  gScore[startIdx] = 0;
  const h0 = heuristic(sx, sy);
  hpush(heap, { i: startIdx, f: (algo === 2 ? h0 : 0 + h0), h: h0 });
  inOpen[startIdx] = 1;
}

// neighbor offsets
const N4 = [[1, 0], [-1, 0], [0, 1], [0, -1]];
const N8 = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]];

// Expand ONE node (pop lowest-f, relax neighbors). Returns false when search ends.
function stepOnce() {
  if (finished) return false;
  // skip stale heap entries (lazy decrease-key): a node already closed is ignored.
  let cur = null;
  while (heap.length) {
    const top = hpop(heap);
    if (closedFlag[top.i]) continue;
    cur = top; break;
  }
  if (!cur) { finished = true; found = false; return false; }

  const ci = cur.i;
  closedFlag[ci] = 1; inOpen[ci] = 0;
  explored++;
  // remember f-range for heat shading (skip start/goal so colors stay distinct)
  if (ci !== startIdx && ci !== goalIdx) {
    if (cur.f > worstF) worstF = cur.f;
    if (bestF <= 1 || cur.f < bestF) bestF = cur.f;
  }

  if (ci === goalIdx) {            // goal popped → optimal (for A*/Dijkstra) → done
    finished = true; found = true;
    reconstruct();
    return false;
  }

  const cx = ci % N, cy = (ci / N) | 0;
  const moves = diag8 ? N8 : N4;
  for (let k = 0; k < moves.length; k++) {
    const nx = cx + moves[k][0], ny = cy + moves[k][1];
    if (nx < 0 || ny < 0 || nx >= N || ny >= N) continue;
    const ni = ny * N + nx;
    if (wall[ni] || closedFlag[ni]) continue;
    // prevent diagonal corner-cutting through wall pairs
    if (moves[k][0] !== 0 && moves[k][1] !== 0) {
      if (wall[idx(cx + moves[k][0], cy)] && wall[idx(cx, cy + moves[k][1])]) continue;
    }
    const stepCost = (moves[k][0] !== 0 && moves[k][1] !== 0) ? MOVE_DIAG : MOVE_STRAIGHT;
    const tentative = gScore[ci] + stepCost;
    if (tentative < gScore[ni]) {
      gScore[ni] = tentative;
      parent[ni] = ci;
      const h = heuristic(nx, ny);
      const f = (algo === 2) ? h : tentative + h;   // greedy: f = h only
      hpush(heap, { i: ni, f, h });                 // push; stale dupes filtered on pop
      inOpen[ni] = 1;
    }
  }
  return true;
}

function reconstruct() {
  pathCells = [];
  let i = goalIdx;
  let guard = 0;
  while (i !== -1 && guard++ < N * N) { pathCells.push(i); i = parent[i]; }
  pathCells.reverse();
  pathLen = pathCells.length;
}

// ============================================================
// PAINT — write state ids into the texture each frame.
// ============================================================
function paint() {
  const d = tex.image.data;
  const span = Math.max(1, worstF - bestF);
  for (let i = 0; i < N * N; i++) {
    let s = EMPTY;
    if (wall[i]) s = WALL;
    else if (closedFlag[i]) {
      // shade by f-rank so the heuristic's pull is visible (low f near goal → hot)
      // we don't store per-cell f, so approximate via g + h at paint time.
      const x = i % N, y = (i / N) | 0;
      const f = gScore[i] + heuristic(x, y);
      let t = (f - bestF) / span;            // 0 = best (hot), 1 = worst (cool)
      t = t < 0 ? 0 : t > 1 ? 1 : t;
      const band = Math.round((1 - t) * (HEAT_BANDS - 1));   // invert: best→hot
      s = HEAT0 + band;
    }
    else if (inOpen[i]) s = OPENF;
    d[i * 4] = s;
    d[i * 4 + 3] = 255;
  }
  // path overlay (revealed progressively for a satisfying trace)
  if (found) {
    const upto = Math.min(pathTraceI, pathCells.length);
    for (let k = 0; k < upto; k++) {
      const pi = pathCells[k];
      if (pi !== startIdx && pi !== goalIdx) d[pi * 4] = PATH;
    }
  }
  // endpoints always on top
  d[startIdx * 4] = START;
  d[goalIdx * 4] = GOAL;
  tex.needsUpdate = true;
}

// ============================================================
// RUN CONTROL — generate, run, reveal path, regenerate.
// ============================================================
let expandsPerFrame = reducedMotion ? 8 : 30;
let postPathHold = 0;           // frames to linger on the finished path
const HOLD_FRAMES = 95;

// expansions/frame cap scales down for bigger grids so it stays smooth.
function expandCap() {
  const base = expandsPerFrame;
  return N >= 161 ? base : (N >= 121 ? Math.round(base * 1.3) : Math.round(base * 1.8));
}

function newRun(regen = true) {
  if (regen) genMazeBy(mazeKind);
  resetSearch();
  postPathHold = 0;
  paint();
  syncReadout();
}

// re-run the SAME maze (e.g. after switching algorithm) so the contrast is fair.
function rerunSameMaze() {
  resetSearch();
  postPathHold = 0;
  paint();
  syncReadout();
}

// ============================================================
// PANEL + HOTKEYS
// ============================================================
const algoEl = document.getElementById("algoname");
const exploredEl = document.getElementById("explored");
const pathEl = document.getElementById("pathlen");

function syncReadout() {
  if (algoEl) algoEl.textContent = ALGOS[algo];
  if (exploredEl) exploredEl.textContent = explored.toLocaleString();
  if (pathEl) pathEl.textContent = found ? pathLen : (finished ? "no path" : "…");
}

// algorithm chips
const algoWrap = document.getElementById("algos");
const algoChips = [];
if (algoWrap) {
  ALGOS.forEach((name, i) => {
    const b = document.createElement("button");
    b.className = "chip" + (i === 0 ? " active" : "");
    b.textContent = name;
    b.addEventListener("click", () => { algo = i; rerunSameMaze(); syncAlgoChips(); });
    algoWrap.appendChild(b);
    algoChips.push(b);
  });
}
function syncAlgoChips() { algoChips.forEach((b, i) => b.classList.toggle("active", i === algo)); }

// maze chips
const mazeWrap = document.getElementById("mazes");
const mazeChips = [];
if (mazeWrap) {
  MAZES.forEach((name, i) => {
    const b = document.createElement("button");
    b.className = "chip" + (i === mazeKind ? " active" : "");
    b.textContent = name;
    b.addEventListener("click", () => { mazeKind = i; newRun(true); syncMazeChips(); });
    mazeWrap.appendChild(b);
    mazeChips.push(b);
  });
}
function syncMazeChips() { mazeChips.forEach((b, i) => b.classList.toggle("active", i === mazeKind)); }

// connectivity toggle (4 vs 8)
const connBtn = document.getElementById("conn");
function syncConn() { if (connBtn) connBtn.textContent = diag8 ? "8-connected" : "4-connected"; }
if (connBtn) connBtn.addEventListener("click", () => { diag8 = !diag8; rerunSameMaze(); syncConn(); });
syncConn();

// new maze button
const newBtn = document.getElementById("newmaze");
if (newBtn) newBtn.addEventListener("click", () => newRun(true));

// grid-size button
const sizeBtn = document.getElementById("size");
function syncSize() { if (sizeBtn) sizeBtn.textContent = `grid: ${N}`; }
if (sizeBtn) sizeBtn.addEventListener("click", () => {
  sizeIdx = (sizeIdx + 1) % SIZES.length;
  N = SIZES[sizeIdx];
  wall = new Uint8Array(N * N);
  tex.dispose();
  tex = makeTex(N);
  material.uniforms.uTex.value = tex;
  newRun(true);
  syncSize();
});
syncSize();

bindRange("speed", (v) => { expandsPerFrame = Math.round(v); }, (v) => `${Math.round(v)}/f`);

// ↑/↓ cycles the algorithm (A* vs Dijkstra vs Greedy) on the SAME maze, so the
// difference in cells-explored is immediately visible. Returns the kiosk label.
setVariantCycler((d) => {
  algo = (algo + d + ALGOS.length) % ALGOS.length;
  syncAlgoChips();
  rerunSameMaze();
  return ALGOS[algo];
});

// color-scheme cycling lives on a button (keeps ↑/↓ for the headline contrast).
const schemeBtn = document.getElementById("scheme");
function syncScheme() { if (schemeBtn) schemeBtn.textContent = SCHEMES[scheme][0]; }
if (schemeBtn) schemeBtn.addEventListener("click", () => {
  scheme = (scheme + 1) % SCHEMES.length;
  applyScheme();
  syncScheme();
});
syncScheme();

function resize(w, h) { material.uniforms.uAspect.value = w / h; }
onResize(renderer, null, resize);
resize(window.innerWidth, window.innerHeight);

// ---------- boot ----------
newRun(true);
liftVeil();

const meter = fpsMeter(document.getElementById("fps"));

loop((dt) => {
  meter(dt);

  if (!finished) {
    // advance a bounded number of expansions so the frontier visibly grows.
    const cap = expandCap();
    for (let k = 0; k < cap; k++) {
      if (!stepOnce()) break;
    }
    paint();
    syncReadout();
  } else if (found && pathTraceI < pathCells.length) {
    // reveal the path a few cells per frame for a clean trace.
    pathTraceI = Math.min(pathCells.length, pathTraceI + pathTraceStep());
    paint();
    if (pathTraceI >= pathCells.length) syncReadout();
  } else {
    // hold the finished picture, then start a fresh maze + search.
    postPathHold++;
    if (postPathHold >= HOLD_FRAMES) newRun(true);
  }

  renderer.render(scene, camera);
});

// cells revealed per frame while tracing the final path (smaller for big grids).
function pathTraceStep() { return N >= 161 ? 3 : (N >= 121 ? 4 : 5); }

// diagnostics — quantifies A*'s efficiency vs Dijkstra (explored cell count).
window.__diag = () => JSON.stringify({
  algo: ALGOS[algo],
  explored,
  pathLen: found ? pathLen : 0,
});
