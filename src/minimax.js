// minimax.js — Minimax + Alpha-Beta pruning, the adversarial game-tree search,
// drawn as a real 3D tree of nodes + edges with the search animating over it.
//
// THE ALGORITHM (accurate):
//   MINIMAX (von Neumann 1928; the standard two-player zero-sum game-tree
//   procedure — Russell & Norvig, "AIMA", ch. 5). Two players alternate moves
//   down a game tree. The MAX player picks the child of greatest value; the MIN
//   player (the opponent) picks the child of least value. Leaves carry a utility
//   from a static evaluation. The minimax value of an internal node is computed
//   bottom-up, the level type alternating with depth (root = MAX at depth 0):
//       value(node) = max over children    if node is a MAX node
//                     min over children    if node is a MIN node
//
//   ALPHA-BETA PRUNING (Knuth & Moore 1975). A depth-first walk carries two
//   bounds: α = best value MAX can already guarantee on the path so far, β = best
//   (lowest) value MIN can already guarantee. At a MAX node α rises to the best
//   child seen; at a MIN node β falls. The instant α ≥ β the remaining siblings
//   are PRUNED — they cannot change the value that propagates to the parent,
//   because the parent already has a better alternative. Alpha-beta returns the
//   SAME minimax value as plain minimax while visiting far fewer nodes
//   (best case ~b^(d/2) vs b^d).
//
// THE VISUAL: a 3D tree (root at top, children fanning down/out). Nodes are
//   instanced spheres — cyan ▲ for MAX levels, magenta ▼ for MIN. Edges are
//   lines. The alpha-beta traversal animates in order: the active node lights
//   up white, values propagate upward, pruned subtrees flash then collapse to
//   dim grey (a satisfying snip), and α/β show on the active path. At the end
//   the optimal principal-variation path glows amber. Then it regenerates.
//   ↑↓ cycles tree-shape presets — and a "no-prune" preset runs plain minimax
//   so you can SEE in the readout how many nodes alpha-beta saves.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { makeRenderer, onResize, loop, fpsMeter, liftVeil, reducedMotion, addGrid, addSun, setVariantCycler } from "./common.js";

// ---- palette (near-black bg, cyan MAX / magenta MIN / amber PV, grey pruned) ----
const C_MAX = new THREE.Color(0x2be4ff);   // cyan  — maximizing levels
const C_MIN = new THREE.Color(0xff2e97);   // magenta — minimizing levels
const C_PV = new THREE.Color(0xffb648);    // amber — principal variation
const C_ACTIVE = new THREE.Color(0xffffff); // white — node under examination
const C_PRUNED = new THREE.Color(0x3a4250); // dim grey — collapsed/cut subtree

const NODE_CAP = 1000;   // hard ceiling on b^d so deep/wide presets never explode
const BASE_STEP = 0.18;  // seconds per traversal event (scaled down for big trees)

// preset tree shapes cycled with ↑↓. prune:false runs plain minimax (visits
// everything) so the node counts in the readout show what alpha-beta saves.
const VARIANTS = [
  { label: "3×4",          b: 3, d: 4, prune: true },
  { label: "2×6 deep",     b: 2, d: 6, prune: true },
  { label: "4×3 wide",     b: 4, d: 3, prune: true },
  { label: "no-prune 3×4", b: 3, d: 4, prune: false },
];

// ---- scene (mirrors nbody.js / grokking.js exactly) ----
const canvas = document.getElementById("scene");
const renderer = makeRenderer(canvas);
const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x05060a, 0.018);

const camera = new THREE.PerspectiveCamera(55, innerWidth / innerHeight, 0.1, 600);
camera.position.set(0, 1, 26);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true; controls.dampingFactor = 0.07;
controls.autoRotate = !reducedMotion; controls.autoRotateSpeed = 0.4;
controls.minDistance = 10; controls.maxDistance = 90;
controls.target.set(0, 0, 0);

scene.add(new THREE.AmbientLight(0x2a3344, 0.9));
const key = new THREE.DirectionalLight(0xdfe8ff, 0.6); key.position.set(4, 12, 10); scene.add(key);
addGrid(scene, { size: 60, divisions: 30, y: -9 });
addSun(scene, { scale: 34, position: [0, 12, -70] });

const group = new THREE.Group(); scene.add(group);

// ---- DIAG (polled by the gallery harness) ----
const DIAG = { depth: 0, branching: 0, visited: 0, pruned: 0, minimax: 0 };

// ---- a numbered/value sprite label (CanvasTexture, same recipe as grokking.js) ----
function makeLabel(text, color = "#9fb0cc", glow = "#2be4ff", scale = 1) {
  const c = document.createElement("canvas"); c.width = 128; c.height = 64;
  const g = c.getContext("2d");
  g.font = "bold 44px 'VT323', monospace"; g.textAlign = "center"; g.textBaseline = "middle";
  g.fillStyle = color; g.shadowColor = glow; g.shadowBlur = 10; g.fillText(text, 64, 34);
  const tex = new THREE.CanvasTexture(c);
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false }));
  sp.scale.set(1.1 * scale, 0.55 * scale, 1);
  sp.userData.tex = tex;
  return sp;
}
function setLabel(sp, text, color = "#dfe8ff", glow = "#2be4ff") {
  const c = sp.userData.tex.image;
  const g = c.getContext("2d");
  g.clearRect(0, 0, c.width, c.height);
  g.font = "bold 40px 'VT323', monospace"; g.textAlign = "center"; g.textBaseline = "middle";
  g.fillStyle = color; g.shadowColor = glow; g.shadowBlur = 10; g.fillText(text, 64, 34);
  sp.userData.tex.needsUpdate = true;
}

// ---- tree node ----
class TreeNode {
  constructor(depth, isMax) {
    this.id = -1;
    this.depth = depth;
    this.isMax = isMax;
    this.children = [];
    this.parent = null;
    this.pos = new THREE.Vector3();
    this.utility = 0;     // leaf utility (static evaluation)
    this.value = NaN;     // resolved minimax value during traversal
    this.visited = false; // reached by the search
    this.pruned = false;  // node lives in a cut subtree (never examined)
    this.onPV = false;    // on the final principal variation
    this.lit = 0;         // 0..1 highlight, decays each frame
  }
  get isLeaf() { return this.children.length === 0; }
}

let nodes = [];   // flat list, index === id

// Build a random game tree with branching b and depth d, capped at NODE_CAP.
function buildTree(b, d) {
  nodes = [];
  const root = new TreeNode(0, true);
  root.id = nodes.length; nodes.push(root);

  let frontier = [root];
  for (let level = 0; level < d; level++) {
    const next = [];
    for (const node of frontier) {
      for (let k = 0; k < b; k++) {
        if (nodes.length >= NODE_CAP) break;
        const child = new TreeNode(node.depth + 1, !node.isMax);
        child.parent = node;
        node.children.push(child);
        child.id = nodes.length; nodes.push(child);
        next.push(child);
      }
      if (nodes.length >= NODE_CAP) break;
    }
    frontier = next;
    if (nodes.length >= NODE_CAP) break;
  }

  // leaves get random integer utilities in [-99, 99]
  for (const n of nodes) if (n.isLeaf) n.utility = Math.round((Math.random() * 2 - 1) * 99);

  layout(root, d);
  return root;
}

// Recursive layout: each subtree gets a horizontal slot sized by its leaf count,
// so siblings never overlap regardless of branching factor. Gentle Z fan by depth
// so the tree reads as a 3D structure, not a flat fan.
function layout(root, depth) {
  const ySpan = 15, yTop = ySpan / 2;
  const dyPer = depth > 0 ? ySpan / depth : 0;
  const width = 24;

  const leafCount = new Map();
  (function count(n) {
    if (n.isLeaf) { leafCount.set(n, 1); return 1; }
    let s = 0; for (const c of n.children) s += count(c);
    leafCount.set(n, s); return s;
  })(root);

  const totalLeaves = leafCount.get(root) || 1;
  const unit = width / totalLeaves;

  let cursor = -width / 2;
  (function place(n) {
    if (n.isLeaf) {
      const x = cursor + unit / 2; cursor += unit;
      n.pos.set(x, yTop - n.depth * dyPer, 0);
      return;
    }
    const start = cursor;
    for (const c of n.children) place(c);
    const span = (leafCount.get(n) || 1) * unit;
    const z = (n.depth % 2 === 0 ? 1 : -1) * n.depth * 0.5;
    n.pos.set(start + span / 2, yTop - n.depth * dyPer, z);
  })(root);
}

// ---- minimax + alpha-beta traversal, recording an event timeline ----
// events: {type, node, alpha?, beta?, value?, from?}
let timeline = [];

function search(root, usePrune) {
  timeline = [];
  let visited = 0, pruned = 0;

  function recurse(node, alpha, beta) {
    visited++;
    timeline.push({ type: "enter", node, alpha, beta });

    if (node.isLeaf) {
      node.value = node.utility;
      timeline.push({ type: "leaf", node, value: node.value });
      return node.value;
    }

    if (node.isMax) {
      let best = -Infinity;
      for (let i = 0; i < node.children.length; i++) {
        const v = recurse(node.children[i], alpha, beta);
        if (v > best) best = v;
        if (v > alpha) alpha = v;
        node.value = best;
        timeline.push({ type: "value", node, value: best, alpha, beta });
        if (usePrune && alpha >= beta) {              // beta cutoff
          pruned += markPruned(node.children, i + 1);
          timeline.push({ type: "prune", node, from: i + 1, alpha, beta });
          break;
        }
      }
      timeline.push({ type: "return", node, value: best });
      return best;
    } else {
      let best = Infinity;
      for (let i = 0; i < node.children.length; i++) {
        const v = recurse(node.children[i], alpha, beta);
        if (v < best) best = v;
        if (v < beta) beta = v;
        node.value = best;
        timeline.push({ type: "value", node, value: best, alpha, beta });
        if (usePrune && alpha >= beta) {              // alpha cutoff
          pruned += markPruned(node.children, i + 1);
          timeline.push({ type: "prune", node, from: i + 1, alpha, beta });
          break;
        }
      }
      timeline.push({ type: "return", node, value: best });
      return best;
    }
  }

  const value = recurse(root, -Infinity, Infinity);
  DIAG.visited = visited;
  DIAG.pruned = pruned;
  DIAG.minimax = value;
  return value;
}

function markPruned(children, from) {
  let count = 0;
  for (let i = from; i < children.length; i++) count += markSubtree(children[i]);
  return count;
}
function markSubtree(node) {
  let count = 1; node.pruned = true;
  for (const c of node.children) count += markSubtree(c);
  return count;
}

// Principal variation: from root follow the child whose value equals the chosen
// (max for MAX, min for MIN), skipping pruned nodes. Flags onPV, returns path.
function computePV(root) {
  const path = [];
  let n = root;
  while (n) {
    n.onPV = true; path.push(n);
    if (n.isLeaf) break;
    let pick = null;
    for (const c of n.children) {
      if (c.pruned || Number.isNaN(c.value)) continue;
      if (pick === null) pick = c;
      else if (n.isMax ? c.value > pick.value : c.value < pick.value) pick = c;
    }
    n = pick;
  }
  return path;
}

// ---- render objects bound to the current tree ----
const dummy = new THREE.Object3D();
const tmpColor = new THREE.Color();

let nodeMesh = null, edgeLines = null, pvLines = null;
let leafLabels = [], abText = null;

function buildRenderObjects() {
  // node spheres (instanced) — frustumCulled off since they spread wide
  const sphereGeo = new THREE.SphereGeometry(0.34, 16, 12);
  const mat = new THREE.MeshBasicMaterial({ transparent: true, blending: THREE.AdditiveBlending, depthWrite: false });
  nodeMesh = new THREE.InstancedMesh(sphereGeo, mat, nodes.length);
  nodeMesh.frustumCulled = false;
  for (const n of nodes) {
    dummy.position.copy(n.pos);
    dummy.scale.setScalar(n.isLeaf ? 0.8 : 1.0);
    dummy.updateMatrix();
    nodeMesh.setMatrixAt(n.id, dummy.matrix);
    nodeMesh.setColorAt(n.id, n.isMax ? C_MAX : C_MIN);
  }
  nodeMesh.instanceMatrix.needsUpdate = true;
  if (nodeMesh.instanceColor) nodeMesh.instanceColor.needsUpdate = true;
  group.add(nodeMesh);

  // edges (parent → child)
  const edgePts = [];
  for (const n of nodes) for (const c of n.children) edgePts.push(n.pos.x, n.pos.y, n.pos.z, c.pos.x, c.pos.y, c.pos.z);
  const edgeGeo = new THREE.BufferGeometry();
  edgeGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(edgePts), 3));
  edgeLines = new THREE.LineSegments(edgeGeo, new THREE.LineBasicMaterial({
    color: 0x33415c, transparent: true, opacity: 0.4, blending: THREE.AdditiveBlending, depthWrite: false,
  }));
  edgeLines.frustumCulled = false;
  group.add(edgeLines);

  // principal-variation overlay (filled in when the run completes)
  const pvGeo = new THREE.BufferGeometry();
  pvGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(0), 3));
  pvLines = new THREE.LineSegments(pvGeo, new THREE.LineBasicMaterial({
    color: 0xffb648, transparent: true, opacity: 0.95, blending: THREE.AdditiveBlending, depthWrite: false,
  }));
  pvLines.frustumCulled = false;
  group.add(pvLines);

  // leaf utility labels (capped so sprite count stays reasonable)
  leafLabels = [];
  const leaves = nodes.filter((n) => n.isLeaf);
  const stride = Math.max(1, Math.ceil(leaves.length / 64));
  for (let i = 0; i < leaves.length; i += stride) {
    const n = leaves[i];
    const sp = makeLabel(String(n.utility), "#9fb0cc", "#2be4ff", 0.9);
    sp.position.copy(n.pos).add(new THREE.Vector3(0, -0.6, 0));
    sp.userData.node = n;
    group.add(sp);
    leafLabels.push(sp);
  }

  // live α/β readout, re-parented to the active node each step
  abText = makeLabel("α=-∞ β=+∞", "#dfe8ff", "#2be4ff", 1.2);
  abText.visible = false;
  group.add(abText);
}

function disposeRenderObjects() {
  for (const obj of [nodeMesh, edgeLines, pvLines]) {
    if (obj) { group.remove(obj); obj.geometry.dispose(); obj.material.dispose(); }
  }
  for (const sp of leafLabels) { group.remove(sp); sp.userData.tex.dispose(); sp.material.dispose(); }
  if (abText) { group.remove(abText); abText.userData.tex.dispose(); abText.material.dispose(); }
  nodeMesh = edgeLines = pvLines = abText = null;
  leafLabels = [];
}

// ---- animation timeline playback ----
let evCursor = 0, evClock = 0, stepDur = BASE_STEP, finished = false, restartAt = 0;

function advanceTimeline(dt, elapsed) {
  if (finished) return;
  evClock += dt;
  while (evClock >= stepDur && evCursor < timeline.length) {
    evClock -= stepDur;
    applyEvent(timeline[evCursor]);
    evCursor++;
  }
  if (evCursor >= timeline.length) finishRun(elapsed);
}

function applyEvent(ev) {
  const n = ev.node;
  switch (ev.type) {
    case "enter": n.visited = true; n.lit = 1; placeAB(n, ev.alpha, ev.beta); break;
    case "leaf":  n.value = ev.value; n.lit = 1; break;
    case "value": n.value = ev.value; n.lit = Math.max(n.lit, 0.7); placeAB(n, ev.alpha, ev.beta); break;
    case "prune": for (let i = ev.from; i < n.children.length; i++) flashSubtree(n.children[i]); break;
    case "return": n.lit = Math.max(n.lit, 0.5); break;
  }
}

function flashSubtree(node) {       // brief flash before the cut subtree collapses to grey
  node.lit = 1;
  for (const c of node.children) flashSubtree(c);
}

function fmtBound(v) { return v === -Infinity ? "-∞" : v === Infinity ? "+∞" : String(v); }
function placeAB(n, alpha, beta) {
  if (!abText) return;
  setLabel(abText, `α=${fmtBound(alpha)} β=${fmtBound(beta)}`);
  abText.position.copy(n.pos).add(new THREE.Vector3(0, 0.9, 0));
  abText.visible = true;
}

function finishRun(elapsed) {
  finished = true;
  if (abText) abText.visible = false;
  const path = computePV(nodes[0]);
  const pts = [];
  for (let i = 0; i + 1 < path.length; i++) {
    const a = path[i].pos, b = path[i + 1].pos;
    pts.push(a.x, a.y, a.z, b.x, b.y, b.z);
  }
  pvLines.geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(pts), 3));
  pvLines.geometry.attributes.position.needsUpdate = true;
  updateReadout();
  restartAt = elapsed + 3.0;        // regenerate after the result settles
}

// Recolor every instance from its animation state each frame so highlight pulses
// and the prune "snip" read smoothly.
function refreshColors(dt) {
  if (!nodeMesh) return;
  const decay = Math.exp(-dt / 0.38);
  for (const n of nodes) {
    n.lit *= decay;
    let c, scaleMul = 1;
    if (n.pruned && n.lit < 0.15) { c = C_PRUNED; scaleMul = 0.45; }      // collapsed
    else if (n.onPV) c = C_PV;
    else if (!n.visited && !n.pruned) { tmpColor.copy(n.isMax ? C_MAX : C_MIN).multiplyScalar(0.32); c = tmpColor; }
    else { tmpColor.copy(n.isMax ? C_MAX : C_MIN).lerp(C_ACTIVE, Math.min(0.85, n.lit)); c = tmpColor; }
    nodeMesh.setColorAt(n.id, c);
    if (scaleMul !== 1) {
      dummy.position.copy(n.pos);
      dummy.scale.setScalar((n.isLeaf ? 0.8 : 1.0) * scaleMul);
      dummy.updateMatrix();
      nodeMesh.setMatrixAt(n.id, dummy.matrix);
    }
  }
  nodeMesh.instanceMatrix.needsUpdate = true;
  if (nodeMesh.instanceColor) nodeMesh.instanceColor.needsUpdate = true;

  for (const sp of leafLabels) sp.material.opacity = sp.userData.node.pruned ? 0.18 : 0.9;
}

// ---- (re)generate a tree + run + render objects ----
let variant = 0;

function regenerate() {
  disposeRenderObjects();
  const v = VARIANTS[variant];
  buildTree(v.b, v.d);
  stepDur = Math.max(0.05, Math.min(BASE_STEP, (BASE_STEP * 60) / nodes.length));
  search(nodes[0], v.prune);
  buildRenderObjects();

  evCursor = 0; evClock = 0; finished = false; restartAt = 0;
  DIAG.depth = v.d; DIAG.branching = v.b;
  updateReadout();
}

// ---- readout (writes into the static HUD nodes in the HTML) ----
const elPreset = document.getElementById("preset-name");
const elNodes = document.getElementById("node-count");
const elVisited = document.getElementById("visited");
const elPruned = document.getElementById("pruned");
const elValue = document.getElementById("mmval");

function updateReadout() {
  const v = VARIANTS[variant];
  if (elPreset) elPreset.textContent = v.label;
  if (elNodes) elNodes.textContent = nodes.length;
  if (elVisited) elVisited.textContent = DIAG.visited;
  if (elPruned) elPruned.textContent = DIAG.pruned;
  if (elValue) elValue.textContent = DIAG.minimax;
}

// ↑↓ cycles tree-shape presets in place; returns the label for the kiosk toast.
setVariantCycler((d) => {
  variant = (variant + d + VARIANTS.length) % VARIANTS.length;
  regenerate();
  return VARIANTS[variant].label;
});

// ---- boot ----
regenerate();
liftVeil();
onResize(renderer, camera);
const meter = fpsMeter(document.getElementById("fps"));

loop((dt, elapsed) => {
  meter(dt);
  advanceTimeline(dt, elapsed);
  refreshColors(dt);
  if (finished && restartAt && elapsed >= restartAt) regenerate();
  controls.update();
  renderer.render(scene, camera);
});

// diagnostics hook for the gallery harness.
window.__diag = () => JSON.stringify({
  depth: DIAG.depth, branching: DIAG.branching, visited: DIAG.visited, pruned: DIAG.pruned,
});
