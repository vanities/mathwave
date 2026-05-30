// epicycles.js — Fourier epicycles: ANY closed curve can be drawn by a chain of
// rotating circles, each turning at an integer frequency. We take a target path,
// compute its complex Discrete Fourier Transform to get each circle's radius and
// phase (cₖ = (1/N)Σ z[n] e^(−2πikn/N)), then sort circles biggest-first and
// chain them tip-to-tip. The end of the last circle traces the original shape.
// This is the geometric heart of the Fourier series. ↑↓ swaps the target shape.
//
// Ref: 3Blue1Brown "But what is a Fourier series" / DFT.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { makeRenderer, onResize, loop, fpsMeter, liftVeil, bindRange, ramp, reducedMotion, addGrid, addSun, setVariantCycler } from "./common.js";

const canvas = document.getElementById("scene");
const renderer = makeRenderer(canvas);
const scene = new THREE.Scene();

const camera = new THREE.PerspectiveCamera(50, innerWidth / innerHeight, 0.1, 600);
camera.position.set(0, 0, 18);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true; controls.dampingFactor = 0.07;
controls.minDistance = 6; controls.maxDistance = 50; controls.autoRotate = false;

scene.add(new THREE.AmbientLight(0x3a2466, 0.9));
addGrid(scene, { size: 40, divisions: 20, y: -9 });
addSun(scene, { scale: 30, position: [0, 8, -56] });

// ---------- target shapes as point lists (complex samples) ----------
const SAMPLES = 256;
function shapePath(kind) {
  const pts = [];
  for (let i = 0; i < SAMPLES; i++) {
    const t = i / SAMPLES * Math.PI * 2; let x, y;
    if (kind === "square") { const s = i / SAMPLES * 4, e = Math.floor(s), f = s - e, S = 5;
      const C = [[-S,-S],[S,-S],[S,S],[-S,S]]; const a = C[e], b = C[(e+1)%4]; x = a[0]+(b[0]-a[0])*f; y = a[1]+(b[1]-a[1])*f; }
    else if (kind === "star") { const k = i / SAMPLES * 5 * Math.PI * 2; const r = 5 * (0.5 + 0.5*Math.cos(k)); x = Math.cos(t)*r; y = Math.sin(t)*r; }
    else if (kind === "heart") { x = 16*Math.sin(t)**3*0.32; y = (13*Math.cos(t)-5*Math.cos(2*t)-2*Math.cos(3*t)-Math.cos(4*t))*0.32; }
    else if (kind === "infinity") { x = 6*Math.cos(t)/(1+Math.sin(t)**2); y = 6*Math.sin(t)*Math.cos(t)/(1+Math.sin(t)**2); }
    else { x = Math.cos(t)*5; y = Math.sin(t)*5; } // circle
    pts.push([x, y]);
  }
  return pts;
}

// ---------- DFT → epicycle list {freq, amp, phase} ----------
function dft(pts) {
  const N = pts.length, out = [];
  for (let k = 0; k < N; k++) {
    let re = 0, im = 0;
    for (let n = 0; n < N; n++) {
      const a = -2 * Math.PI * k * n / N, c = Math.cos(a), s = Math.sin(a);
      re += pts[n][0] * c - pts[n][1] * s;
      im += pts[n][0] * s + pts[n][1] * c;
    }
    re /= N; im /= N;
    // map k to signed frequency so circles alternate ±
    const freq = k <= N/2 ? k : k - N;
    out.push({ freq, amp: Math.hypot(re, im), phase: Math.atan2(im, re) });
  }
  out.sort((a, b) => b.amp - a.amp);   // biggest circles first
  return out;
}

let epi = [];
let numCircles = 60;

// ---------- geometry ----------
const circleLines = new THREE.Group(); scene.add(circleLines);
const armLine = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.6 }));
scene.add(armLine);
const traceGeo = new THREE.BufferGeometry();
const TRACE = SAMPLES;
const tracePos = new Float32Array(TRACE * 3);
traceGeo.setAttribute("position", new THREE.BufferAttribute(tracePos, 3).setUsage(THREE.DynamicDrawUsage));
const traceLine = new THREE.Line(traceGeo, new THREE.LineBasicMaterial({ color: 0x2be4ff, transparent: true, opacity: 0.95, blending: THREE.AdditiveBlending }));
traceLine.frustumCulled = false; scene.add(traceLine);
let traceN = 0;

const circleTemplate = (() => { const p = []; for (let k = 0; k <= 40; k++) { const a = k/40*Math.PI*2; p.push(new THREE.Vector3(Math.cos(a), Math.sin(a), 0)); } return p; })();
function buildCircles() {
  while (circleLines.children.length) { const c = circleLines.children.pop(); c.geometry.dispose(); circleLines.remove(c); }
  const n = Math.min(numCircles, epi.length);
  for (let i = 0; i < n; i++) {
    const g = new THREE.BufferGeometry().setFromPoints(circleTemplate);
    const c = ramp(i / n);
    circleLines.add(new THREE.Line(g, new THREE.LineBasicMaterial({ color: new THREE.Color(c[0],c[1],c[2]), transparent: true, opacity: 0.3 })));
  }
}

function setShape(kind) { epi = dft(shapePath(kind)); buildCircles(); traceN = 0; }

let speed = 1, time = 0;
function update() {
  const n = Math.min(numCircles, epi.length);
  let x = 0, y = 0;
  const arm = [new THREE.Vector3(0,0,0)];
  for (let i = 0; i < n; i++) {
    const e = epi[i];
    const ang = e.freq * time + e.phase;
    const cx = x, cy = y;
    x += e.amp * Math.cos(ang); y += e.amp * Math.sin(ang);
    arm.push(new THREE.Vector3(x, y, 0));
    // position+scale this circle
    const cl = circleLines.children[i];
    const sp = cl.geometry.attributes.position.array;
    for (let k = 0; k <= 40; k++) { const a = k/40*Math.PI*2; sp[k*3]=cx+Math.cos(a)*e.amp; sp[k*3+1]=cy+Math.sin(a)*e.amp; sp[k*3+2]=0; }
    cl.geometry.attributes.position.needsUpdate = true;
  }
  armLine.geometry.setFromPoints(arm);
  // trace
  if (traceN < TRACE) { tracePos.set([x,y,0], traceN*3); traceN++; }
  else { tracePos.copyWithin(0,3); tracePos.set([x,y,0],(TRACE-1)*3); }
  traceLine.geometry.setDrawRange(0, traceN);
  traceLine.geometry.attributes.position.needsUpdate = true;
}

// ---------- panel ----------
const wrap = document.getElementById("shapes");
const SHAPES = ["circle","square","star","heart","infinity"];
let si = 0;
const chips = SHAPES.map((label, i) => {
  const b = document.createElement("button"); b.className = "chip" + (i===0?" active":""); b.textContent = label;
  b.addEventListener("click", () => { si=i; setShape(label); chips.forEach((c,k)=>c.classList.toggle("active",k===i)); });
  wrap.appendChild(b); return b;
});
bindRange("speed", (v) => { speed = v; }, (v) => v.toFixed(2)+"×");
bindRange("circles", (v) => { numCircles = Math.round(v); buildCircles(); traceN = 0; }, (v) => `${Math.round(v)}`);
setVariantCycler((d) => { si = (si+d+SHAPES.length)%SHAPES.length; setShape(SHAPES[si]); chips.forEach((c,k)=>c.classList.toggle("active",k===si)); return SHAPES[si]; });

// ---------- boot ----------
setShape("circle");
liftVeil();
onResize(renderer, camera);
const meter = fpsMeter(document.getElementById("fps"));
loop((dt) => { meter(dt); time += dt * 0.4 * speed; if (time > Math.PI*2) { time -= Math.PI*2; } update(); controls.update(); renderer.render(scene, camera); });
