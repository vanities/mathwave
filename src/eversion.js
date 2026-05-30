// eversion.js — an artistic Morin-style sphere eversion (turning a sphere
// inside-out). NOTE: this is a visual homage, not a certified regular homotopy —
// it morphs through a corrugated halfway model and routes the surface through
// itself. Front faces and back faces are colored differently (magenta vs cyan),
// so you literally watch the inside become the outside. Loops forever.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import {
  makeRenderer, onResize, loop, fpsMeter, liftVeil, bindRange, reducedMotion, addGrid, addSun, setVariantCycler,
} from "./common.js";

const canvas = document.getElementById("scene");
const renderer = makeRenderer(canvas);

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x0a0118, 0.012);

const camera = new THREE.PerspectiveCamera(50, innerWidth / innerHeight, 0.1, 600);
camera.position.set(0, 6, 26);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.07;
controls.autoRotate = !reducedMotion;
controls.autoRotateSpeed = 0.6;
controls.minDistance = 9;
controls.maxDistance = 80;

addGrid(scene, { size: 80, divisions: 40, y: -12 });
addSun(scene, { scale: 50, position: [0, 14, -80] });

// ---------- a UV-sphere grid so we have (theta, phi) per vertex ----------
const ROWS = 130;   // phi  : 0..PI
const COLS = 200;   // theta: 0..2PI
const R = 8;
const vertCount = (ROWS + 1) * (COLS + 1);

const theta = new Float32Array(vertCount);
const phi = new Float32Array(vertCount);
const positions = new Float32Array(vertCount * 3);

let p = 0;
for (let r = 0; r <= ROWS; r++) {
  const ph = (r / ROWS) * Math.PI;
  for (let c = 0; c <= COLS; c++) {
    const th = (c / COLS) * Math.PI * 2;
    phi[p] = ph; theta[p] = th; p++;
  }
}

const indices = [];
const stride = COLS + 1;
for (let r = 0; r < ROWS; r++) {
  for (let c = 0; c < COLS; c++) {
    const a = r * stride + c, b = a + 1, d = a + stride, e = d + 1;
    indices.push(a, d, b, b, d, e);
  }
}

const geo = new THREE.BufferGeometry();
geo.setAttribute("position", new THREE.BufferAttribute(positions, 3).setUsage(THREE.DynamicDrawUsage));
geo.setIndex(indices);

// ---------- front/back colored shader (the eversion reveal) ----------
const uniforms = {
  cFront: { value: new THREE.Color(0xff2e97) }, // outside = magenta
  cBack:  { value: new THREE.Color(0x2be4ff) }, // inside  = cyan
};
const material = new THREE.ShaderMaterial({
  uniforms,
  side: THREE.DoubleSide,
  vertexShader: /* glsl */ `
    varying vec3 vN;
    varying vec3 vP;
    void main() {
      vec4 mv = modelViewMatrix * vec4(position, 1.0);
      vP = mv.xyz;
      vN = normalMatrix * normal;
      gl_Position = projectionMatrix * mv;
    }
  `,
  fragmentShader: /* glsl */ `
    precision highp float;
    uniform vec3 cFront;
    uniform vec3 cBack;
    varying vec3 vN;
    varying vec3 vP;
    void main() {
      vec3 N = normalize(vN);
      if (!gl_FrontFacing) N = -N;
      vec3 base = gl_FrontFacing ? cFront : cBack;
      vec3 L = normalize(vec3(0.4, 0.85, 0.6));
      float dif = clamp(dot(N, L), 0.0, 1.0);
      vec3 V = normalize(-vP);
      float fres = pow(1.0 - clamp(dot(N, V), 0.0, 1.0), 2.4);
      vec3 col = base * (0.30 + 0.9 * dif) + base * fres * 0.7;
      vec3 H = normalize(L + V);
      col += vec3(0.9, 0.95, 1.0) * pow(clamp(dot(N, H), 0.0, 1.0), 40.0) * 0.5;
      col = col / (col + vec3(0.7));        // tonemap
      col = pow(col, vec3(0.4545));         // gamma
      gl_FragColor = vec4(col, 1.0);
    }
  `,
});
const mesh = new THREE.Mesh(geo, material);
scene.add(mesh);

// ---------- the morph ----------
// tau in [0,1] ping-pongs. At tau=0 it's a normal sphere (magenta out); the
// surface corrugates, passes through itself, and arrives inside-out (cyan out).
let K = 6;             // corrugation lobes
let speed = 0.18;      // morph speed
let playing = !reducedMotion;
let tau = 0, dir = 1;

const ease = (x) => x * x * (3 - 2 * x); // smoothstep

function morph() {
  const t = ease(tau);
  const env = Math.sin(Math.PI * tau);        // 0 at ends, 1 at halfway
  const flip = Math.cos(Math.PI * t);          // +1 → -1  (the inside-out turn)
  const corrA = 0.42 * env;                     // radial corrugation amplitude
  const waist = 0.55 * env;                     // vertical waviness to avoid a flat pinch

  for (let i = 0; i < vertCount; i++) {
    const th = theta[i], ph = phi[i];
    const sp = Math.sin(ph), cp = Math.cos(ph);
    const corr = 1 + corrA * Math.sin(K * th) * Math.sin(2 * ph);
    const rr = R * corr;
    const ix = i * 3;
    positions[ix]     = rr * sp * Math.cos(th);
    positions[ix + 1] = rr * (flip * cp) + R * waist * Math.sin(K * th) * sp;
    positions[ix + 2] = rr * sp * Math.sin(th);
  }
  geo.attributes.position.needsUpdate = true;
  geo.computeVertexNormals();
  geo.computeBoundingSphere();
}

// ---------- panel ----------
const LOBES = [4, 6, 8, 12];
const wrap = document.getElementById("lobes");
LOBES.forEach((k, i) => {
  const b = document.createElement("button");
  b.className = "chip" + (k === K ? " active" : "");
  b.textContent = k + " lobes";
  b.addEventListener("click", () => {
    K = k;
    wrap.querySelectorAll(".chip").forEach((c) => c.classList.toggle("active", c === b));
  });
  wrap.appendChild(b);
});

bindRange("speed", (v) => { speed = v; }, (v) => v.toFixed(2) + "×");

const playBtn = document.getElementById("play");
playBtn.classList.toggle("active", playing);
playBtn.addEventListener("click", () => {
  playing = !playing;
  playBtn.textContent = playing ? "pause" : "play";
  playBtn.classList.toggle("active", playing);
});

const home = camera.position.clone();
document.getElementById("reset").addEventListener("click", () => {
  camera.position.copy(home); controls.target.set(0, 0, 0);
});

// ↑/↓ cycle corrugation lobe counts
let _variantIdx = Math.max(0, LOBES.indexOf(K));
setVariantCycler((d) => {
  _variantIdx = (_variantIdx + d + LOBES.length) % LOBES.length;
  K = LOBES[_variantIdx];
  wrap.querySelectorAll(".chip").forEach((c, k) => c.classList.toggle("active", k === _variantIdx));
  return K + " lobes";
});

// ---------- boot ----------
morph();
liftVeil();
onResize(renderer, camera);
const meter = fpsMeter(document.getElementById("fps"));
const tauEl = document.getElementById("tau");
const stateEl = document.getElementById("state");

loop((dt) => {
  meter(dt);
  if (playing) {
    tau += dir * dt * speed;
    if (tau >= 1) { tau = 1; dir = -1; }
    else if (tau <= 0) { tau = 0; dir = 1; }
    morph();
  }
  tauEl.textContent = tau.toFixed(2);
  stateEl.textContent = tau < 0.02 ? "sphere" : tau > 0.98 ? "everted" : "morphing";
  controls.update();
  renderer.render(scene, camera);
});
