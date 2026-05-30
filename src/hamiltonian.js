// hamiltonian.js — Hamiltonian dynamics in phase space.
// Position q and momentum p evolve together: dq/dt = ∂H/∂p, dp/dt = -∂H/∂q.
// We draw the energy landscape H(q,p) as a translucent surface; because energy
// is conserved, each trajectory is a level set — a glowing contour ON the
// surface — and its shadow on the floor is the classic phase portrait. A marker
// rides each orbit. ↑↓ switches systems.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import {
  makeRenderer, onResize, loop, fpsMeter, liftVeil, bindRange, ramp, reducedMotion, addGrid, addSun, setVariantCycler,
} from "./common.js";

const canvas = document.getElementById("scene");
const renderer = makeRenderer(canvas);
const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x0a0118, 0.012);

const camera = new THREE.PerspectiveCamera(52, innerWidth / innerHeight, 0.1, 600);
camera.position.set(13, 11, 15);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true; controls.dampingFactor = 0.07;
controls.autoRotate = !reducedMotion; controls.autoRotateSpeed = 0.5;
controls.minDistance = 7; controls.maxDistance = 70;
controls.target.set(0, 1.5, 0);

scene.add(new THREE.AmbientLight(0x3a2466, 0.8));
const key = new THREE.DirectionalLight(0xfff1dd, 1.0); key.position.set(8, 16, 6); scene.add(key);
const rim = new THREE.DirectionalLight(0x2be4ff, 0.8); rim.position.set(-10, 8, -8); scene.add(rim);
addGrid(scene, { size: 44, divisions: 22, y: -0.02 });
addSun(scene, { scale: 40, position: [0, 12, -60] });

// ---------- systems: H, ∂H/∂q, ∂H/∂p, domain, seed momenta ----------
const SYSTEMS = {
  pendulum: {
    about: "A pendulum. Small swings make closed loops (libration); a hard enough push sends it over the top (rotation). The S-curve between them is the separatrix.",
    H: (q, p) => p * p / 2 + (1 - Math.cos(q)),
    dq: (q, p) => p, dp: (q, p) => -Math.sin(q),
    qmax: Math.PI, pmax: 3.0, seeds: [0.5, 1.0, 1.5, 1.9, 2.1, 2.4],
  },
  harmonic: {
    about: "The simple harmonic oscillator — a mass on a spring. Every orbit is a perfect ellipse; the energy surface is a clean bowl.",
    H: (q, p) => (q * q + p * p) / 2,
    dq: (q, p) => p, dp: (q, p) => -q,
    qmax: 3.2, pmax: 3.2, seeds: [0.6, 1.2, 1.8, 2.4, 3.0],
  },
  duffing: {
    about: "A double-well (Duffing) potential — two stable valleys. Low energy traps you in one well; cross the hump and you orbit both.",
    H: (q, p) => p * p / 2 - q * q / 2 + q * q * q * q / 4,
    dq: (q, p) => p, dp: (q, p) => q - q * q * q,
    qmax: 2.2, pmax: 2.2, seeds: [0.3, 0.7, 1.0, 1.25, 1.5, 1.8],
  },
};
let sysName = "pendulum";
let sys = SYSTEMS.pendulum;

const SX = 3.2, SZ = 3.2, HY = 3.0; // world scaling for q, p, energy

// ---------- energy surface ----------
const N = 90;
const sgeo = new THREE.BufferGeometry();
const spos = new Float32Array(N * N * 3);
const scol = new Float32Array(N * N * 3);
const sidx = [];
for (let i = 0; i < N - 1; i++) for (let j = 0; j < N - 1; j++) {
  const a = i * N + j; sidx.push(a, a + 1, a + N, a + 1, a + N + 1, a + N);
}
sgeo.setIndex(sidx);
sgeo.setAttribute("position", new THREE.BufferAttribute(spos, 3));
sgeo.setAttribute("color", new THREE.BufferAttribute(scol, 3));
const surface = new THREE.Mesh(sgeo, new THREE.MeshStandardMaterial({
  vertexColors: true, roughness: 0.8, metalness: 0.05, side: THREE.DoubleSide,
  transparent: true, opacity: 0.4, depthWrite: false,
}));
scene.add(surface);
const swire = new THREE.Mesh(sgeo, new THREE.MeshBasicMaterial({ color: 0x6a3fae, wireframe: true, transparent: true, opacity: 0.12, depthWrite: false }));
scene.add(swire);

let hMax = 1;
function buildSurface() {
  const { H, qmax, pmax } = sys;
  hMax = H(qmax, pmax);
  let p = 0;
  for (let i = 0; i < N; i++) {
    const q = -qmax + (2 * qmax) * i / (N - 1);
    for (let j = 0; j < N; j++) {
      const pp = -pmax + (2 * pmax) * j / (N - 1);
      const h = H(q, pp);
      spos[p] = (q / qmax) * SX; spos[p + 1] = (h / hMax) * HY; spos[p + 2] = (pp / pmax) * SZ;
      const c = ramp(Math.min(h / hMax, 1));
      scol[p] = c[0]; scol[p + 1] = c[1]; scol[p + 2] = c[2];
      p += 3;
    }
  }
  sgeo.attributes.position.needsUpdate = true;
  sgeo.attributes.color.needsUpdate = true;
  sgeo.computeVertexNormals();
  sgeo.computeBoundingSphere();
}

// ---------- orbits (integrate each seed, draw contour + shadow + marker) ----------
const ORBIT_STEPS = 1400;
const orbitGroup = new THREE.Group();
scene.add(orbitGroup);
let orbits = [];

function rk4(q, p, dt) {
  const { dq, dp } = sys;
  const k1q = dq(q, p), k1p = dp(q, p);
  const k2q = dq(q + k1q*dt/2, p + k1p*dt/2), k2p = dp(q + k1q*dt/2, p + k1p*dt/2);
  const k3q = dq(q + k2q*dt/2, p + k2p*dt/2), k3p = dp(q + k2q*dt/2, p + k2p*dt/2);
  const k4q = dq(q + k3q*dt, p + k3p*dt), k4p = dp(q + k3q*dt, p + k3p*dt);
  return [q + (k1q + 2*k2q + 2*k3q + k4q) * dt/6, p + (k1p + 2*k2p + 2*k3p + k4p) * dt/6];
}

function toWorld(q, p, onFloor) {
  const { qmax, pmax, H } = sys;
  const qw = Math.max(-1, Math.min(1, q / qmax)) * SX;
  const pw = Math.max(-1, Math.min(1, p / pmax)) * SZ;
  const y = onFloor ? 0.04 : (H(q, p) / hMax) * HY + 0.05;
  return [qw, y, pw];
}

function buildOrbits() {
  orbits.forEach((o) => { o.curve.geometry.dispose(); o.shadow.geometry.dispose(); orbitGroup.remove(o.curve, o.shadow, o.marker); });
  orbits = [];
  sys.seeds.forEach((p0, si) => {
    const pts = [];
    let q = 0, p = p0, dt = 0.02;
    for (let s = 0; s < ORBIT_STEPS; s++) {
      pts.push([q, p]);
      [q, p] = rk4(q, p, dt);
      // pendulum: keep q in view by wrapping (rotation orbits)
      if (sys === SYSTEMS.pendulum) { if (q > Math.PI) q -= 2 * Math.PI; if (q < -Math.PI) q += 2 * Math.PI; }
    }
    const energy = sys.H(0, p0);
    const c = ramp(Math.min(energy / hMax, 1));
    const color = new THREE.Color(c[0], c[1], c[2]).lerp(new THREE.Color(1, 1, 1), 0.25);

    // contour on the surface
    const cpos = new Float32Array(pts.length * 3);
    const spos2 = new Float32Array(pts.length * 3);
    pts.forEach(([qq, pp], i) => {
      const w = toWorld(qq, pp, false); cpos.set(w, i * 3);
      const wf = toWorld(qq, pp, true); spos2.set(wf, i * 3);
    });
    const cgeo = new THREE.BufferGeometry(); cgeo.setAttribute("position", new THREE.BufferAttribute(cpos, 3));
    const curve = new THREE.Line(cgeo, new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.9, blending: THREE.AdditiveBlending, depthWrite: false }));
    const fgeo = new THREE.BufferGeometry(); fgeo.setAttribute("position", new THREE.BufferAttribute(spos2, 3));
    const shadow = new THREE.Line(fgeo, new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.5, depthWrite: false }));

    const marker = new THREE.Mesh(new THREE.SphereGeometry(0.12, 14, 12),
      new THREE.MeshBasicMaterial({ color: 0xffffff }));

    orbitGroup.add(curve, shadow, marker);
    orbits.push({ pts, curve, shadow, marker, phase: (si / sys.seeds.length) * pts.length });
  });
}

function load(name) {
  sysName = name; sys = SYSTEMS[name];
  buildSurface(); buildOrbits();
  document.getElementById("sysname").textContent = name;
  document.getElementById("about").textContent = sys.about;
  chips.forEach((c) => c.classList.toggle("active", c.dataset.k === name));
}

// ---------- panel ----------
const wrap = document.getElementById("systems");
const chips = Object.keys(SYSTEMS).map((name) => {
  const b = document.createElement("button");
  b.className = "chip"; b.dataset.k = name; b.textContent = name;
  b.addEventListener("click", () => load(name));
  wrap.appendChild(b);
  return b;
});
let speed = 1;
bindRange("speed", (v) => { speed = v; }, (v) => v.toFixed(2) + "×");
const home = camera.position.clone();
document.getElementById("reset").addEventListener("click", () => { camera.position.copy(home); controls.target.set(0, 1.5, 0); });

setVariantCycler((d) => {
  const names = Object.keys(SYSTEMS);
  const i = (names.indexOf(sysName) + d + names.length) % names.length;
  load(names[i]);
  return names[i];
});

// ---------- boot ----------
load("pendulum");
liftVeil();
onResize(renderer, camera);
const meter = fpsMeter(document.getElementById("fps"));

loop((dt) => {
  meter(dt);
  const adv = speed * 60 * dt;
  for (const o of orbits) {
    o.phase = (o.phase + adv) % o.pts.length;
    const [q, p] = o.pts[o.phase | 0];
    o.marker.position.set(...toWorld(q, p, false));
  }
  controls.update();
  renderer.render(scene, camera);
});
