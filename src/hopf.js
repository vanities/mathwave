// hopf.js — the Hopf fibration, the most beautiful object in topology.
// It maps the 3-sphere S³ onto the ordinary sphere S²; the preimage (fiber) of
// EACH point on S² is a great circle in S³. Stereographically projecting S³→R³
// turns every fiber into a circle in space, and the fibers of any two points are
// LINKED (a Hopf link). Drawing the fibers over a set of base points on S² fills
// space with nested, interlocking tori of circles. Pure 4D structure made visible.
// ↑↓ changes which family of base points we draw.
//
// Math: base point (θ,φ)∈S². Its fiber: for ψ∈[0,2π],
//   q = (cos((θ)/2)·e^{i(ψ)}, sin((θ)/2)·e^{i(ψ+φ)}) ∈ S³ ⊂ C²,
//   then stereographic project (x1,x2,x3,x4)→(x1,x2,x3)/(1−x4).

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { makeRenderer, onResize, loop, fpsMeter, liftVeil, reducedMotion, addGrid, addSun, setVariantCycler } from "./common.js";

const canvas = document.getElementById("scene");
const renderer = makeRenderer(canvas);
const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x0a0118, 0.018);

const camera = new THREE.PerspectiveCamera(52, innerWidth / innerHeight, 0.1, 600);
camera.position.set(0, 6, 16);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true; controls.dampingFactor = 0.07;
controls.autoRotate = !reducedMotion; controls.autoRotateSpeed = 0.5;
controls.minDistance = 5; controls.maxDistance = 60;

scene.add(new THREE.AmbientLight(0x3a2466, 0.9));
addGrid(scene, { size: 50, divisions: 25, y: -10 });
addSun(scene, { scale: 36, position: [0, 12, -64] });

// ---------- fiber of a base point (θ,φ), as a projected circle ----------
const FIBER_SEG = 128;
function fiberPoints(theta, phi) {
  const a = Math.cos(theta / 2), b = Math.sin(theta / 2);
  const pts = [];
  for (let k = 0; k <= FIBER_SEG; k++) {
    const psi = k / FIBER_SEG * Math.PI * 2;
    // q in S^3 ⊂ C^2 → (x1,x2,x3,x4)
    const x1 = a * Math.cos(psi);
    const x2 = a * Math.sin(psi);
    const x3 = b * Math.cos(psi + phi);
    const x4 = b * Math.sin(psi + phi);
    // stereographic S^3 → R^3 from pole x4=1
    const denom = 1 - x4 + 1e-4;
    pts.push(new THREE.Vector3(x1 / denom, x2 / denom, x3 / denom).multiplyScalar(2.2));
  }
  return pts;
}

// ---------- families of base points on S² ----------
const FAMILIES = [
  ["rings", () => { const out = []; for (let r = 1; r <= 4; r++) { const th = r/5*Math.PI; for (let j = 0; j < 8; j++) out.push([th, j/8*Math.PI*2]); } return out; }],
  ["spiral", () => { const out = []; const n = 40; for (let i = 0; i < n; i++) { const th = i/n*Math.PI; out.push([th, i*0.8]); } return out; }],
  ["single ring", () => { const out = []; for (let j = 0; j < 16; j++) out.push([Math.PI/2, j/16*Math.PI*2]); return out; }],
  ["sphere fill", () => { const out = []; for (let i = 0; i < 8; i++) for (let j = 0; j < 10; j++) out.push([(i+0.5)/8*Math.PI, j/10*Math.PI*2]); return out; }],
];
let fam = 0;

const group = new THREE.Group(); scene.add(group);
function build() {
  while (group.children.length) { const c = group.children.pop(); c.geometry.dispose(); c.material.dispose(); group.remove(c); }
  const bases = FAMILIES[fam][1]();
  for (const [th, ph] of bases) {
    const pts = fiberPoints(th, ph);
    // color by base point on S²: hue from φ, lightness from θ
    const r = 0.5 + 0.5 * Math.cos(ph), g = 0.5 + 0.5 * Math.cos(ph + 2.1), bl = 0.5 + 0.5 * Math.cos(ph + 4.2);
    const shade = 0.4 + 0.6 * (th / Math.PI);
    const geo = new THREE.BufferGeometry().setFromPoints(pts);
    group.add(new THREE.Line(geo, new THREE.LineBasicMaterial({ color: new THREE.Color(r*shade, g*shade, bl*shade), transparent: true, opacity: 0.7, blending: THREE.AdditiveBlending, depthWrite: false })));
  }
  if (nameEl) nameEl.textContent = FAMILIES[fam][0];
  if (countEl) countEl.textContent = bases.length;
}

// ---------- panel ----------
const wrap = document.getElementById("families");
const nameEl = document.getElementById("famname");
const countEl = document.getElementById("nfib");
const chips = FAMILIES.map(([label], i) => {
  const b = document.createElement("button"); b.className = "chip" + (i===0?" active":""); b.textContent = label;
  b.addEventListener("click", () => { fam = i; build(); chips.forEach((c,k)=>c.classList.toggle("active",k===i)); });
  wrap.appendChild(b); return b;
});
document.getElementById("reset").addEventListener("click", () => { camera.position.set(0,6,16); controls.target.set(0,0,0); });
setVariantCycler((d) => { fam = (fam+d+FAMILIES.length)%FAMILIES.length; build(); chips.forEach((c,k)=>c.classList.toggle("active",k===fam)); return FAMILIES[fam][0]; });

// ---------- boot ----------
build();
liftVeil();
onResize(renderer, camera);
const meter = fpsMeter(document.getElementById("fps"));
loop((dt) => { meter(dt); group.rotation.y += dt * 0.1; controls.update(); renderer.render(scene, camera); });
