// attractorsong.js — CHAOS YOU CAN HEAR. A strange attractor isn't just a shape,
// it's a trajectory through time — i.e. a melody. We integrate the attractor (RK4),
// fly a playhead along it, and map the head's coordinates to SOUND:
//   x → pitch  (quantized to a pentatonic scale, so chaos sounds musical not random)
//   y → filter cutoff (brightness)
//   z → stereo pan (left ↔ right)
// The actual audio output is drawn back into the 3D scene as a neon OSCILLOSCOPE,
// so you literally watch the waveform the chaos is producing. Synesthesia, real.
// ↑↓ switches attractors — each is a different song. S mutes.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { makeRenderer, onResize, loop, fpsMeter, liftVeil, bindRange, ramp, reducedMotion, addGrid, addSun, setVariantCycler } from "./common.js";
import { installAudioUI, makeVoice, audioOn, getAnalyser, pentatonic } from "./audio.js";

const canvas = document.getElementById("scene");
const renderer = makeRenderer(canvas);
const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x0a0118, 0.012);

const camera = new THREE.PerspectiveCamera(55, innerWidth / innerHeight, 0.1, 1000);
camera.position.set(0, 8, 34);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true; controls.dampingFactor = 0.07;
controls.autoRotate = !reducedMotion; controls.autoRotateSpeed = 0.4;
controls.minDistance = 10; controls.maxDistance = 140;
controls.target.set(0, 0, 0);

scene.add(new THREE.AmbientLight(0x2a1860, 0.8));
addGrid(scene, { size: 90, divisions: 45, y: -16 });
addSun(scene, { scale: 44, position: [0, 16, -90] });

// ---------- attractors ----------
const SYSTEMS = {
  lorenz: { dt: 0.006, start: [0.1, 0, 0], f: (p, k) => { const s=10,r=28,b=8/3; k[0]=s*(p[1]-p[0]); k[1]=p[0]*(r-p[2])-p[1]; k[2]=p[0]*p[1]-b*p[2]; } },
  aizawa: { dt: 0.01, start: [0.1, 0, 0], f: (p, k) => { const a=.95,b=.7,c=.6,d=3.5,e=.25,ff=.1; k[0]=(p[2]-b)*p[0]-d*p[1]; k[1]=d*p[0]+(p[2]-b)*p[1]; k[2]=c+a*p[2]-p[2]**3/3-(p[0]*p[0]+p[1]*p[1])*(1+e*p[2])+ff*p[2]*p[0]**3; } },
  thomas: { dt: 0.02, start: [1.1, 1.1, -0.5], f: (p, k) => { const b=.208186; k[0]=Math.sin(p[1])-b*p[0]; k[1]=Math.sin(p[2])-b*p[1]; k[2]=Math.sin(p[0])-b*p[2]; } },
  dadras: { dt: 0.005, start: [1.1, 2.1, -2], f: (p, k) => { const a=3,b=2.7,c=1.7,d=2,e=9; k[0]=p[1]-a*p[0]+b*p[1]*p[2]; k[1]=c*p[1]-p[0]*p[2]+p[2]; k[2]=d*p[0]*p[1]-e*p[2]; } },
};
const NAMES = Object.keys(SYSTEMS);
let sysIdx = 0;

const N = 60000;
const raw = new Float32Array(N * 3);     // raw coords (for sound)
const disp = new Float32Array(N * 3);    // centered+scaled (for display)
let bounds = { mn: [0,0,0], mx: [1,1,1] };

const geo = new THREE.BufferGeometry();
const dcol = new Float32Array(N * 3);
geo.setAttribute("position", new THREE.BufferAttribute(disp, 3));
geo.setAttribute("color", new THREE.BufferAttribute(dcol, 3));
const ribbon = new THREE.Line(geo, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.55, blending: THREE.AdditiveBlending, depthWrite: false }));
scene.add(ribbon);

const head = new THREE.Mesh(new THREE.SphereGeometry(0.5, 18, 14), new THREE.MeshBasicMaterial({ color: 0xffffff }));
scene.add(head);

const k1=[0,0,0],k2=[0,0,0],k3=[0,0,0],k4=[0,0,0],tmp=[0,0,0];
function integrate() {
  const s = SYSTEMS[NAMES[sysIdx]], dt = s.dt, f = s.f;
  let p = s.start.slice();
  const mn = [1e9,1e9,1e9], mx = [-1e9,-1e9,-1e9];
  for (let i = 0; i < N; i++) {
    raw[i*3]=p[0]; raw[i*3+1]=p[1]; raw[i*3+2]=p[2];
    for (let d=0; d<3; d++){ if(p[d]<mn[d])mn[d]=p[d]; if(p[d]>mx[d])mx[d]=p[d]; }
    f(p,k1);
    for(let d=0;d<3;d++)tmp[d]=p[d]+k1[d]*dt/2; f(tmp,k2);
    for(let d=0;d<3;d++)tmp[d]=p[d]+k2[d]*dt/2; f(tmp,k3);
    for(let d=0;d<3;d++)tmp[d]=p[d]+k3[d]*dt; f(tmp,k4);
    for(let d=0;d<3;d++)p[d]+=(k1[d]+2*k2[d]+2*k3[d]+k4[d])*dt/6;
  }
  bounds = { mn, mx };
  // center + scale for display
  const cx=(mn[0]+mx[0])/2, cy=(mn[1]+mx[1])/2, cz=(mn[2]+mx[2])/2;
  const span = Math.max(mx[0]-mn[0], mx[1]-mn[1], mx[2]-mn[2]) || 1;
  const sc = 24/span;
  for (let i=0;i<N;i++){
    disp[i*3]=(raw[i*3]-cx)*sc; disp[i*3+1]=(raw[i*3+1]-cy)*sc; disp[i*3+2]=(raw[i*3+2]-cz)*sc;
    const c=ramp(i/N); dcol[i*3]=c[0]; dcol[i*3+1]=c[1]; dcol[i*3+2]=c[2];
  }
  geo.attributes.position.needsUpdate = true;
  geo.attributes.color.needsUpdate = true;
  geo.computeBoundingSphere();
}

// ---------- oscilloscope drawn in 3D (so it's in recordings) ----------
const SCOPE = 256;
const scopeGeo = new THREE.BufferGeometry();
const scopePos = new Float32Array(SCOPE * 3);
for (let i=0;i<SCOPE;i++){ scopePos[i*3]=(i/(SCOPE-1)-0.5)*30; scopePos[i*3+1]=16; scopePos[i*3+2]=0; }
scopeGeo.setAttribute("position", new THREE.BufferAttribute(scopePos, 3).setUsage(THREE.DynamicDrawUsage));
const scope = new THREE.Line(scopeGeo, new THREE.LineBasicMaterial({ color: 0x62ffb3, transparent: true, opacity: 0.9, blending: THREE.AdditiveBlending }));
scene.add(scope);
let scopeBuf = null;

// ---------- sound ----------
let voice = null, sub = null;
const SCALE = pentatonic(110, 4);
function clamp01(v){ return Math.max(0, Math.min(1, v)); }
function noteFromX(rx){ const t=clamp01((rx-bounds.mn[0])/((bounds.mx[0]-bounds.mn[0])||1)); return SCALE[Math.min(SCALE.length-1, Math.floor(t*SCALE.length))]; }

function load(i) {
  sysIdx = (i + NAMES.length) % NAMES.length;
  integrate();
  if (nameEl) nameEl.textContent = NAMES[sysIdx];
  chips.forEach((c,k)=>c.classList.toggle("active", k===sysIdx));
}

// ---------- panel ----------
const wrap = document.getElementById("systems");
const nameEl = document.getElementById("sysname");
const chips = NAMES.map((nm,i)=>{ const b=document.createElement("button"); b.className="chip"+(i===0?" active":""); b.textContent=nm; b.addEventListener("click",()=>load(i)); wrap.appendChild(b); return b; });
let speed = 1;
bindRange("speed", (v)=>{ speed=v; }, (v)=>v.toFixed(2)+"×");
setVariantCycler((d)=>{ load(sysIdx+d); return NAMES[sysIdx]; });

// ---------- boot ----------
installAudioUI();
load(0);
liftVeil();
onResize(renderer, camera);
const meter = fpsMeter(document.getElementById("fps"));
const hzEl = document.getElementById("hz");

let idx = 0;
loop((dt) => {
  meter(dt);
  // advance the playhead along the trajectory
  idx = (idx + Math.max(1, Math.round(speed * 90 * dt))) % N;
  head.position.set(disp[idx*3], disp[idx*3+1], disp[idx*3+2]);
  // drive sound from raw coords
  const rx=raw[idx*3], ry=raw[idx*3+1], rz=raw[idx*3+2];
  if (audioOn()) {
    if (!voice) { voice = makeVoice({ type:"sawtooth" }); sub = makeVoice({ type:"sine" }); }
    const f = noteFromX(rx);
    voice.setFreq(f); sub.setFreq(f/2);
    voice.setCutoff(300 + clamp01((ry-bounds.mn[1])/((bounds.mx[1]-bounds.mn[1])||1))*4000);
    const pan = clamp01((rz-bounds.mn[2])/((bounds.mx[2]-bounds.mn[2])||1))*2-1;
    voice.setPan(pan); sub.setPan(pan*0.5);
    voice.setGain(0.22); sub.setGain(0.16);
    if (hzEl) hzEl.textContent = f.toFixed(0) + " Hz";
    // oscilloscope from real output
    const an = getAnalyser();
    if (an) {
      if (!scopeBuf || scopeBuf.length !== an.fftSize) scopeBuf = new Uint8Array(an.fftSize);
      an.getByteTimeDomainData(scopeBuf);
      const step = (scopeBuf.length / SCOPE) | 0;
      for (let i=0;i<SCOPE;i++){ scopePos[i*3+1] = 16 + ((scopeBuf[i*step]-128)/128)*4; }
      scopeGeo.attributes.position.needsUpdate = true;
    }
  } else {
    if (voice) { voice.setGain(0); sub.setGain(0); }
    // flat scope when muted
    for (let i=0;i<SCOPE;i++) scopePos[i*3+1]=16;
    scopeGeo.attributes.position.needsUpdate = true;
    if (hzEl) hzEl.textContent = "— muted —";
  }
  controls.update();
  renderer.render(scene, camera);
});
