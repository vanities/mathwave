// physarum.js — Physarum (slime mold) transport networks, after Jeff Jones (2010).
// Thousands of agents crawl a trail field. Each agent: SENSES the trail at three
// points ahead (front-left / front / front-right), ROTATES toward the strongest,
// STEPS forward, and DEPOSITS trail at its new spot. The trail field then
// DIFFUSES (blur) and DECAYS each frame. From these purely local rules, the
// colony self-organizes into the branching, vein-like networks real slime mold
// builds to solve mazes. GPU ping-pong: agents on the CPU, field on textures.
//
// Ref: Jones, "Characteristics of pattern formation and evolution in
//   approximations of Physarum transport networks" (Artificial Life, 2010).

import * as THREE from "three";
import { makeRenderer, onResize, loop, fpsMeter, liftVeil, bindRange, reducedMotion, setVariantCycler } from "./common.js";

const canvas = document.getElementById("scene");
const renderer = makeRenderer(canvas);
const scene = new THREE.Scene();
const cam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
const quad = new THREE.PlaneGeometry(2, 2);
const VS = `varying vec2 vUv; void main(){ vUv = uv; gl_Position = vec4(position.xy,0.,1.); }`;

// ---------- trail field (ping-pong) ----------
let SW = 0, SH = 0;
const MAXDIM = 700;
function simSize() { const ar = innerWidth / innerHeight; if (ar >= 1) { SW = MAXDIM; SH = Math.round(MAXDIM/ar); } else { SH = MAXDIM; SW = Math.round(MAXDIM*ar); } }
simSize();
const RT = () => new THREE.WebGLRenderTarget(SW, SH, { type: THREE.HalfFloatType, format: THREE.RGBAFormat, minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter, wrapS: THREE.RepeatWrapping, wrapT: THREE.RepeatWrapping, depthBuffer: false });
let trailA = RT(), trailB = RT();

// diffuse + decay shader
const decayU = { uTex: { value: null }, uTexel: { value: new THREE.Vector2(1/SW, 1/SH) }, uDecay: { value: 0.96 } };
const decayScene = new THREE.Scene();
decayScene.add(new THREE.Mesh(quad, new THREE.ShaderMaterial({ uniforms: decayU, vertexShader: VS, fragmentShader: `
  precision highp float; varying vec2 vUv; uniform sampler2D uTex; uniform vec2 uTexel; uniform float uDecay;
  void main(){
    vec3 s = vec3(0.0);
    for(int dx=-1;dx<=1;dx++) for(int dy=-1;dy<=1;dy++) s += texture2D(uTex, vUv + vec2(float(dx),float(dy))*uTexel).rgb;
    s /= 9.0;                       // 3x3 mean blur = diffusion
    gl_FragColor = vec4(s * uDecay, 1.0);   // multiplicative decay
  }` })));

// deposit shader (additive points)
const depU = { };
const depScene = new THREE.Scene();
const AGENTS = 200000;
const agentGeo = new THREE.BufferGeometry();
const agentPosAttr = new THREE.BufferAttribute(new Float32Array(AGENTS * 3), 3).setUsage(THREE.DynamicDrawUsage);
agentGeo.setAttribute("position", agentPosAttr);
const depPoints = new THREE.Points(agentGeo, new THREE.PointsMaterial({ color: new THREE.Color(0.04,0.05,0.06), size: 1, sizeAttenuation: false, blending: THREE.AdditiveBlending, depthWrite: false, transparent: true }));
depScene.add(depPoints);

// display shader (color the trail with the neon ramp)
const dispU = { uTex: { value: null }, uTime: { value: 0 } };
const dispScene = new THREE.Scene();
dispScene.add(new THREE.Mesh(quad, new THREE.ShaderMaterial({ uniforms: dispU, vertexShader: VS, fragmentShader: `
  precision highp float; varying vec2 vUv; uniform sampler2D uTex; uniform float uTime;
  vec3 pal(float t){ vec3 a=vec3(0.05,0.01,0.15),b=vec3(0.55,0.10,0.62),c=vec3(1.0,0.18,0.60),d=vec3(0.16,0.82,0.96),e=vec3(0.6,1.0,0.8);
    t=clamp(t,0.,1.)*4.; if(t<1.)return mix(a,b,t); if(t<2.)return mix(b,c,t-1.); if(t<3.)return mix(c,d,t-2.); return mix(d,e,t-3.); }
  void main(){ float v = texture2D(uTex, vUv).r; gl_FragColor = vec4(pal(pow(clamp(v*1.4,0.,1.),0.7)), 1.0); }` })));

// ---------- agents (CPU) ----------
let ax = new Float32Array(AGENTS), ay = new Float32Array(AGENTS), ah = new Float32Array(AGENTS);
// params (Jones-style)
let sensorDist = 9, sensorAng = 0.4, turnAng = 0.5, stepSize = 1.2;
let trailField = new Float32Array(SW * SH);   // CPU shadow of the trail for sensing

function spawn(kind) {
  for (let i = 0; i < AGENTS; i++) {
    if (kind === "disk") { const r = Math.sqrt(Math.random()) * Math.min(SW,SH) * 0.35, a = Math.random()*Math.PI*2; ax[i] = SW/2 + Math.cos(a)*r; ay[i] = SH/2 + Math.sin(a)*r; ah[i] = a + Math.PI; }
    else if (kind === "ring") { const a = Math.random()*Math.PI*2, r = Math.min(SW,SH)*0.4; ax[i] = SW/2+Math.cos(a)*r; ay[i] = SH/2+Math.sin(a)*r; ah[i] = Math.random()*Math.PI*2; }
    else { ax[i] = Math.random()*SW; ay[i] = Math.random()*SH; ah[i] = Math.random()*Math.PI*2; }
  }
}

// read trail (CPU shadow) with wrap
function sense(x, y) {
  const xi = ((x | 0) % SW + SW) % SW, yi = ((y | 0) % SH + SH) % SH;
  return trailField[yi * SW + xi];
}
function stepAgents() {
  for (let i = 0; i < AGENTS; i++) {
    const h = ah[i], x = ax[i], y = ay[i];
    const fx = x + Math.cos(h)*sensorDist, fy = y + Math.sin(h)*sensorDist;
    const lx = x + Math.cos(h-sensorAng)*sensorDist, ly = y + Math.sin(h-sensorAng)*sensorDist;
    const rx = x + Math.cos(h+sensorAng)*sensorDist, ry = y + Math.sin(h+sensorAng)*sensorDist;
    const F = sense(fx,fy), L = sense(lx,ly), Rr = sense(rx,ry);
    let nh = h;
    if (F > L && F > Rr) {}                                  // keep heading
    else if (F < L && F < Rr) nh += (Math.random() < 0.5 ? -1 : 1) * turnAng;  // both better → random
    else if (L < Rr) nh += turnAng;                          // right stronger
    else if (Rr < L) nh -= turnAng;                          // left stronger
    let nx = x + Math.cos(nh)*stepSize, ny = y + Math.sin(nh)*stepSize;
    nx = (nx % SW + SW) % SW; ny = (ny % SH + SH) % SH;
    ax[i] = nx; ay[i] = ny; ah[i] = nh;
    // write to position buffer in clip space for the deposit pass
    agentPosAttr.array[i*3]   = (nx / SW) * 2 - 1;
    agentPosAttr.array[i*3+1] = (ny / SH) * 2 - 1;
    agentPosAttr.array[i*3+2] = 0;
  }
  agentPosAttr.needsUpdate = true;
}

// read the GPU trail back to the CPU shadow (so agents can sense it)
const readBuf = new Float32Array(SW * SH * 4);
function syncTrailToCPU(rt) {
  renderer.readRenderTargetPixels(rt, 0, 0, SW, SH, readBuf);
  for (let i = 0; i < SW * SH; i++) trailField[i] = readBuf[i*4];
}

function reallocate() {
  simSize();
  trailA.setSize(SW, SH); trailB.setSize(SW, SH);
  decayU.uTexel.value.set(1/SW, 1/SH);
  trailField = new Float32Array(SW * SH);
  // clear targets
  for (const rt of [trailA, trailB]) { renderer.setRenderTarget(rt); renderer.setClearColor(0x000000, 1); renderer.clear(); }
  renderer.setRenderTarget(null);
  spawn(seedMode);
}

// ---------- panel ----------
const wrap = document.getElementById("seeds");
let seedMode = "disk";
const SEEDS = [["disk","disk"],["ring","ring"],["random","random"]];
const chips = SEEDS.map(([label,k],i) => { const b = document.createElement("button"); b.className="chip"+(i===0?" active":""); b.textContent=label; b.addEventListener("click",()=>{ seedMode=k; spawn(k); chips.forEach((c,j)=>c.classList.toggle("active",j===i)); }); wrap.appendChild(b); return b; });
bindRange("sensor", (v) => { sensorAng = v; }, (v) => v.toFixed(2));
bindRange("turn", (v) => { turnAng = v; }, (v) => v.toFixed(2));
bindRange("decay", (v) => { decayU.uDecay.value = v; }, (v) => v.toFixed(3));
setVariantCycler((d) => { const i = (SEEDS.findIndex(s=>s[1]===seedMode)+d+SEEDS.length)%SEEDS.length; seedMode=SEEDS[i][1]; spawn(seedMode); chips.forEach((c,j)=>c.classList.toggle("active",j===i)); return SEEDS[i][0]; });

// ---------- boot ----------
spawn("disk");
for (const rt of [trailA, trailB]) { renderer.setRenderTarget(rt); renderer.setClearColor(0x000000,1); renderer.clear(); }
renderer.setRenderTarget(null);
liftVeil();
window.addEventListener("resize", reallocate);
const meter = fpsMeter(document.getElementById("fps"));

let frame = 0;
loop((dt) => {
  meter(dt);
  // 1) deposit agents onto trailA (additive)
  renderer.autoClear = false;
  renderer.setRenderTarget(trailA);
  renderer.render(depScene, cam);
  // 2) diffuse+decay trailA → trailB
  decayU.uTex.value = trailA.texture;
  renderer.setRenderTarget(trailB);
  renderer.render(decayScene, cam);
  let t = trailA; trailA = trailB; trailB = t;
  renderer.autoClear = true;
  // 3) sync to CPU every other frame (readback is the bottleneck) and step agents
  if ((frame++ & 1) === 0) syncTrailToCPU(trailA);
  stepAgents();
  // 4) display
  dispU.uTex.value = trailA.texture;
  renderer.setRenderTarget(null);
  renderer.render(dispScene, cam);
});
