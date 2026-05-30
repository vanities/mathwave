// gpuflow.js — BIGGER & FASTER. A million particles, all simulated on the GPU.
// Instead of moving points on the CPU (a few thousand, then it chokes), we store
// every particle's position in a floating-point TEXTURE and advance them in a
// fragment shader — so the GPU updates all 1,048,576 of them in parallel, every
// frame, at 60fps. Each particle flows through a divergence-free CURL-NOISE field
// (so they swirl like smoke without clumping), or a vortex / Lorenz / galaxy
// field. Rendered as additive neon points. ↑↓ switches field; the count slider
// goes from 65k to 1M. This is the engine the slower rooms could graduate to.
//
// Tech: three/addons GPUComputationRenderer (ping-pong FBO GPGPU); curl of
// Ashima simplex noise.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GPUComputationRenderer } from "three/addons/misc/GPUComputationRenderer.js";
import { makeRenderer, onResize, loop, fpsMeter, liftVeil, bindRange, reducedMotion, addGrid, addSun, setVariantCycler } from "./common.js";

const canvas = document.getElementById("scene");
const renderer = makeRenderer(canvas);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5)); // fillrate: many additive points

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x0a0118, 0.006);
const camera = new THREE.PerspectiveCamera(55, innerWidth / innerHeight, 0.1, 2000);
camera.position.set(0, 30, 120);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true; controls.dampingFactor = 0.07;
controls.autoRotate = !reducedMotion; controls.autoRotateSpeed = 0.3;
controls.minDistance = 20; controls.maxDistance = 600;
controls.target.set(0, 0, 0);

addGrid(scene, { size: 320, divisions: 40, y: -70 });
addSun(scene, { scale: 90, position: [0, 40, -260] });

const BOUND = 80.0, SPAWN = 34.0;

// ---------- shared GLSL: Ashima simplex noise + curl ----------
const NOISE = `
  vec3 mod289(vec3 x){return x-floor(x*(1.0/289.0))*289.0;}
  vec4 mod289(vec4 x){return x-floor(x*(1.0/289.0))*289.0;}
  vec4 permute(vec4 x){return mod289(((x*34.0)+1.0)*x);}
  vec4 taylorInvSqrt(vec4 r){return 1.79284291400159-0.85373472095314*r;}
  float snoise(vec3 v){
    const vec2 C=vec2(1.0/6.0,1.0/3.0); const vec4 D=vec4(0.0,0.5,1.0,2.0);
    vec3 i=floor(v+dot(v,C.yyy)); vec3 x0=v-i+dot(i,C.xxx);
    vec3 g=step(x0.yzx,x0.xyz); vec3 l=1.0-g; vec3 i1=min(g.xyz,l.zxy); vec3 i2=max(g.xyz,l.zxy);
    vec3 x1=x0-i1+C.xxx; vec3 x2=x0-i2+C.yyy; vec3 x3=x0-D.yyy;
    i=mod289(i);
    vec4 p=permute(permute(permute(i.z+vec4(0.0,i1.z,i2.z,1.0))+i.y+vec4(0.0,i1.y,i2.y,1.0))+i.x+vec4(0.0,i1.x,i2.x,1.0));
    float n_=0.142857142857; vec3 ns=n_*D.wyz-D.xzx;
    vec4 j=p-49.0*floor(p*ns.z*ns.z);
    vec4 x_=floor(j*ns.z); vec4 y_=floor(j-7.0*x_);
    vec4 x=x_*ns.x+ns.yyyy; vec4 y=y_*ns.x+ns.yyyy; vec4 h=1.0-abs(x)-abs(y);
    vec4 b0=vec4(x.xy,y.xy); vec4 b1=vec4(x.zw,y.zw);
    vec4 s0=floor(b0)*2.0+1.0; vec4 s1=floor(b1)*2.0+1.0; vec4 sh=-step(h,vec4(0.0));
    vec4 a0=b0.xzyw+s0.xzyw*sh.xxyy; vec4 a1=b1.xzyw+s1.xzyw*sh.zzww;
    vec3 p0=vec3(a0.xy,h.x); vec3 p1=vec3(a0.zw,h.y); vec3 p2=vec3(a1.xy,h.z); vec3 p3=vec3(a1.zw,h.w);
    vec4 norm=taylorInvSqrt(vec4(dot(p0,p0),dot(p1,p1),dot(p2,p2),dot(p3,p3)));
    p0*=norm.x; p1*=norm.y; p2*=norm.z; p3*=norm.w;
    vec4 m=max(0.6-vec4(dot(x0,x0),dot(x1,x1),dot(x2,x2),dot(x3,x3)),0.0); m=m*m;
    return 42.0*dot(m*m,vec4(dot(p0,x0),dot(p1,x1),dot(p2,x2),dot(p3,x3)));
  }
  vec3 snoiseVec3(vec3 x){ return vec3(snoise(x), snoise(x+vec3(123.4,234.5,345.6)), snoise(x+vec3(456.7,567.8,678.9))); }
  vec3 curlNoise(vec3 p){
    const float e=0.1; vec3 dx=vec3(e,0.0,0.0), dy=vec3(0.0,e,0.0), dz=vec3(0.0,0.0,e);
    vec3 px0=snoiseVec3(p-dx), px1=snoiseVec3(p+dx);
    vec3 py0=snoiseVec3(p-dy), py1=snoiseVec3(p+dy);
    vec3 pz0=snoiseVec3(p-dz), pz1=snoiseVec3(p+dz);
    float x=(py1.z-py0.z)-(pz1.y-pz0.y);
    float y=(pz1.x-pz0.x)-(px1.z-px0.z);
    float z=(px1.y-px0.y)-(py1.x-py0.x);
    return normalize(vec3(x,y,z)/(2.0*e));
  }
`;

const POS_SHADER = `
  ${NOISE}
  uniform float uTime, uDt, uSpeed; uniform int uMode;
  float hash(vec2 p){ return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453); }
  vec3 spawn(vec2 uv){
    float a=hash(uv+0.11)*6.2831853; float r=sqrt(hash(uv+0.27))*${SPAWN.toFixed(1)};
    float h=(hash(uv+0.53)*2.0-1.0)*${SPAWN.toFixed(1)};
    return vec3(cos(a)*r, h, sin(a)*r);
  }
  vec3 field(vec3 p){
    if(uMode==1){ float r=length(p.xz)+0.001; vec3 sw=vec3(-p.z,0.0,p.x)/r; return sw*1.6+curlNoise(p*0.04)*0.6+vec3(0.0,0.4,0.0); }
    else if(uMode==2){ float s=10.0,rr=28.0,b=2.6667; return vec3(s*(p.y-p.x), p.x*(rr-p.z)-p.y, p.x*p.y-b*p.z)*0.018; }
    else if(uMode==3){ float r=length(p.xz)+0.001; vec3 sw=vec3(-p.z,0.0,p.x)/r; vec3 pull=-normalize(vec3(p.x,p.y*2.0,p.z)+1e-4)*0.35; return sw*(2.4/sqrt(r))+pull+curlNoise(p*0.05)*0.3; }
    return curlNoise(p*0.05 + uTime*0.01)*2.0;
  }
  void main(){
    vec2 uv = gl_FragCoord.xy / resolution.xy;
    vec4 t = texture2D( texturePosition, uv );
    vec3 p = t.xyz; float life = t.w;
    vec3 v = field(p) * uSpeed;
    p += v * uDt;
    life -= uDt;
    if( life <= 0.0 || length(p) > ${BOUND.toFixed(1)} ){ p = spawn(uv); life = 2.0 + hash(uv+uTime)*5.0; }
    gl_FragColor = vec4(p, life);
  }
`;

const RENDER_VS = `
  uniform sampler2D texturePosition; uniform float uSize;
  attribute vec2 reference;
  varying float vH; varying float vSpeed;
  void main(){
    vec4 p = texture2D( texturePosition, reference );
    vH = p.y; vSpeed = clamp(p.w*0.15, 0.0, 1.0);
    vec4 mv = modelViewMatrix * vec4(p.xyz, 1.0);
    gl_PointSize = uSize * (300.0 / max(1.0, -mv.z));
    gl_Position = projectionMatrix * mv;
  }
`;
const RENDER_FS = `
  precision highp float; varying float vH; varying float vSpeed;
  vec3 pal(float t){ vec3 a=vec3(0.10,0.02,0.24),b=vec3(0.45,0.10,0.62),c=vec3(1.0,0.18,0.60),d=vec3(0.16,0.82,0.96),e=vec3(0.55,1.0,0.78);
    t=clamp(t,0.0,1.0)*4.0; if(t<1.0)return mix(a,b,t); if(t<2.0)return mix(b,c,t-1.0); if(t<3.0)return mix(c,d,t-2.0); return mix(d,e,t-3.0); }
  void main(){
    vec2 c = gl_PointCoord - 0.5; float d2 = dot(c,c); if(d2>0.25) discard;
    float a = smoothstep(0.25, 0.0, d2);
    vec3 col = pal(fract((vH+${BOUND.toFixed(1)})/${(BOUND*2).toFixed(1)} + 0.1));
    gl_FragColor = vec4(col * (0.5 + vSpeed), a*0.55);
  }
`;

// ---------- build the GPGPU sim at a given square size ----------
let gpu, posVar, points, SIZE = 1024, speed = 1, mode = 0, gpuOK = false, lastErr = "";
function buildSim(size) {
  SIZE = size;
  if (points) { scene.remove(points); points.geometry.dispose(); points.material.dispose(); }
  gpu = new GPUComputationRenderer(SIZE, SIZE, renderer);
  const tex = gpu.createTexture();
  const d = tex.image.data;
  for (let i = 0; i < SIZE * SIZE; i++) {
    const a = Math.random() * Math.PI * 2, r = Math.sqrt(Math.random()) * SPAWN, h = (Math.random() * 2 - 1) * SPAWN;
    d[i*4] = Math.cos(a) * r; d[i*4+1] = h; d[i*4+2] = Math.sin(a) * r; d[i*4+3] = Math.random() * 6;
  }
  posVar = gpu.addVariable("texturePosition", POS_SHADER, tex);
  gpu.setVariableDependencies(posVar, [posVar]);
  Object.assign(posVar.material.uniforms, { uTime: { value: 0 }, uDt: { value: 0.016 }, uSpeed: { value: speed }, uMode: { value: mode } });
  posVar.wrapS = THREE.RepeatWrapping; posVar.wrapT = THREE.RepeatWrapping;
  const err = gpu.init();
  gpuOK = !err; if (err) { lastErr = String(err); console.error("GPGPU init:", err); }

  // render geometry: one point per texel, carrying a 'reference' uv
  const n = SIZE * SIZE;
  const g = new THREE.BufferGeometry();
  const pos = new Float32Array(n * 3);           // dummy (count source)
  const ref = new Float32Array(n * 2);
  for (let i = 0; i < n; i++) { ref[i*2] = ((i % SIZE) + 0.5) / SIZE; ref[i*2+1] = ((Math.floor(i / SIZE)) + 0.5) / SIZE; }
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  g.setAttribute("reference", new THREE.BufferAttribute(ref, 2));
  g.setDrawRange(0, n);
  const mat = new THREE.ShaderMaterial({
    uniforms: { texturePosition: { value: null }, uSize: { value: 1.0 } },
    vertexShader: RENDER_VS, fragmentShader: RENDER_FS,
    transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, depthTest: false,
  });
  points = new THREE.Points(g, mat);
  points.frustumCulled = false;
  scene.add(points);
  if (countEl) countEl.textContent = n.toLocaleString();
}

// ---------- panel ----------
const FIELDS = [["curl smoke", 0], ["vortex", 1], ["lorenz", 2], ["galaxy", 3]];
const wrap = document.getElementById("fields");
const nameEl = document.getElementById("fieldname");
const countEl = document.getElementById("count");
let fi = 0;
const chips = FIELDS.map(([label, m], i) => { const b = document.createElement("button"); b.className="chip"+(i===0?" active":""); b.textContent=label; b.addEventListener("click",()=>{ fi=i; mode=m; posVar.material.uniforms.uMode.value=m; nameEl.textContent=label; chips.forEach((c,k)=>c.classList.toggle("active",k===i)); }); wrap.appendChild(b); return b; });
bindRange("speed", (v)=>{ speed=v; if(posVar) posVar.material.uniforms.uSpeed.value=v; }, (v)=>v.toFixed(2)+"×");
const SIZES = [256, 512, 768, 1024];
bindRange("size", (v)=>{ const s=SIZES[Math.round(v)]; if(s!==SIZE) buildSim(s); }, (v)=>{ const s=SIZES[Math.round(v)]; return (s*s>=1e6)?((s*s/1e6).toFixed(1)+"M"):((s*s/1000)|0)+"k"; });
bindRange("psize", (v)=>{ if(points) points.material.uniforms.uSize.value=v; }, (v)=>v.toFixed(1));
setVariantCycler((d)=>{ fi=(fi+d+FIELDS.length)%FIELDS.length; mode=FIELDS[fi][1]; posVar.material.uniforms.uMode.value=mode; nameEl.textContent=FIELDS[fi][0]; chips.forEach((c,k)=>c.classList.toggle("active",k===fi)); return FIELDS[fi][0]; });

// ---------- boot ----------
buildSim(1024);
liftVeil();
onResize(renderer, camera);
const meter = fpsMeter(document.getElementById("fps"));

window.__diag = () => JSON.stringify({ particles: SIZE*SIZE, gpuOK, mode, err: lastErr || "none" });

let t = 0;
loop((dt) => {
  meter(dt);
  t += dt;
  if (gpu) {
    posVar.material.uniforms.uTime.value = t;
    posVar.material.uniforms.uDt.value = Math.min(dt, 0.033);
    gpu.compute();
    points.material.uniforms.texturePosition.value = gpu.getCurrentRenderTarget(posVar).texture;
  }
  controls.update();
  renderer.render(scene, camera);
});
