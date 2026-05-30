// mandelbrot.js — the Mandelbrot set, the most famous fractal. For each complex
// point c, iterate z ← z² + c from z=0; c is IN the set if z stays bounded
// forever. Color the points that escape by HOW FAST they leave (smooth iteration
// count), and you get the infinitely detailed filigree at the boundary. Computed
// per-pixel on the GPU, with a slow auto-zoom toward a pretty seahorse-valley
// point so it endlessly falls inward. ↑↓ jumps between famous zoom targets.
//
// Ref: Mandelbrot set; smooth (continuous) escape-time coloring.

import * as THREE from "three";
import { makeRenderer, onResize, loop, fpsMeter, liftVeil, bindRange, reducedMotion, setVariantCycler } from "./common.js";

const canvas = document.getElementById("scene");
const renderer = makeRenderer(canvas);
const scene = new THREE.Scene();
const cam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

const uniforms = {
  uRes: { value: new THREE.Vector2(innerWidth, innerHeight) },
  uCenter: { value: new THREE.Vector2(-0.743643887037, 0.131825904205) }, // seahorse valley
  uZoom: { value: 1.6 },
  uIter: { value: 220 },
  uTime: { value: 0 },
};

const frag = `
  precision highp float;
  uniform vec2 uRes, uCenter; uniform float uZoom, uTime; uniform int uIter;
  vec3 pal(float t){
    // cyclic neon palette
    return 0.5 + 0.5*cos(6.2831853*(t + vec3(0.0,0.33,0.66)) + vec3(0.8,0.4,2.0));
  }
  void main(){
    vec2 uv = (gl_FragCoord.xy*2.0 - uRes)/uRes.y;
    vec2 c = uv*uZoom + uCenter;
    vec2 z = vec2(0.0); float it = 0.0;
    for(int i=0;i<1000;i++){
      if(i>=uIter) break;
      z = vec2(z.x*z.x - z.y*z.y, 2.0*z.x*z.y) + c;
      if(dot(z,z) > 256.0) break;
      it += 1.0;
    }
    vec3 col;
    if(it >= float(uIter)-0.5){ col = vec3(0.02,0.01,0.05); }   // inside = near black
    else {
      // smooth iteration count
      float sm = it - log2(log2(dot(z,z))) + 4.0;
      float t = sm*0.025 + uTime*0.03;
      col = pal(fract(t));
      col *= 0.4 + 0.6*clamp(sm/float(uIter)*3.0, 0.0, 1.0);
    }
    col = pow(col, vec3(0.85));
    gl_FragColor = vec4(col, 1.0);
  }
`;

const mat = new THREE.ShaderMaterial({ uniforms, vertexShader: `void main(){ gl_Position = vec4(position.xy,0.,1.); }`, fragmentShader: frag });
scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), mat));

// ---------- zoom targets ----------
const TARGETS = [
  ["seahorse", -0.743643887037, 0.131825904205],
  ["elephant", 0.2820, 0.0100],
  ["triple spiral", -0.088, 0.654],
  ["mini-brot", -1.25066, 0.02012],
  ["full set", -0.5, 0.0],
];
let ti = 0;
let autoZoom = !reducedMotion;
let zoom = 1.6, tzoom = 1.6;
const center = new THREE.Vector2(TARGETS[0][1], TARGETS[0][2]);
const tcenter = center.clone();

function gotoTarget(i) {
  ti = i; tcenter.set(TARGETS[i][1], TARGETS[i][2]);
  tzoom = TARGETS[i][0] === "full set" ? 1.6 : 1.2;   // reset zoom, then auto-dive
  if (nameEl) nameEl.textContent = TARGETS[i][0];
  chips.forEach((c, k) => c.classList.toggle("active", k === i));
}

// manual pan/zoom
canvas.addEventListener("wheel", (e) => { e.preventDefault(); autoZoom = false; autoBtn.classList.remove("active"); tzoom *= (1 + e.deltaY * 0.0012); tzoom = Math.max(2e-6, Math.min(2.5, tzoom)); }, { passive: false });
let drag = false, lx = 0, ly = 0;
canvas.addEventListener("pointerdown", (e) => { drag = true; lx = e.clientX; ly = e.clientY; });
canvas.addEventListener("pointermove", (e) => { if (!drag) return; autoZoom = false; autoBtn.classList.remove("active"); tcenter.x -= (e.clientX - lx)/innerHeight*2*zoom; tcenter.y += (e.clientY - ly)/innerHeight*2*zoom; lx = e.clientX; ly = e.clientY; });
canvas.addEventListener("pointerup", () => (drag = false));

// ---------- panel ----------
const wrap = document.getElementById("targets");
const nameEl = document.getElementById("tname");
const chips = TARGETS.map(([label], i) => { const b = document.createElement("button"); b.className="chip"+(i===0?" active":""); b.textContent=label; b.addEventListener("click",()=>{ gotoTarget(i); autoZoom = true; autoBtn.classList.add("active"); }); wrap.appendChild(b); return b; });
bindRange("iter", (v) => { uniforms.uIter.value = Math.round(v); }, (v) => Math.round(v));
const autoBtn = document.getElementById("auto");
autoBtn.classList.toggle("active", autoZoom);
autoBtn.addEventListener("click", () => { autoZoom = !autoZoom; autoBtn.classList.toggle("active", autoZoom); if (autoZoom) gotoTarget(ti); });
setVariantCycler((d) => { gotoTarget((ti + d + TARGETS.length) % TARGETS.length); autoZoom = true; autoBtn.classList.add("active"); return TARGETS[ti][0]; });

// ---------- boot ----------
liftVeil();
onResize(renderer, cam, (w, h) => uniforms.uRes.value.set(w, h));
const meter = fpsMeter(document.getElementById("fps"));
const zEl = document.getElementById("zoom");

loop((dt) => {
  meter(dt);
  if (autoZoom) { tzoom *= Math.pow(0.5, dt * 0.35); if (tzoom < 2e-6) gotoTarget(ti); }   // dive, then reset
  zoom += (tzoom - zoom) * 0.08;
  center.lerp(tcenter, 0.08);
  uniforms.uZoom.value = zoom; uniforms.uCenter.value.copy(center); uniforms.uTime.value += dt;
  if (zEl) zEl.textContent = (1.6 / zoom).toFixed(0) + "×";
  renderer.render(scene, cam);
});
