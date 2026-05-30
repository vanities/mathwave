// newton.js — Newton's fractal. Newton's method finds a root of f(z) by iterating
//   z ← z − f(z)/f'(z).
// Color each starting point in the complex plane by WHICH root it converges to,
// and shade by HOW MANY steps it took. The basins of attraction interlock in an
// infinitely intricate fractal boundary — between any two basins lies a speck of
// the third, forever. Computed per-pixel on the GPU for f(z)=zⁿ−1. ↑↓ changes n.
//
// Ref: Newton's method on complex polynomials; basins of attraction.

import * as THREE from "three";
import { makeRenderer, onResize, loop, fpsMeter, liftVeil, bindRange, reducedMotion, setVariantCycler } from "./common.js";

const canvas = document.getElementById("scene");
const renderer = makeRenderer(canvas);
const scene = new THREE.Scene();
const cam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

const uniforms = {
  uRes: { value: new THREE.Vector2(innerWidth, innerHeight) },
  uTime: { value: 0 },
  uN: { value: 3 },
  uZoom: { value: 1.4 },
  uCenter: { value: new THREE.Vector2(0, 0) },
  uIter: { value: 40 },
};

const frag = `
  precision highp float;
  uniform vec2 uRes; uniform float uTime, uZoom; uniform vec2 uCenter; uniform int uN, uIter;

  vec2 cmul(vec2 a, vec2 b){ return vec2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x); }
  vec2 cdiv(vec2 a, vec2 b){ float d = dot(b,b) + 1e-12; return vec2(a.x*b.x + a.y*b.y, a.y*b.x - a.x*b.y)/d; }
  vec2 cpow(vec2 z, int n){ vec2 r = vec2(1.0,0.0); for(int i=0;i<12;i++){ if(i>=n) break; r = cmul(r, z); } return r; }

  vec3 pal(float t){
    vec3 a=vec3(1.0,0.18,0.60), b=vec3(0.16,0.82,0.96), c=vec3(0.6,1.0,0.78), d=vec3(0.70,0.45,1.0),
         e=vec3(1.0,0.75,0.3), f=vec3(0.95,0.95,0.95);
    if(t<0.18) return a; if(t<0.36) return b; if(t<0.54) return c; if(t<0.72) return d; if(t<0.9) return e; return f;
  }

  void main(){
    vec2 uv = (gl_FragCoord.xy * 2.0 - uRes) / uRes.y;
    vec2 z = uv * uZoom + uCenter;
    int conv = -1; int steps = 0;
    for(int i=0;i<80;i++){
      if(i>=uIter) break;
      // f = z^n - 1 ; f' = n z^(n-1)
      vec2 zn = cpow(z, uN);
      vec2 f = zn - vec2(1.0, 0.0);
      vec2 df = vec2(float(uN),0.0);
      df = cmul(df, cpow(z, uN-1));
      z = z - cdiv(f, df);
      steps = i;
      // close enough to a root of unity? roots are e^{2πik/n}
      // detect convergence by small |f|
      if(dot(f,f) < 1e-6){ break; }
    }
    // classify by angle of final z (≈ a root of unity)
    float ang = atan(z.y, z.x);                 // -π..π
    float root = floor((ang/ (2.0*3.14159265) + 0.5) * float(uN) + 0.5);
    float hue = root / float(uN);
    vec3 base = pal(fract(hue + 0.02));
    float shade = 1.0 - float(steps) / float(uIter);   // faster convergence brighter
    vec3 colr = base * (0.25 + 0.85 * shade);
    colr = pow(colr, vec3(0.85));
    gl_FragColor = vec4(colr, 1.0);
  }
`;

const mat = new THREE.ShaderMaterial({ uniforms, vertexShader: `void main(){ gl_Position = vec4(position.xy,0.,1.); }`, fragmentShader: frag });
scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), mat));

// ---------- pan/zoom ----------
let zoom = 1.4, tzoom = 1.4; const center = new THREE.Vector2(0, 0);
canvas.addEventListener("wheel", (e) => { e.preventDefault(); tzoom *= (1 + e.deltaY * 0.0015); tzoom = Math.max(0.02, Math.min(4, tzoom)); }, { passive: false });
let drag = false, lx = 0, ly = 0;
canvas.addEventListener("pointerdown", (e) => { drag = true; lx = e.clientX; ly = e.clientY; });
canvas.addEventListener("pointermove", (e) => { if (!drag) return; center.x -= (e.clientX - lx) / innerHeight * 2 * zoom; center.y += (e.clientY - ly) / innerHeight * 2 * zoom; lx = e.clientX; ly = e.clientY; });
canvas.addEventListener("pointerup", () => (drag = false));

// ---------- panel ----------
bindRange("iter", (v) => { uniforms.uIter.value = Math.round(v); }, (v) => Math.round(v));
const nEl = document.getElementById("nval");
const wrap = document.getElementById("degrees");
const DEGS = [3,4,5,6,7,8];
let di = 0;
const chips = DEGS.map((nn, i) => { const b = document.createElement("button"); b.className="chip"+(i===0?" active":""); b.textContent="z^"+nn+"−1"; b.addEventListener("click",()=>{ di=i; uniforms.uN.value=nn; nEl.textContent=nn; chips.forEach((c,k)=>c.classList.toggle("active",k===i)); }); wrap.appendChild(b); return b; });
document.getElementById("reset").addEventListener("click", () => { tzoom = 1.4; center.set(0,0); });
setVariantCycler((d) => { di = (di+d+DEGS.length)%DEGS.length; uniforms.uN.value = DEGS[di]; nEl.textContent = DEGS[di]; chips.forEach((c,k)=>c.classList.toggle("active",k===di)); return "z^"+DEGS[di]+"−1"; });

// ---------- boot ----------
liftVeil();
onResize(renderer, cam, (w, h) => uniforms.uRes.value.set(w, h));
const meter = fpsMeter(document.getElementById("fps"));
loop((dt) => {
  meter(dt);
  zoom += (tzoom - zoom) * 0.1; uniforms.uZoom.value = zoom; uniforms.uCenter.value.copy(center);
  uniforms.uTime.value += dt;
  renderer.render(scene, cam);
});
