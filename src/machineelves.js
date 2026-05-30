// machineelves.js — "hyperspace," after the DMT machine-elf phenomenology
// (Strassman, McKenna; trip-report corpora): a rush through a kaleidoscopic
// TUNNEL into a vaulted CHAMBER filled with fast, self-transforming, jewelled
// fractal ENTITIES that weave and beckon. We build it honestly from real math —
// no hand-waving:
//   • RAYMARCHED scene: a signed-distance tunnel you fly down forever.
//   • KALEIDOSCOPE: polar-angle mirror symmetry (mod into N wedges) → the
//     constantly-folding mandala walls.
//   • ENTITIES: a kaleidoscopic-IFS (kIFS) fold-and-scale fractal SDF — the
//     classic recipe for those impossible self-similar "beings."
//   • Chrome + jewel palette, fast morph, heavy bloom-ish glow via step glow.
// ↑↓ changes the symmetry order (how many-fold the elves are). Heavy shader —
// DPR capped. NOTE: intense strobing visuals; honors prefers-reduced-motion.

import * as THREE from "three";
import { makeRenderer, onResize, loop, fpsMeter, liftVeil, bindRange, reducedMotion, setVariantCycler } from "./common.js";

const canvas = document.getElementById("scene");
const renderer = makeRenderer(canvas);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.25)); // raymarch is heavy

const scene = new THREE.Scene();
const cam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

const uniforms = {
  uRes: { value: new THREE.Vector2(innerWidth, innerHeight) },
  uTime: { value: 0 },
  uSym: { value: 6.0 },        // kaleidoscope fold count
  uSpeed: { value: 1.0 },
  uMorph: { value: 1.0 },
  uSteps: { value: 90 },
};

const frag = `
  precision highp float;
  uniform vec2 uRes; uniform float uTime, uSym, uSpeed, uMorph; uniform int uSteps;
  #define PI 3.14159265

  mat2 rot(float a){ float c=cos(a), s=sin(a); return mat2(c,-s,s,c); }

  // jewel/chrome palette
  vec3 pal(float t){
    return 0.5 + 0.5*cos(2.0*PI*(t + vec3(0.0,0.33,0.67)) + vec3(0.9, 2.1, 4.2));
  }

  // kaleidoscopic-IFS "entity": fold space against rotating planes + scale, the
  // standard recipe for self-similar fractal creatures.
  float entity(vec3 p, float t){
    float scale = 1.0;
    float orbit = 1e9;
    for(int i=0;i<7;i++){
      p = abs(p);                                   // fold into the positive octant
      if(p.x < p.y) p.xy = p.yx;
      if(p.x < p.z) p.xz = p.zx;
      if(p.y < p.z) p.yz = p.zy;
      p.xy *= rot(0.5 + 0.25*sin(t*0.7));           // writhing rotation (the "weaving")
      p.yz *= rot(0.3 + 0.20*cos(t*0.5));
      p = p*1.9 - vec3(1.3, 0.9, 1.1)*(1.0+0.15*sin(t*0.6));  // scale + translate (IFS)
      scale *= 1.9;
      orbit = min(orbit, length(p));
    }
    return (length(p) - 1.2)/scale - 0.001*orbit*uMorph;
  }

  // the kaleidoscope TUNNEL: mirror the polar angle into uSym wedges, carve a pipe
  vec2 fold(vec2 q){
    float a = atan(q.y, q.x);
    float r = length(q);
    float seg = PI/uSym;
    a = mod(a, 2.0*seg); a = abs(a - seg);          // mirror symmetry
    return vec2(cos(a), sin(a))*r;
  }

  float map(vec3 p, float t, out float kind){
    // travel down +z; swirl the cross-section over time
    vec3 q = p;
    q.xy *= rot(q.z*0.06 + t*0.2);
    q.xy = fold(q.xy);
    float tunnel = 3.4 - length(q.xy);              // pipe radius 3.4 (walls)
    // entities live in the wedge, repeating along z so you keep meeting them
    vec3 e = q;
    e.z = mod(e.z + t*4.0, 5.0) - 2.5;              // closer spacing → meet them constantly
    e.xy -= vec2(2.0, 0.0);                          // sit them in the wedge
    float ent = entity(e, t);
    kind = ent < tunnel ? 1.0 : 0.0;
    return min(tunnel, ent);
  }

  void main(){
    vec2 uv = (gl_FragCoord.xy*2.0 - uRes)/uRes.y;
    float t = uTime*uSpeed;
    vec3 ro = vec3(0.0, 0.0, t*4.0);               // fly forward forever
    vec3 rd = normalize(vec3(uv, 1.3));
    rd.xy *= rot(t*0.15);                            // barrel roll

    float dist = 0.1, glow = 0.0, entGlow = 0.0; float kind = 0.0; bool hit=false; int used=0;
    for(int i=0;i<160;i++){
      if(i>=uSteps) break; used=i;
      vec3 p = ro + rd*dist;
      float k; float d = map(p, t, k);
      glow    += 0.010/(1.0+d*d*22.0);                       // tight tunnel glow
      if(k>0.5) entGlow += 0.020/(1.0+d*d*60.0);             // sharp jewel glow on entities
      if(d < 0.0012*dist){ hit=true; kind=k; break; }
      dist += d*0.85;
      if(dist > 70.0) break;
    }

    vec3 p = ro + rd*dist;
    // cheap normal for shading the hit
    vec3 n = vec3(0.0);
    if(hit){ vec2 e=vec2(0.004,0.0); float kk;
      n = normalize(vec3(map(p+e.xyy,t,kk)-map(p-e.xyy,t,kk), map(p+e.yxy,t,kk)-map(p-e.yxy,t,kk), map(p+e.yyx,t,kk)-map(p-e.yyx,t,kk))); }

    float hue = fract(dist*0.035 + t*0.06 + (kind>0.5?0.45:0.0));
    vec3 base = pal(hue);
    vec3 col = vec3(0.0);
    if(hit){
      vec3 lig = normalize(vec3(0.4,0.7,-0.5));
      float dif = clamp(dot(n,lig),0.0,1.0);
      float fres = pow(1.0-abs(dot(rd,n)),3.0);
      col = base*(0.35 + 0.9*dif);
      // chrome+jewel sheen, much stronger on entities
      col += vec3(1.0,0.95,1.0)*fres*(kind>0.5?1.1:0.45);
      if(kind>0.5) col *= 1.4;
      else col *= 0.6 + 0.5*sin(dist*1.8 + t*2.5);   // mandala banding on walls
    }
    float ao = 1.0 - float(used)/float(uSteps);
    col *= (0.35 + 0.65*ao);
    // additive neon halos — entity glow is saturated, not milky
    col += base * glow * 0.9;
    col += pal(fract(hue+0.55)) * entGlow * 2.2;
    col = col/(col+vec3(0.7));                        // tonemap
    col = pow(col, vec3(0.42));                       // gamma → vivid
    col = mix(vec3(dot(col,vec3(0.33))), col, 1.45);  // saturation boost
    col *= 1.0 - 0.55*dot(uv*0.6, uv*0.6);            // vignette → the rush
    gl_FragColor = vec4(col, 1.0);
  }
`;

const mat = new THREE.ShaderMaterial({ uniforms, vertexShader: `void main(){ gl_Position=vec4(position.xy,0.,1.); }`, fragmentShader: frag });
scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), mat));

// ---------- panel ----------
const SYMS = [3, 4, 5, 6, 8, 12];
let si = 3;
const wrap = document.getElementById("syms");
const nameEl = document.getElementById("symname");
const chips = SYMS.map((n, i) => { const b = document.createElement("button"); b.className="chip"+(i===3?" active":""); b.textContent=n+"-fold"; b.addEventListener("click",()=>{ si=i; uniforms.uSym.value=n; nameEl.textContent=n+"-fold"; chips.forEach((c,k)=>c.classList.toggle("active",k===i)); }); wrap.appendChild(b); return b; });
bindRange("speed", (v)=>{ uniforms.uSpeed.value=v; }, (v)=>v.toFixed(2)+"×");
bindRange("morph", (v)=>{ uniforms.uMorph.value=v; }, (v)=>v.toFixed(2));
bindRange("steps", (v)=>{ uniforms.uSteps.value=Math.round(v); }, (v)=>Math.round(v));
setVariantCycler((d)=>{ si=(si+d+SYMS.length)%SYMS.length; uniforms.uSym.value=SYMS[si]; nameEl.textContent=SYMS[si]+"-fold"; chips.forEach((c,k)=>c.classList.toggle("active",k===si)); return SYMS[si]+"-fold"; });

if (reducedMotion) uniforms.uSpeed.value = 0.15;   // calm it way down for reduced-motion

// ---------- boot ----------
liftVeil();
onResize(renderer, cam, (w,h)=>uniforms.uRes.value.set(w,h));
const meter = fpsMeter(document.getElementById("fps"));
window.__diag = () => JSON.stringify({ sym: uniforms.uSym.value, steps: uniforms.uSteps.value });
let t = 0;
loop((dt) => { meter(dt); t += dt; uniforms.uTime.value = t; renderer.render(scene, cam); });
