// earthbound.js — EarthBound / MOTHER 2 style battle backgrounds.
// Two procedural layers, each warped by HDMA-style per-row sine distortion
// (horizontal/vertical/interlaced) and colored by a CYCLING palette gradient —
// exactly the trio of tricks the SNES used. Extremely weird on purpose; record
// with R and drop the .webm straight into your RPG.
//
// Technique refs (HDMA row distortion + palette rotation):
//   https://github.com/gjtorikian/Earthbound-Battle-Backgrounds-JS
//   https://github.com/pk-hack/CoilSnake/wiki/Tutorial:-Battle-Backgrounds

import * as THREE from "three";
import { makeRenderer, onResize, loop, fpsMeter, liftVeil, bindRange, reducedMotion, setVariantCycler } from "./common.js";

const canvas = document.getElementById("scene");
const renderer = makeRenderer(canvas);

const scene = new THREE.Scene();
const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

const uniforms = {
  uResolution: { value: new THREE.Vector2(innerWidth, innerHeight) },
  uTime:  { value: 0 },
  uSpeed: { value: 1.0 },
  uAmp:   { value: 0.5 },
  uFreq:  { value: 8.0 },
  uModeA: { value: 2 }, uDistA: { value: 1 },
  uModeB: { value: 0 }, uDistB: { value: 2 },
  uPalette: { value: 0 },
};

const vert = /* glsl */ `void main(){ gl_Position = vec4(position.xy, 0.0, 1.0); }`;

const frag = /* glsl */ `
  precision highp float;
  uniform vec2  uResolution;
  uniform float uTime, uSpeed, uAmp, uFreq;
  uniform int   uModeA, uDistA, uModeB, uDistB, uPalette;

  #define PI 3.14159265

  // cycling palettes — palette rotation = the +t fed into fract() at the call site
  vec3 pal(float t, int which){
    t = fract(t);
    if(which == 0){                       // vaporwave loop
      vec3 a=vec3(0.10,0.02,0.24), b=vec3(0.45,0.10,0.62), c=vec3(1.00,0.18,0.60),
           d=vec3(0.16,0.82,0.96), e=vec3(0.55,1.00,0.78);
      float x=t*5.0;
      if(x<1.0) return mix(a,b,x);
      if(x<2.0) return mix(b,c,x-1.0);
      if(x<3.0) return mix(c,d,x-2.0);
      if(x<4.0) return mix(d,e,x-3.0);
      return mix(e,a,x-4.0);
    } else if(which == 1){                // rainbow (IQ cosine palette)
      return 0.5 + 0.5*cos(2.0*PI*(t + vec3(0.0,0.33,0.67)));
    }                                     // acid sunset
    return 0.5 + 0.5*cos(2.0*PI*(t*vec3(1.0,0.7,0.4) + vec3(0.0,0.15,0.5)));
  }

  float pattern(vec2 p, int mode, float t){
    if(mode == 0){                        // concentric rings
      return 0.5 + 0.5*sin(length(p)*uFreq - t*2.0);
    } else if(mode == 1){                 // diamond grid
      vec2 q = fract(p*uFreq*0.12) - 0.5;
      return 0.5 + 0.5*sin((abs(q.x)+abs(q.y))*9.0 - t*2.0);
    } else if(mode == 2){                 // plasma
      float v = sin(p.x*uFreq*0.10 + t) + sin(p.y*uFreq*0.12 - t)
              + sin((p.x+p.y)*uFreq*0.08 + t) + sin(length(p)*uFreq*0.10 - t);
      return 0.5 + 0.25*v;
    } else if(mode == 3){                 // spiral
      return 0.5 + 0.5*sin(atan(p.y,p.x)*6.0 + length(p)*uFreq*0.12 - t*2.0);
    }                                     // checker waves
    return 0.5 + 0.5*sin(p.x*uFreq*0.10 + t)*sin(p.y*uFreq*0.10 - t);
  }

  vec2 distort(vec2 p, int kind, float t){
    if(kind == 1){                        // horizontal, smooth
      p.x += uAmp*sin(p.y*0.06*uFreq + t*2.0);
    } else if(kind == 2){                 // vertical, smooth (refract/split)
      p.y += uAmp*sin(p.x*0.06*uFreq + t*2.0);
    } else if(kind == 3){                 // horizontal, interlaced (helix)
      float s = mod(floor(gl_FragCoord.y), 2.0) < 1.0 ? 1.0 : -1.0;
      p.x += s*uAmp*sin(p.y*0.06*uFreq + t*2.0);
    }
    return p;                             // kind 0 = none
  }

  void main(){
    vec2 uv = (gl_FragCoord.xy - 0.5*uResolution)/uResolution.y * 6.0;
    float t = uTime*uSpeed;

    vec2 pa = distort(uv, uDistA, t);
    vec3 ca = pal(pattern(pa, uModeA, t) + t*0.10, uPalette);

    vec2 pb = distort(uv*1.3 + vec2(2.0,-1.0), uDistB, t*1.27);
    vec3 cb = pal(pattern(pb, uModeB, t*1.13) + t*0.07 + 0.5, uPalette);

    vec3 col = mix(ca, cb, 0.5);
    col = pow(col, vec3(0.85));           // a touch of contrast for the CRT
    gl_FragColor = vec4(col, 1.0);
  }
`;

const material = new THREE.ShaderMaterial({ uniforms, vertexShader: vert, fragmentShader: frag });
scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material));

// ---------- presets (named after EarthBound locales) ----------
const PRESETS = [
  ["onett",    { mA: 2, dA: 1, mB: 0, dB: 2, pal: 0 }],
  ["giygas",   { mA: 3, dA: 3, mB: 4, dB: 1, pal: 2 }],
  ["saturn",   { mA: 1, dA: 2, mB: 1, dB: 1, pal: 1 }],
  ["caverns",  { mA: 0, dA: 1, mB: 3, dB: 3, pal: 0 }],
  ["acid rain",{ mA: 4, dA: 3, mB: 2, dB: 2, pal: 1 }],
  ["the void", { mA: 3, dA: 2, mB: 0, dB: 3, pal: 2 }],
];
function applyPreset(p) {
  uniforms.uModeA.value = p.mA; uniforms.uDistA.value = p.dA;
  uniforms.uModeB.value = p.mB; uniforms.uDistB.value = p.dB;
  uniforms.uPalette.value = p.pal;
  palIdx = p.pal;
}

// ---------- panel ----------
const wrap = document.getElementById("presets");
const chips = PRESETS.map(([name, cfg], i) => {
  const b = document.createElement("button");
  b.className = "chip" + (i === 0 ? " active" : "");
  b.textContent = name;
  b.addEventListener("click", () => {
    applyPreset(cfg);
    chips.forEach((c) => c.classList.toggle("active", c === b));
    nameEl.textContent = name;
  });
  wrap.appendChild(b);
  return b;
});

bindRange("speed", (v) => { uniforms.uSpeed.value = v; }, (v) => v.toFixed(2) + "×");
bindRange("amp",   (v) => { uniforms.uAmp.value = v; }, (v) => v.toFixed(2));
bindRange("freq",  (v) => { uniforms.uFreq.value = v; }, (v) => v.toFixed(0));

let palIdx = 0;
const PAL_NAMES = ["vapor", "rainbow", "acid"];
const palBtn = document.getElementById("palette");
palBtn.addEventListener("click", () => {
  palIdx = (palIdx + 1) % 3;
  uniforms.uPalette.value = palIdx;
  palBtn.textContent = "palette: " + PAL_NAMES[palIdx];
});

let playing = !reducedMotion;
const playBtn = document.getElementById("play");
playBtn.classList.toggle("active", playing);
playBtn.addEventListener("click", () => {
  playing = !playing;
  playBtn.textContent = playing ? "pause" : "play";
  playBtn.classList.toggle("active", playing);
});

const nameEl = document.getElementById("presetname");

// ↑/↓ cycle the presets
let _variantIdx = 0;
setVariantCycler((d) => {
  _variantIdx = (_variantIdx + d + PRESETS.length) % PRESETS.length;
  chips[_variantIdx].click();
  return PRESETS[_variantIdx][0];
});

// ---------- boot ----------
applyPreset(PRESETS[0][1]);
liftVeil();
onResize(renderer, camera, (w, h) => uniforms.uResolution.value.set(w, h));
const meter = fpsMeter(document.getElementById("fps"));

let clock = 0;
loop((dt) => {
  meter(dt);
  if (playing) clock += dt;
  uniforms.uTime.value = clock;
  renderer.render(scene, camera);
});
