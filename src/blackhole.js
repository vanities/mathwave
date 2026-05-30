// blackhole.js — a Schwarzschild black hole, gravitationally lensing light.
// Each pixel shoots a photon; instead of travelling straight, the photon is bent
// by gravity as it marches. Using the standard ray-marching approximation
//   a = −1.5 · h² · Rs · r̂ / r⁵      (h = |r × v|, the angular momentum)
// the photon curves around the hole — bending the starfield behind it into an
// EINSTEIN RING, swallowing anything that crosses the event horizon (black), and
// lighting up an ACCRETION DISK in the equatorial plane. All on the GPU.
// ↑↓ changes the camera inclination to the disk.
//
// Refs: rantonels "starless"; oseiskar/black-hole; Schwarzschild geodesics.

import * as THREE from "three";
import { makeRenderer, onResize, loop, fpsMeter, liftVeil, bindRange, reducedMotion, setVariantCycler } from "./common.js";

const canvas = document.getElementById("scene");
const renderer = makeRenderer(canvas);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.4)); // heavy shader
const scene = new THREE.Scene();
const cam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

const uniforms = {
  uRes: { value: new THREE.Vector2(innerWidth, innerHeight) },
  uTime: { value: 0 },
  uIncl: { value: 0.32 },     // camera inclination above the disk plane
  uSteps: { value: 240 },
};

const frag = `
  precision highp float;
  uniform vec2 uRes; uniform float uTime, uIncl; uniform int uSteps;
  const float Rs = 1.0;          // Schwarzschild radius (event horizon)
  const float DISK_IN = 2.2, DISK_OUT = 7.0;

  // hash starfield
  float hash(vec2 p){ p = fract(p*vec2(123.34,456.21)); p += dot(p, p+45.32); return fract(p.x*p.y); }
  vec3 stars(vec3 dir){
    vec2 uv = vec2(atan(dir.z,dir.x), asin(clamp(dir.y,-1.,1.)));
    float s = 0.0;
    for(float l=0.0;l<3.0;l++){
      vec2 g = floor(uv*(60.0+l*80.0));
      float h = hash(g+l*17.0);
      s += smoothstep(0.998-l*0.0006, 1.0, h);
    }
    vec3 sky = mix(vec3(0.02,0.01,0.06), vec3(0.06,0.02,0.13), dir.y*0.5+0.5);
    return sky + vec3(0.9,0.85,1.0)*s;
  }
  vec3 diskColor(float r, float ang){
    float t = (r - DISK_IN)/(DISK_OUT - DISK_IN);
    vec3 hot = vec3(1.0,0.95,0.7), warm = vec3(1.0,0.45,0.15), cool = vec3(0.7,0.1,0.5);
    vec3 c = t < 0.5 ? mix(hot, warm, t*2.0) : mix(warm, cool, (t-0.5)*2.0);
    float swirl = 0.6 + 0.4*sin(ang*3.0 - uTime*2.0 + r*2.0);   // turbulent banding
    return c * swirl * (1.2 - t);
  }

  void main(){
    vec2 uv = (gl_FragCoord.xy*2.0 - uRes)/uRes.y;
    // camera orbiting at radius, looking at the hole, inclined by uIncl
    float ci = cos(uIncl), si = sin(uIncl);
    vec3 ro = vec3(0.0, si*9.0, -ci*9.0);
    vec3 ww = normalize(-ro);
    vec3 uu = normalize(cross(vec3(0.0,1.0,0.0), ww));
    vec3 vv = cross(ww, uu);
    vec3 rd = normalize(uv.x*uu + uv.y*vv + 1.6*ww);

    vec3 pos = ro; vec3 vel = rd;
    vec3 color = vec3(0.0); bool done = false;
    float dt = 0.16;
    for(int i=0;i<400;i++){
      if(i>=uSteps || done) break;
      float r = length(pos);
      if(r < Rs){ color = vec3(0.0); done = true; break; }      // event horizon
      // gravitational bending: a = -1.5 h^2 Rs * pos / r^5
      vec3 h = cross(pos, vel); float h2 = dot(h,h);
      vec3 acc = -1.5 * h2 * Rs * pos / pow(r, 5.0);
      vec3 prev = pos;
      vel += acc * dt; pos += vel * dt;
      // accretion-disk crossing (y=0 plane)
      if(prev.y * pos.y < 0.0){
        float tt = prev.y / (prev.y - pos.y);
        vec3 hit = mix(prev, pos, tt);
        float rr = length(hit.xz);
        if(rr > DISK_IN && rr < DISK_OUT){
          color += diskColor(rr, atan(hit.z, hit.x));
          done = true; break;
        }
      }
      if(r > 30.0){ color = stars(normalize(vel)); done = true; break; }  // escaped
    }
    if(!done) color = stars(normalize(vel));
    color = color/(color+vec3(1.0));        // tonemap
    color = pow(color, vec3(0.45));         // gamma
    gl_FragColor = vec4(color, 1.0);
  }
`;
const mat = new THREE.ShaderMaterial({ uniforms, vertexShader: `void main(){ gl_Position=vec4(position.xy,0.,1.); }`, fragmentShader: frag });
scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), mat));

// ---------- panel ----------
bindRange("steps", (v) => { uniforms.uSteps.value = Math.round(v); }, (v) => Math.round(v));
let inclTarget = 0.32;
const INCLS = [0.05, 0.2, 0.32, 0.6, 1.0];
let ii = 2;
const wrap = document.getElementById("views");
const chips = INCLS.map((v, i) => { const b = document.createElement("button"); b.className="chip"+(i===2?" active":""); b.textContent=(v*57|0)+"°"; b.addEventListener("click",()=>{ ii=i; inclTarget=v; chips.forEach((c,k)=>c.classList.toggle("active",k===i)); }); wrap.appendChild(b); return b; });
setVariantCycler((d) => { ii=(ii+d+INCLS.length)%INCLS.length; inclTarget=INCLS[ii]; chips.forEach((c,k)=>c.classList.toggle("active",k===ii)); return (INCLS[ii]*57|0)+"° tilt"; });

// ---------- boot ----------
liftVeil();
onResize(renderer, cam, (w, h) => uniforms.uRes.value.set(w, h));
const meter = fpsMeter(document.getElementById("fps"));
loop((dt) => {
  meter(dt);
  uniforms.uIncl.value += (inclTarget - uniforms.uIncl.value) * 0.05;
  uniforms.uTime.value += dt;
  renderer.render(scene, cam);
});
