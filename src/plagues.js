// plagues.js — 災厄 / The Ten Plagues of Egypt, as a cycling showcase of real
// simulations. Each plague is a distinct, self-contained sim; exactly ONE runs
// at a time (the others are disposed/hidden) so the room holds 60fps. ↑↓ (or the
// auto-advance timer, ~10s) switches plague; the slider scales the current
// plague's intensity; "again" reseeds the active scene.
//
// The plagues shipped here:
//   血 Water to Blood — Gray-Scott reaction-diffusion (Turing, 1952; Pearson,
//        Science 1993), copied from src/reaction.js's GPU ping-pong, tuned so a
//        crimson B-species blooms through a blue water field. The water turns to
//        blood as the pattern spreads.
//   蛙 Frogs — ballistic particle hops: many small bodies leap on parabolic arcs
//        and scatter, a heaving carpet of frogs (CPU agents → one THREE.Points).
//   蝗 Locusts — Reynolds boids (SIGGRAPH '87) copied from src/boids.js: a dark
//        devouring swarm sweeping the view (CPU flocking → one THREE.Points).
//   雹 Hail & Fire — fire-laced hail: a THREE.Points field of icy + ember streaks
//        raining down, with bright impact flashes when they strike the ground.
//   闇 Darkness — "darkness that can be felt": roiling value-noise fog drowns the
//        scene to near-black, only faint cyan edges of drifting motes surviving.
//   過越 Passover — the death of the firstborn: a field of lights where the
//        unmarked are extinguished one by one, while lintel-marked lamps stay lit.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import {
  makeRenderer, onResize, loop, fpsMeter, liftVeil, bindRange,
  reducedMotion, setVariantCycler,
} from "./common.js";

const canvas = document.getElementById("scene");
const renderer = makeRenderer(canvas);
renderer.autoClear = true;

// ============================================================
// shared 3D scene (used by every plague except blood, which is a 2D GPU field
// rendered straight to screen with its own ortho camera)
// ============================================================
const scene = new THREE.Scene();
scene.fog = null;
const camera = new THREE.PerspectiveCamera(58, innerWidth / innerHeight, 0.1, 2000);
camera.position.set(0, 28, 96);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true; controls.dampingFactor = 0.08;
controls.autoRotate = false;            // plagues drive their own motion
controls.minDistance = 20; controls.maxDistance = 360;
controls.target.set(0, 6, 0);

const ambient = new THREE.AmbientLight(0xffffff, 0.5);
scene.add(ambient);
const key = new THREE.DirectionalLight(0xffffff, 0.8);
key.position.set(10, 24, 14); scene.add(key);

// a ground quad many plagues rain onto / hop across
const groundGeo = new THREE.PlaneGeometry(420, 420).rotateX(-Math.PI / 2);
const groundMat = new THREE.MeshStandardMaterial({
  color: 0x07060a, roughness: 1.0, metalness: 0.0,
  transparent: true, opacity: 0.92,
});
const ground = new THREE.Mesh(groundGeo, groundMat);
ground.position.y = -2;
scene.add(ground);

// blood's full-screen quad lives in its own ortho pass
const orthoCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
const fsQuad = new THREE.PlaneGeometry(2, 2);
const VSHADER = `varying vec2 vUv; void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`;

// ============================================================
// small shared helpers
// ============================================================
function rand(a, b) { return a + Math.random() * (b - a); }
const _tmpColor = new THREE.Color();

// build a soft round sprite texture (white core → transparent) for Points
function discTexture(inner = 0.0, hardness = 0.5) {
  const c = document.createElement("canvas");
  c.width = c.height = 64;
  const g = c.getContext("2d");
  const grad = g.createRadialGradient(32, 32, inner * 32, 32, 32, 32);
  grad.addColorStop(0, "rgba(255,255,255,1)");
  grad.addColorStop(hardness, "rgba(255,255,255,0.85)");
  grad.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = grad; g.beginPath(); g.arc(32, 32, 32, 0, Math.PI * 2); g.fill();
  const tex = new THREE.CanvasTexture(c);
  return tex;
}
const DISC = discTexture(0.0, 0.45);
const STREAK = (() => {
  // vertical streak sprite for falling hail/fire
  const c = document.createElement("canvas");
  c.width = 16; c.height = 64;
  const g = c.getContext("2d");
  const grad = g.createLinearGradient(0, 0, 0, 64);
  grad.addColorStop(0, "rgba(255,255,255,0)");
  grad.addColorStop(0.55, "rgba(255,255,255,0.9)");
  grad.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = grad; g.fillRect(5, 0, 6, 64);
  return new THREE.CanvasTexture(c);
})();

// ============================================================
// PLAGUE 1 — 血 WATER TO BLOOD  (Gray-Scott GPU ping-pong)
//   copied from src/reaction.js: half-float RT ping-pong, A=1 water,
//   B blooms; display ramps deep-water-blue → crimson → bright blood.
// ============================================================
const Blood = (() => {
  let SW = 0, SH = 0;
  const SIM_MAX_BASE = 512;
  let simMax = SIM_MAX_BASE;
  function simSize() {
    const ar = innerWidth / innerHeight;
    if (ar >= 1) { SW = simMax; SH = Math.max(2, Math.round(simMax / ar)); }
    else { SH = simMax; SW = Math.max(2, Math.round(simMax * ar)); }
  }
  const RT_OPTS = {
    type: THREE.HalfFloatType, format: THREE.RGBAFormat,
    minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter,
    wrapS: THREE.RepeatWrapping, wrapT: THREE.RepeatWrapping,
    depthBuffer: false, stencilBuffer: false,
  };
  let rtA = null, rtB = null;

  const simU = {
    uTex: { value: null }, uTexel: { value: new THREE.Vector2(1, 1) },
    uF: { value: 0.0367 }, uK: { value: 0.0649 }, uDA: { value: 1.0 }, uDB: { value: 0.5 }, uDt: { value: 1.0 },
  };
  const simScene = new THREE.Scene();
  simScene.add(new THREE.Mesh(fsQuad, new THREE.ShaderMaterial({
    uniforms: simU, vertexShader: VSHADER,
    fragmentShader: `
      precision highp float; varying vec2 vUv;
      uniform sampler2D uTex; uniform vec2 uTexel;
      uniform float uF, uK, uDA, uDB, uDt;
      void main(){
        vec2 c = texture2D(uTex, vUv).xy;
        vec2 lap = vec2(0.0);
        lap += texture2D(uTex, vUv + vec2(-uTexel.x, 0.0)).xy * 0.2;
        lap += texture2D(uTex, vUv + vec2( uTexel.x, 0.0)).xy * 0.2;
        lap += texture2D(uTex, vUv + vec2( 0.0,-uTexel.y)).xy * 0.2;
        lap += texture2D(uTex, vUv + vec2( 0.0, uTexel.y)).xy * 0.2;
        lap += texture2D(uTex, vUv + vec2(-uTexel.x,-uTexel.y)).xy * 0.05;
        lap += texture2D(uTex, vUv + vec2( uTexel.x,-uTexel.y)).xy * 0.05;
        lap += texture2D(uTex, vUv + vec2(-uTexel.x, uTexel.y)).xy * 0.05;
        lap += texture2D(uTex, vUv + vec2( uTexel.x, uTexel.y)).xy * 0.05;
        lap -= c;
        float a = c.x, b = c.y, abb = a * b * b;
        a += (uDA * lap.x - abb + uF * (1.0 - a)) * uDt;
        b += (uDB * lap.y + abb - (uF + uK) * b) * uDt;
        gl_FragColor = vec4(clamp(a,0.0,1.0), clamp(b,0.0,1.0), 0.0, 1.0);
      }`,
  })));

  // display: water (low B) = deep blue; blooming B = crimson → bright blood
  const dispU = { uTex: { value: null }, uTime: { value: 0 } };
  const dispMat = new THREE.ShaderMaterial({
    uniforms: dispU, vertexShader: VSHADER,
    fragmentShader: `
      precision highp float; varying vec2 vUv;
      uniform sampler2D uTex; uniform float uTime;
      void main(){
        vec2 cab = texture2D(uTex, vUv).xy;
        float b = cab.y;
        // water base — dark teal/blue, faint ripple from chemical A
        vec3 water = mix(vec3(0.015,0.045,0.085), vec3(0.03,0.10,0.16), cab.x);
        // blood ramp: clotted maroon -> crimson -> arterial red
        float t = smoothstep(0.06, 0.28, b);
        vec3 blood = mix(vec3(0.20,0.01,0.02), vec3(0.62,0.04,0.05), t);
        blood = mix(blood, vec3(0.92,0.12,0.10), smoothstep(0.22, 0.42, b));
        vec3 col = mix(water, blood, smoothstep(0.05, 0.16, b));
        col += blood * 0.25 * smoothstep(0.3, 0.5, b);   // glow in the thick of it
        gl_FragColor = vec4(col, 1.0);
      }`,
  });
  const dispScene = new THREE.Scene();
  dispScene.add(new THREE.Mesh(fsQuad, dispMat));

  const copyU = { uTex: { value: null } };
  const copyScene = new THREE.Scene();
  copyScene.add(new THREE.Mesh(fsQuad, new THREE.ShaderMaterial({
    uniforms: copyU, vertexShader: VSHADER,
    fragmentShader: `precision highp float; varying vec2 vUv; uniform sampler2D uTex;
      void main(){ gl_FragColor = texture2D(uTex, vUv); }`,
  })));

  function seedField() {
    const data = new Float32Array(SW * SH * 4);
    for (let i = 0; i < SW * SH; i++) { data[i * 4] = 1; data[i * 4 + 1] = 0; data[i * 4 + 3] = 1; }
    // a wound of B that will spread like blood through the water
    const splats = 30;
    for (let s = 0; s < splats; s++) {
      const cx = (Math.random() * SW) | 0, cy = (Math.random() * SH) | 0, r = (4 + Math.random() * 8) | 0;
      for (let y = -r; y <= r; y++) for (let x = -r; x <= r; x++) {
        if (x * x + y * y > r * r) continue;
        const px = ((cx + x) % SW + SW) % SW, py = ((cy + y) % SH + SH) % SH;
        data[(py * SW + px) * 4 + 1] = 1.0;
      }
    }
    const tex = new THREE.DataTexture(data, SW, SH, THREE.RGBAFormat, THREE.FloatType);
    tex.needsUpdate = true;
    copyU.uTex.value = tex;
    for (const rt of [rtA, rtB]) { renderer.setRenderTarget(rt); renderer.render(copyScene, orthoCam); }
    renderer.setRenderTarget(null);
    tex.dispose();
  }

  function allocate() {
    simSize();
    if (!rtA) { rtA = new THREE.WebGLRenderTarget(SW, SH, RT_OPTS); rtB = new THREE.WebGLRenderTarget(SW, SH, RT_OPTS); }
    else { rtA.setSize(SW, SH); rtB.setSize(SW, SH); }
    simU.uTexel.value.set(1 / SW, 1 / SH);
    seedField();
  }

  let iters = 12;
  let t = 0;

  return {
    name: "Water to Blood", jp: "血", english: "Water to Blood",
    is2D: true,
    enter() { allocate(); t = 0; },
    exit() { /* keep RTs allocated for fast re-entry; nothing in the 3D scene to hide */ },
    reseed() { seedField(); t = 0; },
    setIntensity(v) { iters = Math.max(2, Math.round(2 + v * 22)); },   // sim sub-steps / frame
    onResize() { if (rtA) allocate(); },
    step(dt) {
      for (let i = 0; i < iters; i++) {
        simU.uTex.value = rtA.texture;
        renderer.setRenderTarget(rtB);
        renderer.render(simScene, orthoCam);
        const tmp = rtA; rtA = rtB; rtB = tmp;
      }
      t += dt;
    },
    render() {
      dispU.uTex.value = rtA.texture; dispU.uTime.value = t;
      renderer.setRenderTarget(null);
      renderer.render(dispScene, orthoCam);
    },
  };
})();

// ============================================================
// PLAGUE 2 — 蛙 FROGS  (ballistic hops; one THREE.Points)
//   each frog sits on the ground, waits, then launches on a parabolic arc
//   (gravity), lands, and waits again. A heaving, scattering carpet.
// ============================================================
const Frogs = (() => {
  const SPAN = 120, GRAV = -55;
  let N = 1200;
  let px, py, pz, vx, vy, vz, grounded, wait, hue;
  let points = null, geo = null, posAttr = null, colAttr = null, sizeAttr = null;
  let intensity = 0.5;

  function alloc() {
    px = new Float32Array(N); py = new Float32Array(N); pz = new Float32Array(N);
    vx = new Float32Array(N); vy = new Float32Array(N); vz = new Float32Array(N);
    grounded = new Uint8Array(N); wait = new Float32Array(N); hue = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      px[i] = rand(-SPAN, SPAN); pz[i] = rand(-SPAN, SPAN); py[i] = -2;
      grounded[i] = 1; wait[i] = rand(0, 2.5); hue[i] = rand(0.22, 0.42); // greens
    }
  }
  function build() {
    if (points) { scene.remove(points); geo.dispose(); points.material.dispose(); }
    geo = new THREE.BufferGeometry();
    posAttr = new THREE.BufferAttribute(new Float32Array(N * 3), 3).setUsage(THREE.DynamicDrawUsage);
    colAttr = new THREE.BufferAttribute(new Float32Array(N * 3), 3).setUsage(THREE.DynamicDrawUsage);
    sizeAttr = new THREE.BufferAttribute(new Float32Array(N), 1).setUsage(THREE.DynamicDrawUsage);
    geo.setAttribute("position", posAttr);
    geo.setAttribute("color", colAttr);
    geo.setAttribute("asize", sizeAttr);
    const mat = new THREE.PointsMaterial({
      size: 3.2, map: DISC, vertexColors: true, transparent: true,
      depthWrite: false, sizeAttenuation: true, alphaTest: 0.02,
    });
    points = new THREE.Points(geo, mat);
    points.frustumCulled = false; points.visible = false;
    scene.add(points);
  }

  function jump(i) {
    grounded[i] = 0;
    const ang = rand(0, Math.PI * 2);
    const power = rand(10, 24) * (0.6 + intensity);
    vx[i] = Math.cos(ang) * power * 0.45;
    vz[i] = Math.sin(ang) * power * 0.45;
    vy[i] = rand(18, 34) * (0.7 + intensity * 0.6);
  }

  return {
    name: "Frogs", jp: "蛙", english: "Frogs",
    is2D: false,
    enter() {
      ambient.intensity = 0.55; key.intensity = 0.7; key.color.setHex(0xbfe0c0);
      renderer.setClearColor(0x05080a, 1); scene.fog = new THREE.FogExp2(0x05080a, 0.0055);
      ground.visible = true; groundMat.color.setHex(0x06100a); groundMat.opacity = 0.95;
      camera.position.set(0, 26, 104); controls.target.set(0, 4, 0);
      if (!points) build();
      points.visible = true;
    },
    exit() { if (points) points.visible = false; },
    reseed() { alloc(); },
    setIntensity(v) { intensity = v; },
    onResize() {},
    step(dt) {
      const h = Math.min(dt, 1 / 30);
      for (let i = 0; i < N; i++) {
        if (grounded[i]) {
          wait[i] -= h * (0.5 + intensity * 1.5);
          if (wait[i] <= 0) jump(i);
        } else {
          vy[i] += GRAV * h;
          px[i] += vx[i] * h; py[i] += vy[i] * h; pz[i] += vz[i] * h;
          if (py[i] <= -2 && vy[i] < 0) {
            py[i] = -2; grounded[i] = 1; wait[i] = rand(0.15, 1.6);
            vx[i] = vz[i] = vy[i] = 0;
            // wrap stragglers back into the field
            if (Math.abs(px[i]) > SPAN) px[i] = rand(-SPAN, SPAN);
            if (Math.abs(pz[i]) > SPAN) pz[i] = rand(-SPAN, SPAN);
          }
        }
      }
    },
    draw() {
      const pa = posAttr.array, ca = colAttr.array;
      for (let i = 0; i < N; i++) {
        pa[i * 3] = px[i]; pa[i * 3 + 1] = py[i]; pa[i * 3 + 2] = pz[i];
        const air = grounded[i] ? 0 : 1;
        _tmpColor.setHSL(hue[i], 0.7, 0.30 + air * 0.20);
        ca[i * 3] = _tmpColor.r; ca[i * 3 + 1] = _tmpColor.g; ca[i * 3 + 2] = _tmpColor.b;
      }
      posAttr.needsUpdate = true; colAttr.needsUpdate = true;
    },
    render() { renderer.setRenderTarget(null); renderer.render(scene, camera); },
  };
})();

// ============================================================
// PLAGUE 3 — 蝗 LOCUSTS  (Reynolds boids; one THREE.Points)
//   separation / alignment / cohesion (SIGGRAPH '87), copied from boids.js,
//   plus a slow sweep drift so the swarm crosses the view as a dark cloud.
// ============================================================
const Locusts = (() => {
  const BOUND = 60;
  let N = 2200;
  let perception = 8, sepW = 1.8, aliW = 1.0, cohW = 0.9, maxSpeed = 1.4, maxForce = 0.05;
  let px, py, pz, vx, vy, vz;
  let points = null, geo = null, posAttr = null, colAttr = null;
  let intensity = 0.5;
  let sweep = -BOUND;

  function alloc() {
    px = new Float32Array(N); py = new Float32Array(N); pz = new Float32Array(N);
    vx = new Float32Array(N); vy = new Float32Array(N); vz = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      px[i] = rand(-BOUND, BOUND); py[i] = rand(-BOUND * 0.4, BOUND * 0.7); pz[i] = rand(-BOUND, BOUND);
      vx[i] = rand(-1, 1); vy[i] = rand(-1, 1); vz[i] = rand(-1, 1);
    }
  }
  function build() {
    if (points) { scene.remove(points); geo.dispose(); points.material.dispose(); }
    geo = new THREE.BufferGeometry();
    posAttr = new THREE.BufferAttribute(new Float32Array(N * 3), 3).setUsage(THREE.DynamicDrawUsage);
    colAttr = new THREE.BufferAttribute(new Float32Array(N * 3), 3).setUsage(THREE.DynamicDrawUsage);
    geo.setAttribute("position", posAttr);
    geo.setAttribute("color", colAttr);
    const mat = new THREE.PointsMaterial({
      size: 1.7, map: DISC, vertexColors: true, transparent: true,
      depthWrite: false, sizeAttenuation: true, alphaTest: 0.04, opacity: 0.95,
    });
    points = new THREE.Points(geo, mat);
    points.frustumCulled = false; points.visible = false;
    scene.add(points);
  }

  // O(N²) is too much at 2k; use a coarse uniform grid for neighbor queries.
  // cell size = perception; only scan the 27 neighboring cells.
  const CELL = () => Math.max(4, perception);
  let cellMap = new Map();
  function rebuildGrid() {
    cellMap.clear();
    const c = CELL();
    for (let i = 0; i < N; i++) {
      const k = `${Math.floor(px[i] / c)},${Math.floor(py[i] / c)},${Math.floor(pz[i] / c)}`;
      let arr = cellMap.get(k); if (!arr) { arr = []; cellMap.set(k, arr); }
      arr.push(i);
    }
  }
  function step3() {
    rebuildGrid();
    const c = CELL(), per2 = perception * perception, halfPer = perception * 0.5;
    for (let i = 0; i < N; i++) {
      let sx = 0, sy = 0, sz = 0, ax = 0, ay = 0, az = 0, cx = 0, cy = 0, cz = 0, n = 0, ns = 0;
      const gx = Math.floor(px[i] / c), gy = Math.floor(py[i] / c), gz = Math.floor(pz[i] / c);
      for (let ox = -1; ox <= 1; ox++) for (let oy = -1; oy <= 1; oy++) for (let oz = -1; oz <= 1; oz++) {
        const arr = cellMap.get(`${gx + ox},${gy + oy},${gz + oz}`);
        if (!arr) continue;
        for (let t = 0; t < arr.length; t++) {
          const j = arr[t]; if (j === i) continue;
          const dx = px[j] - px[i], dy = py[j] - py[i], dz = pz[j] - pz[i];
          const d2 = dx * dx + dy * dy + dz * dz;
          if (d2 > per2) continue;
          const d = Math.sqrt(d2) + 1e-6;
          if (d < halfPer) { sx -= dx / d; sy -= dy / d; sz -= dz / d; ns++; }
          ax += vx[j]; ay += vy[j]; az += vz[j];
          cx += px[j]; cy += py[j]; cz += pz[j];
          n++;
        }
      }
      let fx = 0, fy = 0, fz = 0;
      if (n > 0) {
        ax /= n; ay /= n; az /= n; fx += (ax - vx[i]) * aliW; fy += (ay - vy[i]) * aliW; fz += (az - vz[i]) * aliW;
        cx = cx / n - px[i]; cy = cy / n - py[i]; cz = cz / n - pz[i]; fx += cx * 0.02 * cohW; fy += cy * 0.02 * cohW; fz += cz * 0.02 * cohW;
      }
      if (ns > 0) { fx += sx * sepW; fy += sy * sepW; fz += sz * sepW; }
      // sweep: bias the whole swarm to drift across +x like a passing cloud
      fx += 0.012 * (0.4 + intensity);
      // soft box boundary (wrap on x for an endless sweep)
      if (py[i] > BOUND) fy -= (py[i] - BOUND) * 0.02;
      if (py[i] < -BOUND) fy += (-BOUND - py[i]) * 0.02;
      if (pz[i] > BOUND) fz -= (pz[i] - BOUND) * 0.02;
      if (pz[i] < -BOUND) fz += (-BOUND - pz[i]) * 0.02;
      const fm = Math.hypot(fx, fy, fz); if (fm > maxForce) { fx = fx / fm * maxForce; fy = fy / fm * maxForce; fz = fz / fm * maxForce; }
      vx[i] += fx; vy[i] += fy; vz[i] += fz;
      const ms = maxSpeed * (0.7 + intensity);
      const sp = Math.hypot(vx[i], vy[i], vz[i]); if (sp > ms) { vx[i] = vx[i] / sp * ms; vy[i] = vy[i] / sp * ms; vz[i] = vz[i] / sp * ms; }
      px[i] += vx[i]; py[i] += vy[i]; pz[i] += vz[i];
      if (px[i] > BOUND * 1.4) px[i] = -BOUND * 1.4;   // wrap the sweep
    }
  }

  return {
    name: "Locusts", jp: "蝗", english: "Locusts",
    is2D: false,
    enter() {
      ambient.intensity = 0.7; key.intensity = 0.6; key.color.setHex(0xd8b24a);
      renderer.setClearColor(0x120d04, 1); scene.fog = new THREE.FogExp2(0x1a1206, 0.006);
      ground.visible = true; groundMat.color.setHex(0x18120a); groundMat.opacity = 0.85;
      camera.position.set(0, 18, 92); controls.target.set(0, 6, 0);
      if (!points) build();
      points.visible = true;
    },
    exit() { if (points) points.visible = false; },
    reseed() { alloc(); sweep = -BOUND; },
    setIntensity(v) { intensity = v; },
    onResize() {},
    step(dt) { step3(); },
    draw() {
      const pa = posAttr.array, ca = colAttr.array;
      for (let i = 0; i < N; i++) {
        pa[i * 3] = px[i]; pa[i * 3 + 1] = py[i]; pa[i * 3 + 2] = pz[i];
        const sp = Math.min(Math.hypot(vx[i], vy[i], vz[i]) / 2.0, 1);
        // sickly amber-on-black: dark bodies, faint amber sheen where fast
        _tmpColor.setHSL(0.11, 0.85, 0.12 + sp * 0.30);
        ca[i * 3] = _tmpColor.r; ca[i * 3 + 1] = _tmpColor.g; ca[i * 3 + 2] = _tmpColor.b;
      }
      posAttr.needsUpdate = true; colAttr.needsUpdate = true;
    },
    render() { renderer.setRenderTarget(null); renderer.render(scene, camera); },
  };
})();

// ============================================================
// PLAGUE 4 — 雹 HAIL & FIRE  (falling THREE.Points + impact flashes)
//   half the particles are ice (white-blue), half ember (orange). They fall,
//   strike the ground (y=-2), and recycle to the top. Each ember impact pops a
//   brief additive flash on the ground.
// ============================================================
const Hail = (() => {
  const SPAN = 130, TOP = 150, FLOOR = -2;
  let N = 2600;
  let px, py, pz, vy, isFire, spin;
  let points = null, geo = null, posAttr = null, colAttr = null;
  let intensity = 0.5;

  // impact flashes: a small recycled THREE.Points of additive sprites on ground
  const FLASH_N = 240;
  let fpx, fpy, fpz, flife, fhead = 0;
  let flashes = null, fGeo = null, fPos = null, fCol = null;

  function alloc() {
    px = new Float32Array(N); py = new Float32Array(N); pz = new Float32Array(N);
    vy = new Float32Array(N); isFire = new Uint8Array(N); spin = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      px[i] = rand(-SPAN, SPAN); pz[i] = rand(-SPAN, SPAN); py[i] = rand(FLOOR, TOP);
      isFire[i] = Math.random() < 0.4 ? 1 : 0;
      vy[i] = -(rand(45, 80) * (isFire[i] ? 0.9 : 1.0));
      spin[i] = rand(-6, 6);
    }
    fpx = new Float32Array(FLASH_N); fpy = new Float32Array(FLASH_N); fpz = new Float32Array(FLASH_N);
    flife = new Float32Array(FLASH_N); fhead = 0;
  }
  function build() {
    if (points) { scene.remove(points); geo.dispose(); points.material.dispose(); }
    geo = new THREE.BufferGeometry();
    posAttr = new THREE.BufferAttribute(new Float32Array(N * 3), 3).setUsage(THREE.DynamicDrawUsage);
    colAttr = new THREE.BufferAttribute(new Float32Array(N * 3), 3).setUsage(THREE.DynamicDrawUsage);
    geo.setAttribute("position", posAttr);
    geo.setAttribute("color", colAttr);
    const mat = new THREE.PointsMaterial({
      size: 3.4, map: STREAK, vertexColors: true, transparent: true,
      depthWrite: false, blending: THREE.AdditiveBlending, sizeAttenuation: true,
    });
    points = new THREE.Points(geo, mat);
    points.frustumCulled = false; points.visible = false;
    scene.add(points);

    if (flashes) { scene.remove(flashes); fGeo.dispose(); flashes.material.dispose(); }
    fGeo = new THREE.BufferGeometry();
    fPos = new THREE.BufferAttribute(new Float32Array(FLASH_N * 3), 3).setUsage(THREE.DynamicDrawUsage);
    fCol = new THREE.BufferAttribute(new Float32Array(FLASH_N * 3), 3).setUsage(THREE.DynamicDrawUsage);
    fGeo.setAttribute("position", fPos);
    fGeo.setAttribute("color", fCol);
    const fmat = new THREE.PointsMaterial({
      size: 14, map: DISC, vertexColors: true, transparent: true,
      depthWrite: false, blending: THREE.AdditiveBlending, sizeAttenuation: true,
    });
    flashes = new THREE.Points(fGeo, fmat);
    flashes.frustumCulled = false; flashes.visible = false;
    scene.add(flashes);
  }
  function spawnFlash(x, z, fire) {
    const i = fhead; fhead = (fhead + 1) % FLASH_N;
    fpx[i] = x; fpy[i] = FLOOR + 0.5; fpz[i] = z; flife[i] = fire ? 1.0 : 0.55;
  }

  return {
    name: "Hail & Fire", jp: "雹", english: "Hail & Fire",
    is2D: false,
    enter() {
      ambient.intensity = 0.4; key.intensity = 0.5; key.color.setHex(0xbcd2ff);
      renderer.setClearColor(0x03040a, 1); scene.fog = new THREE.FogExp2(0x05060f, 0.004);
      ground.visible = true; groundMat.color.setHex(0x05060c); groundMat.opacity = 0.9;
      camera.position.set(0, 24, 110); controls.target.set(0, 14, 0);
      if (!points) build();
      points.visible = true; flashes.visible = true;
    },
    exit() { if (points) points.visible = false; if (flashes) flashes.visible = false; },
    reseed() { alloc(); },
    setIntensity(v) { intensity = v; },
    onResize() {},
    step(dt) {
      const h = Math.min(dt, 1 / 30);
      const speed = 1 + intensity * 1.4;
      for (let i = 0; i < N; i++) {
        py[i] += vy[i] * h * speed;
        if (py[i] <= FLOOR) {
          if (Math.random() < 0.5) spawnFlash(px[i], pz[i], isFire[i]);
          py[i] = TOP + rand(0, 30);
          px[i] = rand(-SPAN, SPAN); pz[i] = rand(-SPAN, SPAN);
        }
      }
      for (let i = 0; i < FLASH_N; i++) if (flife[i] > 0) flife[i] = Math.max(0, flife[i] - h * 2.4);
    },
    draw() {
      const pa = posAttr.array, ca = colAttr.array;
      for (let i = 0; i < N; i++) {
        pa[i * 3] = px[i]; pa[i * 3 + 1] = py[i]; pa[i * 3 + 2] = pz[i];
        if (isFire[i]) { ca[i * 3] = 1.0; ca[i * 3 + 1] = 0.42 + Math.random() * 0.12; ca[i * 3 + 2] = 0.08; }
        else { ca[i * 3] = 0.72; ca[i * 3 + 1] = 0.85; ca[i * 3 + 2] = 1.0; }
      }
      posAttr.needsUpdate = true; colAttr.needsUpdate = true;
      const fp = fPos.array, fc = fCol.array;
      for (let i = 0; i < FLASH_N; i++) {
        fp[i * 3] = fpx[i]; fp[i * 3 + 1] = fpy[i]; fp[i * 3 + 2] = fpz[i];
        const l = flife[i];
        // ember flashes burn orange, ice flashes flare pale blue
        fc[i * 3] = l; fc[i * 3 + 1] = l * 0.5; fc[i * 3 + 2] = l * 0.25;
      }
      fPos.needsUpdate = true; fCol.needsUpdate = true;
    },
    render() { renderer.setRenderTarget(null); renderer.render(scene, camera); },
  };
})();

// ============================================================
// PLAGUE 5 — 闇 DARKNESS  ("darkness that can be felt")
//   drifting motes seen only at their faint cyan edges, drowned by a roiling
//   value-noise dark veil that thickens over the whole frame. A full-screen
//   noise quad is drawn additively (very dark) over the 3D motes each frame.
// ============================================================
const Darkness = (() => {
  let N = 900;
  let px, py, pz, vx, vy, vz, ph;
  let points = null, geo = null, posAttr = null, colAttr = null;
  let intensity = 0.5, t = 0;
  const SPAN = 80;

  // full-screen roiling-fog quad (drawn in its own ortho pass, multiplicative)
  const veilU = { uTime: { value: 0 }, uThick: { value: 0.6 }, uAspect: { value: 1 } };
  const veilMat = new THREE.ShaderMaterial({
    uniforms: veilU, vertexShader: VSHADER, transparent: true,
    depthTest: false, depthWrite: false,
    blending: THREE.NormalBlending,
    fragmentShader: `
      precision highp float; varying vec2 vUv;
      uniform float uTime, uThick, uAspect;
      // cheap value noise
      float h(vec2 p){ return fract(sin(dot(p, vec2(127.1,311.7)))*43758.5453); }
      float vn(vec2 p){
        vec2 i=floor(p), f=fract(p); f=f*f*(3.0-2.0*f);
        float a=h(i), b=h(i+vec2(1,0)), c=h(i+vec2(0,1)), d=h(i+vec2(1,1));
        return mix(mix(a,b,f.x), mix(c,d,f.x), f.y);
      }
      float fbm(vec2 p){ float s=0.0,a=0.5; for(int k=0;k<5;k++){ s+=a*vn(p); p*=2.03; a*=0.5; } return s; }
      void main(){
        vec2 uv = vUv; uv.x *= uAspect;
        float n = fbm(uv*3.0 + vec2(uTime*0.05, uTime*0.03));
        n += 0.5*fbm(uv*7.0 - vec2(uTime*0.08, 0.0));
        // darkness pools where noise is high; thickness drowns everything
        float dark = smoothstep(0.2, 1.1, n*0.7 + uThick);
        float alpha = clamp(dark, 0.0, 0.985);
        // a whisper of cold edge light in the thin spots
        vec3 edge = vec3(0.02,0.06,0.08) * (1.0 - dark);
        gl_FragColor = vec4(edge, alpha);
      }`,
  });
  const veilScene = new THREE.Scene();
  veilScene.add(new THREE.Mesh(fsQuad, veilMat));

  function alloc() {
    px = new Float32Array(N); py = new Float32Array(N); pz = new Float32Array(N);
    vx = new Float32Array(N); vy = new Float32Array(N); vz = new Float32Array(N); ph = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      px[i] = rand(-SPAN, SPAN); py[i] = rand(-SPAN * 0.5, SPAN * 0.6); pz[i] = rand(-SPAN, SPAN);
      vx[i] = rand(-0.4, 0.4); vy[i] = rand(-0.2, 0.2); vz[i] = rand(-0.4, 0.4); ph[i] = rand(0, 6.28);
    }
  }
  function build() {
    if (points) { scene.remove(points); geo.dispose(); points.material.dispose(); }
    geo = new THREE.BufferGeometry();
    posAttr = new THREE.BufferAttribute(new Float32Array(N * 3), 3).setUsage(THREE.DynamicDrawUsage);
    colAttr = new THREE.BufferAttribute(new Float32Array(N * 3), 3).setUsage(THREE.DynamicDrawUsage);
    geo.setAttribute("position", posAttr);
    geo.setAttribute("color", colAttr);
    const mat = new THREE.PointsMaterial({
      size: 2.0, map: DISC, vertexColors: true, transparent: true,
      depthWrite: false, blending: THREE.AdditiveBlending, sizeAttenuation: true,
    });
    points = new THREE.Points(geo, mat);
    points.frustumCulled = false; points.visible = false;
    scene.add(points);
  }

  return {
    name: "Darkness", jp: "闇", english: "Darkness",
    is2D: false,
    enter() {
      ambient.intensity = 0.18; key.intensity = 0.12;
      renderer.setClearColor(0x010103, 1); scene.fog = new THREE.FogExp2(0x000000, 0.012);
      ground.visible = false;
      camera.position.set(0, 8, 70); controls.target.set(0, 0, 0);
      if (!points) build();
      points.visible = true; t = 0;
    },
    exit() { if (points) points.visible = false; ground.visible = true; },
    reseed() { alloc(); t = 0; },
    setIntensity(v) { intensity = v; },     // drives veil thickness
    onResize() {},
    step(dt) {
      t += dt;
      const h = Math.min(dt, 1 / 30);
      for (let i = 0; i < N; i++) {
        px[i] += vx[i] * h * 6 + Math.sin(t * 0.4 + ph[i]) * 0.02;
        py[i] += vy[i] * h * 6; pz[i] += vz[i] * h * 6;
        if (px[i] > SPAN) px[i] = -SPAN; if (px[i] < -SPAN) px[i] = SPAN;
        if (py[i] > SPAN * 0.7) py[i] = -SPAN * 0.5; if (py[i] < -SPAN * 0.6) py[i] = SPAN * 0.6;
        if (pz[i] > SPAN) pz[i] = -SPAN; if (pz[i] < -SPAN) pz[i] = SPAN;
      }
    },
    draw() {
      const pa = posAttr.array, ca = colAttr.array;
      for (let i = 0; i < N; i++) {
        pa[i * 3] = px[i]; pa[i * 3 + 1] = py[i]; pa[i * 3 + 2] = pz[i];
        const f = 0.10 + 0.10 * (0.5 + 0.5 * Math.sin(t + ph[i]));
        ca[i * 3] = f * 0.3; ca[i * 3 + 1] = f * 0.9; ca[i * 3 + 2] = f;   // cold cyan edges
      }
      posAttr.needsUpdate = true; colAttr.needsUpdate = true;
    },
    render() {
      renderer.setRenderTarget(null);
      renderer.clear();
      renderer.render(scene, camera);                      // faint motes
      // thickness rises with intensity and breathes a little
      veilU.uTime.value = t;
      veilU.uAspect.value = innerWidth / innerHeight;
      veilU.uThick.value = 0.35 + intensity * 0.55 + 0.06 * Math.sin(t * 0.5);
      const prevAutoClear = renderer.autoClear; renderer.autoClear = false;
      renderer.render(veilScene, orthoCam);                // drown it in dark
      renderer.autoClear = prevAutoClear;
    },
  };
})();

// ============================================================
// PLAGUE 6 — 過越 PASSOVER  (death of the firstborn; somber finale)
//   a grid-field of household lamps. The Destroyer passes: unmarked lamps are
//   extinguished one by one (snuffed to ember-dark), but lamps "marked" with
//   blood on the lintel stay burning. One THREE.Points; brightness per lamp.
// ============================================================
const Passover = (() => {
  const COLS = 46, ROWS = 30, GAP = 4.4;
  let N = COLS * ROWS;
  let px, py, pz, lit, marked, flick, fade;
  let points = null, geo = null, posAttr = null, colAttr = null;
  let intensity = 0.5, t = 0, sweepZ = 0;

  function alloc() {
    N = COLS * ROWS;
    px = new Float32Array(N); py = new Float32Array(N); pz = new Float32Array(N);
    lit = new Float32Array(N); marked = new Uint8Array(N); flick = new Float32Array(N); fade = new Float32Array(N);
    const w = (COLS - 1) * GAP, d = (ROWS - 1) * GAP;
    for (let r = 0; r < ROWS; r++) for (let c = 0; c < COLS; c++) {
      const i = r * COLS + c;
      px[i] = c * GAP - w / 2; pz[i] = r * GAP - d / 2; py[i] = 0;
      lit[i] = 1; fade[i] = 1; flick[i] = rand(0, 6.28);
      marked[i] = Math.random() < 0.18 ? 1 : 0;   // the marked households
    }
    sweepZ = -d / 2 - 8; t = 0;
  }
  function build() {
    if (points) { scene.remove(points); geo.dispose(); points.material.dispose(); }
    geo = new THREE.BufferGeometry();
    posAttr = new THREE.BufferAttribute(new Float32Array(N * 3), 3).setUsage(THREE.DynamicDrawUsage);
    colAttr = new THREE.BufferAttribute(new Float32Array(N * 3), 3).setUsage(THREE.DynamicDrawUsage);
    geo.setAttribute("position", posAttr);
    geo.setAttribute("color", colAttr);
    const mat = new THREE.PointsMaterial({
      size: 6.5, map: DISC, vertexColors: true, transparent: true,
      depthWrite: false, blending: THREE.AdditiveBlending, sizeAttenuation: true,
    });
    points = new THREE.Points(geo, mat);
    points.frustumCulled = false; points.visible = false;
    scene.add(points);
  }

  return {
    name: "Passover", jp: "過越", english: "Passover",
    is2D: false,
    enter() {
      ambient.intensity = 0.12; key.intensity = 0.1;
      renderer.setClearColor(0x040206, 1); scene.fog = new THREE.FogExp2(0x040206, 0.004);
      ground.visible = true; groundMat.color.setHex(0x050208); groundMat.opacity = 0.95;
      camera.position.set(0, 56, 86); controls.target.set(0, 0, 0);
      if (!points) { alloc(); build(); }
      points.visible = true;
    },
    exit() { if (points) points.visible = false; },
    reseed() { alloc(); },
    setIntensity(v) { intensity = v; },    // speed of the passing
    onResize() {},
    step(dt) {
      const h = Math.min(dt, 1 / 30);
      t += h;
      const d = (ROWS - 1) * GAP;
      // the Destroyer advances in +z; lamps it has passed get a chance to die
      sweepZ += h * (3 + intensity * 9);
      for (let i = 0; i < N; i++) {
        if (lit[i] > 0 && !marked[i] && pz[i] < sweepZ) {
          // snuff stochastically once passed (so it ripples, not a hard line)
          if (Math.random() < h * 3.5) lit[i] = 0;
        }
        // fade toward target brightness
        const target = lit[i] > 0 ? 1 : 0;
        fade[i] += (target - fade[i]) * Math.min(1, h * 4);
      }
      if (sweepZ > d / 2 + 12) sweepZ = -d / 2 - 8;   // (won't relight; reseed for a fresh pass)
    },
    draw() {
      const pa = posAttr.array, ca = colAttr.array;
      for (let i = 0; i < N; i++) {
        pa[i * 3] = px[i]; pa[i * 3 + 1] = py[i]; pa[i * 3 + 2] = pz[i];
        const fl = 0.85 + 0.15 * Math.sin(t * 6 + flick[i]);
        const b = fade[i] * fl;
        if (marked[i]) {
          // marked lamps burn with a protective blood-warm glow
          ca[i * 3] = 1.0 * b; ca[i * 3 + 1] = 0.30 * b; ca[i * 3 + 2] = 0.12 * b;
        } else {
          // ordinary lamps: warm flame -> cold dead ember as fade drops
          ca[i * 3] = (1.0 * b) + (1 - fade[i]) * 0.03;
          ca[i * 3 + 1] = 0.62 * b;
          ca[i * 3 + 2] = 0.22 * b;
        }
      }
      posAttr.needsUpdate = true; colAttr.needsUpdate = true;
    },
    render() { renderer.setRenderTarget(null); renderer.render(scene, camera); },
  };
})();

// ============================================================
// orchestration — one active plague at a time
// ============================================================
const PLAGUES = [Blood, Frogs, Locusts, Hail, Darkness, Passover];
let idx = 0;
let current = null;

function activate(i, reseed = true) {
  const next = PLAGUES[((i % PLAGUES.length) + PLAGUES.length) % PLAGUES.length];
  if (current && current !== next && current.exit) current.exit();
  idx = PLAGUES.indexOf(next);
  current = next;
  if (reseed && current.reseed) current.reseed();
  if (current.enter) current.enter();
  if (current.setIntensity) current.setIntensity(intensity);
  updateLabel();
}

// --- label card (Japanese + English) ---
const nameEl = document.getElementById("pname");
const jpEl = document.getElementById("pjp");
const idxEl = document.getElementById("pidx");
const readoutEl = document.getElementById("preadout");
function updateLabel() {
  if (nameEl) nameEl.textContent = current.english;
  if (jpEl) jpEl.textContent = current.jp;
  if (idxEl) idxEl.textContent = `${idx + 1}/${PLAGUES.length}`;
  if (readoutEl) readoutEl.textContent = current.english;
  syncChips();
}

// --- panel: plague chips ---
const wrap = document.getElementById("plaguelist");
let chips = [];
if (wrap) {
  chips = PLAGUES.map((p, i) => {
    const b = document.createElement("button");
    b.className = "chip" + (i === 0 ? " active" : "");
    b.textContent = `${p.jp} ${p.english.split(" ")[0]}`;
    b.title = p.english;
    b.addEventListener("click", () => activate(i));
    wrap.appendChild(b);
    return b;
  });
}
function syncChips() { chips.forEach((c, k) => c.classList.toggle("active", k === idx)); }

// --- intensity slider (declared before activate uses it; no TDZ) ---
let intensity = 0.5;
bindRange("intensity", (v) => { intensity = v; if (current && current.setIntensity) current.setIntensity(v); },
  (v) => `${Math.round(v * 100)}%`);

// --- reset button ---
const resetBtn = document.getElementById("reset");
if (resetBtn) resetBtn.addEventListener("click", () => { if (current && current.reseed) current.reseed(); auto = AUTO_SECS; });

// --- auto-advance toggle ---
let autoEnabled = !reducedMotion;
const autoBtn = document.getElementById("autobtn");
if (autoBtn) {
  autoBtn.classList.toggle("active", autoEnabled);
  autoBtn.textContent = autoEnabled ? "auto ◉" : "auto ○";
  autoBtn.addEventListener("click", () => {
    autoEnabled = !autoEnabled;
    autoBtn.classList.toggle("active", autoEnabled);
    autoBtn.textContent = autoEnabled ? "auto ◉" : "auto ○";
    auto = AUTO_SECS;
  });
}

// --- ↑↓ cycles plague ---
setVariantCycler((d) => {
  activate(idx + d);
  auto = AUTO_SECS;       // reset the timer on manual change
  return `${current.jp} · ${current.english}`;
});

// ============================================================
// boot + loop
// ============================================================
activate(0);
liftVeil();

// reallocate the GPU field on resize (mirror reaction.js — don't pass a
// callback into onResize's camera slot); 3D plagues just need the camera synced
onResize(renderer, camera, () => { if (current && current.onResize) current.onResize(); });

const meter = fpsMeter(document.getElementById("fps"));

const AUTO_SECS = 10;
let auto = AUTO_SECS;

window.__diag = () => JSON.stringify({ plague: current ? current.english : null, jp: current ? current.jp : null, idx });

loop((dt) => {
  meter(dt);
  // auto-advance
  if (autoEnabled) {
    auto -= dt;
    if (auto <= 0) { auto = AUTO_SECS; activate(idx + 1); }
  }
  current.step(dt);
  if (current.draw) current.draw();
  controls.update();
  current.render();
});
