// brimstone.js — Sodom & Gomorrah: ballistic fire-and-brimstone rain razes a city to ash.
// Method: a deterministic CPU particle/physics sim rendered with real Three.js geometry.
// The city is ONE THREE.InstancedMesh of boxes (a precomputed jittered-grid skyline on a
// ground plane). Falling brimstone is ONE THREE.Points integrated under constant gravity
// (ballistic: p += v·dt, v.y -= g·dt). On ground/building contact a meteor triggers an
// IMPACT — a one-shot PointLight flash + a burst spawned into a shared ember Points pool —
// and DAMAGES the struck building (it shrinks/darkens toward ash). Rising embers + dark
// smoke (two more Points pools) bleed off burning buildings; the sky glows hellish red and
// warms as the city falls. When the city is fully razed the scene rebuilds and the
// judgment begins again — a continuous filmable cycle.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { makeRenderer, onResize, loop, fpsMeter, liftVeil, bindRange, reducedMotion, setVariantCycler, addGrid } from "./common.js";

const canvas = document.getElementById("scene");
const renderer = makeRenderer(canvas);
renderer.setClearColor(0x07030a, 1); // near-black with a red bruise
renderer.toneMappingExposure = 1.1;

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x180408, 0.0019); // smoky hellish haze

const camera = new THREE.PerspectiveCamera(58, innerWidth / innerHeight, 0.1, 6000);
camera.position.set(0, 150, 380);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true; controls.dampingFactor = 0.05;
controls.target.set(0, 30, 0);
controls.autoRotate = !reducedMotion; controls.autoRotateSpeed = 0.22;
controls.minDistance = 120; controls.maxDistance = 900;

// ---- params ----
const params = {
  cityN: 900,    // building count (slider: "city")
  rain: 1.0,     // rain rate multiplier (slider)
  g: 90,         // gravity for ballistic fall
  angle: 0.18,   // lateral lean of the rain
  startRuin: 0,  // fraction of city pre-destroyed at spawn (presets)
  intensity: 1.0 // global ferocity (meteor budget multiplier)
};

const CITY_HALF = 230;   // half-extent of city footprint on each axis
const GROUND_Y = 0;

// ---- shared scratch ----
const dummy = new THREE.Object3D();
const tmpColor = new THREE.Color();

// ===================================================================
// CITY — one InstancedMesh of unit boxes, scaled per-building.
// ===================================================================
const MAX_CITY = 1600;
let city;
let bx, by, bz;        // base position (footprint + ground anchor)
let bBaseH, bH;        // base height + current (damaged) height
let bBaseW, bBaseD;    // footprint width/depth
let bBurn, bFire;      // burn 0..1 (0 lit, 1 ash) + active fire timer
let bTower, bAlive;
let cityCount = 0;
let razed = 0;

function buildCity(n) {
  n = Math.min(n | 0, MAX_CITY);
  if (!bx) {
    bx = new Float32Array(MAX_CITY); by = new Float32Array(MAX_CITY); bz = new Float32Array(MAX_CITY);
    bBaseH = new Float32Array(MAX_CITY); bH = new Float32Array(MAX_CITY);
    bBaseW = new Float32Array(MAX_CITY); bBaseD = new Float32Array(MAX_CITY);
    bBurn = new Float32Array(MAX_CITY); bFire = new Float32Array(MAX_CITY);
    bTower = new Uint8Array(MAX_CITY); bAlive = new Uint8Array(MAX_CITY);
  }

  if (city) { scene.remove(city); city.geometry.dispose(); city.material.dispose(); }
  const g = new THREE.BoxGeometry(1, 1, 1);
  const m = new THREE.MeshStandardMaterial({ metalness: 0.18, roughness: 0.82, emissive: 0x000000 });
  city = new THREE.InstancedMesh(g, m, MAX_CITY);
  city.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  city.frustumCulled = false;
  scene.add(city);

  // jittered grid so the skyline reads as a city; taller toward the centre + a few towers
  const cols = Math.max(4, Math.round(Math.sqrt(n)));
  const cell = (CITY_HALF * 2) / cols;
  let i = 0;
  for (let gx = 0; gx < cols && i < n; gx++) {
    for (let gz = 0; gz < cols && i < n; gz++) {
      const jx = (Math.random() - 0.5) * cell * 0.55;
      const jz = (Math.random() - 0.5) * cell * 0.55;
      const px = -CITY_HALF + cell * 0.5 + gx * cell + jx;
      const pz = -CITY_HALF + cell * 0.5 + gz * cell + jz;
      const r = Math.hypot(px, pz) / CITY_HALF;
      const tower = Math.random() < 0.04;
      const baseH = tower ? 90 + Math.random() * 110 : (14 + Math.random() * 56) * (1.15 - r * 0.55);
      bx[i] = px; bz[i] = pz; by[i] = GROUND_Y;
      bBaseH[i] = Math.max(8, baseH); bH[i] = bBaseH[i];
      bBaseW[i] = cell * (0.34 + Math.random() * 0.34);
      bBaseD[i] = cell * (0.34 + Math.random() * 0.34);
      bBurn[i] = 0; bFire[i] = 0; bTower[i] = tower ? 1 : 0; bAlive[i] = 1;
      i++;
    }
  }
  cityCount = i;
  razed = 0;

  // optional pre-ruin (presets like "the ruin")
  if (params.startRuin > 0) {
    const k = Math.floor(cityCount * params.startRuin);
    for (let s = 0; s < k; s++) damageBuilding((Math.random() * cityCount) | 0, 0.6 + Math.random() * 0.5, false);
  }

  // hide unused instances (scale ~0); write + paint everything once
  for (let j = 0; j < MAX_CITY; j++) {
    if (j >= cityCount) {
      dummy.position.set(0, -9999, 0);
      dummy.scale.set(0.0001, 0.0001, 0.0001);
      dummy.rotation.set(0, 0, 0);
      dummy.updateMatrix();
      city.setMatrixAt(j, dummy.matrix);
    } else writeBuilding(j);
  }
  city.instanceMatrix.needsUpdate = true;
  city.setColorAt(0, tmpColor.set(0x222428)); // force instanceColor allocation
  for (let j = 0; j < cityCount; j++) paintBuilding(j);
  city.instanceColor.needsUpdate = true;
}

// write one building's transform (origin at its base; height grows upward)
function writeBuilding(i) {
  const h = Math.max(0.001, bH[i]);
  dummy.position.set(bx[i], by[i] + h * 0.5, bz[i]);
  dummy.scale.set(bBaseW[i], h, bBaseD[i]);
  dummy.rotation.set(0, 0, 0);
  dummy.updateMatrix();
  city.setMatrixAt(i, dummy.matrix);
}

// colour by burn level: lit slate -> ember-glow when on fire -> charcoal ash.
function paintBuilding(i) {
  const burn = bBurn[i];
  const fire = bFire[i] > 0 ? Math.min(1, bFire[i] / 1.6) : 0;
  const lit = 0.16 + 0.10 * (1 - burn);
  let r = lit, gg = lit * 1.02, b = lit * 1.15;
  const ash = 0.06 + burn * 0.02;
  r = r * (1 - burn) + ash * burn;
  gg = gg * (1 - burn) + ash * burn;
  b = b * (1 - burn) + ash * burn;
  r += fire * 0.55; gg += fire * 0.18; b += fire * 0.02;
  tmpColor.setRGB(Math.min(1, r), Math.min(1, gg), Math.min(1, b));
  city.setColorAt(i, tmpColor);
}

// damage a building; emit=true spawns embers/smoke + a flash from the strike.
function damageBuilding(i, amount, emit) {
  if (i < 0 || i >= cityCount || !bAlive[i]) return;
  const wasStanding = bBurn[i] < 0.999;
  bBurn[i] = Math.min(1, bBurn[i] + amount);
  bFire[i] = Math.max(bFire[i], 1.6); // ignite / re-stoke
  bH[i] = Math.max(bBaseH[i] * (1 - bBurn[i] * 0.86), bBaseH[i] * 0.08);
  if (bBurn[i] >= 0.999 && wasStanding) razed++;
  writeBuilding(i);
  if (city) city.instanceMatrix.needsUpdate = true;
  if (emit) {
    const topY = by[i] + bH[i];
    spawnEmberBurst(bx[i], topY, bz[i], 14 + (Math.random() * 12 | 0));
    spawnSmoke(bx[i], topY, bz[i], 6);
    flash(bx[i], topY + 6, bz[i]);
  }
}

// ===================================================================
// BRIMSTONE RAIN — one THREE.Points, ballistic fall under gravity.
// ===================================================================
const MAX_METEORS = 320;
let mPos, mVel, mLife, mActive;
let meteorGeom, meteorPts, meteorCol;
let meteorCount = 0; // cumulative impacts

function buildMeteors() {
  if (meteorPts) { scene.remove(meteorPts); meteorGeom.dispose(); meteorPts.material.dispose(); }
  mPos = new Float32Array(MAX_METEORS * 3);
  mVel = new Float32Array(MAX_METEORS * 3);
  mLife = new Float32Array(MAX_METEORS);
  mActive = new Uint8Array(MAX_METEORS);
  meteorCol = new Float32Array(MAX_METEORS * 3);
  for (let i = 0; i < MAX_METEORS; i++) mPos[i * 3 + 1] = -9999;
  meteorGeom = new THREE.BufferGeometry();
  meteorGeom.setAttribute("position", new THREE.BufferAttribute(mPos, 3).setUsage(THREE.DynamicDrawUsage));
  meteorGeom.setAttribute("color", new THREE.BufferAttribute(meteorCol, 3));
  const mat = new THREE.PointsMaterial({
    size: 13, map: glowTex, vertexColors: true, transparent: true, opacity: 1.0,
    blending: THREE.AdditiveBlending, depthWrite: false, sizeAttenuation: true
  });
  meteorPts = new THREE.Points(meteorGeom, mat);
  meteorPts.frustumCulled = false;
  scene.add(meteorPts);
}

function spawnMeteor() {
  let i = -1;
  for (let k = 0; k < MAX_METEORS; k++) if (!mActive[k]) { i = k; break; }
  if (i < 0) return;
  mActive[i] = 1;
  const sx = (Math.random() - 0.5) * CITY_HALF * 2.1;
  const sz = (Math.random() - 0.5) * CITY_HALF * 2.1;
  mPos[i * 3] = sx + params.angle * 600;        // lean the rain
  mPos[i * 3 + 1] = 540 + Math.random() * 220;   // high in the sky
  mPos[i * 3 + 2] = sz;
  mVel[i * 3] = -params.angle * 130 + (Math.random() - 0.5) * 18;
  mVel[i * 3 + 1] = -(70 + Math.random() * 50);
  mVel[i * 3 + 2] = (Math.random() - 0.5) * 18;
  mLife[i] = 8;
  meteorCol[i * 3] = 1.0; meteorCol[i * 3 + 1] = 0.85 + Math.random() * 0.15; meteorCol[i * 3 + 2] = 0.55;
}

// height of city under a point (top of nearest building whose footprint we're over)
let _lastHitIdx = -1;
function groundOrBuildingHit(x, z, y) {
  if (y <= GROUND_Y + 1.5) return GROUND_Y;
  let best = -1, bestTop = GROUND_Y;
  for (let i = 0; i < cityCount; i++) {
    if (!bAlive[i]) continue;
    if (Math.abs(x - bx[i]) < bBaseW[i] * 0.6 && Math.abs(z - bz[i]) < bBaseD[i] * 0.6) {
      const top = by[i] + bH[i];
      if (y <= top + 4 && top > bestTop) { bestTop = top; best = i; }
    }
  }
  if (best >= 0) { _lastHitIdx = best; return bestTop; }
  return null;
}

function nearestBuilding(x, z, maxR) {
  let best = -1, bd = maxR * maxR;
  for (let i = 0; i < cityCount; i++) {
    if (!bAlive[i] || bBurn[i] >= 1) continue;
    const dx = x - bx[i], dz = z - bz[i], d2 = dx * dx + dz * dz;
    if (d2 < bd) { bd = d2; best = i; }
  }
  return best;
}

function stepMeteors(dt) {
  const g = params.g;
  for (let i = 0; i < MAX_METEORS; i++) {
    if (!mActive[i]) continue;
    mVel[i * 3 + 1] -= g * dt;                 // gravity
    mPos[i * 3] += mVel[i * 3] * dt;
    mPos[i * 3 + 1] += mVel[i * 3 + 1] * dt;
    mPos[i * 3 + 2] += mVel[i * 3 + 2] * dt;
    mLife[i] -= dt;
    if (Math.random() < 0.6) spawnEmberAt(mPos[i * 3], mPos[i * 3 + 1], mPos[i * 3 + 2], 1, 0.6); // trail

    _lastHitIdx = -1;
    const hitY = groundOrBuildingHit(mPos[i * 3], mPos[i * 3 + 2], mPos[i * 3 + 1]);
    const reached = hitY !== null && mPos[i * 3 + 1] <= hitY + 2;
    if (reached || mLife[i] <= 0 || mPos[i * 3 + 1] < GROUND_Y - 5) {
      const ix = mPos[i * 3], iy = hitY === null ? GROUND_Y : Math.max(GROUND_Y, hitY), iz = mPos[i * 3 + 2];
      let tgt = _lastHitIdx;
      if (tgt < 0) tgt = nearestBuilding(ix, iz, 26);
      if (tgt >= 0) damageBuilding(tgt, 0.42 + Math.random() * 0.3, true);
      else { spawnEmberBurst(ix, iy + 2, iz, 12); spawnSmoke(ix, iy + 2, iz, 4); flash(ix, iy + 8, iz); }
      meteorCount++;
      mActive[i] = 0;
      mPos[i * 3 + 1] = -9999;
    }
  }
  meteorGeom.attributes.position.needsUpdate = true;
  meteorGeom.attributes.color.needsUpdate = true;
}

// ===================================================================
// EMBERS — one THREE.Points pool (sparks from fire + impacts + trails).
// ===================================================================
const MAX_EMBERS = 4000;
let ePos, eVel, eLife, eMax, eActive, eHead = 0;
let emberGeom, emberPts, emberCol;

function buildEmbers() {
  if (emberPts) { scene.remove(emberPts); emberGeom.dispose(); emberPts.material.dispose(); }
  ePos = new Float32Array(MAX_EMBERS * 3);
  eVel = new Float32Array(MAX_EMBERS * 3);
  eLife = new Float32Array(MAX_EMBERS);
  eMax = new Float32Array(MAX_EMBERS);
  eActive = new Uint8Array(MAX_EMBERS);
  emberCol = new Float32Array(MAX_EMBERS * 3);
  for (let i = 0; i < MAX_EMBERS; i++) ePos[i * 3 + 1] = -9999;
  emberGeom = new THREE.BufferGeometry();
  emberGeom.setAttribute("position", new THREE.BufferAttribute(ePos, 3).setUsage(THREE.DynamicDrawUsage));
  emberGeom.setAttribute("color", new THREE.BufferAttribute(emberCol, 3));
  const mat = new THREE.PointsMaterial({
    size: 4.6, map: glowTex, vertexColors: true, transparent: true, opacity: 0.95,
    blending: THREE.AdditiveBlending, depthWrite: false, sizeAttenuation: true
  });
  emberPts = new THREE.Points(emberGeom, mat);
  emberPts.frustumCulled = false;
  scene.add(emberPts);
}

function spawnEmberAt(x, y, z, count, speed) {
  for (let c = 0; c < count; c++) {
    const i = eHead; eHead = (eHead + 1) % MAX_EMBERS;
    eActive[i] = 1;
    ePos[i * 3] = x + (Math.random() - 0.5) * 3;
    ePos[i * 3 + 1] = y;
    ePos[i * 3 + 2] = z + (Math.random() - 0.5) * 3;
    eVel[i * 3] = (Math.random() - 0.5) * 16 * speed;
    eVel[i * 3 + 1] = (8 + Math.random() * 30) * speed;
    eVel[i * 3 + 2] = (Math.random() - 0.5) * 16 * speed;
    eLife[i] = 0.6 + Math.random() * 1.2;
    eMax[i] = eLife[i];
  }
}
function spawnEmberBurst(x, y, z, count) {
  for (let c = 0; c < count; c++) {
    const i = eHead; eHead = (eHead + 1) % MAX_EMBERS;
    eActive[i] = 1;
    ePos[i * 3] = x; ePos[i * 3 + 1] = y; ePos[i * 3 + 2] = z;
    const a = Math.random() * Math.PI * 2, sp = 20 + Math.random() * 70, up = 0.4 + Math.random() * 0.9;
    eVel[i * 3] = Math.cos(a) * sp;
    eVel[i * 3 + 1] = sp * up;
    eVel[i * 3 + 2] = Math.sin(a) * sp;
    eLife[i] = 0.7 + Math.random() * 1.3;
    eMax[i] = eLife[i];
  }
}

function stepEmbers(dt) {
  for (let i = 0; i < MAX_EMBERS; i++) {
    if (!eActive[i]) continue;
    eVel[i * 3 + 1] -= 26 * dt;
    eVel[i * 3] *= 0.98; eVel[i * 3 + 2] *= 0.98;
    ePos[i * 3] += eVel[i * 3] * dt;
    ePos[i * 3 + 1] += eVel[i * 3 + 1] * dt;
    ePos[i * 3 + 2] += eVel[i * 3 + 2] * dt;
    eLife[i] -= dt;
    const t = Math.max(0, eLife[i] / eMax[i]); // 1 -> 0
    emberCol[i * 3] = 1.0;
    emberCol[i * 3 + 1] = 0.25 + t * 0.6;
    emberCol[i * 3 + 2] = t * t * 0.25;
    if (eLife[i] <= 0 || ePos[i * 3 + 1] < GROUND_Y - 4) { eActive[i] = 0; ePos[i * 3 + 1] = -9999; }
  }
  emberGeom.attributes.position.needsUpdate = true;
  emberGeom.attributes.color.needsUpdate = true;
}

// ===================================================================
// SMOKE — one THREE.Points pool (dark drifting plumes).
// ===================================================================
const MAX_SMOKE = 2600;
let sPos, sVel, sLife, sMax, sActive, sHead = 0;
let smokeGeom, smokePts, smokeCol;

function buildSmoke() {
  if (smokePts) { scene.remove(smokePts); smokeGeom.dispose(); smokePts.material.dispose(); }
  sPos = new Float32Array(MAX_SMOKE * 3);
  sVel = new Float32Array(MAX_SMOKE * 3);
  sLife = new Float32Array(MAX_SMOKE);
  sMax = new Float32Array(MAX_SMOKE);
  sActive = new Uint8Array(MAX_SMOKE);
  smokeCol = new Float32Array(MAX_SMOKE * 3);
  for (let i = 0; i < MAX_SMOKE; i++) sPos[i * 3 + 1] = -9999;
  smokeGeom = new THREE.BufferGeometry();
  smokeGeom.setAttribute("position", new THREE.BufferAttribute(sPos, 3).setUsage(THREE.DynamicDrawUsage));
  smokeGeom.setAttribute("color", new THREE.BufferAttribute(smokeCol, 3));
  const mat = new THREE.PointsMaterial({
    size: 34, map: softTex, vertexColors: true, transparent: true, opacity: 0.42,
    blending: THREE.NormalBlending, depthWrite: false, sizeAttenuation: true
  });
  smokePts = new THREE.Points(smokeGeom, mat);
  smokePts.frustumCulled = false;
  scene.add(smokePts);
}

function spawnSmoke(x, y, z, count) {
  for (let c = 0; c < count; c++) {
    const i = sHead; sHead = (sHead + 1) % MAX_SMOKE;
    sActive[i] = 1;
    sPos[i * 3] = x + (Math.random() - 0.5) * 8;
    sPos[i * 3 + 1] = y + Math.random() * 4;
    sPos[i * 3 + 2] = z + (Math.random() - 0.5) * 8;
    sVel[i * 3] = (Math.random() - 0.5) * 6 + params.angle * 10;
    sVel[i * 3 + 1] = 10 + Math.random() * 18;
    sVel[i * 3 + 2] = (Math.random() - 0.5) * 6;
    sLife[i] = 2.4 + Math.random() * 2.6;
    sMax[i] = sLife[i];
  }
}

function stepSmoke(dt) {
  for (let i = 0; i < MAX_SMOKE; i++) {
    if (!sActive[i]) continue;
    sVel[i * 3 + 1] *= 0.99;
    sPos[i * 3] += sVel[i * 3] * dt;
    sPos[i * 3 + 1] += sVel[i * 3 + 1] * dt;
    sPos[i * 3 + 2] += sVel[i * 3 + 2] * dt;
    sLife[i] -= dt;
    const t = Math.max(0, sLife[i] / sMax[i]); // 1 -> 0
    smokeCol[i * 3] = 0.10 + t * 0.22;
    smokeCol[i * 3 + 1] = 0.08 + t * 0.08;
    smokeCol[i * 3 + 2] = 0.08 + t * 0.02;
    if (sLife[i] <= 0) { sActive[i] = 0; sPos[i * 3 + 1] = -9999; }
  }
  smokeGeom.attributes.position.needsUpdate = true;
  smokeGeom.attributes.color.needsUpdate = true;
}

// ===================================================================
// IMPACT FLASH — a small ring of recycled PointLights pulsed per strike.
// ===================================================================
const FLASHES = 5;
let flashLights = [], flashLife = [], flashHead = 0;
function buildFlashes() {
  for (const l of flashLights) scene.remove(l);
  flashLights = []; flashLife = [];
  for (let i = 0; i < FLASHES; i++) {
    const l = new THREE.PointLight(0xffae5a, 0, 320, 2.0);
    l.position.set(0, -9999, 0);
    scene.add(l); flashLights.push(l); flashLife.push(0);
  }
}
function flash(x, y, z) {
  const i = flashHead; flashHead = (flashHead + 1) % FLASHES;
  flashLights[i].position.set(x, y, z);
  flashLights[i].intensity = 9 + Math.random() * 6;
  flashLife[i] = 0.32;
}
function stepFlashes(dt) {
  for (let i = 0; i < FLASHES; i++) {
    if (flashLife[i] > 0) {
      flashLife[i] -= dt;
      flashLights[i].intensity = Math.max(0, flashLights[i].intensity * (1 - dt * 7));
      if (flashLife[i] <= 0) { flashLights[i].intensity = 0; flashLights[i].position.y = -9999; }
    }
  }
}

// ===================================================================
// SKY — a hellish gradient dome + horizon glow + lights.
// ===================================================================
let skyDome, hellLight, ambient, horizonGlow;
function buildSky() {
  const g = new THREE.SphereGeometry(2600, 32, 16);
  const pos = g.attributes.position;
  const col = new Float32Array(pos.count * 3);
  for (let i = 0; i < pos.count; i++) {
    const y = pos.getY(i) / 2600;            // -1..1
    const h = Math.max(0, 1 - Math.abs(y));   // peaks at horizon
    const low = Math.max(0, -y);              // below-horizon glow
    col[i * 3] = 0.04 + h * 0.42 + low * 0.30;
    col[i * 3 + 1] = 0.01 + h * 0.10 + low * 0.06;
    col[i * 3 + 2] = 0.03 + h * 0.04;
  }
  g.setAttribute("color", new THREE.BufferAttribute(col, 3));
  skyDome = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.BackSide, fog: false }));
  scene.add(skyDome);

  hellLight = new THREE.DirectionalLight(0xff5a2a, 0.9);
  hellLight.position.set(60, 400, 120);
  scene.add(hellLight);
  ambient = new THREE.AmbientLight(0x40160c, 0.7);
  scene.add(ambient);
  horizonGlow = new THREE.PointLight(0xff7a30, 0.6, 1400, 2.0);
  horizonGlow.position.set(0, 18, 0);
  scene.add(horizonGlow);
}

// ===================================================================
// procedural sprite textures (no external assets).
// ===================================================================
function makeGlowTexture() {
  const c = document.createElement("canvas"); c.width = c.height = 64;
  const ctx = c.getContext("2d");
  const grd = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
  grd.addColorStop(0.0, "rgba(255,255,255,1)");
  grd.addColorStop(0.25, "rgba(255,220,150,0.95)");
  grd.addColorStop(0.6, "rgba(255,120,40,0.5)");
  grd.addColorStop(1.0, "rgba(255,60,10,0)");
  ctx.fillStyle = grd; ctx.fillRect(0, 0, 64, 64);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace;
  return t;
}
function makeSoftTexture() {
  const c = document.createElement("canvas"); c.width = c.height = 64;
  const ctx = c.getContext("2d");
  const grd = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
  grd.addColorStop(0.0, "rgba(255,255,255,0.9)");
  grd.addColorStop(0.5, "rgba(255,255,255,0.35)");
  grd.addColorStop(1.0, "rgba(255,255,255,0)");
  ctx.fillStyle = grd; ctx.fillRect(0, 0, 64, 64);
  return new THREE.CanvasTexture(c);
}
const glowTex = makeGlowTexture();
const softTex = makeSoftTexture();

// ===================================================================
// PILLAR OF SALT — a lone white column left standing (Lot's wife).
// ===================================================================
let pillar;
function buildPillar() {
  if (pillar) return;
  const g = new THREE.CylinderGeometry(4, 6, 46, 10);
  const m = new THREE.MeshStandardMaterial({ color: 0xeef0f4, roughness: 0.85, emissive: 0x222428, emissiveIntensity: 0.4 });
  pillar = new THREE.Mesh(g, m);
  pillar.position.set(CITY_HALF * 0.86, 23, -CITY_HALF * 0.78);
  scene.add(pillar);
}

// ===================================================================
// per-frame city bookkeeping: burning buildings emit + smolder out.
// ===================================================================
let paintAccum = 0;
function stepCity(dt) {
  let moved = false;
  for (let i = 0; i < cityCount; i++) {
    if (bFire[i] > 0) {
      bFire[i] = Math.max(0, bFire[i] - dt * 0.22);
      if (bBurn[i] < 0.999) {
        const wasStanding = true;
        bBurn[i] = Math.min(1, bBurn[i] + dt * 0.05);
        bH[i] = Math.max(bBaseH[i] * (1 - bBurn[i] * 0.86), bBaseH[i] * 0.08);
        if (bBurn[i] >= 0.999 && wasStanding) razed++;
        writeBuilding(i); moved = true;
      }
      if (Math.random() < 0.5 * params.intensity) spawnEmberAt(bx[i], by[i] + bH[i], bz[i], 1, 0.7);
      if (Math.random() < 0.35) spawnSmoke(bx[i], by[i] + bH[i], bz[i], 1);
    }
  }
  if (moved) city.instanceMatrix.needsUpdate = true;
  paintAccum += dt;
  if (paintAccum > 0.1) { // repaint a few times/sec (cheap)
    paintAccum = 0;
    for (let i = 0; i < cityCount; i++) paintBuilding(i);
    city.instanceColor.needsUpdate = true;
  }
  const frac = cityCount ? razed / cityCount : 0; // warm the world as it burns
  if (hellLight) hellLight.intensity = 0.9 + frac * 0.8;
  if (horizonGlow) horizonGlow.intensity = 0.6 + frac * 1.4;
}

// ===================================================================
// rain emission + full-destruction loop
// ===================================================================
let spawnAccum = 0, ruinHold = 0;
function emitRain(dt) {
  spawnAccum += 26 * params.rain * params.intensity * dt; // meteors/sec
  let budget = 6;
  while (spawnAccum >= 1 && budget-- > 0) { spawnMeteor(); spawnAccum -= 1; }
}
function maybeRebuild(dt) {
  const frac = cityCount ? razed / cityCount : 0;
  if (frac >= 0.985) { ruinHold += dt; if (ruinHold > 3.2) { ruinHold = 0; restart(); } }
  else ruinHold = 0;
}
function restart() {
  buildCity(params.cityN);
  for (let i = 0; i < MAX_METEORS; i++) { mActive[i] = 0; mPos[i * 3 + 1] = -9999; }
  meteorGeom.attributes.position.needsUpdate = true;
}

// ===================================================================
// panel / readout
// ===================================================================
let curPreset = "the judgment";
bindRange("rain", (v) => { params.rain = v; }, (v) => v.toFixed(1) + "×");
bindRange("city", (v) => { params.cityN = v | 0; buildCity(params.cityN); }, (v) => `${v | 0}`);

const PRESETS = [
  { name: "the judgment", rain: 1.0, intensity: 1.0, angle: 0.18, startRuin: 0.0 },
  { name: "firestorm",    rain: 2.4, intensity: 1.8, angle: 0.10, startRuin: 0.0 },
  { name: "the ruin",     rain: 0.7, intensity: 0.9, angle: 0.18, startRuin: 0.72 },
  { name: "rain of fire", rain: 1.4, intensity: 1.2, angle: 0.55, startRuin: 0.0 },
];
let PRESET_I = 0;
setVariantCycler((d) => {
  PRESET_I = (PRESET_I + d + PRESETS.length) % PRESETS.length;
  const p = PRESETS[PRESET_I];
  params.rain = p.rain; params.intensity = p.intensity; params.angle = p.angle; params.startRuin = p.startRuin;
  curPreset = p.name;
  if (booted) restart();
  return p.name;
});

const resetBtn = document.getElementById("reset");
if (resetBtn) resetBtn.addEventListener("click", () => restart());

// diag hook
window.__diag = () => JSON.stringify({
  preset: curPreset,
  destroyed: cityCount ? Math.round((razed / cityCount) * 100) : 0,
  impacts: meteorCount
});

// ===================================================================
// boot
// ===================================================================
let booted = false;
buildSky();
addGrid(scene, { size: 1000, divisions: 50, y: 0 }); // charred ground plane
buildEmbers();
buildSmoke();
buildMeteors();
buildFlashes();
buildCity(params.cityN);
buildPillar();
booted = true;
liftVeil();
onResize(renderer, camera);

const meter = fpsMeter(document.getElementById("fps"));
const pctEl = document.getElementById("pct");
const impEl = document.getElementById("imp");

loop((dt) => {
  meter(dt);
  controls.update();
  emitRain(dt);
  stepMeteors(dt);
  stepCity(dt);
  stepEmbers(dt);
  stepSmoke(dt);
  stepFlashes(dt);
  maybeRebuild(dt);
  if (pctEl) pctEl.textContent = cityCount ? Math.round((razed / cityCount) * 100) : 0;
  if (impEl) impEl.textContent = meteorCount;
  renderer.render(scene, camera);
});
