// hero.js — the landing-page hero: an endless neon wireframe terrain
// scrolling toward the camera under a sliced sun. Pure outrun.

import * as THREE from "three";
import { reducedMotion, addSun } from "./common.js";

const canvas = document.getElementById("hero");
if (canvas) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  const scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0x0a0118, 0.028);

  const camera = new THREE.PerspectiveCamera(62, 1, 0.1, 300);
  camera.position.set(0, 2.4, 9);
  camera.lookAt(0, 1.2, -20);

  const fit = () => {
    const r = canvas.getBoundingClientRect();
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(r.width, r.height, false);
    camera.aspect = r.width / r.height;
    camera.updateProjectionMatrix();
  };

  addSun(scene, { scale: 34, position: [0, 11, -64] });

  // --- grid terrain as pure quads (no diagonals) ---
  const COLS = 56;   // across (x)
  const ROWS = 90;   // into the distance (z)
  const SX = 1.15;   // spacing x
  const SZ = 1.25;   // spacing z
  const count = COLS * ROWS;

  const positions = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);

  // ridged hills on the sides, flat "road" valley down the middle
  function heightAt(x, zPhase) {
    const valley = Math.min(Math.abs(x) / 7, 1);             // 0 center → 1 edges
    const ridge =
      Math.sin(x * 0.55 + zPhase * 0.9) * 1.1 +
      Math.sin(x * 0.21 - zPhase * 0.6) * 1.6 +
      Math.sin(zPhase * 1.3 + x * 0.13) * 0.9;
    return ridge * valley * valley * 1.25;
  }

  function rebuild(scroll) {
    let p = 0;
    for (let r = 0; r < ROWS; r++) {
      const z = -r * SZ;
      const zPhase = r * SZ + scroll;
      for (let c = 0; c < COLS; c++) {
        const x = (c - (COLS - 1) / 2) * SX;
        const y = heightAt(x, zPhase);
        positions[p] = x; positions[p + 1] = y; positions[p + 2] = z;
        // color: cyan (near) → magenta (far), brighter on the peaks
        const depth = r / ROWS;
        const peak = Math.min(Math.abs(y) / 3, 1);
        colors[p]     = 0.16 + depth * 0.84 + peak * 0.0;
        colors[p + 1] = 0.82 - depth * 0.62;
        colors[p + 2] = 0.96 - depth * 0.36 + peak * 0.04;
        p += 3;
      }
    }
    geo.attributes.position.needsUpdate = true;
    geo.attributes.color.needsUpdate = true;
  }

  // line index: connect each node to its right + forward neighbor
  const idx = [];
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      const a = r * COLS + c;
      if (c < COLS - 1) idx.push(a, a + 1);
      if (r < ROWS - 1) idx.push(a, a + COLS);
    }
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(positions, 3).setUsage(THREE.DynamicDrawUsage));
  geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  geo.setIndex(idx);

  const terrain = new THREE.LineSegments(
    geo,
    new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.9 })
  );
  scene.add(terrain);

  fit();
  window.addEventListener("resize", fit);

  rebuild(0);
  if (reducedMotion) {
    renderer.render(scene, camera);
  } else {
    let scroll = 0;
    const animate = () => {
      scroll += 0.05;                     // roll the hills toward us
      rebuild(scroll % SZ === 0 ? scroll : scroll); // continuous phase scroll
      renderer.render(scene, camera);
      requestAnimationFrame(animate);
    };
    animate();
  }
}
