// ps1.js — a PlayStation-1 rendering aesthetic for ＭＡＴＨＷＡＶＥ, built from the
// ACTUAL PS1 hardware quirks (not a vague "retro" filter):
//
//   • VERTEX SNAPPING — the PS1 GPU had no floating-point; vertices were snapped to
//     a low-res integer grid in screen space. That quantization is the iconic
//     "wobble"/jitter as things move. Reproduced in the vertex shader by snapping
//     clip-space xy to a grid → ps1ify().
//   • LOW FRAMEBUFFER — render the scene into a tiny (~1/scale) target, then upscale
//     with NEAREST filtering → chunky pixels.
//   • 15-BIT COLOR + ORDERED DITHER — console output was 5 bits/channel with a 4×4
//     Bayer dither to hide banding. We quantize + Bayer-dither in the final pass.
//   • (Heavy distance fog is set per-scene to hide the short draw distance.)
//
// makePS1Pipeline(renderer, scene, camera, opts) returns { render, setSize } shaped
// like an EffectComposer, so a room can drop it in where bloom used to be.
// Refs: PSX/"PsxDither" shader lore, retro-rendering write-ups (Bisqwit et al.).

import * as THREE from "three";

// 4×4 Bayer matrix as a tiny tiling texture (avoids dynamic GLSL matrix indexing).
function bayerTexture() {
  const m = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5];
  const data = new Uint8Array(16 * 4);
  for (let i = 0; i < 16; i++) {
    const v = Math.round((m[i] / 16) * 255);
    data[i * 4] = v; data[i * 4 + 1] = v; data[i * 4 + 2] = v; data[i * 4 + 3] = 255;
  }
  const tex = new THREE.DataTexture(data, 4, 4, THREE.RGBAFormat);
  tex.minFilter = THREE.NearestFilter; tex.magFilter = THREE.NearestFilter;
  tex.wrapS = THREE.RepeatWrapping; tex.wrapT = THREE.RepeatWrapping;
  tex.needsUpdate = true;
  return tex;
}

export function makePS1Pipeline(renderer, scene, camera, { scale = 4, levels = 32, srgb = true } = {}) {
  const sizeFor = () => [
    Math.max(1, Math.floor(window.innerWidth / scale)),
    Math.max(1, Math.floor(window.innerHeight / scale)),
  ];
  let [w, h] = sizeFor();
  const rt = new THREE.WebGLRenderTarget(w, h, {
    minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter, depthBuffer: true,
  });

  const quadScene = new THREE.Scene();
  const quadCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  const mat = new THREE.ShaderMaterial({
    uniforms: {
      tDiffuse: { value: rt.texture },
      tBayer: { value: bayerTexture() },
      uLow: { value: new THREE.Vector2(w, h) },
      uLevels: { value: levels },
    },
    depthTest: false, depthWrite: false,
    vertexShader: `
      varying vec2 vUv;
      void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }
    `,
    fragmentShader: `
      precision highp float;
      uniform sampler2D tDiffuse, tBayer;
      uniform vec2 uLow; uniform float uLevels;
      varying vec2 vUv;
      void main(){
        vec3 c = texture2D(tDiffuse, vUv).rgb;
        c = clamp(c, 0.0, 1.0);
        ${srgb ? "c = pow(c, vec3(0.4545));   // linear → sRGB (Three skips output encoding for raw ShaderMaterial; skip this when the source shader already outputs display colour)" : "// (srgb:false) source already outputs display colour — no extra gamma"}
        vec2 tp = floor(vUv * uLow);              // low-res texel coordinate
        float b = texture2D(tBayer, tp / 4.0).r;  // ordered-dither threshold, tiles every 4
        c += (b - 0.5) / uLevels;                 // Bayer dither in display space
        c = floor(c * uLevels) / uLevels;         // crunch to ~15-bit color
        gl_FragColor = vec4(clamp(c, 0.0, 1.0), 1.0);
      }
    `,
  });
  quadScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), mat));

  function render() {
    renderer.setRenderTarget(rt);
    renderer.render(scene, camera);
    renderer.setRenderTarget(null);
    renderer.render(quadScene, quadCam);
  }
  function setSize() {
    [w, h] = sizeFor();
    rt.setSize(w, h);
    mat.uniforms.uLow.value.set(w, h);
  }
  return { render, setSize, renderTarget: rt };
}

// Inject PS1 vertex-snapping into any material via onBeforeCompile. `snap` is the
// snap-grid resolution in clip space (smaller = chunkier wobble; ~200 reads well).
export function ps1ify(material, { snap = 200 } = {}) {
  material.onBeforeCompile = (shader) => {
    shader.uniforms.uSnap = { value: snap };
    shader.vertexShader =
      "uniform float uSnap;\n" +
      shader.vertexShader.replace(
        "#include <project_vertex>",
        `#include <project_vertex>
        {
          float _w = max(gl_Position.w, 1e-4);
          vec2 _g = vec2(uSnap);
          gl_Position.xy = floor((gl_Position.xy / _w) * _g) / _g * _w;  // PS1 vertex snap
        }`
      );
  };
  material.needsUpdate = true;
  return material;
}
