// audio.js — give the math a VOICE. A tiny shared WebAudio layer so any room
// can SING. Browsers block audio until a user gesture, so the AudioContext is
// created lazily on the first click/keypress and faded in. Press S to mute.
//
// A "voice" is osc → lowpass filter → gain → (stereo pan) → master → analyser →
// out. Rooms drive setFreq / setCutoff / setPan / setGain from their math each
// frame; the shared AnalyserNode lets a room draw the actual output waveform as
// a neon oscilloscope (so the sound is also SEEN, and lands in recordings).

let ctx = null, master = null, analyser = null, on = false, started = false;
const voices = [];

export function audioOn() { return on && ctx && ctx.state === "running"; }
export function getAnalyser() { return analyser; }

// a MediaStream audio track of the master mix, so the video recorder can capture sound
let streamDest = null;
export function getAudioStream() {
  if (!started || !ctx.createMediaStreamDestination) return null;
  if (!streamDest) { streamDest = ctx.createMediaStreamDestination(); master.connect(streamDest); }
  return streamDest.stream;
}

function build() {
  ctx = new (window.AudioContext || window.webkitAudioContext)();
  master = ctx.createGain(); master.gain.value = 0;
  analyser = ctx.createAnalyser(); analyser.fftSize = 2048; analyser.smoothingTimeConstant = 0.6;
  master.connect(analyser); analyser.connect(ctx.destination);
  started = true;
}

export function ensureAudio() {
  if (!started) build();
  if (ctx.state !== "running") ctx.resume();
  on = true;
  master.gain.setTargetAtTime(0.35, ctx.currentTime, 0.4);
  updateBadge();
}

export function muteToggle() {
  if (!started) { ensureAudio(); return; }
  on = !on;
  master.gain.setTargetAtTime(on ? 0.35 : 0.0, ctx.currentTime, 0.15);
  if (on && ctx.state !== "running") ctx.resume();
  updateBadge();
}

// a voice: osc -> filter -> gain -> pan -> master
export function makeVoice({ type = "sawtooth", freq = 220, cutoff = 1200 } = {}) {
  if (!started) build();
  const osc = ctx.createOscillator(); osc.type = type; osc.frequency.value = freq;
  const filt = ctx.createBiquadFilter(); filt.type = "lowpass"; filt.frequency.value = cutoff; filt.Q.value = 6;
  const g = ctx.createGain(); g.gain.value = 0;
  const pan = ctx.createStereoPanner ? ctx.createStereoPanner() : null;
  osc.connect(filt); filt.connect(g);
  if (pan) { g.connect(pan); pan.connect(master); } else { g.connect(master); }
  osc.start();
  const v = {
    setFreq: (f) => osc.frequency.setTargetAtTime(Math.max(20, f), ctx.currentTime, 0.03),
    setCutoff: (c) => filt.frequency.setTargetAtTime(Math.max(80, c), ctx.currentTime, 0.04),
    setPan: (p) => pan && pan.pan.setTargetAtTime(Math.max(-1, Math.min(1, p)), ctx.currentTime, 0.06),
    setGain: (x) => g.gain.setTargetAtTime(Math.max(0, x), ctx.currentTime, 0.06),
    osc, filt, g, pan,
  };
  voices.push(v);
  return v;
}

// musical helper: a pentatonic-minor ladder of frequencies over `octaves`
export function pentatonic(root = 110, octaves = 4) {
  const steps = [0, 3, 5, 7, 10], out = [];
  for (let o = 0; o < octaves; o++) for (const s of steps) out.push(root * Math.pow(2, (o * 12 + s) / 12));
  return out;
}

// ---------- the on-screen badge + S-to-mute ----------
let badge = null;
function updateBadge() {
  if (!badge) return;
  badge.textContent = audioOn() ? "♪ sound on · S" : (started ? "♪ muted · S" : "♪ click for sound");
  badge.classList.toggle("on", audioOn());
}
export function installAudioUI() {
  const setup = () => {
    badge = document.createElement("div");
    badge.className = "audio-badge";
    badge.textContent = "♪ click for sound";
    badge.addEventListener("click", ensureAudio);
    document.body.appendChild(badge);
    // first gesture anywhere wakes the audio (autoplay policy)
    window.addEventListener("pointerdown", ensureAudio, { once: true });
    window.addEventListener("keydown", (e) => {
      if (e.metaKey || e.ctrlKey) return;
      const t = document.activeElement; if (t && /INPUT|TEXTAREA|SELECT/.test(t.tagName)) return;
      if (e.key === "s" || e.key === "S") { e.preventDefault(); muteToggle(); }
    });
    updateBadge();
  };
  if (document.readyState !== "loading") setup();
  else window.addEventListener("DOMContentLoaded", setup);
}

// let the (audio-agnostic) recorder in common.js pull the sound track
window.__mwAudioStream = () => (audioOn() ? getAudioStream() : null);

// debug hook
window.__audio = () => JSON.stringify({ started, on, state: ctx ? ctx.state : "none", voices: voices.length });
