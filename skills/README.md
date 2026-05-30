# mathwave skills

Reusable build knowledge for [mathwave](../) — the buildless Three.js gallery of
mathematical/scientific visualizations. Each skill captures **how a class of room
is built**, so a new room can be added without rediscovering the conventions.

These are plain Markdown `SKILL.md` files with YAML frontmatter (`name` +
`description`). To have **Claude Code auto-load** them while you work in this repo,
copy or symlink this folder into `.claude/skills/`:

```bash
mkdir -p .claude && ln -s ../skills .claude/skills
```

## The skills

| Skill | Use it when… |
|-------|--------------|
| [mathwave-overview](mathwave-overview/SKILL.md) | starting any new room — conventions, file layout, kiosk nav, the `ROOMS` array |
| [raymarched-shader-room](raymarched-shader-room/SKILL.md) | the room is a fragment-shader SDF/raymarch (fractals, volumes) — template: `src/fractal.js` |
| [gpu-pingpong-sim-room](gpu-pingpong-sim-room/SKILL.md) | the room is a GPU field simulation on swapped float textures — template: `src/reaction.js` |
| [cpu-physics-3d-room](cpu-physics-3d-room/SKILL.md) | the room is CPU physics on real 3D geometry — template: `src/nbody.js` |
| [grid-cells-room](grid-cells-room/SKILL.md) | the room is a CPU integer/cell grid shown via a DataTexture — template: `src/wolfram.js` |
| [ps1-pipeline](ps1-pipeline/SKILL.md) | you want the PlayStation-1 look (low-res + dither + vertex snap) — `src/ps1.js` |
| [verify-on-screen](verify-on-screen/SKILL.md) | before claiming a room works — the canary discipline that catches silent failures |

## Third-party Three.js / shader skills (found online)

Other people have published Claude Code skills for Three.js and GLSL. License status
matters for a public repo:

- **[freshtechbro/claudedesignskills](https://github.com/freshtechbro/claudedesignskills)** — **MIT**.
  The 3D-relevant skills are vendored here under [`third-party/claudedesignskills/`](third-party/claudedesignskills/)
  with the original LICENSE preserved.
- **[CloudAI-X/threejs-skills](https://github.com/CloudAI-X/threejs-skills)** — substantive
  (10 skills: fundamentals, shaders, lighting, materials, post-processing…), but **no LICENSE
  file** → all rights reserved. **Not copied** — clone it yourself if you want it.
- **[dgreenheck/webgpu-claude-skill](https://github.com/dgreenheck/webgpu-claude-skill)** —
  WebGPU + TSL (Three r183+), **no LICENSE file**. **Not copied** — link only.
- **[VoltAgent/awesome-agent-skills](https://github.com/VoltAgent/awesome-agent-skills)** —
  a 1000+ community-skill index, if you want more.

> Why not just copy them all? Code without a license is **all-rights-reserved** by
> default — vendoring it into a public repo isn't permitted. Only the MIT-licensed
> set is safe to include.
