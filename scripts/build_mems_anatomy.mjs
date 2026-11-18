import { writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import * as k from "/Users/kiwi/.gemini/config/skills/anatomy/kit/iso-kit.mjs";
import { kitScript } from "/Users/kiwi/.gemini/config/skills/anatomy/scripts/inline-kit.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = resolve(HERE, "..");
const theme = process.argv.includes("--light") ? "light" : "dark";

const WIDTH = 760;
const HEIGHT = 520;

// Bench / Base plate
const BASE = { x: -150, y: -90, w: 300, d: 180, r: 10 };
const BASE_H = 12;
const FOOT_H = 6;

// Sternal surface pad (tissue mimic)
const PAD = { x: -110, y: -65, w: 220, d: 130, r: 8 };
const PAD_Z = 0;
const PAD_H = 4;

// Smartphone chassis (placed on chest)
const PHONE = { x: -70, y: -40, w: 140, d: 80, r: 6 };
const PHONE_Z = PAD_Z + PAD_H;
const PHONE_H = 8;

// Cutaway MEMS sensor package inside phone
const CHIP = { x: -20, y: -18, w: 40, d: 36, r: 2 };
const CHIP_Z = PHONE_Z + PHONE_H;
const CHIP_H = 5;

// Suspended proof mass inside MEMS chamber
const MASS = { x: -12, y: -10, w: 24, d: 20, r: 1.5 };
const MASS_Z = CHIP_Z + 1.2;
const MASS_H = 2.4;

const P = k.fitProjection(
  [
    ...k.boxCorners(BASE, -BASE_H - FOOT_H, 0),
    [BASE.x, BASE.y, 0],
    [BASE.x + BASE.w, BASE.y + BASE.d, 0],
    [PHONE.x, PHONE.y, PHONE_Z + PHONE_H],
    [PHONE.x + PHONE.w, PHONE.y + PHONE.d, PHONE_Z + PHONE_H],
    [CHIP.x + CHIP.w + 20, CHIP.y + CHIP.d + 20, CHIP_Z + CHIP_H + 25]
  ],
  WIDTH,
  HEIGHT,
  { pad: 40, azimuth: 44, elevation: 32 }
);

const at = (point) => k.iso(point, P);
const S = (paths, style) => k.solidSvg(paths, style);
const L = (d, style) => k.lineSvg(d, style);
const D = (points, style) => k.dotsSvg(points, style);
const byDepth = (points) => [...points].sort((a, b) => k.depthOf([a[0], a[1], 0], P) - k.depthOf([b[0], b[1], 0], P));

// 1. Heavy Base Plate & Vibration Isolation Feet
function drawBase() {
  const out = [`<path class="iso-halo" d="${k.haloOf(BASE, -BASE_H - FOOT_H, BASE_H, P)}"/>`];
  
  // 4 Vibration Isolation Bushing Feet
  for (const [x, y] of byDepth(k.corners(BASE, 26))) {
    out.push(S(k.cylinder(x, y, 9, -BASE_H - FOOT_H, 2.2, P, 28), { tone: "lo" }));
    out.push(S(k.cylinder(x, y, 7.2, -BASE_H - FOOT_H + 2.2, FOOT_H - 2.2, P, 28), { tone: "mid" }));
  }
  
  // Base slab
  out.push(S(k.slabOf(BASE, -BASE_H, BASE_H, P, 8, 1.8), { tone: "mid" }));
  out.push(L(k.planOutline(k.insetPlan(BASE, 6), 0, P), { tone: "lo" }));
  
  // Corner mounting screws
  const screws = k.corners(BASE, 12);
  out.push(L(screws.map(([x, y]) => k.ring(x, y, 2.4, 0, P, 16)).join(""), { tone: "lo" }));
  out.push(D(screws.map(([x, y]) => at([x, y, 0])), { size: 0.55 }));
  
  // Front engraved calibration strip
  const strip = { x: -130, y: BASE.y + BASE.d - 16, w: 260, d: 8, r: 1.2 };
  out.push(S(k.slabOf(strip, 0, 0.6, P, 3), { tone: "lo", crease: "none" }));
  const ticks = k.topTicks(strip.x + 6, strip.x + strip.w - 6, 5, 5, strip.y + 1, 0.6, [2.2, 4.2], P);
  out.push(L(ticks.minor, { tone: "lo" }), L(ticks.major, { tone: "mid" }));
  
  return out.join("");
}

// 2. Thoracic Sternal Bed & Tissue Interface Pad
function drawSternumPad() {
  const out = [];
  out.push(S(k.slabOf(PAD, PAD_Z, PAD_H, P, 6, 1.2), { tone: "lo" }));
  out.push(L(k.planOutline(k.insetPlan(PAD, 4), PAD_Z + PAD_H, P), { tone: "faint" }));
  
  // Sternal ridge guideline (Costal cartilage markers)
  for (const y of [-40, -15, 15, 40]) {
    const p1 = at([-95, y, PAD_Z + PAD_H]);
    const p2 = at([-75, y, PAD_Z + PAD_H]);
    const p3 = at([75, y, PAD_Z + PAD_H]);
    const p4 = at([95, y, PAD_Z + PAD_H]);
    out.push(L(`M${p1[0]},${p1[1]}L${p2[0]},${p2[1]}M${p3[0]},${p3[1]}L${p4[0]},${p4[1]}`, { tone: "faint" }));
  }
  return out.join("");
}

// 3. Phone Chassis with Bevels and Display Window
function drawPhoneChassis() {
  const out = [];
  // Phone body
  out.push(S(k.slabOf(PHONE, PHONE_Z, PHONE_H, P, 6, 1.4), { tone: "mid" }));
  
  // Screen perimeter bezel
  const screen = k.insetPlan(PHONE, 5);
  out.push(S(k.slabOf(screen, PHONE_Z + PHONE_H, 0.4, P, 4), { tone: "lo", crease: "none" }));
  out.push(L(k.planOutline(k.insetPlan(screen, 2), PHONE_Z + PHONE_H + 0.4, P), { tone: "hi" }));
  
  // Top earpiece slot & camera dot
  out.push(L(`M${at([-15, PHONE.y + 3, PHONE_Z + PHONE_H])[0]},${at([-15, PHONE.y + 3, PHONE_Z + PHONE_H])[1]}L${at([15, PHONE.y + 3, PHONE_Z + PHONE_H])[0]},${at([15, PHONE.y + 3, PHONE_Z + PHONE_H])[1]}`, { tone: "lo" }));
  out.push(D([at([0, PHONE.y + 2, PHONE_Z + PHONE_H])], { size: 0.5, tone: "lo" }));
  
  return out.join("");
}

// 4. Cutaway MEMS Chip Package Housing
function drawMemsChamber() {
  const out = [];
  // Ceramic MEMS enclosure
  out.push(S(k.slabOf(CHIP, CHIP_Z, CHIP_H, P, 3, 0.8), { tone: "mid" }));
  
  // Internal cavity cutaway where proof mass hangs
  const cavity = k.insetPlan(CHIP, 3.5);
  out.push(L(k.planOutline(cavity, CHIP_Z + CHIP_H, P), { tone: "hi" }));
  
  // 4 Corner Gold Wirebonds connecting MEMS die to ASIC
  const cb = k.corners(CHIP, 2.5);
  for (const [cx, cy] of cb) {
    out.push(D([at([cx, cy, CHIP_Z + CHIP_H])], { size: 0.6, tone: "lit" }));
  }
  
  // Differential capacitive sensing combs (Fixed Stator Electrodes)
  // Left comb fingers (X-axis sense)
  for (let dy = -7; dy <= 7; dy += 3.5) {
    const f1 = at([CHIP.x + 4, CHIP.y + CHIP.d / 2 + dy, CHIP_Z + 1.8]);
    const f2 = at([CHIP.x + 9, CHIP.y + CHIP.d / 2 + dy, CHIP_Z + 1.8]);
    out.push(L(`M${f1[0]},${f1[1]}L${f2[0]},${f2[1]}`, { tone: "lo" }));
  }
  // Right comb fingers (X-axis sense)
  for (let dy = -7; dy <= 7; dy += 3.5) {
    const f1 = at([CHIP.x + CHIP.w - 4, CHIP.y + CHIP.d / 2 + dy, CHIP_Z + 1.8]);
    const f2 = at([CHIP.x + CHIP.w - 9, CHIP.y + CHIP.d / 2 + dy, CHIP_Z + 1.8]);
    out.push(L(`M${f1[0]},${f1[1]}L${f2[0]},${f2[1]}`, { tone: "lo" }));
  }
  
  return out.join("");
}

// 5. Moving Proof Mass & Suspension Spring Assembly (Live animated part)
function drawProofMass() {
  const out = [];
  // Suspended polysilicon proof mass (The Lit Focal Element)
  out.push(S(k.slabOf(MASS, 0, MASS_H, P, 2, 0.4), { tone: "lit" }));
  
  // Moving capacitive comb teeth extending from mass
  for (let dy = -5.5; dy <= 5.5; dy += 3.5) {
    const l1 = at([MASS.x, MASS.y + MASS.d / 2 + dy, MASS_H * 0.5]);
    const l2 = at([MASS.x - 3.5, MASS.y + MASS.d / 2 + dy, MASS_H * 0.5]);
    const r1 = at([MASS.x + MASS.w, MASS.y + MASS.d / 2 + dy, MASS_H * 0.5]);
    const r2 = at([MASS.x + MASS.w + 3.5, MASS.y + MASS.d / 2 + dy, MASS_H * 0.5]);
    out.push(L(`M${l1[0]},${l1[1]}L${l2[0]},${l2[1]}M${r1[0]},${r1[1]}L${r2[0]},${r2[1]}`, { tone: "hi" }));
  }
  
  // Center fiducial etched on seismic mass
  const mc = at([MASS.x + MASS.w / 2, MASS.y + MASS.d / 2, MASS_H]);
  out.push(D([mc], { size: 0.7, tone: "lit" }));
  out.push(L(k.ring(MASS.x + MASS.w / 2, MASS.y + MASS.d / 2, 2.8, MASS_H, P, 16), { tone: "mid" }));
  
  return out.join("");
}

// 6. 3-Axis Orthogonal Coordinate Reticle (X, Y, Z Indicators)
function drawAxesReticle() {
  const out = [];
  const origin = [-85, 45, 10];
  const o_iso = at(origin);
  const x_iso = at([origin[0] + 32, origin[1], origin[1] > 0 ? origin[2] : origin[2]]);
  const y_iso = at([origin[0], origin[1] + 32, origin[2]]);
  const z_iso = at([origin[0], origin[1], origin[2] + 32]);
  
  // Axis lines
  out.push(L(`M${o_iso[0]},${o_iso[1]}L${x_iso[0]},${x_iso[1]}`, { tone: "mid" }));
  out.push(L(`M${o_iso[0]},${o_iso[1]}L${y_iso[0]},${y_iso[1]}`, { tone: "mid" }));
  out.push(L(`M${o_iso[0]},${o_iso[1]}L${z_iso[0]},${z_iso[1]}`, { tone: "hi" }));
  
  // Terminal balls
  out.push(D([x_iso, y_iso, z_iso], { size: 0.65, tone: "lit" }));
  
  return out.join("");
}

// Assemble Static SVG and Dynamic Groups
const staticSvg = [
  drawBase(),
  drawSternumPad(),
  drawPhoneChassis(),
  drawMemsChamber(),
  drawAxesReticle()
].join("");

const massSvg = drawProofMass();

// Dial Callout & Gauge Face for Aortic Ejection Recoil
const GAUGE = [105, -35];
const GAUGE_R = 24;
const GAUGE_Z = 65;
const GAUGE_H = 8;

function drawDialStand() {
  const out = [];
  // Support pillar from bench
  out.push(S(k.cylinder(GAUGE[0], GAUGE[1], 4.5, 0, GAUGE_Z, P, 24), { tone: "mid" }));
  // Gauge bezel casing
  out.push(S(k.cylinder(GAUGE[0], GAUGE[1], GAUGE_R, GAUGE_Z, GAUGE_H, P, 36), { tone: "mid" }));
  // Glass dial face recessed
  out.push(L(k.ring(GAUGE[0], GAUGE[1], GAUGE_R - 2.5, GAUGE_Z + GAUGE_H, P, 36), { tone: "lo" }));
  // Dial ticks (0 to 100 mG)
  // radialTicks(cx, cy, radius, count, every, z, lengths, projection)
  const ticks = k.radialTicks(GAUGE[0], GAUGE[1], GAUGE_R - 3.5, 24, 6, GAUGE_Z + GAUGE_H, [2.0, 3.8], P);
  out.push(L(ticks.minor, { tone: "lo" }), L(ticks.major, { tone: "mid" }));
  return out.join("");
}

const needlePath = (theta) => {
  const tipX = GAUGE[0] + (GAUGE_R - 5) * Math.sin(theta);
  const tipY = GAUGE[1] - (GAUGE_R - 5) * Math.cos(theta);
  const p1 = at([GAUGE[0], GAUGE[1], GAUGE_Z + GAUGE_H + 0.4]);
  const p2 = at([tipX, tipY, GAUGE_Z + GAUGE_H + 0.4]);
  return `M${p1[0]},${p1[1]}L${p2[0]},${p2[1]}`;
};

const dialSvg = drawDialStand();

const svg = k.figureSvg({
  width: WIDTH,
  height: HEIGHT,
  body: [
    staticSvg,
    dialSvg,
    `<g data-part="needle" class="iso-lit">${L(needlePath(0), { tone: "lit" })}</g>`,
    `<g data-part="mems-mass">${massSvg}</g>`
  ]
});

// Interactive Physical Simulation Driver Script
const live = `
const P = ${JSON.stringify(P)};
const MASS_Z_BASE = ${MASS_Z};
const GAUGE = ${JSON.stringify(GAUGE)};
const GAUGE_R = ${GAUGE_R};
const GAUGE_Z = ${GAUGE_Z};
const GAUGE_H = ${GAUGE_H};

const stage = document.querySelector(".iso-stage");
const svgEl = stage.querySelector("svg");
const massEl = svgEl.querySelector('[data-part="mems-mass"]');
const needleEl = svgEl.querySelector('[data-part="needle"] path');
const readout = document.querySelector("[data-readout]");

let t = 0;
let lastTime = 0;
let heartRateBpm = 72;
let mode = 'normal'; // 'normal', 'ao_click', 'respiration'

// Physiological cardiac mechanics model:
// SCG: Sternal displacement from Aortic Valve Opening (AO) & Isovolumetric Contraction
function scgAcceleration(time) {
  const period = 60 / heartRateBpm;
  const phase = (time % period) / period; // 0 to 1
  
  // AO (Aortic Opening) sharp recoil wave around phase = 0.12 - 0.22
  let a_ao = 0;
  if (phase > 0.10 && phase < 0.28) {
    const sub = (phase - 0.10) / 0.18;
    a_ao = Math.sin(sub * Math.PI * 4) * Math.exp(-sub * 3.5) * 1.8;
  }
  
  // AC (Aortic Closure) secondary wave around phase = 0.42 - 0.52
  let a_ac = 0;
  if (phase > 0.42 && phase < 0.56) {
    const sub = (phase - 0.42) / 0.14;
    a_ac = Math.sin(sub * Math.PI * 3) * Math.exp(-sub * 3.0) * 0.7;
  }
  
  // Thoracic respiratory baseline heave (0.25 Hz)
  const a_resp = Math.sin(time * 2 * Math.PI * 0.25) * 0.35;
  
  return {
    z_acc: a_ao + a_ac + a_resp,
    ao_active: Math.abs(a_ao) > 0.4,
    resp: a_resp
  };
}

function needleAt(theta) {
  const tipX = GAUGE[0] + (GAUGE_R - 5) * Math.sin(theta);
  const tipY = GAUGE[1] - (GAUGE_R - 5) * Math.cos(theta);
  const p1 = iso([GAUGE[0], GAUGE[1], GAUGE_Z + GAUGE_H + 0.4], P);
  const p2 = iso([tipX, tipY, GAUGE_Z + GAUGE_H + 0.4], P);
  return "M" + p1[0] + "," + p1[1] + "L" + p2[0] + "," + p2[1];
}

function renderFrame(now) {
  const dt = lastTime ? Math.min((now - lastTime) / 1000, 0.05) : 0.016;
  lastTime = now;
  t += dt;
  
  const sim = scgAcceleration(t);
  
  // Z-displacement of proof mass (in micro-units translated to world units)
  const zOffset = sim.z_acc * 3.8;
  // Inertial X tilt from transverse shear
  const xOffset = Math.sin(t * 1.5) * 0.6;
  
  // Translate proof mass inside MEMS chamber
  massEl.setAttribute("transform", translateAlong(P, [xOffset, 0, MASS_Z_BASE + zOffset]));
  
  // Needle tracks recoil magnitude in mG
  const needleAngle = (sim.z_acc / 2.0) * (Math.PI * 0.7);
  needleEl.setAttribute("d", needleAt(needleAngle));
  
  // Dynamic physiological readout
  const statusLabel = sim.ao_active ? "AO Recoil (Aortic Ejection Click)" : (Math.abs(sim.resp) > 0.2 ? "Thoracic Breathing Wave" : "Diastolic Ventricular Filling");
  const accValue = (Math.abs(sim.z_acc) * 42.5).toFixed(1);
  const text = statusLabel + " · Z-Acc: " + (sim.z_acc > 0 ? "+" : "") + accValue + " mG";
  
  if (readout.textContent !== text) {
    readout.textContent = text;
  }
  
  requestAnimationFrame(renderFrame);
}

requestAnimationFrame(renderFrame);

// Interactive mouse scrubbing to simulate postural tilt
stage.addEventListener("pointermove", (e) => {
  const rect = stage.getBoundingClientRect();
  const relX = (e.clientX - rect.left) / rect.width;
  heartRateBpm = 60 + relX * 45; // 60 to 105 BPM
});
`;

const body = k.plateHtml({
  fig: "Fig 2",
  title: "SCG MEMS Mechanics",
  hint: "Hover: modulate BPM",
  readout: "Diastolic Filling · Z: +14 mG",
  keys: [
    { mark: "lit", label: "Polysilicon Proof Mass: deflects vertically along Z-axis under aortic blood ejection recoil" },
    { mark: "raised", label: "Differential Capacitive Combs: transduces sub-milligravitational micro-displacements into pico-Farad shifts" },
    { mark: "wire", label: "Costal Cartilage Bed: sternal bone plate coupling mechanical cardiac impulses directly to phone chassis" }
  ],
  caption:
    "Anatomy of chest-placed tri-axial accelerometer sensing. When the left ventricle contracts during cardiac systole, blood accelerates into the ascending aorta. Newton's third law drives a micro-vibrational recoil wave (1.5–15 Hz) through the sternum into the smartphone chassis. The suspended MEMS proof mass deflects relative to fixed stator combs, modulating capacitance ΔC = εA/d. Member B's zero-phase Butterworth filter eliminates DC gravity (1g) and high-frequency tremor, isolating genuine systolic AO peaks.",
  body: `<div class="iso-stage" tabindex="0" role="img" aria-label="Isometric technical anatomy of smartphone MEMS accelerometer recording cardiac sternal recoil">${svg}</div>`
});

const customCss = `
<style>
.iso-page { width: min(840px, 94vw) !important; margin: 40px auto 60px !important; }
.iso-plate { padding: 44px 20px 48px !important; }
.iso-plate-corners { inset: 14px 18px !important; }
.iso-plate-corner { font-size: 10px !important; max-width: 48%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.iso-plate-corner[data-corner="title"] { text-align: right; }
.iso-plate-corner[data-corner="readout"] { text-align: right; }
</style>
`;

const htmlContent = k.pageHtml({
  title: "Anatomy: Sternal Accelerometer Cardiac Sensing",
  theme,
  body,
  script: kitScript() + live,
  width: 840
}).replace("</head>", customCss + "</head>");

const outPath = join(PROJECT_ROOT, "public", theme === "light" ? "mems_scg_anatomy-light.html" : "mems_scg_anatomy.html");
writeFileSync(outPath, htmlContent);
console.log("Wrote", outPath);
