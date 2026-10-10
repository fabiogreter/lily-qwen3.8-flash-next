// Brush geometry for the ink look: section strokes, the ensō and the
// schematic's flow strokes. Deterministic (seeded), so a re-render never
// changes a stroke.

/** Deterministic noise in [0, 1) (a small LCG). */
export function noise(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a * 1664525 + 1013904223) >>> 0;
    return a / 4294967296;
  };
}

type Pt = readonly [number, number];
const poly = (pts: readonly Pt[], digits = 1): string => pts.map((p) => p[0].toFixed(digits) + ' ' + p[1].toFixed(digits)).join(' L');

/** A brushed horizontal stroke for section dividers, in a 1000 x 14 box: lands thick-ish, tapers to a dry end. */
export function strokePath(seed: number): string {
  const r = noise(seed * 97 + 13);
  const n = 40;
  const top: Pt[] = [];
  const bot: Pt[] = [];
  for (let i = 0; i <= n; i++) {
    const u = i / n;
    const x = u * 1000;
    const w = (u < 0.06 ? 2.2 + u * 30 : 4 * (1 - u) + 0.6) * (0.8 + 0.4 * r());
    const y = 7 + Math.sin(u * 3 + seed) * 0.8;
    top.push([x, y - w / 2]);
    bot.push([x, y + w / 2]);
  }
  return 'M' + poly(top, 3) + ' L' + poly(bot.reverse(), 3) + 'Z';
}

// The ensō, in a 320 x 320 box: a band of varying width around a circle.
const C = 160;
const R = 118;
const START = (200 * Math.PI) / 180;
const SWEEP = (346 * Math.PI) / 180;

/**
 * Filled outline of the ensō closed to `frac` (0..1 of the full sweep). The
 * brush touches down narrow with a rounded landing, swells into a small pool
 * of ink, settles, and lifts at the end.
 */
export function ensoBand(frac: number, seed: number, scale: number): string {
  const r = noise(seed);
  const f = Math.max(0.012, Math.min(1, frac));
  const n = Math.max(10, Math.round(220 * f));
  const outer: Pt[] = [];
  const inner: Pt[] = [];
  let w0 = 0;
  let rr0 = R;
  for (let i = 0; i <= n; i++) {
    const u = (i / n) * f; // progress over the full sweep
    const a = START + u * SWEEP;
    const local = i / n; // progress over what is drawn
    let w = 26 * (1 - 0.6 * u) * scale;
    const land = Math.min(1, u / 0.05);
    w *= 0.42 + 0.58 * Math.sin((land * Math.PI) / 2);
    w *= 1 + 0.2 * Math.exp(-Math.pow((u - 0.055) / 0.03, 2));
    if (local > 0.9) w *= Math.max(0.15, (1 - local) * 10);
    w *= 0.9 + 0.2 * r();
    const rr = R + Math.sin(u * 7 + seed) * 2.2;
    if (i === 0) {
      w0 = w;
      rr0 = rr;
    }
    outer.push([C + Math.cos(a) * (rr + w / 2), C + Math.sin(a) * (rr + w / 2)]);
    inner.push([C + Math.cos(a) * (rr - w / 2), C + Math.sin(a) * (rr - w / 2)]);
  }
  // Rounded cap behind the landing point, from the inner edge back to the outer.
  const cap: Pt[] = [];
  const c0: Pt = [C + Math.cos(START) * rr0, C + Math.sin(START) * rr0];
  const nrm: Pt = [Math.cos(START), Math.sin(START)];
  const tan: Pt = [-Math.sin(START), Math.cos(START)];
  for (let k = 1; k < 10; k++) {
    const phi = Math.PI - (k / 10) * Math.PI;
    const rad = w0 / 2;
    cap.push([
      c0[0] + nrm[0] * rad * Math.cos(phi) - tan[0] * rad * 1.3 * Math.sin(phi),
      c0[1] + nrm[1] * rad * Math.cos(phi) - tan[1] * rad * 1.3 * Math.sin(phi),
    ]);
  }
  return 'M' + poly([...outer, ...inner.reverse(), ...cap]) + 'Z';
}

/** The faint full circle behind the ensō (an arc over the whole sweep). */
export function ensoGhost(): string {
  const p0: Pt = [C + Math.cos(START) * R, C + Math.sin(START) * R];
  const p1: Pt = [C + Math.cos(START + SWEEP) * R, C + Math.sin(START + SWEEP) * R];
  return `M${p0[0]} ${p0[1]} A${R} ${R} 0 1 1 ${p1[0]} ${p1[1]}`;
}

/** Brush stroke width for a throughput on the schematic's request path. */
export function flowWidth(tps: number): number {
  return 1.4 + Math.sqrt(Math.max(0, tps || 0)) * 0.24;
}

/** A horizontal brush stroke from x1 to x2 at height y, ending in an arrowhead. */
export function flowPath(x1: number, y: number, x2: number, w: number): string {
  const head = Math.max(7, w * 1.3);
  const xe = x2 - head;
  const top: Pt[] = [];
  const bot: Pt[] = [];
  const n = 12;
  for (let i = 0; i <= n; i++) {
    const u = i / n;
    const x = x1 + (xe - x1) * u;
    const ww = w * (0.7 + 0.3 * Math.sin((Math.min(1, u * 3) * Math.PI) / 2)) * (1 - 0.12 * u);
    top.push([x, y - ww / 2]);
    bot.push([x, y + ww / 2]);
  }
  const hw = Math.max(w * 0.95, 5);
  return (
    'M' +
    poly(top) +
    ` L${xe.toFixed(1)} ${(y - hw).toFixed(1)} L${x2} ${y} L${xe.toFixed(1)} ${(y + hw).toFixed(1)} L` +
    poly(bot.reverse()) +
    'Z'
  );
}
