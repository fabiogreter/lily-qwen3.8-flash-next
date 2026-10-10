// Chart geometry. Pure, so the gap rules are testable without a DOM.

export type Num = number | null | undefined;

/** The smallest "nice" axis maximum that holds `v`. */
export function niceMax(v: number): number {
  if (!v || v <= 0) return 1;
  const p = Math.pow(10, Math.floor(Math.log10(v)));
  // Maxima whose quarters are clean (the axis has ticks at 0, 1/4, 1/2, 3/4, 1).
  for (const m of [1, 1.2, 1.6, 2, 2.4, 3.2, 4, 6, 8, 10]) if (m * p >= v) return m * p;
  return 10 * p;
}

/** Runs of consecutive non-null values, as [start, end] index pairs (inclusive). A null is a gap. */
export function runs(values: readonly Num[]): [number, number][] {
  const out: [number, number][] = [];
  let start = -1;
  values.forEach((v, i) => {
    if (v == null) {
      if (start >= 0) out.push([start, i - 1]);
      start = -1;
    } else if (start < 0) {
      start = i;
    }
  });
  if (start >= 0) out.push([start, values.length - 1]);
  return out;
}

export interface SeriesPaths {
  /** Line path: one `M` subpath per run of two or more buckets. */
  d: string;
  /** Closed area path under the line, same runs; empty unless requested. */
  area: string;
  /** Isolated buckets (a run of one): drawn as dots, since a line needs two points. */
  dots: [number, number][];
}

const f = (n: number): string => n.toFixed(1);

/**
 * Paths for one series. `x(i)` and `y(v)` map a bucket index and a value to
 * pixels; `baseY` is where the area closes. A null bucket splits the line: it
 * is a gap, never interpolated across and never drawn as zero.
 */
export function seriesPaths(
  values: readonly Num[],
  x: (i: number) => number,
  y: (v: number) => number,
  baseY: number,
  withArea: boolean,
): SeriesPaths {
  let d = '';
  let area = '';
  const dots: [number, number][] = [];
  for (const [a, b] of runs(values)) {
    const pts: [number, number][] = [];
    for (let i = a; i <= b; i++) pts.push([x(i), y(values[i] as number)]);
    const first = pts[0] as [number, number];
    if (pts.length === 1) {
      dots.push(first);
      continue;
    }
    const last = pts[pts.length - 1] as [number, number];
    const line = pts.map((p) => f(p[0]) + ' ' + f(p[1])).join(' L');
    d += 'M' + line;
    if (withArea) area += `M${f(first[0])} ${f(baseY)} L${line} L${f(last[0])} ${f(baseY)}Z`;
  }
  return { d, area, dots };
}

/** Axis value label. */
export function axisValue(v: Num, opts: { percent?: boolean; top: number }): string {
  if (v == null) return '–';
  if (opts.percent) return Math.round(v * 100) + '%';
  if (v >= 1000) return (v / 1000).toFixed(1) + 'K';
  if (v >= 100) return String(Math.round(v));
  return opts.top <= 5 ? v.toFixed(2) : String(Math.round(v * 10) / 10);
}

/** Index of the bucket nearest to pixel `px`, clamped to the data. */
export function nearestIndex(px: number, left: number, innerWidth: number, count: number): number {
  if (count <= 1 || innerWidth <= 0) return 0;
  const i = Math.round(((px - left) / innerWidth) * (count - 1));
  return Math.max(0, Math.min(count - 1, i));
}

/**
 * Bucket indices for `count` evenly spaced x-axis labels over `n` buckets,
 * without repeats (one bucket gives one label, two give two).
 */
export function tickIndices(n: number, count: number): number[] {
  if (n <= 0) return [];
  const k = Math.max(1, Math.min(count, n));
  if (k === 1) return [0];
  const out: number[] = [];
  for (let i = 0; i < k; i++) {
    const idx = Math.round((i * (n - 1)) / (k - 1));
    if (out[out.length - 1] !== idx) out.push(idx);
  }
  return out;
}
