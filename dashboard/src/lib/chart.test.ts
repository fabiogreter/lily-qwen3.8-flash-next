import { describe, expect, it } from 'vitest';
import { axisValue, nearestIndex, niceMax, runs, seriesPaths, tickIndices } from './chart';

const x = (i: number): number => i * 10;
const y = (v: number): number => 100 - v;

describe('runs', () => {
  it('splits at nulls', () => {
    expect(runs([1, 2, null, 3, 4, 5, null, null, 6])).toEqual([
      [0, 1],
      [3, 5],
      [8, 8],
    ]);
  });
  it('handles all-null, empty and gapless input', () => {
    expect(runs([null, null])).toEqual([]);
    expect(runs([])).toEqual([]);
    expect(runs([1, 2, 3])).toEqual([[0, 2]]);
  });
});

describe('seriesPaths', () => {
  it('a gap splits the path into separate subpaths and is never bridged', () => {
    const p = seriesPaths([10, 20, null, 30, 40], x, y, 100, false);
    expect(p.d).toBe('M0.0 90.0 L10.0 80.0M30.0 70.0 L40.0 60.0');
    expect(p.d.match(/M/g)).toHaveLength(2);
    expect(p.dots).toEqual([]);
    expect(p.area).toBe('');
  });
  it('a lone bucket becomes a dot, not a line', () => {
    const p = seriesPaths([null, 10, null, 20, 30], x, y, 100, false);
    expect(p.dots).toEqual([[10, 90]]);
    expect(p.d).toBe('M30.0 80.0 L40.0 70.0');
  });
  it('only lone buckets: no line at all', () => {
    const p = seriesPaths([5, null, 6, null, 7], x, y, 100, true);
    expect(p.d).toBe('');
    expect(p.area).toBe('');
    expect(p.dots).toHaveLength(3);
  });
  it('null is not drawn as zero', () => {
    const p = seriesPaths([null, null], x, y, 100, true);
    expect(p).toEqual({ d: '', area: '', dots: [] });
  });
  it('the area closes to the baseline for each run', () => {
    const p = seriesPaths([10, 20, null, 30, 40], x, y, 100, true);
    expect(p.area).toBe('M0.0 100.0 L0.0 90.0 L10.0 80.0 L10.0 100.0ZM30.0 100.0 L30.0 70.0 L40.0 60.0 L40.0 100.0Z');
  });
});

describe('axis helpers', () => {
  it('niceMax rounds up to a readable axis maximum', () => {
    expect(niceMax(0)).toBe(1);
    expect(niceMax(152)).toBe(160);
    expect(niceMax(121)).toBe(160);
    expect(niceMax(3281)).toBe(4000);
    expect(niceMax(78)).toBe(80);
    expect(niceMax(0.7)).toBe(0.8);
  });
  it('axisValue formats ticks', () => {
    expect(axisValue(null, { top: 10 })).toBe('–');
    expect(axisValue(0.55, { top: 1, percent: true })).toBe('55%');
    expect(axisValue(3000, { top: 4000 })).toBe('3.0K');
    expect(axisValue(150, { top: 200 })).toBe('150');
    expect(axisValue(2.5, { top: 4 })).toBe('2.50');
    expect(axisValue(37.45, { top: 80 })).toBe('37.5');
  });
  it('nearestIndex clamps to the data', () => {
    expect(nearestIndex(0, 46, 500, 11)).toBe(0);
    expect(nearestIndex(46 + 250, 46, 500, 11)).toBe(5);
    expect(nearestIndex(9999, 46, 500, 11)).toBe(10);
    expect(nearestIndex(100, 46, 500, 1)).toBe(0);
  });
});

describe('tickIndices', () => {
  it('spreads labels evenly and includes both ends', () => {
    expect(tickIndices(61, 5)).toEqual([0, 15, 30, 45, 60]);
    expect(tickIndices(15, 3)).toEqual([0, 7, 14]);
  });
  it('never repeats an index for small data', () => {
    expect(tickIndices(1, 5)).toEqual([0]);
    expect(tickIndices(2, 5)).toEqual([0, 1]);
    expect(tickIndices(3, 5)).toEqual([0, 1, 2]);
    expect(tickIndices(4, 5)).toEqual([0, 1, 2, 3]);
    expect(tickIndices(0, 5)).toEqual([]);
    for (let n = 1; n < 12; n++) {
      const t = tickIndices(n, 5);
      expect(new Set(t).size).toBe(t.length);
    }
  });
});
