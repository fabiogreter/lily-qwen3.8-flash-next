import { describe, expect, it } from 'vitest';
import { ensoBand, flowPath, flowWidth, noise, strokePath } from './brush';

describe('brush geometry', () => {
  it('is deterministic', () => {
    expect(strokePath(3)).toBe(strokePath(3));
    expect(ensoBand(0.5, 21, 1)).toBe(ensoBand(0.5, 21, 1));
    const a = noise(5);
    const b = noise(5);
    expect([a(), a(), a()]).toEqual([b(), b(), b()]);
  });
  it('different seeds give different strokes', () => {
    expect(strokePath(1)).not.toBe(strokePath(2));
  });
  it('the ensō draws a closed path with finite coordinates even at 0 and 1', () => {
    for (const f of [0, 0.01, 0.5, 1]) {
      const d = ensoBand(f, 21, 1);
      expect(d.startsWith('M')).toBe(true);
      expect(d.endsWith('Z')).toBe(true);
      expect(d).not.toMatch(/NaN|Infinity/);
    }
  });
  it('flow stroke width grows with throughput; the path is well formed', () => {
    expect(flowWidth(0)).toBeLessThan(flowWidth(100));
    expect(flowWidth(-5)).toBe(flowWidth(0));
    expect(flowPath(130, 158, 166, flowWidth(80))).not.toMatch(/NaN/);
  });
});
