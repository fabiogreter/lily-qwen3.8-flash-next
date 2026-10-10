import { describe, expect, it } from 'vitest';
import { hashFor, parseHash, VIEWS } from './view';

describe('view hash', () => {
  it('round-trips every view', () => {
    for (const v of VIEWS) expect(parseHash(hashFor(v))).toBe(v);
  });
  it('accepts a hash without the slash and ignores a query or subpath', () => {
    expect(parseHash('#sessions')).toBe('sessions');
    expect(parseHash('#/engine?x=1')).toBe('engine');
    expect(parseHash('#/settings/foo')).toBe('settings');
  });
  it('falls back to the overview', () => {
    expect(parseHash('')).toBe('overview');
    expect(parseHash('#/nonsense')).toBe('overview');
    expect(parseHash('#')).toBe('overview');
  });
});
