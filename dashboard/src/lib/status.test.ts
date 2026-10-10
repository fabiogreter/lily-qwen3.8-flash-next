import { describe, expect, it } from 'vitest';
import { isStale, isStaleAge, levelOf, STALE_AFTER_POLLS, statusAgeMs } from './status';

describe('status levels', () => {
  it('lily names, an icon and a label for each', () => {
    expect(levelOf('normal')).toEqual({ cls: 'st-good', icon: '✓', label: 'normal' });
    expect(levelOf('warning')).toEqual({ cls: 'st-warn', icon: '!', label: 'warning' });
    expect(levelOf('critical')).toEqual({ cls: 'st-crit', icon: '×', label: 'critical' });
  });
});

describe('staleness', () => {
  it('the age is the client clock minus the status clock', () => {
    expect(statusAgeMs({ now: 1000 }, 1500)).toBe(500);
    expect(statusAgeMs({ now: 2000 }, 1500)).toBe(-500);
  });
  it('stale after three poll intervals, not before', () => {
    expect(STALE_AFTER_POLLS).toBe(3);
    expect(isStaleAge(6000, 2000)).toBe(false);
    expect(isStaleAge(6001, 2000)).toBe(true);
    expect(isStale({ now: 0 }, 7000, 2000)).toBe(true);
    expect(isStale({ now: 5000 }, 7000, 2000)).toBe(false);
  });
  it('a server clock ahead of the client is not stale', () => {
    expect(isStale({ now: 10_000 }, 0, 2000)).toBe(false);
  });
});
