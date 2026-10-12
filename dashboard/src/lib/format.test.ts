import { describe, expect, it } from 'vitest';
import { ago, agoSince, bounds, bucketLabel, bytes, clock, count, dateTime, DASH, duration, energy, label, ms, pct, rate, show, timeLabel, tokens, whole } from './format';

// The test setup pins TZ=UTC, so local time equals UTC here.
const T = Date.UTC(2026, 9, 10, 14, 5, 9); // Sat 10 Oct 2026, 14:05:09

describe('bytes', () => {
  it('scales units and keeps one decimal below 100', () => {
    expect(bytes(0)).toBe('0 B');
    expect(bytes(512)).toBe('512 B');
    expect(bytes(1536)).toBe('1.5 KB');
    expect(bytes(10.2 * 1024 ** 3)).toBe('10.2 GB');
    expect(bytes(100 * 1024 ** 3)).toBe('100 GB');
    expect(bytes(2.5 * 1024 ** 4)).toBe('2.5 TB');
    expect(bytes(99.97 * 1024)).toBe('100 KB');
  });
  it('shows a dash for no value', () => {
    expect(bytes(null)).toBe(DASH);
    expect(bytes(undefined)).toBe(DASH);
  });
});

describe('tokens', () => {
  it('abbreviates by magnitude', () => {
    expect(tokens(999)).toBe('999');
    expect(tokens(1500)).toBe('1.5K');
    expect(tokens(12_400)).toBe('12K');
    expect(tokens(2_340_000)).toBe('2.3M');
    expect(tokens(null)).toBe(DASH);
  });
});

describe('rate and pct', () => {
  it('rate: one decimal below 100, whole above', () => {
    expect(rate(69.94)).toBe('69.9');
    expect(rate(99.96)).toBe('100');
    expect(rate(152.4)).toBe('152');
    expect(rate(0)).toBe('0.0');
    expect(rate(null)).toBe(DASH);
  });
  it('pct: a ratio as a percentage', () => {
    expect(pct(0.874)).toBe('87%');
    expect(pct(0.874, 1)).toBe('87.4%');
    expect(pct(0)).toBe('0%');
    expect(pct(null)).toBe(DASH);
  });
});

describe('ago and duration', () => {
  it('ago picks the largest sensible unit', () => {
    expect(ago(5)).toBe('5s ago');
    expect(ago(59.6)).toBe('59s ago');
    expect(ago(150)).toBe('2m ago');
    expect(ago(3599)).toBe('59m ago');
    expect(ago(120)).toBe('2m ago');
    expect(ago(7200)).toBe('2h ago');
    expect(ago(3 * 86400)).toBe('3d ago');
    expect(ago(null)).toBe(DASH);
  });
  it('duration: two units at most', () => {
    expect(duration(40)).toBe('40s');
    expect(duration(12 * 60)).toBe('12m');
    expect(duration(3 * 3600 + 20 * 60)).toBe('3h 20m');
    expect(duration(4 * 86400 + 3 * 3600 + 1200)).toBe('4d 3h');
    expect(duration(1800)).toBe('30m');
  });
});

describe('time labels by range', () => {
  it('15m has seconds, 1h and 24h minutes, 7d weekday and hour', () => {
    expect(timeLabel(T, '15m')).toBe('14:05:09');
    expect(timeLabel(T, '1h')).toBe('14:05');
    expect(timeLabel(T, '24h')).toBe('14:05');
    expect(timeLabel(T, '7d')).toBe('Sat 14h');
  });
  it('clock and dateTime', () => {
    expect(clock(T)).toBe('14:05:09');
    expect(dateTime(T)).toBe('10 Oct, 14:05');
  });
});

describe('misc', () => {
  it('energy switches to kWh', () => {
    expect(energy(38.6)).toBe('39 Wh');
    expect(energy(1500)).toBe('1.50 kWh');
    expect(energy(null)).toBe(DASH);
  });
  it('show renders setting values', () => {
    expect(show(true)).toBe('on');
    expect(show(false)).toBe('off');
    expect(show(1.05)).toBe('1.05');
    expect(show('30m')).toBe('30m');
    expect(show(null)).toBe(DASH);
  });
  it('label and bucketLabel', () => {
    expect(label('repetition_penalty')).toBe('repetition penalty');
    expect(bucketLabel('15m')).toBe('1 min');
    expect(bucketLabel('24h')).toBe('15 min');
    expect(bucketLabel('7d')).toBe('1 h');
  });
});

describe('agoSince and whole', () => {
  it('agoSince is relative to the given clock, never negative', () => {
    expect(agoSince(T - 150_000, T)).toBe('2m ago');
    expect(agoSince(T + 5_000, T)).toBe('0s ago');
    expect(agoSince(null, T)).toBe(DASH);
  });
  it('whole rounds and appends a unit', () => {
    expect(whole(1568.6, 'MHz')).toBe('1569 MHz');
    expect(whole(71.4)).toBe('71');
    expect(whole(null, 'MHz')).toBe(DASH);
  });

  it('bounds: both, one side or none', () => {
    expect(bounds(0, 1)).toBe('0 to 1');
    expect(bounds(0, null)).toBe('at least 0');
    expect(bounds(null, 2)).toBe('at most 2');
    expect(bounds(null, null)).toBeNull();
  });
});

describe('ms', () => {
  it('keeps one decimal below 10 and rounds above', () => {
    expect(ms(0.249)).toBe('0.2');
    expect(ms(9.96)).toBe('10');
    expect(ms(1908.586)).toBe('1909');
    expect(ms(null)).toBe(DASH);
  });
});

describe('count', () => {
  it('uses the singular for exactly one', () => {
    expect(count(0, 'entry', 'entries')).toBe('0 entries');
    expect(count(1, 'entry', 'entries')).toBe('1 entry');
    expect(count(2, 'entry', 'entries')).toBe('2 entries');
  });
});
