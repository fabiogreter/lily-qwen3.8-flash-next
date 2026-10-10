import { describe, expect, it } from 'vitest';
import { debugStr, DurationError, parseDurationSecs } from './duration';

// Cases mirror src/serve.rs `parse_duration_secs`.
describe('parseDurationSecs', () => {
  it('units: s m h d, bare number = seconds', () => {
    expect(parseDurationSecs('45s')).toBe(45);
    expect(parseDurationSecs('90m')).toBe(5400);
    expect(parseDurationSecs('12h')).toBe(43_200);
    expect(parseDurationSecs('3d')).toBe(259_200);
    expect(parseDurationSecs('90')).toBe(90);
    expect(parseDurationSecs('0')).toBe(0);
  });

  it('decimals, trimming, whitespace before the unit, any case', () => {
    expect(parseDurationSecs('1.5h')).toBe(5400);
    expect(parseDurationSecs('.5h')).toBe(1800);
    expect(parseDurationSecs('5.')).toBe(5);
    expect(parseDurationSecs('  30m  ')).toBe(1800);
    expect(parseDurationSecs('30 m')).toBe(1800);
    expect(parseDurationSecs('2 H')).toBe(7200);
    expect(parseDurationSecs('1D')).toBe(86_400);
  });

  it('truncates to whole seconds', () => {
    expect(parseDurationSecs('1.9s')).toBe(1);
    expect(parseDurationSecs('0.5s')).toBe(0);
  });

  const fail = (text: string): string => {
    try {
      parseDurationSecs(text);
    } catch (e) {
      expect(e).toBeInstanceOf(DurationError);
      return (e as Error).message;
    }
    throw new Error(`expected ${JSON.stringify(text)} to be refused`);
  };

  it('invalid numbers: the message quotes the trimmed text', () => {
    expect(fail('')).toBe('invalid duration ""');
    expect(fail('   ')).toBe('invalid duration ""');
    expect(fail('abc')).toBe('invalid duration "abc"');
    expect(fail('s')).toBe('invalid duration "s"');
    expect(fail('.')).toBe('invalid duration "."');
    expect(fail('1.2.3')).toBe('invalid duration "1.2.3"');
    expect(fail(' -5 ')).toBe('invalid duration "-5"');
    expect(fail('1.2.3m')).toBe('invalid duration "1.2.3m"');
  });

  it('unknown units: lower-cased, trimmed, quoted', () => {
    expect(fail('5x')).toBe('unknown duration unit "x" (use s, m, h or d)');
    expect(fail('5 MS')).toBe('unknown duration unit "ms" (use s, m, h or d)');
    expect(fail('1e5')).toBe('unknown duration unit "e5" (use s, m, h or d)');
    expect(fail('3 days')).toBe('unknown duration unit "days" (use s, m, h or d)');
    expect(fail('3constructor')).toBe('unknown duration unit "constructor" (use s, m, h or d)');
  });

  it('debugStr escapes like Rust {:?}', () => {
    expect(debugStr('a"b')).toBe('"a\\"b"');
    expect(debugStr('a\\b')).toBe('"a\\\\b"');
    expect(debugStr('a\nb\tc')).toBe('"a\\nb\\tc"');
    expect(debugStr('é')).toBe('"é"');
    expect(fail('5"')).toBe('unknown duration unit "\\"" (use s, m, h or d)');
  });
});
