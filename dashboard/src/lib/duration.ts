// A port of `parse_duration_secs` (src/serve.rs), so the page and the mock
// refuse what the server refuses, with the same messages:
//
//   trim; split at the first character that is not an ASCII digit or '.';
//   the head must parse as a float (Rust `f64::from_str`: digits with at most
//   one '.', so "1.5", ".5", "5." but not "." or "1.2.3"); the tail, trimmed
//   and lower-cased, is "" or "s" (seconds), "m", "h" or "d"; the result is
//   at most a year (`MAX_DURATION_SECS`); truncated to whole seconds (`as u64`).
//
// Errors carry the same text as lily's (the context line `invalid duration
// {text:?}` and `unknown duration unit {unit:?} (use s, m, h or d)`; `{:?}` of
// a string is its quoted, escaped form).

export class DurationError extends Error {}

/** Rust's `{:?}` for a str: double quotes, with `"` `\` and control characters escaped. */
export function debugStr(s: string): string {
  let out = '"';
  for (const ch of s) {
    const c = ch.codePointAt(0) as number;
    if (ch === '"') out += '\\"';
    else if (ch === '\\') out += '\\\\';
    else if (ch === '\n') out += '\\n';
    else if (ch === '\r') out += '\\r';
    else if (ch === '\t') out += '\\t';
    else if (c < 0x20 || c === 0x7f) out += `\\u{${c.toString(16)}}`;
    else out += ch;
  }
  return out + '"';
}

// Rust's trim() removes Unicode White_Space; JS trim() removes WhiteSpace and
// LineTerminator, which is the same set for practical input.
const SCALES = new Map<string, number>([
  ['', 1],
  ['s', 1],
  ['m', 60],
  ['h', 3600],
  ['d', 86_400],
]);
const FLOAT = /^(\d+(\.\d*)?|\.\d+)$/;

export function parseDurationSecs(input: string): number {
  const text = input.trim();
  const i = text.search(/[^0-9.]/);
  const digits = i < 0 ? text : text.slice(0, i);
  const unit = i < 0 ? '' : text.slice(i);
  if (!FLOAT.test(digits)) throw new DurationError(`invalid duration ${debugStr(text)}`);
  const value = Number(digits);
  const u = unit.trim().replace(/[A-Z]/g, (c) => c.toLowerCase()); // to_ascii_lowercase
  const scale = SCALES.get(u);
  if (scale === undefined) throw new DurationError(`unknown duration unit ${debugStr(u)} (use s, m, h or d)`);
  const secs = value * scale;
  if (secs > MAX_DURATION_SECS) throw new DurationError(`duration ${debugStr(text)} is longer than the 365-day cap`);
  return Math.floor(secs);
}

/** lily's `MAX_DURATION_SECS`: a year. */
export const MAX_DURATION_SECS = 365 * 86_400;
