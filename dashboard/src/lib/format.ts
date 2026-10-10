// Display formatting. Pure functions; `null` (no value) renders as an en dash.
// Times are shown in the browser's local time zone.

import type { Range, SettingValue } from '../api/types';

export const DASH = '–';

const pad2 = (n: number): string => String(n).padStart(2, '0');

export function bytes(b: number | null | undefined): string {
  if (b == null) return DASH;
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let i = 0;
  let v = b;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return (v >= 99.95 || i === 0 ? String(Math.round(v)) : v.toFixed(1)) + ' ' + units[i];
}

export function tokens(n: number | null | undefined): string {
  if (n == null) return DASH;
  if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (n >= 1e4) return Math.round(n / 1e3) + 'K';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return String(Math.round(n));
}

/** A ratio 0..1 as a percentage. */
export function pct(x: number | null | undefined, digits = 0): string {
  return x == null ? DASH : (x * 100).toFixed(digits) + '%';
}

/** Tokens per second: one decimal below 100, whole numbers above. */
export function rate(x: number | null | undefined): string {
  if (x == null) return DASH;
  return x >= 99.95 ? String(Math.round(x)) : x.toFixed(1);
}

/** Seconds since something happened. */
export function ago(s: number | null | undefined): string {
  if (s == null) return DASH;
  // Floor, so a boundary never reads "60s ago" or "60m ago".
  if (s < 60) return Math.floor(s) + 's ago';
  if (s < 3600) return Math.floor(s / 60) + 'm ago';
  if (s < 86400) return Math.floor(s / 3600) + 'h ago';
  return Math.floor(s / 86400) + 'd ago';
}

/** Time since t, both unix ms (e.g. a session's last use against the server's clock). */
export function agoSince(t: number | null | undefined, now: number): string {
  return t == null ? DASH : ago(Math.max(0, (now - t) / 1000));
}

/** A whole number with a unit, e.g. "1569 MHz"; the unit is dropped with the value. */
export function whole(x: number | null | undefined, unit = ''): string {
  return x == null ? DASH : Math.round(x) + (unit ? ' ' + unit : '');
}

/** A span of seconds: "4d 3h", "3h 20m", "12m", "40s". */
export function duration(s: number | null | undefined): string {
  if (s == null) return DASH;
  if (s < 60) return Math.floor(s) + 's';
  const d = Math.floor(s / 86400);
  const h = Math.floor((s % 86400) / 3600);
  const m = Math.floor((s % 3600) / 60);
  return d ? `${d}d ${h}h` : h ? `${h}h ${m}m` : `${m}m`;
}

const WEEKDAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];

/** An x-axis or tooltip time for a chart of the given range. */
export function timeLabel(t: number, range: Range): string {
  const d = new Date(t);
  const hh = pad2(d.getHours());
  const mm = pad2(d.getMinutes());
  if (range === '7d') return `${WEEKDAYS[d.getDay()]} ${hh}h`;
  if (range === '15m') return `${hh}:${mm}:${pad2(d.getSeconds())}`;
  return `${hh}:${mm}`;
}

/** Wall-clock time of day, HH:MM:SS. */
export function clock(t: number): string {
  const d = new Date(t);
  return [d.getHours(), d.getMinutes(), d.getSeconds()].map(pad2).join(':');
}

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

/** "10 Oct, 14:20" */
export function dateTime(t: number): string {
  const d = new Date(t);
  return `${d.getDate()} ${MONTHS[d.getMonth()]}, ${pad2(d.getHours())}:${pad2(d.getMinutes())}`;
}

/** Watt-hours as Wh or kWh. */
export function energy(wh: number | null | undefined): string {
  if (wh == null) return DASH;
  return wh >= 1000 ? (wh / 1000).toFixed(2) + ' kWh' : Math.round(wh) + ' Wh';
}

/** A setting value for display. */
export function show(v: SettingValue | null | undefined): string {
  if (typeof v === 'boolean') return v ? 'on' : 'off';
  return v == null ? DASH : String(v);
}

/** "repetition_penalty" -> "repetition penalty" */
export function label(name: string): string {
  return name.replace(/_/g, ' ');
}

/** Bucket width of a range as a short label ("1 min"). */
export function bucketLabel(range: Range): string {
  return { '15m': '10 s', '1h': '1 min', '24h': '15 min', '7d': '1 h' }[range];
}
