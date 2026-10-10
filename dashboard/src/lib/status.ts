// Status presentation shared by StatusMark and the engine schematic, and the
// staleness rule for the live status.

import type { MemoryPressure, Status } from '../api/types';

export interface Level {
  /** CSS class setting `--c`, the status colour (never used for text). */
  cls: 'st-good' | 'st-warn' | 'st-crit';
  icon: string;
  label: string;
}

const LEVELS: Record<MemoryPressure, Level> = {
  normal: { cls: 'st-good', icon: '✓', label: 'normal' },
  warning: { cls: 'st-warn', icon: '!', label: 'warning' },
  critical: { cls: 'st-crit', icon: '×', label: 'critical' },
};

export function levelOf(level: MemoryPressure): Level {
  return LEVELS[level];
}

/** The status is stale when the server's own clock is older than this many poll intervals. */
export const STALE_AFTER_POLLS = 3;

/** How old `status.now` is against the client's clock (ms; negative when the server is ahead). */
export function statusAgeMs(status: Pick<Status, 'now'>, clientNow: number): number {
  return clientNow - status.now;
}

export function isStaleAge(ageMs: number, pollMs: number): boolean {
  return ageMs > STALE_AFTER_POLLS * pollMs;
}

export function isStale(status: Pick<Status, 'now'>, clientNow: number, pollMs: number): boolean {
  return isStaleAge(statusAgeMs(status, clientNow), pollMs);
}
