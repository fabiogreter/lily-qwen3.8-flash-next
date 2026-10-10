import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { createPoller } from './poller';

describe('createPoller', () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it('runs at once, then every interval while visible', () => {
    const run = vi.fn();
    const p = createPoller({ run, intervalMs: 2000, isVisible: () => true });
    p.start();
    expect(run).toHaveBeenCalledTimes(1);
    vi.advanceTimersByTime(6000);
    expect(run).toHaveBeenCalledTimes(4);
    p.stop();
    vi.advanceTimersByTime(10_000);
    expect(run).toHaveBeenCalledTimes(4);
  });

  it('does not poll while the page is hidden (only the first fetch runs)', () => {
    const run = vi.fn();
    let visible = false;
    const p = createPoller({ run, intervalMs: 2000, isVisible: () => visible });
    p.start();
    expect(run).toHaveBeenCalledTimes(1);
    vi.advanceTimersByTime(20_000);
    expect(run).toHaveBeenCalledTimes(1);
    visible = true;
    vi.advanceTimersByTime(2000);
    expect(run).toHaveBeenCalledTimes(2);
    p.stop();
  });

  it('runs immediately when the tab becomes visible again', () => {
    const run = vi.fn();
    let visible = false;
    const p = createPoller({ run, intervalMs: 60_000, isVisible: () => visible });
    p.start();
    run.mockClear();
    visible = true;
    document.dispatchEvent(new Event('visibilitychange'));
    expect(run).toHaveBeenCalledTimes(1);
    p.stop();
    document.dispatchEvent(new Event('visibilitychange'));
    expect(run).toHaveBeenCalledTimes(1);
  });

  it('start is idempotent', () => {
    const run = vi.fn();
    const p = createPoller({ run, intervalMs: 1000, isVisible: () => true });
    p.start();
    p.start();
    vi.advanceTimersByTime(3000);
    expect(run).toHaveBeenCalledTimes(4);
    p.stop();
  });
});
