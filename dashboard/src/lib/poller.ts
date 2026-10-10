// Visibility-aware polling. The dashboard polls only while its tab is visible:
// a hidden tab must cost lily nothing (docs/dashboard.md § 5, § 2).

export interface PollerOptions {
  run: () => void | Promise<void>;
  intervalMs: number;
  /** Defaults to `document.visibilityState === 'visible'`. */
  isVisible?: () => boolean;
}

export interface Poller {
  start(): void;
  stop(): void;
}

const documentVisible = (): boolean => typeof document === 'undefined' || document.visibilityState === 'visible';

/**
 * `start()` runs once at once (so a tab opened in the background still has
 * something to show), then every `intervalMs` while visible. A tick that finds
 * the page hidden does nothing; becoming visible again runs immediately.
 */
export function createPoller(opts: PollerOptions): Poller {
  const isVisible = opts.isVisible ?? documentVisible;
  let timer: ReturnType<typeof setInterval> | null = null;

  const tick = (): void => {
    if (isVisible()) void opts.run();
  };
  const onVisibility = (): void => tick();

  return {
    start() {
      if (timer !== null) return;
      void opts.run();
      timer = setInterval(tick, opts.intervalMs);
      if (typeof document !== 'undefined') document.addEventListener('visibilitychange', onVisibility);
    },
    stop() {
      if (timer === null) return;
      clearInterval(timer);
      timer = null;
      if (typeof document !== 'undefined') document.removeEventListener('visibilitychange', onVisibility);
    },
  };
}
