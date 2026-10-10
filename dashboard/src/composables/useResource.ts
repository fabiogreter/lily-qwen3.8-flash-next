import { onBeforeUnmount, onMounted, shallowRef, watch, type Ref, type WatchSource } from 'vue';
import { errorMessage, isNamed } from '../api/client';
import { createPoller } from '../lib/poller';

export interface Resource<T> {
  /** The last good answer. A refetch keeps it on screen until the next one arrives. */
  data: Ref<T | null>;
  /** Message of the last failed fetch; cleared by the next success. */
  error: Ref<string | null>;
  /** Fetch now, replacing a request in flight. */
  refresh(): Promise<void>;
}

export interface ResourceOptions {
  /** Poll while the tab is visible; omit to fetch on mount and on `sources` changes only. */
  intervalMs?: number;
  /** A change in any of these refetches (e.g. the range). */
  sources?: WatchSource[];
  /**
   * A change in any of these first drops the current value (and any request in
   * flight), so what was fetched for the old key is never shown under the new
   * one (e.g. the selected session). List them in `sources` too.
   */
  resetOn?: WatchSource[];
  /** Give up on a request after this long (default 5000 ms); the timeout is an error. */
  timeoutMs?: number;
}

export const DEFAULT_TIMEOUT_MS = 5000;

/**
 * A fetched value with an optional visible-only poll. A newer request always
 * wins: an answer to a superseded request (older range, older poll) is dropped.
 * Every request has a deadline, so a hung server shows as an error instead of
 * a page that quietly stops updating.
 */
export function useResource<T>(fetcher: (signal: AbortSignal) => Promise<T | null>, opts: ResourceOptions = {}): Resource<T> {
  const data = shallowRef<T | null>(null) as Ref<T | null>;
  const error = shallowRef<string | null>(null);
  let seq = 0;
  let ctl: AbortController | null = null;
  let inFlight = false;

  async function load(): Promise<void> {
    ctl?.abort();
    const mine = new AbortController();
    ctl = mine;
    const id = ++seq;
    inFlight = true;
    const signal = AbortSignal.any([mine.signal, AbortSignal.timeout(opts.timeoutMs ?? DEFAULT_TIMEOUT_MS)]);
    try {
      const v = await fetcher(signal);
      if (id !== seq) return;
      data.value = v;
      error.value = null;
    } catch (e) {
      if (id !== seq || mine.signal.aborted) return;
      error.value = isNamed(e, 'TimeoutError') ? 'lily did not answer in time' : errorMessage(e);
    } finally {
      if (id === seq) inFlight = false;
    }
  }

  const poller = opts.intervalMs
    ? createPoller({
        // A slow answer is never stacked with another poll.
        run: () => (inFlight ? undefined : load()),
        intervalMs: opts.intervalMs,
      })
    : null;

  function drop(): void {
    seq++;
    ctl?.abort();
    inFlight = false;
    data.value = null;
    error.value = null;
  }

  // Registered before `sources`, so the old value is gone before the refetch starts.
  if (opts.resetOn?.length) watch(opts.resetOn, drop);
  if (opts.sources?.length) watch(opts.sources, () => void load());
  onMounted(() => (poller ? poller.start() : void load()));
  onBeforeUnmount(() => {
    poller?.stop();
    seq++;
    ctl?.abort();
  });

  return { data, error, refresh: load };
}
