import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { mount } from '@vue/test-utils';
import { defineComponent, h, ref } from 'vue';
import { useResource, type Resource } from './useResource';

function host<T>(fetcher: (s: AbortSignal) => Promise<T | null>, opts: Parameters<typeof useResource>[1]) {
  let res!: Resource<T>;
  const wrapper = mount(
    defineComponent({
      setup() {
        res = useResource(fetcher, opts);
        return () => h('div');
      },
    }),
  );
  return { res, wrapper };
}

const tick = async (): Promise<void> => {
  await Promise.resolve();
  await Promise.resolve();
  await Promise.resolve();
};

describe('useResource', () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it('fetches on mount, keeps the previous value while refetching, and records errors without dropping data', async () => {
    let n = 0;
    let fail = false;
    const { res, wrapper } = host(async () => {
      n++;
      if (fail) throw new Error('boom');
      return n;
    }, {});
    await tick();
    expect(res.data.value).toBe(1);
    fail = true;
    await res.refresh();
    expect(res.data.value).toBe(1); // previous render stays
    expect(res.error.value).toBe('boom');
    fail = false;
    await res.refresh();
    expect(res.data.value).toBe(3);
    expect(res.error.value).toBeNull();
    wrapper.unmount();
  });

  it('refetches when a source changes and drops the answer of a superseded request', async () => {
    const range = ref('1h');
    const resolvers: Record<string, (v: string) => void> = {};
    const { res, wrapper } = host<string>(
      () =>
        new Promise((resolve) => {
          resolvers[range.value] = resolve;
        }),
      { sources: [range] },
    );
    await tick();
    range.value = '24h';
    await tick();
    resolvers['24h']!('24h data');
    await tick();
    expect(res.data.value).toBe('24h data');
    resolvers['1h']!('stale 1h data'); // the old request answers late
    await tick();
    expect(res.data.value).toBe('24h data');
    wrapper.unmount();
  });

  it('polls on the interval only while visible and never stacks requests', async () => {
    let calls = 0;
    const pending: { release: (() => void) | null } = { release: null };
    const { wrapper } = host(
      () => {
        calls++;
        return new Promise<number>((resolve) => {
          pending.release = () => resolve(calls);
        });
      },
      { intervalMs: 1000 },
    );
    expect(calls).toBe(1);
    vi.advanceTimersByTime(3000); // first request still in flight
    expect(calls).toBe(1);
    pending.release!();
    await tick();
    vi.advanceTimersByTime(1000);
    expect(calls).toBe(2);
    wrapper.unmount();
    pending.release?.();
    vi.advanceTimersByTime(5000);
    expect(calls).toBe(2); // stopped on unmount
  });

  it('does not poll in a hidden tab', async () => {
    const vis = vi.spyOn(document, 'visibilityState', 'get').mockReturnValue('hidden');
    let calls = 0;
    const { wrapper } = host(
      async () => {
        calls++;
        return calls;
      },
      { intervalMs: 1000 },
    );
    await tick();
    expect(calls).toBe(1);
    vi.advanceTimersByTime(10_000);
    expect(calls).toBe(1);
    vis.mockReturnValue('visible');
    document.dispatchEvent(new Event('visibilitychange'));
    await tick();
    expect(calls).toBe(2);
    wrapper.unmount();
    vis.mockRestore();
  });

  it('resetOn drops the old value at once and a late answer for the old key never lands', async () => {
    const key = ref('a');
    const resolvers: Record<string, (v: string) => void> = {};
    const { res, wrapper } = host<string>(
      () =>
        new Promise((resolve) => {
          resolvers[key.value] = resolve;
        }),
      { sources: [key], resetOn: [key] },
    );
    await tick();
    resolvers['a']!('data for a');
    await tick();
    expect(res.data.value).toBe('data for a');
    key.value = 'b';
    await tick();
    expect(res.data.value).toBeNull(); // not a's data under b
    resolvers['b']!('data for b');
    await tick();
    expect(res.data.value).toBe('data for b');
    key.value = 'c';
    await tick();
    expect(res.data.value).toBeNull();
    resolvers['b']!('late b'); // an answer for a superseded key
    await tick();
    expect(res.data.value).toBeNull();
    wrapper.unmount();
  });

  it('without resetOn the previous value stays while the new key loads', async () => {
    const key = ref('a');
    const { res, wrapper } = host<string>(async () => key.value, { sources: [key] });
    await tick();
    expect(res.data.value).toBe('a');
    key.value = 'b';
    await tick();
    expect(res.data.value).toBe('b');
    wrapper.unmount();
  });
});

describe('useResource timeout', () => {
  it('a request that never answers becomes an error after the deadline, keeping the last value', async () => {
    let first = true;
    const { res, wrapper } = host<number>(
      (signal) => {
        if (first) {
          first = false;
          return Promise.resolve(1);
        }
        return new Promise((_, reject) => signal.addEventListener('abort', () => reject(signal.reason)));
      },
      { timeoutMs: 20 },
    );
    await vi.waitFor(() => expect(res.data.value).toBe(1));
    void res.refresh();
    await vi.waitFor(() => expect(res.error.value).toBe('lily did not answer in time'), { timeout: 1000 });
    expect(res.data.value).toBe(1);
    wrapper.unmount();
  });

  it('a polled request is retried after a timeout (the in-flight flag is released)', async () => {
    let calls = 0;
    const { res, wrapper } = host<number>(
      (signal) => {
        calls++;
        return new Promise((_, reject) => signal.addEventListener('abort', () => reject(signal.reason)));
      },
      { timeoutMs: 20, intervalMs: 60 },
    );
    await vi.waitFor(() => expect(calls).toBeGreaterThanOrEqual(2), { timeout: 2000 });
    expect(res.error.value).toBe('lily did not answer in time');
    wrapper.unmount();
  });
});
