import { flushPromises, mount } from '@vue/test-utils';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import App from './App.vue';
import { ApiError } from './api/client';
import type { Status } from './api/types';
import { MockBackend } from './mock/backend';

describe('App status notice', () => {
  beforeEach(() => {
    history.replaceState(null, '', '#/engine');
  });
  afterEach(() => {
    vi.useRealTimers();
    history.replaceState(null, '', '#');
  });

  it('says so when the status answer is old on the client clock', async () => {
    class Old extends MockBackend {
      async status(): Promise<Status> {
        const s = await super.status();
        return { ...s, now: Date.now() - 60_000 };
      }
    }
    const w = mount(App, { props: { backend: new Old() } });
    await flushPromises();
    const note = w.find('.inline-error');
    expect(note.exists()).toBe(true);
    expect(note.text()).toBe('The last status is 1m old. Showing the last values.');
    w.unmount();
  });

  it('no notice for a fresh status; an error after a good one keeps the values and says so', async () => {
    vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval', 'setTimeout', 'clearTimeout'] });
    const backend = new MockBackend();
    const w = mount(App, { props: { backend } });
    await vi.advanceTimersByTimeAsync(10);
    expect(w.find('.inline-error').exists()).toBe(false);
    expect(w.find('.schematic').exists()).toBe(true);
    vi.spyOn(backend, 'status').mockRejectedValue(new ApiError(0, 'lily is not reachable'));
    await vi.advanceTimersByTimeAsync(2000);
    expect(w.find('.inline-error').text()).toBe('lily is not reachable. Showing the last values.');
    expect(w.find('.schematic').exists()).toBe(true); // last values stay on screen
    w.unmount();
  });

  it('the nav switches views through the hash', async () => {
    const w = mount(App, { props: { backend: new MockBackend() } });
    await flushPromises();
    const btn = w.findAll('nav button').find((b) => b.text() === 'requests')!;
    await btn.trigger('click');
    expect(location.hash).toBe('#/requests');
    await flushPromises();
    expect(w.text()).toContain('Recent requests');
    w.unmount();
  });
});
