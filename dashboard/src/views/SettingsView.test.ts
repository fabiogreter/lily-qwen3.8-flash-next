import { flushPromises, mount } from '@vue/test-utils';
import { describe, expect, it } from 'vitest';
import { vi } from 'vitest';
import { ApiError } from '../api/client';
import type { Setting } from '../api/types';
import { MockBackend } from '../mock/backend';
import { dashboardKey } from '../composables/context';
import { ref } from 'vue';
import SettingsView from './SettingsView.vue';

async function setup(backend: MockBackend = new MockBackend({ clock: () => 0 })) {
  const wrapper = mount(SettingsView, {
    global: {
      provide: {
        [dashboardKey as symbol]: {
          backend,
          range: ref('1h'),
          status: { data: ref(null), error: ref(null), refresh: async () => {} },
        },
      },
    },
  });
  await flushPromises();
  return { wrapper, backend };
}

const row = (wrapper: Awaited<ReturnType<typeof setup>>['wrapper'], flag: string) =>
  wrapper.findAll('.set-row').find((r) => r.text().includes(flag))!;

describe('SettingsView', () => {
  it('lists editable settings by group, read-only ones apart, and the browser preference group', async () => {
    const { wrapper } = await setup();
    const heads = wrapper.findAll('h2').map((h) => h.text());
    expect(heads).toEqual(['Dashboard (this browser)', 'Sampling', 'Thinking', 'Lifecycle', 'Caches', 'Restart required', 'Change log']);
    expect(row(wrapper, '--max-seq').find('input').exists()).toBe(false);
    expect(row(wrapper, '--top-p').find('input').exists()).toBe(true);
  });

  it('an out-of-range edit shows the server message inline and leaves the value alone', async () => {
    const { wrapper } = await setup();
    const r = row(wrapper, '--top-p');
    const input = r.find('input');
    await input.setValue('5');
    await input.trigger('keydown.enter');
    await flushPromises();
    expect(r.find('[role="alert"]').text()).toBe('top_p must be between 0 and 1');
    expect(input.attributes('aria-invalid')).toBe('true');
    expect(r.text()).toContain('default'); // still the default source
  });

  it('a valid edit is applied, marked as database, logged, and can be reset', async () => {
    const { wrapper } = await setup();
    const r = row(wrapper, '--top-p');
    await r.find('input').setValue('0.8');
    await r.find('input').trigger('keydown.enter');
    await flushPromises();
    expect(r.find('[role="alert"]').exists()).toBe(false);
    expect(r.text()).toContain('database');
    expect(wrapper.find('.log li').text()).toContain('0.95 → 0.8');
    await r.find('button.btn').trigger('click');
    await flushPromises();
    expect(row(wrapper, '--top-p').text()).not.toContain('database');
  });

  it('a toggle commits at once', async () => {
    const { wrapper } = await setup();
    const r = row(wrapper, '--thinking-nudges');
    await r.find('input[type="checkbox"]').setValue(false);
    await flushPromises();
    expect(r.text()).toContain('off');
    expect(r.text()).toContain('database');
  });

  it('the texture switch changes <html data-paper> and is separate from the server settings', async () => {
    const { wrapper } = await setup();
    const grain = wrapper.findAll('.seg button').find((b) => b.text() === 'grain')!;
    await grain.trigger('click');
    expect(document.documentElement.dataset.paper).toBe('grain');
    expect(localStorage.getItem('lily-dashboard-paper')).toBe('grain');
  });

  it('shows when a change applies, per row, in the row tooltip', async () => {
    const { wrapper } = await setup();
    const title = (flag: string) => row(wrapper, flag).find('.nm').attributes('title');
    expect(title('--pin-hold')).toContain('Applies: hold re-armed on change');
    expect(title('--idle-unload')).toContain('Applies: timer re-armed on change');
    expect(title('--disk-cache-ttl')).toContain('Applies: next expiry pass');
    expect(title('--top-p')).toBe('Applies: next request');
    expect(title('--temperature')).toBe('Default when a request sets none. Applies: next request');
  });

  it('every group with an editable setting renders, including Engine and unknown groups', async () => {
    class Extra extends MockBackend {
      async settings() {
        const r = await super.settings();
        const base = r.settings.find((x) => x.name === 'top_k')!;
        const extra = (name: string, group: string): Setting => ({ ...base, name, group, flag: `--${name}` });
        return { settings: [...r.settings, extra('engine_knob', 'Engine'), extra('odd_knob', 'Telemetry')] };
      }
    }
    const { wrapper } = await setup(new Extra({ clock: () => 0 }));
    const heads = wrapper.findAll('h2').map((h) => h.text());
    expect(heads).toEqual(['Dashboard (this browser)', 'Sampling', 'Thinking', 'Lifecycle', 'Caches', 'Engine', 'Telemetry', 'Restart required', 'Change log']);
    expect(row(wrapper, '--engine_knob').find('input').exists()).toBe(true);
    expect(row(wrapper, '--odd_knob').find('input').exists()).toBe(true);
  });

  it('a refused toggle snaps back to the stored value and shows the refusal', async () => {
    const backend = new MockBackend({ clock: () => 0 });
    vi.spyOn(backend, 'updateSettings').mockRejectedValue(new ApiError(400, 'thinking_nudges is locked'));
    const { wrapper } = await setup(backend);
    const r = row(wrapper, '--thinking-nudges');
    const box = r.find('input[type="checkbox"]');
    expect((box.element as HTMLInputElement).checked).toBe(true);
    await box.setValue(false);
    await flushPromises();
    expect((box.element as HTMLInputElement).checked).toBe(true);
    expect(r.find('[role="alert"]').text()).toBe('thinking_nudges is locked');
  });

  it('a refused select snaps back to the stored value', async () => {
    const backend = new MockBackend({ clock: () => 0 });
    vi.spyOn(backend, 'updateSettings').mockRejectedValue(new ApiError(400, 'refused'));
    const { wrapper } = await setup(backend);
    const r = row(wrapper, '--reasoning-effort');
    const sel = r.find('select');
    expect((sel.element as HTMLSelectElement).value).toBe('low');
    await sel.setValue('medium');
    await flushPromises();
    expect((sel.element as HTMLSelectElement).value).toBe('low');
    expect(r.find('[role="alert"]').text()).toBe('refused');
  });

  it('a second edit while one is in flight sends nothing; the field is readonly and aria-busy, not disabled', async () => {
    const backend = new MockBackend({ clock: () => 0 });
    let release!: () => void;
    const gate = new Promise<void>((r) => (release = r));
    const real = backend.updateSettings.bind(backend);
    const spy = vi.spyOn(backend, 'updateSettings').mockImplementation(async (patch) => {
      await gate;
      return real(patch);
    });
    const { wrapper } = await setup(backend);
    const r = row(wrapper, '--top-p');
    const input = r.find('input');
    await input.setValue('0.7');
    await input.trigger('keydown.enter');
    await input.trigger('keydown.enter');
    expect(spy).toHaveBeenCalledTimes(1);
    expect(input.attributes('readonly')).toBeDefined();
    expect(input.attributes('aria-busy')).toBe('true');
    expect(input.attributes('disabled')).toBeUndefined();
    release();
    await flushPromises();
    expect(input.attributes('readonly')).toBeUndefined();
    expect(row(wrapper, '--top-p').text()).toContain('database');
  });

  it('a refused duration shows lily\'s parser message', async () => {
    const { wrapper } = await setup();
    const r = row(wrapper, '--idle-unload');
    await r.find('input').setValue('5x');
    await r.find('input').trigger('keydown.enter');
    await flushPromises();
    expect(r.find('[role="alert"]').text()).toBe('unknown duration unit "x" (use s, m, h or d)');
  });

  it('links to the licences', async () => {
    const { wrapper } = await setup();
    expect(wrapper.find('a[href$="assets/licenses.txt"]').exists()).toBe(true);
  });
});
