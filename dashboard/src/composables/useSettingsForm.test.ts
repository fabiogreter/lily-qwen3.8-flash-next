import { describe, expect, it, vi } from 'vitest';
import { ref } from 'vue';
import { ApiError } from '../api/client';
import { MockBackend } from '../mock/backend';
import type { Setting } from '../api/types';
import { coerce, useSettingsForm } from './useSettingsForm';

async function setup() {
  const backend = new MockBackend({ clock: () => 0 });
  const settings = ref<Setting[] | null>((await backend.settings()).settings);
  const onChanged = vi.fn();
  const form = useSettingsForm(backend, settings, onChanged);
  const get = (name: string): Setting => settings.value!.find((s) => s.name === name)!;
  return { backend, settings, onChanged, form, get };
}

describe('coerce', () => {
  it('numbers: text becomes a number, junk and blanks are refused', () => {
    expect(coerce({ type: 'number' }, '0.9')).toEqual({ ok: true, value: 0.9 });
    expect(coerce({ type: 'int' }, ' 20 ')).toEqual({ ok: true, value: 20 });
    expect(coerce({ type: 'number' }, '')).toEqual({ ok: false, message: 'Enter a number.' });
    expect(coerce({ type: 'number' }, 'abc')).toEqual({ ok: false, message: 'Enter a number.' });
    expect(coerce({ type: 'number' }, 'Infinity')).toMatchObject({ ok: false });
  });
  it('bool and text', () => {
    expect(coerce({ type: 'bool' }, true)).toEqual({ ok: true, value: true });
    expect(coerce({ type: 'bool' }, false)).toEqual({ ok: true, value: false });
    expect(coerce({ type: 'duration' }, '30m')).toEqual({ ok: true, value: '30m' });
    expect(coerce({ type: 'enum' }, 'low')).toEqual({ ok: true, value: 'low' });
  });
});

describe('useSettingsForm', () => {
  it('surfaces the backend refusal inline, keeps the draft and the old value', async () => {
    const { form, get, onChanged } = await setup();
    form.drafts.top_p = '5';
    const ok = await form.commit(get('top_p'), '5');
    expect(ok).toBe(false);
    expect(form.errors.top_p).toBe('top_p must be between 0 and 1');
    expect(form.drafts.top_p).toBe('5');
    expect(get('top_p').value).toBe(0.95);
    expect(onChanged).not.toHaveBeenCalled();
    expect(form.busy.top_p).toBe(false);
  });

  it('a local type error never reaches the backend', async () => {
    const { form, get, backend } = await setup();
    const spy = vi.spyOn(backend, 'updateSettings');
    expect(await form.commit(get('top_p'), 'x')).toBe(false);
    expect(form.errors.top_p).toBe('Enter a number.');
    expect(spy).not.toHaveBeenCalled();
  });

  it('a valid change replaces the setting, clears draft and error, and reports the change', async () => {
    const { form, get, onChanged } = await setup();
    form.errors.top_p = 'old refusal';
    form.drafts.top_p = '0.9';
    expect(await form.commit(get('top_p'), '0.9')).toBe(true);
    expect(get('top_p')).toMatchObject({ value: 0.9, source: 'database' });
    expect(form.errors.top_p).toBeUndefined();
    expect(form.drafts.top_p).toBeUndefined();
    expect(onChanged).toHaveBeenCalledTimes(1);
  });

  it('committing the current value is a no-op for the server', async () => {
    const { form, get, backend } = await setup();
    const spy = vi.spyOn(backend, 'updateSettings');
    expect(await form.commit(get('top_k'), '20')).toBe(true);
    expect(spy).not.toHaveBeenCalled();
  });

  it('toggles and selects commit their value', async () => {
    const { form, get } = await setup();
    expect(await form.commit(get('thinking_nudges'), false)).toBe(true);
    expect(get('thinking_nudges').value).toBe(false);
    expect(await form.commit(get('reasoning_effort'), 'medium')).toBe(true);
    expect(get('reasoning_effort').value).toBe('medium');
  });

  it('a refused enum or duration shows the server message', async () => {
    const { form, get } = await setup();
    expect(await form.commit(get('idle_unload'), 'soon')).toBe(false);
    expect(form.errors.idle_unload).toContain('duration');
  });

  it('reset returns to the flag or default and reports the change', async () => {
    const { form, get, onChanged } = await setup();
    expect(get('repetition_penalty')).toMatchObject({ value: 1.05, source: 'database' });
    expect(await form.reset(get('repetition_penalty'))).toBe(true);
    expect(get('repetition_penalty')).toMatchObject({ value: 1, source: 'default' });
    expect(onChanged).toHaveBeenCalledTimes(1);
  });

  it('a commit while another is in flight for the same setting is refused without a request', async () => {
    const { form, get, backend } = await setup();
    let release!: () => void;
    const gate = new Promise<void>((r) => (release = r));
    const real = backend.updateSettings.bind(backend);
    const spy = vi.spyOn(backend, 'updateSettings').mockImplementation(async (p) => {
      await gate;
      return real(p);
    });
    const first = form.commit(get('top_p'), '0.5');
    expect(form.busy.top_p).toBe(true);
    expect(await form.commit(get('top_p'), '0.6')).toBe(false);
    expect(await form.reset(get('top_p'))).toBe(false);
    release();
    expect(await first).toBe(true);
    expect(spy).toHaveBeenCalledTimes(1);
    expect(get('top_p').value).toBe(0.5);
    expect(form.busy.top_p).toBe(false);
  });

  it('different settings do not block each other', async () => {
    const { form, get } = await setup();
    const a = form.commit(get('top_p'), '0.5');
    const b = form.commit(get('top_k'), '30');
    expect(await Promise.all([a, b])).toEqual([true, true]);
  });

  it('after a 5xx the view is asked to reload; after a 4xx it is not', async () => {
    const backend = new MockBackend({ clock: () => 0 });
    const settings = ref<Setting[] | null>((await backend.settings()).settings);
    const onUncertain = vi.fn();
    const form = useSettingsForm(backend, settings, vi.fn(), { onUncertain });
    const top = settings.value!.find((s) => s.name === 'top_p')!;
    const spy = vi.spyOn(backend, 'updateSettings');
    for (const [status, reloads] of [[400, 0], [503, 1], [504, 2], [500, 3]] as const) {
      spy.mockRejectedValueOnce(new ApiError(status, `answered ${status}`));
      expect(await form.commit(top, '0.5')).toBe(false);
      expect(form.errors.top_p).toBe(`answered ${status}`);
      expect(onUncertain).toHaveBeenCalledTimes(reloads);
    }
    vi.spyOn(backend, 'resetSetting').mockRejectedValueOnce(new ApiError(504, 'unknown'));
    expect(await form.reset(top)).toBe(false);
    expect(onUncertain).toHaveBeenCalledTimes(4);
    expect(form.busy.top_p).toBe(false);
  });

  it('a change the confirmation cancels is not sent', async () => {
    const backend = new MockBackend({ clock: () => 0 });
    const settings = ref<Setting[] | null>((await backend.settings()).settings);
    const confirm = vi.fn(() => false);
    const form = useSettingsForm(backend, settings, vi.fn(), { confirm });
    const spy = vi.spyOn(backend, 'updateSettings');
    const idle = settings.value!.find((s) => s.name === 'idle_unload')!;
    expect(await form.commit(idle, '1m')).toBe(false);
    expect(confirm).toHaveBeenCalledWith(idle, '1m');
    expect(spy).not.toHaveBeenCalled();
    expect(form.busy.idle_unload).toBeFalsy();
  });
});
