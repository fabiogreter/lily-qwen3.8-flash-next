import { flushPromises, mount } from '@vue/test-utils';
import { describe, expect, it, vi } from 'vitest';
import { ref } from 'vue';
import * as fx from '../api/fixtures';
import { HttpBackend } from '../api/http';
import type { ErrorBody, SettingsPatch } from '../api/types';
import { dashboardKey } from '../composables/context';
import SettingsView from './SettingsView.vue';

// The Settings view over the real HTTP backend, against what lily's server
// serializes (the fixtures are generated from its Rust types).

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
}

interface Server {
  /** Answers a PATCH body; the default accepts with the fixture's update. */
  patch?: (body: SettingsPatch) => Response;
  /** The change log's answer. */
  log?: () => Response;
  /** The live status the view reads (the idle unload's warning). */
  status?: { state: string; idle_for_s: number };
}

async function setup(server: Server = {}) {
  const writes: { method: string; url: string; body: string | undefined; type: string | undefined }[] = [];
  const reads: string[] = [];
  const fetch = vi.fn(async (url: string, init?: RequestInit) => {
    const method = init?.method ?? 'GET';
    const type = (init?.headers as Record<string, string> | undefined)?.['Content-Type'];
    if (method !== 'GET') writes.push({ method, url, body: init?.body as string | undefined, type });
    else reads.push(url);
    if (method === 'GET' && url === '/v1/lily/settings') return json(fx.settings);
    if (method === 'GET' && url === '/v1/lily/settings/log') return server.log ? server.log() : json(fx.settingsLog);
    if (method === 'PATCH') {
      const body = JSON.parse(String(init?.body)) as SettingsPatch;
      return server.patch ? server.patch(body) : json(fx.settingsUpdate);
    }
    if (method === 'DELETE') return json(fx.settingReset);
    return json(fx.errorUnknown, 404);
  });
  const backend = new HttpBackend({ fetch });
  const wrapper = mount(SettingsView, {
    global: {
      provide: {
        [dashboardKey as symbol]: {
          backend,
          range: ref('1h'),
          status: { data: ref(server.status ?? null), error: ref(null), refresh: async () => {} },
        },
      },
    },
  });
  await flushPromises();
  return { wrapper, writes, reads };
}

type Wrapper = Awaited<ReturnType<typeof setup>>['wrapper'];
const row = (wrapper: Wrapper, flag: string) => wrapper.findAll('.set-row').find((r) => r.find('.flagname').exists() && r.find('.flagname').text() === flag)!;

async function enter(wrapper: Wrapper, flag: string, text: string): Promise<void> {
  const input = row(wrapper, flag).find('input');
  await input.setValue(text);
  await input.trigger('keydown.enter');
  await flushPromises();
}

describe('SettingsView over the HTTP backend', () => {
  it("shows the server's settings: runtime ones editable, sources, bounds, an unset choice", async () => {
    const { wrapper } = await setup();
    // Runtime settings get a field, restart-only ones a value.
    expect(row(wrapper, '--temperature').find('input').exists()).toBe(true);
    expect(row(wrapper, '--max-seq').find('input').exists()).toBe(false);
    expect(row(wrapper, '--queue').find('input').exists()).toBe(false);
    expect(row(wrapper, '--durable-min-tokens').find('input').exists()).toBe(true);
    // The database's overrides can be reset; a flag's value cannot.
    expect(row(wrapper, '--pin-hold').text()).toContain('database');
    expect(row(wrapper, '--pin-hold').find('button.btn').exists()).toBe(true);
    expect(row(wrapper, '--idle-unload').find('button.btn').exists()).toBe(false);
    // A lower bound alone reads as such, not "0 to null".
    expect(row(wrapper, '--temperature').text()).toContain('at least 0');
    expect(row(wrapper, '--temperature').text()).not.toContain('null');
    // No effort set: the select says so instead of showing the first choice.
    const effort = row(wrapper, '--reasoning-effort').find('select').element as HTMLSelectElement;
    expect(effort.value).toBe('');
    expect(effort.selectedOptions[0]?.textContent).toBe('unset');
    // The change log.
    expect(wrapper.findAll('.log li')).toHaveLength(fx.settingsLog.entries.length);
  });

  it('a change is a PATCH of {name: value} through the write guard; the answer replaces the row', async () => {
    const { wrapper, writes } = await setup();
    await enter(wrapper, '--repetition-penalty', '1.1');
    expect(writes).toEqual([{ method: 'PATCH', url: '/v1/lily/settings', body: '{"repetition_penalty":1.1}', type: 'application/json' }]);
    // The row shows what the server answered.
    const shown = (row(wrapper, '--repetition-penalty').find('input').element as HTMLInputElement).value;
    expect(shown).toBe(String(fx.settingsUpdate.updated[0]!.value));
    expect(row(wrapper, '--repetition-penalty').find('[role="alert"]').exists()).toBe(false);
  });

  it("refusals show the server's message inline on the row: 400, 503, 504", async () => {
    // (A 409 is for a restart-only setting, which has no field here; the
    // HTTP backend's own tests cover it.)
    const answers: [string, string, Response, ErrorBody, boolean][] = [
      ['--top-p', '5', json(fx.errorInvalid, 400), fx.errorInvalid, false],
      ['--temperature', '0.5', json(fx.errorNoDatabase, 503), fx.errorNoDatabase, true],
      ['--repetition-penalty', '1.2', json(fx.errorOutcomeUnknown, 504), fx.errorOutcomeUnknown, true],
    ];
    for (const [flag, text, response, expected, reloads] of answers) {
      const { wrapper, reads } = await setup({ patch: () => response.clone() });
      const before = reads.length;
      await enter(wrapper, flag, text);
      const r = row(wrapper, flag);
      expect(r.find('[role="alert"]').text()).toBe(expected.error.message);
      expect(r.find('input').attributes('aria-invalid')).toBe('true');
      // After a 5xx the page reloads what the server holds, settings and log.
      const again = reads.slice(before).sort();
      expect(again).toEqual(reloads ? ['/v1/lily/settings', '/v1/lily/settings/log'] : []);
    }
  });

  it('an idle unload shorter than the idle time asks first: lily will unload now', async () => {
    const confirm = vi.spyOn(window, 'confirm');
    // Idle 20 minutes: 30m is fine, 10m unloads at once.
    const status = { state: 'ready', idle_for_s: 1200 };
    confirm.mockReturnValue(false);
    const { wrapper, writes } = await setup({ status });
    await enter(wrapper, '--idle-unload', '10m');
    expect(confirm).toHaveBeenCalledTimes(1);
    expect(confirm.mock.calls[0]![0]).toContain('lily will unload now');
    expect(writes).toEqual([]);
    confirm.mockReturnValue(true);
    await enter(wrapper, '--idle-unload', '15m');
    expect(writes).toHaveLength(1);
    confirm.mockClear();
    await enter(wrapper, '--idle-unload', '45m');
    expect(confirm).not.toHaveBeenCalled();
    // An unloaded or busy engine, or never unloading: nothing to warn about.
    const idle = await setup({ status: { state: 'unloaded', idle_for_s: 99_999 } });
    await enter(idle.wrapper, '--idle-unload', '1m');
    await enter(idle.wrapper, '--idle-unload', '0');
    expect(confirm).not.toHaveBeenCalled();
    confirm.mockRestore();
  });

  it('a reset is a DELETE with the JSON content type and no body; the row goes back to its default', async () => {
    const { wrapper, writes } = await setup();
    await row(wrapper, '--pin-hold').find('button.btn').trigger('click');
    await flushPromises();
    expect(writes).toEqual([{ method: 'DELETE', url: '/v1/lily/settings/pin_hold', body: undefined, type: 'application/json' }]);
    const r = row(wrapper, '--pin-hold');
    expect((r.find('input').element as HTMLInputElement).value).toBe(String(fx.settingReset.setting.value));
    expect(r.text()).toContain(fx.settingReset.setting.source);
    expect(r.find('button.btn').exists()).toBe(false);
  });

  it('without the database the change log says why instead of disappearing', async () => {
    const off = {
      error: { message: "the change log is kept in the recorder's database, which is off (--stats-db off)" },
    };
    const { wrapper } = await setup({ log: () => json(off, 503) });
    expect(wrapper.find('.log').exists()).toBe(false);
    expect(wrapper.text()).toContain(off.error.message);
  });
});
