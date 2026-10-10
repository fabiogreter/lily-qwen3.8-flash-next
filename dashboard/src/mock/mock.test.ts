import { describe, expect, it } from 'vitest';
import { ApiError } from '../api/client';
import { RANGES } from '../api/types';
import { MockBackend } from './backend';
import { buildHistory, RANGE_CFG, summarize } from './data';

const NOW = Date.UTC(2026, 9, 10, 14, 20, 0);
const clock = (): number => NOW;
const make = (c: () => number = clock) => new MockBackend({ clock: c });

describe('mock determinism', () => {
  it('histories are identical on every call', () => {
    for (const r of RANGES) expect(buildHistory(r, NOW)).toEqual(buildHistory(r, NOW));
  });

  it('a bucket depends only on its own time: the grid slides without the curves changing', () => {
    const a = buildHistory('1h', NOW);
    const b = buildHistory('1h', NOW + 5 * 60e3);
    const overlap = a.points.slice(5);
    expect(b.points.slice(0, overlap.length)).toEqual(overlap);
  });

  it('the n-th status is the same for every backend', async () => {
    const a = make();
    const b = make();
    for (let i = 0; i < 12; i++) expect(await a.status()).toEqual(await b.status());
  });

  it('status alternates six busy polls and three idle ones', async () => {
    const m = make();
    const states: string[] = [];
    for (let i = 0; i < 9; i++) states.push((await m.status()).state);
    expect(states).toEqual(['busy', 'busy', 'busy', 'busy', 'busy', 'busy', 'ready', 'ready', 'ready']);
  });

  it('status.now is the clock, so the page does not see the mock as stale', async () => {
    expect((await make().status()).now).toBe(NOW);
  });

  it('sessions, requests and session histories are stable', async () => {
    const a = make();
    const b = make();
    const sa = await a.sessions();
    expect(sa).toEqual(await b.sessions());
    const id = sa.sessions[0]!.lineage!;
    expect(await a.sessionHistory(id, '1h')).toEqual(await b.sessionHistory(id, '1h'));
    expect(await a.requests(40)).toEqual(await b.requests(40));
  });
});

describe('mock follows the contract', () => {
  it('ranges use the contract buckets: 15m 60 s, 1h 60 s, 24h 900 s, 7d 3600 s', () => {
    expect(RANGE_CFG['15m'].step).toBe(60e3);
    expect(RANGE_CFG['1h'].step).toBe(60e3);
    expect(RANGE_CFG['24h'].step).toBe(900e3);
    expect(RANGE_CFG['7d'].step).toBe(3600e3);
    expect(buildHistory('15m', NOW).points).toHaveLength(15);
    expect(buildHistory('1h', NOW).points).toHaveLength(60);
    expect(buildHistory('24h', NOW).points).toHaveLength(96);
    expect(buildHistory('7d', NOW).points).toHaveLength(168);
  });

  it('the history is the complete contiguous grid ending at now, oldest first', () => {
    for (const r of RANGES) {
      const h = buildHistory(r, NOW);
      const step = h.step_s * 1000;
      for (let i = 1; i < h.points.length; i++) expect(h.points[i]!.t - h.points[i - 1]!.t).toBe(step);
      const last = h.points[h.points.length - 1]!.t;
      expect(last).toBeLessThanOrEqual(NOW);
      expect(NOW - last).toBeLessThan(step);
    }
  });

  it('idle buckets have null rates, not zero; buckets with no data have every value null and counts 0', () => {
    const h = buildHistory('7d', NOW);
    const empty = h.points.filter((p) => p.cpu_load == null);
    expect(empty.length).toBeGreaterThan(0); // the mock's "server was off" stretches
    for (const p of empty) {
      expect(p).toMatchObject({
        busy_share: 0,
        requests: 0,
        decode_tokens: 0,
        prefill_tokens: 0,
        sessions_max: 0,
        decode_tps_avg: null,
        decode_tps_max: null,
        prefill_tps_avg: null,
        acceptance: null,
        cache_hit: null,
        gpu_active: null,
        memory_pressure: null,
        swap_bytes: null,
        power_w: null,
        gpu_freq_mhz: null,
      });
    }
    const idle = h.points.filter((p) => p.cpu_load != null && p.busy_share === 0);
    for (const p of idle) expect(p.decode_tps_avg).toBeNull();
  });

  it('memory pressure uses lily names only', () => {
    const seen = new Set(buildHistory('7d', NOW).points.map((p) => p.memory_pressure));
    for (const v of seen) expect([null, 'normal', 'warning', 'critical']).toContain(v);
  });

  it('summary: average is busy-weighted, max is the best bucket', () => {
    const h = buildHistory('24h', NOW);
    const busy = h.points.filter((p) => p.decode_tps_avg != null);
    expect(h.summary.decode.max).toBe(Math.max(...busy.map((p) => p.decode_tps_max ?? 0)));
    const lo = Math.min(...busy.map((p) => p.decode_tps_avg ?? 0));
    const hi = Math.max(...busy.map((p) => p.decode_tps_avg ?? 0));
    expect(h.summary.decode.avg).toBeGreaterThanOrEqual(lo);
    expect(h.summary.decode.avg).toBeLessThanOrEqual(hi);
  });

  it('summary of nothing: no averages, no max, zero totals', () => {
    const s = summarize([]);
    expect(s.decode).toEqual({ avg: null, max: null });
    expect(s.requests).toBe(0);
    expect(s.energy_wh).toBeNull();
  });

  it('a session history has gaps, null machine fields and an unknown lineage is a 404', async () => {
    const m = make();
    const { sessions } = await m.sessions();
    const h = await m.sessionHistory(sessions[0]!.lineage!, '24h');
    expect(h.points.some((p) => p.decode_tps_avg == null)).toBe(true);
    expect(h.points.some((p) => p.decode_tps_avg != null)).toBe(true);
    for (const p of h.points) {
      if (p.decode_tps_avg == null) expect(p.decode_tokens).toBe(0);
      expect([p.cpu_load, p.gpu_active, p.memory_pressure, p.swap_bytes, p.power_w, p.gpu_power_w, p.gpu_freq_mhz]).toEqual([null, null, null, null, null, null, null]);
    }
    await expect(m.sessionHistory(1, '1h')).rejects.toMatchObject({ status: 404 });
  });

  it('sessions use unix-ms times against the clock and one entry has no lineage', async () => {
    const { sessions } = await make().sessions();
    for (const s of sessions) {
      expect(s.last_used_t).toBeLessThanOrEqual(NOW);
      expect(s.created_t).toBeLessThanOrEqual(s.last_used_t);
    }
    expect(sessions.filter((s) => s.lineage == null)).toHaveLength(1);
  });

  it('requests: nullable rates and draft counts, no derived acceptance, cancelled_by from lily', async () => {
    const { requests } = await make().requests(40);
    expect(requests.some((r) => r.drafted_tokens == null && r.accepted_tokens == null)).toBe(true);
    expect(requests.some((r) => r.prefill_tps == null)).toBe(true);
    for (const r of requests) {
      expect(r).not.toHaveProperty('acceptance');
      expect([null, 'client', 'shutdown']).toContain(r.cancelled_by);
    }
  });

  it('status: tiers.disk can be present, no `pinned` field, power is an object, sampler is named', async () => {
    const st = await make().status();
    expect(st).not.toHaveProperty('pinned');
    expect(st.tiers.disk).not.toBeNull();
    expect(st.machine.sampler).toBe('ok');
    expect(st.machine.power).toMatchObject({ cpu_w: expect.any(Number), sys_w: expect.any(Number) });
    expect(st.machine.memory_pressure).toBe('normal');
  });
});

describe('mock settings', () => {
  it('rejects an out-of-range number with the bounds in the message and names the setting', async () => {
    const m = make();
    const e = await m.updateSettings({ top_p: 5 }).catch((x: unknown) => x);
    expect(e).toBeInstanceOf(ApiError);
    expect((e as ApiError).status).toBe(400);
    expect((e as ApiError).message).toBe('top_p must be between 0 and 1');
    expect((e as ApiError).setting).toBe('top_p');
    const { settings } = await m.settings();
    expect(settings.find((s) => s.name === 'top_p')!.value).toBe(0.95);
  });

  it('rejects a fractional int, a bad enum, a non-boolean', async () => {
    const m = make();
    await expect(m.updateSettings({ top_k: 20.5 })).rejects.toThrow('whole number');
    await expect(m.updateSettings({ reasoning_effort: 'extreme' })).rejects.toThrow('one of low, medium, xhigh');
    await expect(m.updateSettings({ thinking_nudges: 'yes' })).rejects.toThrow('true or false');
  });

  it('durations are validated as lily parses them, with its messages', async () => {
    const m = make();
    await expect(m.updateSettings({ idle_unload: 'soon' })).rejects.toThrow('invalid duration "soon"');
    await expect(m.updateSettings({ idle_unload: '5x' })).rejects.toThrow('unknown duration unit "x" (use s, m, h or d)');
    await expect(m.updateSettings({ pin_hold: '' })).rejects.toThrow('invalid duration ""');
    const res = await m.updateSettings({ pin_hold: ' 1.5 H ' });
    expect(res.updated[0]).toMatchObject({ name: 'pin_hold', value: '1.5 H', value_s: 5400 });
    const bare = await m.updateSettings({ disk_cache_ttl: '90' });
    expect(bare.updated[0]).toMatchObject({ value: '90', value_s: 90 });
  });

  it('a restart-only setting is 409, an unknown one 404', async () => {
    const m = make();
    await expect(m.updateSettings({ max_seq: 4096 })).rejects.toMatchObject({ status: 409, setting: 'max_seq' });
    await expect(m.updateSettings({ nope: 1 })).rejects.toMatchObject({ status: 404, setting: 'nope' });
    await expect(m.resetSetting('max_seq')).rejects.toMatchObject({ status: 409 });
    await expect(m.resetSetting('nope')).rejects.toMatchObject({ status: 404 });
  });

  it('a patch is all or none', async () => {
    const m = make();
    await expect(m.updateSettings({ temperature: 0.5, top_p: 9 })).rejects.toThrow();
    const { settings } = await m.settings();
    expect(settings.find((s) => s.name === 'temperature')!.value).toBe(1);
  });

  it('a valid change is stored, marked as database, logged; reset undoes it', async () => {
    const m = make();
    const before = (await m.settingsLog()).entries.length;
    const res = await m.updateSettings({ top_p: 0.8 });
    expect(res.updated).toHaveLength(1);
    expect(res.updated[0]).toMatchObject({ name: 'top_p', value: 0.8, source: 'database' });
    const log = (await m.settingsLog()).entries;
    expect(log).toHaveLength(before + 1);
    expect(log[0]).toMatchObject({ name: 'top_p', old: 0.95, new: 0.8, source: 'dashboard' });
    const back = await m.resetSetting('top_p');
    expect(back.setting).toMatchObject({ value: 0.95, source: 'default' });
  });

  it('reset returns to the flag value (and its seconds) when there is one', async () => {
    const m = make();
    await m.updateSettings({ idle_unload: '2h' });
    const back = await m.resetSetting('idle_unload');
    expect(back.setting).toMatchObject({ value: '30m', value_s: 1800, source: 'flag' });
  });

  it('numbers typed as strings are accepted and stored as numbers', async () => {
    const res = await make().updateSettings({ top_k: '30' });
    expect(res.updated[0]!.value).toBe(30);
  });

  it('every setting is fully specified: explicit nulls, applies per spec 7.1, value_s only for durations', async () => {
    const { settings } = await make().settings();
    const by = (n: string) => settings.find((s) => s.name === n)!;
    expect(by('idle_unload').applies).toBe('timer re-armed on change');
    expect(by('pin_hold').applies).toBe('next release');
    expect(by('disk_cache_ttl').applies).toBe('next expiry pass');
    expect(by('temperature').applies).toBe('next request');
    expect(by('thinking_nudges').applies).toBe('next request');
    expect(by('max_seq').applies).toBeNull();
    for (const s of settings) {
      for (const k of ['min', 'max', 'step', 'options', 'help', 'applies', 'value_s', 'flag_value'] as const) expect(s[k]).not.toBeUndefined();
      expect(s.value_s === null).toBe(s.type !== 'duration');
    }
    expect(by('idle_unload').value_s).toBe(1800);
    expect(by('pin_hold').value_s).toBe(60);
    expect(by('disk_cache_ttl').value_s).toBe(259200);
  });
});
