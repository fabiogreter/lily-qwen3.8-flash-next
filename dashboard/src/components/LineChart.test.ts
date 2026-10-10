import { mount } from '@vue/test-utils';
import { describe, expect, it } from 'vitest';
import type { HistoryPoint } from '../api/types';
import LineChart from './LineChart.vue';

function pt(i: number, v: number | null): HistoryPoint {
  return {
    t: Date.UTC(2026, 9, 10, 14, i),
    busy_share: v == null ? 0 : 0.5,
    requests: 0,
    decode_tps_avg: v,
    decode_tps_max: v == null ? null : v * 1.2,
    prefill_tps_avg: null,
    prefill_tps_max: null,
    decode_tokens: 0,
    prefill_tokens: 0,
    acceptance: null,
    cache_hit: null,
    queue_ms_p50: null,
    queue_ms_max: null,
    sessions_max: 0,
    cpu_load: 0,
    gpu_active: null,
    memory_pressure: null,
    swap_bytes: 0,
    power_w: null,
    gpu_power_w: null,
    gpu_freq_mhz: null,
  };
}

const series = [
  { key: 'decode_tps_avg', label: 'average', color: 'var(--s1)', area: true },
  { key: 'decode_tps_max', label: 'max', color: 'var(--s2)' },
] as const;

describe('LineChart', () => {
  const points = [pt(0, 50), pt(1, 60), pt(2, null), pt(3, 70), pt(4, null), pt(5, 80), pt(6, 90)];
  const mountIt = (props = {}) => mount(LineChart, { props: { title: 'Decode', unit: 'tok/s', points, series: [...series], range: '1h', ...props } });

  it('draws gaps as gaps: two line subpaths per series and a dot for the lone bucket', () => {
    const w = mountIt();
    const lines = w.findAll('path[fill="none"]').filter((p) => p.attributes('stroke') === 'var(--s1)');
    expect(lines).toHaveLength(1);
    expect(lines[0]!.attributes('d')!.match(/M/g)).toHaveLength(2);
    expect(w.findAll('circle').filter((c) => c.attributes('fill') === 'var(--s1)')).toHaveLength(1);
  });

  it('has a legend for two series, none for one', () => {
    expect(mountIt().find('.legend').exists()).toBe(true);
    expect(mountIt({ series: [series[0]] }).find('.legend').exists()).toBe(false);
  });

  it('a table twin lists every bucket, gaps as a dash', () => {
    const w = mountIt();
    const rows = w.findAll('tbody tr');
    expect(rows).toHaveLength(7);
    expect(rows[2]!.text()).toContain('–');
    expect(rows[0]!.text()).toContain('50');
  });

  it('arrow keys move the crosshair and show a tooltip', async () => {
    const w = mountIt();
    const svg = w.find('svg');
    await svg.trigger('keydown', { key: 'ArrowLeft' });
    expect(w.find('.tip').exists()).toBe(true);
    expect(w.find('.tip .val').text()).toBe('80'); // last bucket is 90; one step left is 80
    await svg.trigger('blur');
    expect(w.find('.tip').exists()).toBe(false);
  });

  it('one bucket gives one x label, two give two, no repeats', () => {
    expect(mountIt({ points: [pt(0, 5)] }).findAll('text.axis').filter((t) => /^\d\d:\d\d$/.test(t.text()))).toHaveLength(1);
    expect(mountIt({ points: [pt(0, 5), pt(1, 6)] }).findAll('text.axis').filter((t) => /^\d\d:\d\d$/.test(t.text()))).toHaveLength(2);
  });

  it('the tooltip readout is also in a polite live region', async () => {
    const w = mountIt();
    const live = w.find('[aria-live="polite"]');
    expect(live.exists()).toBe(true);
    expect(live.text()).toBe('');
    await w.find('svg').trigger('keydown', { key: 'ArrowLeft' });
    expect(live.text()).toBe('14:05: average 80, max 96');
    expect(w.find('.tip').attributes('aria-hidden')).toBe('true');
  });

  it('empty data renders without throwing', () => {
    expect(() => mountIt({ points: [] })).not.toThrow();
  });
});
