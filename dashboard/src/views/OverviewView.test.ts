import { flushPromises, mount } from '@vue/test-utils';
import { describe, expect, it } from 'vitest';
import { ref } from 'vue';
import { dashboardKey } from '../composables/context';
import type { Status } from '../api/types';
import { MockBackend } from '../mock/backend';
import EngineSchematic from '../components/EngineSchematic.vue';
import OverviewView from './OverviewView.vue';

const clock = (): number => Date.UTC(2026, 9, 10, 14, 20);

/** A server whose sampler failed and whose disk tier is off: everything optional is null. */
async function bare(): Promise<Status> {
  const s = await new MockBackend({ clock }).status();
  return {
    ...s,
    idle_unload_s: null,
    pin_hold_s: null,
    live: { ...s.live, decode_tps: 0, prefill_tps: 0, acceptance: null },
    machine: {
      sampler: 'failed',
      cpu_load: null,
      gpu_active: null,
      gpu_freq_mhz: null,
      memory_pressure: null,
      ram_total_bytes: null,
      ram_used_bytes: null,
      lily_resident_bytes: null,
      swap_used_bytes: null,
      power: null,
      temp_c: null,
    },
    tiers: { ...s.tiers, disk: null },
  };
}

class NoHistoryMax extends MockBackend {
  async history(range: Parameters<MockBackend['history']>[0]) {
    const h = await super.history(range);
    return { ...h, summary: { ...h.summary, decode: { avg: null, max: null }, prefill: { avg: null, max: null }, energy_wh: null, power_avg_w: null, lily_energy_wh: null } };
  }
}

describe('null-safe rendering', () => {
  it('the Overview renders with no sampler, no disk tier and no throughput maximum', async () => {
    const backend = new NoHistoryMax({ clock });
    const wrapper = mount(OverviewView, {
      global: { provide: { [dashboardKey as symbol]: { backend, range: ref('1h'), status: { data: ref(await bare()), error: ref(null), refresh: async () => {} } } } },
    });
    await flushPromises();
    const text = wrapper.text();
    expect(text).toContain('unavailable');
    expect(text).toContain('disk off');
    expect(text).not.toMatch(/NaN|undefined|null|Infinity/);
    // the ring has no history maximum and nothing live: it does not divide by zero
    expect(wrapper.find('svg[role="img"]').attributes('aria-label')).toContain('0 percent');
  });

  it('with live decode but no history maximum the ring closes at the live rate', async () => {
    const s = await bare();
    s.live.decode_tps = 50;
    const backend = new NoHistoryMax({ clock });
    const wrapper = mount(OverviewView, {
      global: { provide: { [dashboardKey as symbol]: { backend, range: ref('1h'), status: { data: ref(s), error: ref(null), refresh: async () => {} } } } },
    });
    await flushPromises();
    expect(wrapper.find('svg[role="img"]').attributes('aria-label')).toContain('100 percent');
  });

  it('the schematic renders with nulls and a zero-capacity queue', async () => {
    const s = await bare();
    s.queue = { waiting: 1, running: 0, max_batch: 4, capacity: 0 };
    const wrapper = mount(EngineSchematic, { props: { st: s, mtp: null, bind: null, ngram: null } });
    const html = wrapper.html();
    expect(html).not.toMatch(/NaN|Infinity|undefined/);
    expect(wrapper.text()).toContain('n/a');
    expect(wrapper.text()).toContain('off');
  });

  it("lily's own power shows as an estimate next to the system's, and not at all without one", async () => {
    const s = await new MockBackend({ clock }).status();
    s.machine.power = {
      cpu_w: 7.5,
      gpu_w: 48.25,
      ane_w: 0,
      sys_w: 71.5,
      rest_w: 15.75,
      lily: { w: 50.5, cpu_w: 3.25, gpu_w: 47.25 },
    };
    const schematic = mount(EngineSchematic, { props: { st: s, mtp: 2, bind: null, ngram: null } });
    expect(schematic.text()).toContain('lily CPU+GPU 50.5 W (est.)');
    const backend = new MockBackend({ clock });
    const overview = mount(OverviewView, {
      global: { provide: { [dashboardKey as symbol]: { backend, range: ref('1h'), status: { data: ref(s), error: ref(null), refresh: async () => {} } } } },
    });
    await flushPromises();
    expect(overview.text()).toMatch(/lily CPU\+GPU 50\.5 W \(est\.\)/);
    // The system's remainder is shown as such, attributed to no one.
    expect(overview.text()).toMatch(/rest of system 15\.8 W \(not attributed\)/);
    // The history's power chart has lily's series, and the summary its energy.
    expect(overview.text()).toContain('lily CPU+GPU (est.)');
    expect(overview.text()).toMatch(/lily CPU\+GPU \d+ Wh \(est\.\)/);

    s.machine.power = { ...s.machine.power, lily: null };
    const without = mount(EngineSchematic, { props: { st: s, mtp: 2, bind: null, ngram: null } });
    expect(without.text()).not.toContain('lily CPU+GPU');
    expect(without.text()).toContain('71.5 W');

    // Without a system power there is no remainder either.
    s.machine.power = { ...s.machine.power, sys_w: null, rest_w: null };
    const unknown = mount(OverviewView, {
      global: { provide: { [dashboardKey as symbol]: { backend, range: ref('1h'), status: { data: ref(s), error: ref(null), refresh: async () => {} } } } },
    });
    await flushPromises();
    expect(unknown.text()).not.toContain('rest of system');
  });

  it('pin hold and clocks are formatted, not raw', async () => {
    const s = await new MockBackend({ clock }).status();
    s.state = 'ready';
    s.pin_hold_s = 125;
    s.machine.gpu_freq_mhz = 1568.6;
    s.machine.temp_c = { cpu: 64.4, gpu: 71.6 };
    const wrapper = mount(EngineSchematic, { props: { st: s, mtp: 2, bind: null, ngram: null } });
    expect(wrapper.text()).toContain('hold 2m');
    expect(wrapper.text()).toContain('1569 MHz');
    expect(wrapper.text()).toContain('CPU 64° · GPU 72°C');
  });
});
