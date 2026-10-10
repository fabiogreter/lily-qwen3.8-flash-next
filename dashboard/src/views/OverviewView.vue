<script setup lang="ts">
import { computed } from 'vue';
import type { HistoryPoint } from '../api/types';
import Enso from '../components/Enso.vue';
import InkMeter from '../components/InkMeter.vue';
import { avgMaxSeries } from '../components/chartSeries';
import LineChart, { type SeriesDef } from '../components/LineChart.vue';
import RangeSwitch from '../components/RangeSwitch.vue';
import SectionHead from '../components/SectionHead.vue';
import StatusMark from '../components/StatusMark.vue';
import { HISTORY_POLL_MS, useDashboard } from '../composables/context';
import { useResource } from '../composables/useResource';
import { bucketLabel, bytes, duration, energy, pct, rate, timeLabel, tokens, whole } from '../lib/format';

type PressurePoint = Pick<HistoryPoint, 'memory_pressure'>;

const { backend, status: statusRes, range } = useDashboard();
const history = useResource((signal) => backend.history(range.value, signal), { intervalMs: HISTORY_POLL_MS, sources: [range] });

const st = statusRes.data;
const hist = history.data;

// The ring closes at the interval's best decode rate; without one it closes at the live rate.
const ensoMax = computed(() => hist.value?.summary.decode.max ?? st.value?.live.decode_tps ?? 0);
const stateLabel = computed(() => {
  const s = st.value?.state;
  if (!s) return '';
  if (s === 'busy') return 'Serving';
  const name = { ready: 'Ready', unloaded: 'Unloaded', loading: 'Loading', stopping: 'Stopping' }[s];
  return s === 'ready' ? name + ', idle' : name;
});

const pressureSummary = computed(() => {
  const h = hist.value;
  if (!h) return '';
  const known = h.points.filter((p) => p.memory_pressure != null);
  if (!known.length) return 'No readings in this interval.';
  const warn = known.filter((p) => p.memory_pressure === 'warning').length;
  const crit = known.filter((p) => p.memory_pressure === 'critical').length;
  if (!warn && !crit) return 'Normal for the whole interval.';
  const b = bucketLabel(h.range);
  const parts = [warn ? `${warn} ${b} bucket${warn > 1 ? 's' : ''} at warning` : '', crit ? `${crit} critical` : ''].filter(Boolean);
  const swap = Math.max(...known.map((p) => p.swap_bytes ?? 0));
  return `Normal except ${parts.join(' and ')}; swap peaked at ${bytes(swap)}.`;
});
const stripColour = (p: PressurePoint): string =>
  p.memory_pressure === 'critical' ? 'var(--crit)' : p.memory_pressure === 'warning' ? 'var(--warn)' : p.memory_pressure === 'normal' ? 'var(--hair)' : 'transparent';

const decodeSeries = avgMaxSeries('decode_tps_avg', 'decode_tps_max');
const prefillSeries = avgMaxSeries('prefill_tps_avg', 'prefill_tps_max');
const loadSeries: SeriesDef[] = [
  { key: 'gpu_active', label: 'GPU', color: 'var(--s1)', area: true },
  { key: 'cpu_load', label: 'CPU', color: 'var(--s2)', width: 1.5 },
];
const powerSeries: SeriesDef[] = [
  { key: 'power_w', label: 'system', color: 'var(--s1)', area: true },
  { key: 'gpu_power_w', label: 'GPU', color: 'var(--s2)', width: 1.5 },
];
</script>

<template>
  <main v-if="st">
    <div class="statusline">
      <span class="st"><span class="dot" :class="{ busy: st.state === 'busy' }"></span>{{ stateLabel }}</span>
      <span class="sep">·</span><span>{{ st.model }}</span>
      <span class="sep">·</span><span>up {{ duration(st.uptime_s) }}</span>
      <span class="sep">·</span
      ><span
        >weights {{ st.tiers.weights.pinned ? 'pinned' : 'not pinned' }}<template v-if="st.state !== 'busy' && st.pin_hold_s != null">, released in {{ duration(st.pin_hold_s) }}</template></span
      >
      <template v-if="st.idle_unload_s"
        ><span class="sep">·</span><span>unloads after {{ duration(st.idle_unload_s) }} idle</span></template
      >
    </div>

    <section class="hero">
      <div class="enso-block">
        <div class="enso-wrap" :title="`The ring closes at the ${range} maximum, ${rate(ensoMax)} tok/s`">
          <Enso :value="st.live.decode_tps" :max="ensoMax" />
          <div class="enso-center">
            <div class="big">{{ rate(st.live.decode_tps) }}</div>
            <div class="unit">decode tok/s, now</div>
          </div>
        </div>
        <div class="enso-foot">
          <span
            >prefill
            <template v-if="st.live.prefill_tps"
              ><b>{{ rate(st.live.prefill_tps) }}</b
              >tok/s</template
            ><b v-else>idle</b></span
          >
          <span
            >acceptance <b>{{ pct(st.live.acceptance) }}</b></span
          >
          <span
            ><b>{{ st.live.requests_per_min }}</b
            >req/min</span
          >
        </div>
      </div>
      <div class="tiles">
        <div class="tile">
          <div class="k">Sessions</div>
          <div class="v">{{ st.queue.running }}<small>of {{ st.queue.max_batch }} batch rows</small></div>
          <div class="m">{{ st.tiers.gpu.entries }} resident · {{ st.tiers.disk ? st.tiers.disk.entries + ' on disk' : 'disk off' }} · queue {{ st.queue.waiting }}/{{ st.queue.capacity }}</div>
        </div>
        <div class="tile">
          <div class="k">Load</div>
          <div class="v">{{ pct(st.machine.gpu_active) }}<small>GPU</small></div>
          <div class="m">CPU {{ pct(st.machine.cpu_load) }} · {{ whole(st.machine.gpu_freq_mhz, 'MHz') }}</div>
        </div>
        <div class="tile">
          <div class="k">Power</div>
          <div v-if="st.machine.power" class="v">{{ st.machine.power.sys_w === null ? '–' : rate(st.machine.power.sys_w) }}<small>W system</small></div>
          <div v-else class="v" :title="`Sampler ${st.machine.sampler}`"><small>unavailable</small></div>
          <div class="m">
            <template v-if="st.machine.power">GPU {{ rate(st.machine.power.gpu_w) }} W</template>
            <template v-if="st.machine.power && st.machine.temp_c"> · </template>
            <template v-if="st.machine.temp_c">GPU {{ whole(st.machine.temp_c.gpu) }}° · CPU {{ whole(st.machine.temp_c.cpu) }}°</template>
          </div>
        </div>
        <div class="tile">
          <div class="k">Memory</div>
          <div class="v"><StatusMark :level="st.machine.memory_pressure" /></div>
          <div class="m">
            {{ bytes(st.machine.ram_used_bytes) }} of {{ bytes(st.machine.ram_total_bytes) }} · lily {{ bytes(st.machine.lily_resident_bytes) }} · swap
            {{ bytes(st.machine.swap_used_bytes) }}
          </div>
        </div>
      </div>
    </section>

    <section class="section">
      <SectionHead title="Throughput" :seed="1"><RangeSwitch v-model="range" /></SectionHead>
      <template v-if="hist">
        <div class="summary">
          <div>
            <div class="k" title="Tokens over the time a request was decoding, not wall time">decode, average</div>
            <div class="v">{{ rate(hist.summary.decode.avg) }}<small>tok/s</small></div>
          </div>
          <div>
            <div class="k" title="The best 1-minute bucket in the interval">decode, max</div>
            <div class="v">{{ rate(hist.summary.decode.max) }}<small>tok/s</small></div>
          </div>
          <div>
            <div class="k" title="Per computed token: a cache hit does not inflate it">prefill, average</div>
            <div class="v">{{ rate(hist.summary.prefill.avg) }}<small>max {{ rate(hist.summary.prefill.max) }}</small></div>
          </div>
          <div>
            <div class="k">tokens</div>
            <div class="v">{{ tokens(hist.summary.tokens_generated) }}<small>out, {{ tokens(hist.summary.tokens_prefilled) }} in</small></div>
          </div>
          <div>
            <div class="k">requests</div>
            <div class="v">{{ hist.summary.requests.toLocaleString('en-US') }}<small>busy {{ pct(hist.summary.busy_share) }}</small></div>
          </div>
          <div>
            <div class="k">energy</div>
            <div class="v">{{ energy(hist.summary.energy_wh) }}<small>avg {{ rate(hist.summary.power_avg_w) }} W</small></div>
          </div>
        </div>
        <div class="stack" style="margin-top: 22px">
          <LineChart title="Decode" unit="tok/s" :points="hist.points" :range="hist.range" :height="240" :series="decodeSeries" />
          <LineChart title="Prefill" unit="tok/s" :points="hist.points" :range="hist.range" :height="150" :series="prefillSeries" />
        </div>
      </template>
    </section>

    <div class="lower">
      <section class="section">
        <SectionHead title="Caches" :seed="3" />
        <div class="tiers">
          <div class="tier">
            <div class="name">GPU sessions</div>
            <div class="val">{{ bytes(st.tiers.gpu.used_bytes) }} of {{ bytes(st.tiers.gpu.budget_bytes) }} · {{ st.tiers.gpu.entries }}/{{ st.tiers.gpu.max_entries }}</div>
            <div class="meter"><InkMeter :parts="[{ v: st.tiers.gpu.used_bytes, color: 'var(--s1)' }]" :total="st.tiers.gpu.budget_bytes ?? 0" /></div>
          </div>
          <div class="tier">
            <div class="name">
              Disk tier
              <span v-if="st.tiers.disk" class="legend"
                ><span><span class="key" style="border-color: var(--s1)"></span>evicted</span
                ><span><span class="key" style="border-color: var(--s2)"></span>durable</span></span
              >
            </div>
            <div v-if="st.tiers.disk" class="val">{{ bytes(st.tiers.disk.used_bytes) }} of {{ bytes(st.tiers.disk.budget_bytes) }} · {{ st.tiers.disk.durable_entries }} durable</div>
            <div v-else class="val na">off</div>
            <div v-if="st.tiers.disk" class="meter">
              <InkMeter
                :parts="[
                  { v: st.tiers.disk.used_bytes - st.tiers.disk.durable_bytes, color: 'var(--s1)' },
                  { v: st.tiers.disk.durable_bytes, color: 'var(--s2)' },
                ]"
                :total="st.tiers.disk.budget_bytes"
              />
            </div>
          </div>
          <div class="tier">
            <div class="name">n-gram table</div>
            <div v-if="st.tiers.ngram" class="val">{{ pct(st.tiers.ngram.resident_share) }} of {{ bytes(st.tiers.ngram.bytes) }} resident</div>
            <div v-else class="val na">none</div>
            <div v-if="st.tiers.ngram" class="meter"><InkMeter :parts="[{ v: st.tiers.ngram.resident_share, color: 'var(--s1)' }]" :total="1" /></div>
          </div>
          <div class="tier">
            <div class="name">Weights</div>
            <div class="val">{{ bytes(st.tiers.weights.bytes) }} · <StatusMark v-if="st.tiers.weights.pinned" level="normal" text="pinned" /><span v-else>not pinned</span></div>
          </div>
          <div class="tier">
            <div class="name">Expert cache</div>
            <div v-if="!st.tiers.experts" class="val na" title="Small-machine mode only">n/a</div>
            <div v-else class="val">{{ pct(st.tiers.experts.hit_rate) }} hit · {{ st.tiers.experts.used }} of {{ st.tiers.experts.slots }} slots</div>
          </div>
        </div>
      </section>

      <section v-if="hist" class="section">
        <SectionHead title="Machine" :seed="6" :aside="hist.range" />
        <div class="stack">
          <LineChart title="Load" unit="%" :points="hist.points" :range="hist.range" :y-max="1" percent :height="130" :series="loadSeries" />
          <LineChart title="Power" unit="W" :points="hist.points" :range="hist.range" :height="130" :series="powerSeries" />
          <div>
            <div class="chart-title">
              <h3>Memory pressure</h3>
              <div class="legend"><StatusMark level="normal" /><StatusMark level="warning" /><StatusMark level="critical" /></div>
            </div>
            <div class="strip" role="img" :aria-label="pressureSummary">
              <span
                v-for="p in hist.points"
                :key="p.t"
                :title="timeLabel(p.t, hist.range) + ': ' + (p.memory_pressure ?? 'no reading')"
                :style="{ background: stripColour(p) }"
              ></span>
            </div>
            <div class="muted pressure-summary">{{ pressureSummary }}</div>
          </div>
        </div>
      </section>
    </div>
  </main>
  <p v-else-if="statusRes.error.value" class="inline-error" role="alert">{{ statusRes.error.value }}</p>
</template>

<style scoped>
.statusline {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
  color: var(--ink-2);
  margin-top: 22px;
  font-size: 14px;
}
.statusline .st {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  font-family: var(--f-head);
  font-size: 18px;
  color: var(--ink);
}
.statusline .st .dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  border: 1.5px solid var(--ai);
  display: inline-block;
}
.statusline .st .dot.busy {
  background: var(--ai);
}
.statusline .sep {
  color: var(--base);
}

/* Hero: the ensō beside four quiet blocks */
.hero {
  display: grid;
  grid-template-columns: minmax(280px, 5fr) minmax(0, 7fr);
  gap: 48px;
  align-items: center;
  margin-top: 18px;
}
.hero > * {
  min-width: 0;
}
.enso-wrap {
  position: relative;
  max-width: 340px;
  margin: 0 auto;
}
.enso-wrap :deep(svg) {
  display: block;
  width: 100%;
  height: auto;
}
.enso-center {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  pointer-events: none;
}
.enso-center .big {
  font-family: var(--f-head);
  font-weight: var(--w-head);
  font-size: 64px;
  line-height: 1;
  font-variant-numeric: tabular-nums;
}
.enso-center .unit {
  color: var(--ink-2);
  margin-top: 6px;
}
.enso-foot {
  display: flex;
  justify-content: center;
  gap: 22px;
  flex-wrap: wrap;
  color: var(--ink-2);
  font-size: 13.5px;
  margin-top: -30px;
  position: relative;
}
.enso-foot b {
  color: var(--ink);
  font-family: var(--f-num);
  font-weight: var(--w-num);
  font-size: 17px;
  font-variant-numeric: tabular-nums;
  margin: 0 2px;
}
.tiles {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0 40px;
}
.tile {
  padding: 20px 0 22px;
  border-top: 1px solid var(--hair);
}
.tile:nth-child(-n + 2) {
  border-top: 0;
}
.tile .k {
  color: var(--ink-2);
  font-size: 13px;
  letter-spacing: 0.02em;
  margin-bottom: 4px;
}
.tile .v {
  font-family: var(--f-num);
  font-weight: var(--w-num);
  font-size: 34px;
  line-height: 1.15;
  font-variant-numeric: tabular-nums;
}
.tile .v small {
  font-size: 14px;
  color: var(--ink-2);
  margin-left: 5px;
  font-family: var(--f-label);
}
.tile .v :deep(.status) {
  font-size: 24px;
}
.tile .v :deep(.status .ico) {
  width: 20px;
  height: 20px;
  font-size: 13px;
}
.tile .m {
  color: var(--ink-2);
  font-size: 13px;
  margin-top: 4px;
  font-variant-numeric: tabular-nums;
}

/* Lower: caches and machine, quieter */
.lower {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
  gap: 0 64px;
}
.lower > * {
  min-width: 0;
}
.lower :deep(.section-head h2) {
  font-size: 19px;
}
.tiers {
  display: grid;
  gap: 16px;
}
.tier {
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 2px 14px;
  align-items: center;
}
.tier .name {
  font-family: var(--f-head);
  font-size: 15px;
}
.tier .val {
  color: var(--ink-2);
  font-variant-numeric: tabular-nums;
  text-align: right;
  font-size: 13px;
}
.tier .na {
  color: var(--ink-2);
  font-style: italic;
}
.tier .meter {
  grid-column: 1 / -1;
}
.tier .meter :deep(svg) {
  display: block;
  width: 100%;
  height: 16px;
}
.legend {
  display: inline-flex;
  gap: 10px;
  margin-left: 8px;
  color: var(--ink-2);
  font-family: var(--f-label);
  font-size: 12.5px;
}
.legend .key {
  display: inline-block;
  width: 16px;
  height: 0;
  border-top: 2px solid;
  vertical-align: middle;
  margin-right: 6px;
  border-radius: 2px;
}
.chart-title {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 6px;
  flex-wrap: wrap;
}
.chart-title h3 {
  font-size: 16px;
}
.chart-title .legend {
  margin-left: 0;
  gap: 14px;
}
.strip {
  display: flex;
  gap: 1px;
  height: 14px;
  margin-top: 6px;
}
.strip span {
  flex: 1;
  border-radius: 1px;
}
.pressure-summary {
  font-size: 12.5px;
  margin-top: 6px;
}

@media (max-width: 980px) {
  .hero {
    grid-template-columns: 1fr;
    gap: 20px;
  }
  .lower {
    grid-template-columns: minmax(0, 1fr);
  }
}
</style>
