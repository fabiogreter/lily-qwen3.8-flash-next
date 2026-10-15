<script setup lang="ts">
import { computed } from 'vue';
import type { Status } from '../api/types';
import { flowPath, flowWidth } from '../lib/brush';
import { bytes, duration, pct, rate, whole } from '../lib/format';
import { levelOf } from '../lib/status';

// The engine as a diagram: clients, front, queue, batch, the GPU engine, its
// tiers and the machine beneath. Brush width is tokens per second on the
// request path; hairlines are what the engine reads or moves (no rate is
// measured for those); dashed boxes are not active in this mode.
const props = defineProps<{
  st: Status;
  /** MTP drafts per step (`mtp_drafts` setting), when known. */
  mtp: number | null;
  /** Listen address (`bind` setting), when known. */
  bind: string | null;
  /** n-gram table mode (`ngram_table` setting), when known. */
  ngram: string | null;
}>();

const busy = computed(() => props.st.state === 'busy');
const tps = computed(() => (props.st.live.decode_tps || 0) + (props.st.live.prefill_tps || 0));
const pipe = computed(() => {
  const w = flowWidth(tps.value);
  return ([[130, 166], [310, 346], [490, 526], [670, 706]] as const).map(([a, b]) => flowPath(a, 158, b, w));
});
const gpu = computed(() => props.st.tiers.gpu);
const disk = computed(() => props.st.tiers.disk);
const m = computed(() => props.st.machine);
const share = (part: number, whole: number): number => (whole > 0 ? part / whole : 0);
const gpuShare = computed(() => share(gpu.value.used_bytes, gpu.value.budget_bytes ?? 0));
const diskShare = computed(() => (disk.value ? share(disk.value.used_bytes - disk.value.durable_bytes, disk.value.budget_bytes) : 0));
const durShare = computed(() => (disk.value ? share(disk.value.durable_bytes, disk.value.budget_bytes) : 0));
const queueShare = computed(() => share(props.st.queue.waiting, props.st.queue.capacity));
const slots = computed(() => Array.from({ length: props.st.queue.max_batch }, (_, i) => i < props.st.queue.running));
const pressure = computed(() => (m.value.memory_pressure ? levelOf(m.value.memory_pressure) : null));
const power = computed(() => m.value.power);
const cpuLoad = computed(() => m.value.cpu_load ?? 0);
const gpuActive = computed(() => m.value.gpu_active ?? 0);
const accept = computed(() => props.st.live.acceptance);
const pinned = computed(() => props.st.tiers.weights.pinned);
const temps = computed(() => (m.value.temp_c ? `CPU ${Math.round(m.value.temp_c.cpu)}° · GPU ${Math.round(m.value.temp_c.gpu)}°C` : 'temperatures n/a'));
</script>

<template>
  <svg class="schematic" viewBox="0 24 1180 536" role="img" aria-label="Lily engine schematic with live values">
    <title>Brush width: tokens per second on the request path (prefill and decode). Hairlines: what the engine reads or moves, not measured as a rate. Dashed: not active in this mode.</title>
    <defs>
      <marker id="hairhead" viewBox="0 0 10 10" refX="8" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
        <path d="M0 1L9 5L0 9z" class="arrowhead" />
      </marker>
    </defs>

    <!-- machine ground -->
    <line class="hair" x1="830" y1="384" x2="830" y2="430" />
    <line class="hair" x1="565" y1="384" x2="565" y2="430" />
    <rect class="ground" x="20" y="430" width="1140" height="110" rx="4" />
    <text class="n-title" x="40" y="460">Machine</text>
    <g transform="translate(200,456)">
      <text class="n-sub" x="0" y="4">CPU load</text>
      <text class="n-val" x="0" y="31">{{ pct(m.cpu_load) }}</text>
      <rect class="track" x="0" y="44" width="110" height="3" />
      <rect class="fill" x="0" y="43" :width="Math.max(3, 110 * cpuLoad)" height="5" filter="url(#ink-line)" />
    </g>
    <g transform="translate(340,456)">
      <text class="n-sub" x="0" y="4">GPU active</text>
      <text class="n-val" x="0" y="31">{{ pct(m.gpu_active) }}</text>
      <rect class="track" x="0" y="44" width="110" height="3" />
      <rect class="fill" x="0" y="43" :width="Math.max(3, 110 * gpuActive)" height="5" filter="url(#ink-line)" />
    </g>
    <g transform="translate(480,456)">
      <text class="n-sub" x="0" y="4">Power, system</text>
      <text class="n-val" x="0" y="31">{{ power ? (power.sys_w === null ? '–' : power.sys_w.toFixed(1) + ' W') : 'n/a' }}</text>
      <text v-if="power" class="n-small" x="0" y="54">CPU {{ power.cpu_w.toFixed(1) }} · GPU {{ power.gpu_w.toFixed(1) }} W</text>
      <text v-if="power?.lily" class="n-small" x="0" y="72">lily CPU+GPU {{ power.lily.w.toFixed(1) }} W (est.)</text>
    </g>
    <g transform="translate(660,456)" :class="pressure?.cls">
      <text class="n-sub" x="0" y="4">Memory pressure</text>
      <template v-if="pressure">
        <circle class="ring" cx="8" cy="24" r="7.5" fill="none" stroke-width="1.5" />
        <text x="8" y="28.5" text-anchor="middle" font-size="11" font-weight="500">{{ pressure.icon }}</text>
        <text class="n-val" x="22" y="31">{{ pressure.label }}</text>
      </template>
      <text v-else class="n-val" x="0" y="31">–</text>
      <text class="n-small" x="0" y="54">swap {{ bytes(m.swap_used_bytes) }}</text>
    </g>
    <g transform="translate(830,456)">
      <text class="n-sub" x="0" y="4">Memory</text>
      <text class="n-val" x="0" y="31">{{ bytes(m.ram_used_bytes) }}</text>
      <text class="n-small" x="0" y="54">of {{ bytes(m.ram_total_bytes) }} · lily {{ bytes(m.lily_resident_bytes) }}</text>
    </g>
    <g transform="translate(1010,456)">
      <text class="n-sub" x="0" y="4">GPU clock, temps</text>
      <text class="n-val" x="0" y="31">{{ whole(m.gpu_freq_mhz, 'MHz') }}</text>
      <text class="n-small" x="0" y="54">{{ temps }}</text>
    </g>

    <!-- request path: brush strokes -->
    <g filter="url(#ink-line)">
      <path v-for="(d, i) in pipe" :key="i" :d="d" :class="['brush', busy ? '' : 'idle']" />
    </g>
    <template v-if="busy">
      <path class="flow" d="M134 158H156" /><path class="flow" d="M314 158H336" />
      <path class="flow" d="M494 158H516" /><path class="flow" d="M674 158H696" />
    </template>
    <text class="e-label" x="702" y="108" text-anchor="end">{{ Math.round(tps) }} tok/s</text>
    <line class="hair" x1="690" y1="114" x2="690" y2="148" style="stroke: var(--hair)" />

    <!-- engine and tiers: hairlines -->
    <path class="hair" d="M950 104C970 104 970 90 986 90" marker-end="url(#hairhead)" marker-start="url(#hairhead)" />
    <text class="e-label" x="968" y="78" text-anchor="middle">state</text>
    <path class="hair" d="M1060 140V172" marker-end="url(#hairhead)" />
    <path class="hair" d="M1090 172V140" marker-end="url(#hairhead)" />
    <text class="e-label" x="1052" y="161" text-anchor="end">evict</text>
    <text class="e-label" x="1098" y="161">restore</text>
    <path class="hair" d="M565 300C565 270 700 270 740 258" marker-end="url(#hairhead)" />
    <text class="e-label" x="470" y="288">weights / token</text>
    <path class="hair" d="M830 300V260" marker-end="url(#hairhead)" />
    <text class="e-label" x="838" y="284">rows / token</text>
    <path class="hair dash" d="M1000 320C1000 290 960 280 952 258" />

    <!-- clients -->
    <rect class="node" x="20" y="120" width="110" height="76" rx="3" />
    <text class="n-title" x="34" y="142">Clients</text>
    <text class="n-val" x="34" y="168">{{ st.live.requests_per_min }}</text>
    <text class="n-sub" x="34" y="186">requests / min</text>

    <!-- http front -->
    <rect class="node" x="170" y="120" width="140" height="76" rx="3" />
    <text class="n-title" x="184" y="142">HTTP front</text>
    <text class="n-small" x="184" y="166">{{ bind ?? 'lily' }}</text>
    <text class="n-sub" x="184" y="186">OpenAI API, /v1</text>

    <!-- queue -->
    <rect class="node" x="350" y="120" width="140" height="76" rx="3" />
    <text class="n-title" x="364" y="142">Queue</text>
    <text class="n-val" x="364" y="168">{{ st.queue.waiting }}<tspan class="n-sub" dx="5">of {{ st.queue.capacity }} waiting</tspan></text>
    <rect class="track" x="364" y="180" width="112" height="3" />
    <rect v-if="st.queue.waiting" class="fill" x="364" y="179" :width="Math.max(4, 112 * queueShare)" height="5" />

    <!-- batch -->
    <rect class="node" x="530" y="120" width="140" height="76" rx="3" />
    <text class="n-title" x="544" y="142">Batch</text>
    <text class="n-val" x="544" y="168">{{ st.queue.running }}<tspan class="n-sub" dx="5">of {{ st.queue.max_batch }} running</tspan></text>
    <rect v-for="(on, i) in slots" :key="i" :class="['slot', on ? 'on' : '']" :x="544 + i * 22" y="177" width="18" height="9" rx="1" />

    <!-- engine -->
    <rect class="node engine" x="710" y="60" width="240" height="196" rx="4" />
    <text class="n-title" x="728" y="86">GPU engine</text>
    <text class="n-sub" x="728" y="106">{{ st.model }}</text>
    <text class="n-big" x="728" y="142">{{ rate(st.live.decode_tps) }}</text>
    <text class="n-sub" x="728" y="158">decode tok/s</text>
    <text class="n-val" x="852" y="142">{{ st.live.prefill_tps ? Math.round(st.live.prefill_tps) : '–' }}</text>
    <text class="n-sub" x="852" y="158">prefill tok/s</text>
    <line class="hair" x1="728" x2="932" y1="170" y2="170" style="stroke: var(--hair)" />
    <text class="n-small" x="728" y="190">{{ mtp != null ? 'MTP, ' + mtp + ' drafts per step' : 'MTP' }}</text>
    <text class="n-small" x="932" y="190" text-anchor="end">{{ accept != null ? pct(accept) + ' accepted' : 'idle' }}</text>
    <rect class="track" x="728" y="199" width="204" height="3" />
    <rect v-if="accept != null" class="fill" x="728" y="198" :width="204 * accept" height="5" filter="url(#ink-line)" />
    <text class="n-small" x="728" y="228">GPU {{ pct(m.gpu_active) }} · {{ whole(m.gpu_freq_mhz, 'MHz') }}</text>
    <text class="n-small" x="728" y="246">{{ pinned ? 'weights pinned' : 'weights not pinned' }}{{ st.pin_hold_s != null ? ', hold ' + duration(st.pin_hold_s) : '' }}</text>

    <!-- gpu sessions -->
    <rect class="node" x="990" y="40" width="170" height="100" rx="3" />
    <text class="n-title" x="1004" y="62">GPU sessions</text>
    <text class="n-val" x="1004" y="89">{{ gpu.entries }}<tspan class="n-sub" dx="5">of {{ gpu.max_entries }} resident</tspan></text>
    <rect class="track" x="1004" y="102" width="142" height="3" />
    <rect class="fill" x="1004" y="101" :width="142 * gpuShare" height="5" filter="url(#ink-line)" />
    <text class="n-sub" x="1004" y="127">{{ bytes(gpu.used_bytes) }} of {{ bytes(gpu.budget_bytes) }}</text>

    <!-- disk -->
    <template v-if="disk">
      <rect class="node" x="990" y="176" width="170" height="112" rx="3" />
      <text class="n-title" x="1004" y="198">Disk tier</text>
      <text class="n-val" x="1004" y="225">{{ disk.entries }}<tspan class="n-sub" dx="5">{{ disk.entries === 1 ? 'entry' : 'entries' }}, {{ disk.durable_entries }} durable</tspan></text>
      <rect class="track" x="1004" y="238" width="142" height="3" />
      <rect v-if="diskShare > 0" class="fill" x="1004" y="237" :width="Math.max(2, 142 * diskShare)" height="5" />
      <rect v-if="durShare > 0" class="fill-2" :x="1004 + (diskShare > 0 ? Math.max(2, 142 * diskShare) + 2 : 0)" y="237" :width="Math.max(2, 142 * durShare)" height="5" />
      <text class="n-sub" x="1004" y="262">{{ bytes(disk.used_bytes) }} of {{ bytes(disk.budget_bytes) }}</text>
      <line x1="1004" x2="1016" y1="276" y2="276" stroke="var(--s1)" stroke-width="2" /><text class="n-sub" x="1020" y="280">evicted</text>
      <line x1="1072" x2="1084" y1="276" y2="276" stroke="var(--s2)" stroke-width="2" /><text class="n-sub" x="1088" y="280">durable</text>
    </template>
    <template v-else>
      <rect class="node off" x="990" y="176" width="170" height="64" rx="3" />
      <text class="n-title off-text" x="1004" y="198">Disk tier</text>
      <text class="n-sub" x="1004" y="218">off</text>
    </template>

    <!-- weights -->
    <rect class="node" x="470" y="300" width="190" height="84" rx="3" />
    <text class="n-title" x="484" y="322">Weights</text>
    <text class="n-val" x="484" y="349">{{ bytes(st.tiers.weights.bytes) }}</text>
    <text class="n-sub" x="484" y="368">{{ pinned ? 'pinned (mlock)' : 'not pinned' }}</text>

    <!-- ngram -->
    <rect class="node" x="700" y="300" width="210" height="84" rx="3" />
    <text class="n-title" x="714" y="322">{{ ngram ? 'n-gram table, ' + ngram : 'n-gram table' }}</text>
    <text class="n-val" x="714" y="349">{{ pct(st.tiers.ngram?.resident_share) }}<tspan class="n-sub" dx="5">resident</tspan></text>
    <rect class="track" x="714" y="359" width="182" height="3" />
    <rect class="fill" x="714" y="358" :width="182 * (st.tiers.ngram?.resident_share ?? 0)" height="5" filter="url(#ink-line)" />
    <text class="n-sub" x="714" y="378">{{ bytes(st.tiers.ngram?.bytes) }} in page cache</text>

    <!-- expert cache (small-machine mode only) -->
    <rect :class="['node', st.tiers.experts ? '' : 'off']" x="950" y="320" width="210" height="64" rx="3" />
    <text :class="['n-title', st.tiers.experts ? '' : 'off-text']" x="964" y="344">Expert cache</text>
    <text v-if="!st.tiers.experts" class="n-sub" x="964" y="364">n/a<title>Small-machine mode only</title></text>
    <text v-else class="n-small" x="964" y="364">{{ pct(st.tiers.experts.hit_rate) }} hit · {{ st.tiers.experts.used }} of {{ st.tiers.experts.slots }} slots</text>
  </svg>
</template>

<style scoped>
.schematic {
  display: block;
  width: 100%;
  min-width: 940px;
  height: auto;
}
text {
  fill: var(--ink);
  font-family: var(--f-label);
}
.n-title {
  font-family: var(--f-head);
  font-weight: var(--w-head);
  font-size: 16px;
}
.n-sub {
  font-size: 12px;
  fill: var(--ink-2);
}
.n-val {
  font-family: var(--f-num);
  font-weight: var(--w-num);
  font-size: 21px;
  font-variant-numeric: tabular-nums;
}
.n-big {
  font-family: var(--f-head);
  font-weight: var(--w-head);
  font-size: 38px;
  font-variant-numeric: tabular-nums;
}
.n-small {
  font-size: 12.5px;
  fill: var(--ink-2);
  font-variant-numeric: tabular-nums;
}
.ring {
  stroke: var(--c);
}
.e-label {
  font-size: 12px;
  fill: var(--ink-2);
  font-style: italic;
}
.node {
  fill: var(--node);
  stroke: var(--base);
  stroke-width: 1;
}
.node.engine {
  stroke: var(--ai);
  stroke-width: 1.5;
}
.node.off {
  fill: none;
  stroke: var(--base);
  stroke-dasharray: 3 5;
}
.off-text {
  fill: var(--ink-2) !important;
}
.ground {
  fill: var(--node);
  stroke: var(--hair);
}
.hair {
  fill: none;
  stroke: var(--ink-3);
  stroke-width: 1;
}
.hair.dash {
  stroke-dasharray: 3 4;
}
.brush {
  fill: var(--ink);
}
.brush.idle {
  opacity: 0.32;
}
.arrowhead {
  fill: var(--ink-3);
}
.track {
  fill: var(--hair);
}
.fill {
  fill: var(--s1);
  opacity: 0.8;
}
.fill-2 {
  fill: var(--s2);
}
.slot {
  fill: none;
  stroke: var(--base);
}
.slot.on {
  fill: var(--ai);
  stroke: none;
}
.flow {
  fill: none;
  stroke: var(--paper);
  stroke-width: 2;
  stroke-dasharray: 1.5 16;
  stroke-linecap: round;
  animation: flow 1.2s linear infinite;
  opacity: 0.9;
}
@keyframes flow {
  to {
    stroke-dashoffset: -17.5;
  }
}
@media (prefers-reduced-motion: reduce) {
  .flow {
    display: none;
  }
}
</style>
