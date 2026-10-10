<script lang="ts">
import type { HistoryPoint } from '../api/types';

/** Keys of a history point that hold a number or null. */
export type SeriesKey = {
  [K in keyof HistoryPoint]: HistoryPoint[K] extends number | null ? K : never;
}[keyof HistoryPoint];

export interface SeriesDef {
  key: SeriesKey;
  label: string;
  /** A CSS colour (token), e.g. `var(--s1)`. */
  color: string;
  /** Fill under the line. */
  area?: boolean;
  width?: number;
}
</script>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue';
import type { Range } from '../api/types';
import { axisValue, nearestIndex, niceMax, seriesPaths, tickIndices } from '../lib/chart';
import { timeLabel } from '../lib/format';

// Chart rules: one value axis; a legend when there are two or more series;
// null buckets are gaps (never zero, never bridged); text in ink tokens; a
// crosshair with a tooltip on hover or arrow keys; the same numbers in a table.
const props = withDefaults(
  defineProps<{
    title: string;
    unit: string;
    points: HistoryPoint[];
    series: SeriesDef[];
    range: Range;
    yMax?: number;
    percent?: boolean;
    height?: number;
  }>(),
  { height: 180, percent: false },
);

const el = ref<HTMLElement | null>(null);
const w = ref(560);
const hover = ref<number | null>(null);
let ro: ResizeObserver | null = null;

onMounted(() => {
  if (typeof ResizeObserver === 'undefined' || !el.value) return;
  ro = new ResizeObserver((entries) => {
    const width = entries[0]?.contentRect.width;
    if (width) w.value = Math.max(240, width);
  });
  ro.observe(el.value);
});
onBeforeUnmount(() => ro?.disconnect());

const pad = { l: 46, r: 44, t: 12, b: 26 };
const innerW = computed(() => w.value - pad.l - pad.r);
const innerH = computed(() => props.height - pad.t - pad.b);
const n = computed(() => props.points.length);

const x = (i: number): number => pad.l + (n.value > 1 ? (i * innerW.value) / (n.value - 1) : 0);
const top = computed(() => {
  if (props.yMax) return props.yMax;
  let m = 0;
  for (const p of props.points) for (const s of props.series) m = Math.max(m, p[s.key] ?? 0);
  return niceMax(m);
});
const y = (v: number): number => pad.t + innerH.value - (v / top.value) * innerH.value;
const fmtV = (v: number | null | undefined): string => axisValue(v, { percent: props.percent, top: top.value });

const ticks = computed(() => [0, 0.25, 0.5, 0.75, 1].map((k) => ({ v: k * top.value, y: y(k * top.value) })));
const xticks = computed(() => {
  const idx = tickIndices(n.value, w.value < 420 ? 3 : 5);
  return idx.map((i, k) => ({
    x: x(i),
    label: timeLabel((props.points[i] as HistoryPoint).t, props.range),
    anchor: k === 0 ? 'start' : k === idx.length - 1 ? 'end' : 'middle',
  }));
});

const paths = computed(() =>
  props.series.map((s) => ({
    ...s,
    ...seriesPaths(
      props.points.map((p) => p[s.key]),
      x,
      y,
      y(0),
      !!s.area,
    ),
  })),
);

function move(ev: PointerEvent): void {
  const r = (ev.currentTarget as Element).getBoundingClientRect();
  hover.value = nearestIndex(ev.clientX - r.left, pad.l, innerW.value, n.value);
}
function key(ev: KeyboardEvent): void {
  if (!n.value) return;
  if (ev.key !== 'ArrowLeft' && ev.key !== 'ArrowRight') return;
  const cur = hover.value ?? n.value - 1;
  hover.value = ev.key === 'ArrowLeft' ? Math.max(0, cur - 1) : Math.min(n.value - 1, cur + 1);
  ev.preventDefault();
}

const tip = computed(() => {
  if (hover.value == null) return null;
  const p = props.points[hover.value];
  if (!p) return null;
  const hx = x(hover.value);
  return {
    x: hx,
    left: hx > w.value - 180 ? hx - 164 : hx + 14,
    t: timeLabel(p.t, props.range),
    rows: props.series.map((s) => ({
      label: s.label,
      color: s.color,
      v: fmtV(p[s.key]),
      y: p[s.key] == null ? null : y(p[s.key] as number),
    })),
  };
});

/** The tooltip as one sentence, for the live region. */
const tipText = computed(() => (tip.value ? `${tip.value.t}: ${tip.value.rows.map((r) => `${r.label} ${r.v}`).join(', ')}` : ''));

/** Direct label at the right end of the first series (the story). */
const lastLabel = computed(() => {
  const s = props.series[0];
  if (!s) return null;
  for (let i = props.points.length - 1; i >= 0; i--) {
    const v = (props.points[i] as HistoryPoint)[s.key];
    if (v != null) return { x: x(i), y: y(v), v: fmtV(v) };
  }
  return null;
});
</script>

<template>
  <div ref="el" class="chart">
    <div class="chart-title">
      <h3>
        {{ title }} <span class="muted unit">{{ unit }}</span>
      </h3>
      <div v-if="series.length > 1" class="legend">
        <span v-for="s in series" :key="s.key"
          ><span class="key" :style="{ borderColor: s.color, borderTopWidth: (s.width || 2) + 'px' }"></span>{{ s.label }}</span
        >
      </div>
    </div>
    <svg
      :width="w"
      :height="height"
      :viewBox="'0 0 ' + w + ' ' + height"
      tabindex="0"
      role="img"
      :aria-label="title + ' chart, use arrow keys to read values'"
      @pointermove="move"
      @pointerleave="hover = null"
      @keydown="key"
      @blur="hover = null"
    >
      <g>
        <line v-for="t in ticks" :key="'g' + t.v" :x1="pad.l" :x2="w - pad.r + 4" :y1="t.y" :y2="t.y" :stroke="t.v === 0 ? 'var(--base)' : 'var(--hair)'" stroke-width="1" />
        <text v-for="t in ticks" :key="'t' + t.v" :x="pad.l - 8" :y="t.y + 4" text-anchor="end" class="axis">{{ fmtV(t.v) }}</text>
        <text v-for="t in xticks" :key="'x' + t.x" :x="t.x" :y="height - 6" :text-anchor="t.anchor" class="axis">{{ t.label }}</text>
      </g>
      <g v-for="p in paths" :key="p.key">
        <path v-if="p.area" :d="p.area" :fill="p.color" opacity="0.07" />
        <path :d="p.d" fill="none" :stroke="p.color" :stroke-width="p.width || 2" stroke-linejoin="round" stroke-linecap="round" filter="url(#ink-line)" />
        <circle v-for="(d, i) in p.dots" :key="i" :cx="d[0]" :cy="d[1]" r="2.2" :fill="p.color" />
      </g>
      <text v-if="lastLabel && hover == null" :x="lastLabel.x + 7" :y="lastLabel.y + 4" text-anchor="start" class="last">{{ lastLabel.v }}</text>
      <g v-if="tip">
        <line :x1="tip.x" :x2="tip.x" :y1="pad.t" :y2="pad.t + innerH" stroke="var(--ink-3)" stroke-width="1" />
        <template v-for="r in tip.rows" :key="r.label">
          <circle v-if="r.y != null" :cx="tip.x" :cy="r.y" r="4" :fill="r.color" stroke="var(--paper)" stroke-width="2" />
        </template>
      </g>
      <rect :x="pad.l" :y="0" :width="Math.max(0, w - pad.l - pad.r)" :height="height" fill="transparent" />
    </svg>
    <div class="sr-only" aria-live="polite" aria-atomic="true">{{ tipText }}</div>
    <div v-if="tip" class="tip" aria-hidden="true" :style="{ left: tip.left + 'px', top: '34px' }">
      <div class="t">{{ tip.t }}</div>
      <div v-for="r in tip.rows" :key="r.label" class="row">
        <span class="key" :style="{ borderColor: r.color }"></span><span class="val">{{ r.v }}</span><span class="nm">{{ r.label }}</span>
      </div>
    </div>
    <details class="tbl">
      <summary>table</summary>
      <div class="scroll">
        <table class="ink">
          <thead>
            <tr>
              <th class="l">time</th>
              <th v-for="s in series" :key="s.key">{{ s.label }}</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="p in points" :key="p.t">
              <td class="l">{{ timeLabel(p.t, range) }}</td>
              <td v-for="s in series" :key="s.key">{{ fmtV(p[s.key]) }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </details>
  </div>
</template>

<style scoped>
.chart {
  position: relative;
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
  color: var(--ink);
}
.unit {
  font-size: 12.5px;
  font-family: var(--f-label);
  font-weight: var(--w-body);
}
.legend {
  display: flex;
  gap: 14px;
  color: var(--ink-2);
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
svg {
  display: block;
  width: 100%;
  overflow: visible;
}
svg:focus {
  outline: none;
}
svg:focus-visible {
  outline: 1px solid var(--ai);
  outline-offset: 4px;
}
.axis {
  fill: var(--ink-2);
  font-size: 11.5px;
  font-variant-numeric: tabular-nums;
}
.last {
  fill: var(--ink-2);
  font-size: 12px;
  font-variant-numeric: tabular-nums;
}
.tip {
  position: absolute;
  pointer-events: none;
  background: var(--paper-lamp);
  border: 1px solid var(--hair);
  box-shadow: 0 6px 18px var(--shadow);
  padding: 8px 10px;
  font-size: 12.5px;
  min-width: 140px;
  z-index: 5;
  border-radius: 2px;
}
.tip .t {
  color: var(--ink-2);
  margin-bottom: 4px;
}
.tip .row {
  display: flex;
  align-items: center;
  gap: 8px;
}
.tip .val {
  font-family: var(--f-num);
  font-variant-numeric: tabular-nums;
  font-weight: 500;
  color: var(--ink);
  min-width: 48px;
}
.tip .nm {
  color: var(--ink-2);
}
.tip .key {
  display: inline-block;
  width: 12px;
  border-top: 2px solid;
}
details.tbl {
  margin-top: 4px;
  color: var(--ink-2);
  font-size: 12px;
}
details.tbl summary {
  cursor: pointer;
  width: max-content;
}
details.tbl .scroll {
  max-height: 200px;
  overflow: auto;
  margin-top: 6px;
}
</style>
