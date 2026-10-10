<script setup lang="ts">
import { computed } from 'vue';

export interface MeterPart {
  v: number;
  color: string;
}

const props = defineProps<{ parts: MeterPart[]; total: number }>();

/** Ink parts laid end to end over a hairline track; a part under 0.05 % is not drawn. */
const segs = computed(() => {
  let x = 0;
  const out: { x: number; w: number; color: string }[] = [];
  props.parts.forEach((p, i) => {
    const w = props.total ? Math.max(0, p.v / props.total) * 1000 : 0;
    if (w > 0.5) out.push({ x, w: Math.max(0, w - (i < props.parts.length - 1 ? 4 : 0)), color: p.color });
    x += w;
  });
  return out;
});
</script>

<template>
  <svg viewBox="0 0 1000 22" preserveAspectRatio="none" aria-hidden="true">
    <line x1="0" y1="11" x2="1000" y2="11" stroke="var(--base)" stroke-width="1" vector-effect="non-scaling-stroke" />
    <rect v-for="(s, i) in segs" :key="i" :x="s.x" y="5" :width="s.w" height="12" rx="2" :fill="s.color" filter="url(#ink-line)" opacity="0.78" />
  </svg>
</template>
