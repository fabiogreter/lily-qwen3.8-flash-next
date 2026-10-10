<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue';
import { ensoBand, ensoGhost } from '../lib/brush';

// The ensō: a brushed circle that closes as the live value approaches the
// interval's maximum. It eases to each new value; with prefers-reduced-motion
// it jumps.
const props = defineProps<{ value: number; max: number }>();

const shown = ref(0);
let raf = 0;

const target = computed(() => (props.max ? Math.max(0, Math.min(1, (props.value || 0) / props.max)) : 0));

function animate(): void {
  const to = target.value;
  const reduced = typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches;
  cancelAnimationFrame(raf);
  if (reduced) {
    shown.value = to;
    return;
  }
  const from = shown.value;
  const t0 = performance.now();
  const dur = 900;
  const step = (now: number): void => {
    const k = Math.min(1, (now - t0) / dur);
    const e = 1 - Math.pow(1 - k, 3);
    shown.value = from + (to - from) * e;
    if (k < 1) raf = requestAnimationFrame(step);
  };
  raf = requestAnimationFrame(step);
}

watch(target, animate);
onMounted(animate);
onBeforeUnmount(() => cancelAnimationFrame(raf));

const band = computed(() => ensoBand(shown.value, 21, 1));
const ghost = ensoGhost();
const label = computed(() => `Ensō gauge: ${Math.round(target.value * 100)} percent of the interval maximum`);
</script>

<template>
  <svg viewBox="0 0 320 320" role="img" :aria-label="label">
    <path :d="ghost" fill="none" stroke="var(--ink)" stroke-width="1" opacity="0.14" />
    <path :d="band" fill="var(--ink)" filter="url(#ink-strong)" opacity="0.92" />
  </svg>
</template>
