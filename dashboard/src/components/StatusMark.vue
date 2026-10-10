<script setup lang="ts">
import { computed } from 'vue';
import type { MemoryPressure } from '../api/types';
import { levelOf } from '../lib/status';

// Status is never colour alone: an icon and a label come with it, and the
// colour only draws the ring (text stays in ink). null = no reading.
const props = defineProps<{ level: MemoryPressure | null; text?: string }>();

const lv = computed(() => (props.level ? levelOf(props.level) : null));
</script>

<template>
  <span v-if="lv" class="status" :class="lv.cls"><span class="ico" aria-hidden="true">{{ lv.icon }}</span><span class="lbl">{{ text ?? lv.label }}</span></span>
  <span v-else class="status"><span class="lbl">{{ text ?? '–' }}</span></span>
</template>

<style>
.st-good {
  --c: var(--good);
}
.st-warn {
  --c: var(--warn);
}
.st-crit {
  --c: var(--crit);
}
.status {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}
.status .ico {
  font-size: 12px;
  line-height: 1;
  display: inline-grid;
  place-items: center;
  width: 16px;
  height: 16px;
  border-radius: 50%;
  border: 1.5px solid var(--c);
  color: var(--ink);
  font-weight: 500;
}
.status .lbl {
  color: var(--ink);
}
</style>
