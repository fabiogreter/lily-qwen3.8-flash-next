<script setup lang="ts">
import { computed } from 'vue';
import { strokePath } from '../lib/brush';

const props = withDefaults(defineProps<{ seed?: number }>(), { seed: 1 });
const d = computed(() => strokePath(props.seed));
</script>

<template>
  <svg viewBox="0 0 1000 14" preserveAspectRatio="none" aria-hidden="true">
    <path :d="d" fill="var(--ink)" opacity="0.7" filter="url(#ink-stroke)" />
  </svg>
</template>
