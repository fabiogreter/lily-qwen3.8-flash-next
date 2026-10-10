<script setup lang="ts">
import BrushStroke from './BrushStroke.vue';

withDefaults(defineProps<{ title: string; seed?: number; aside?: string; hint?: string }>(), { seed: 1 });
</script>

<template>
  <div class="section-head">
    <h2 :title="hint">{{ title }}</h2>
    <BrushStroke class="stroke" :seed="seed" />
    <span v-if="aside" class="aside">{{ aside }}</span>
    <slot />
  </div>
</template>
