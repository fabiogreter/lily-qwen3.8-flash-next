<script setup lang="ts">
import { RANGES, type Range } from '../api/types';

const model = defineModel<Range>({ required: true });
</script>

<template>
  <span class="seg" role="group" aria-label="Range">
    <button v-for="r in RANGES" :key="r" type="button" :aria-pressed="model === r" @click="model = r">{{ r }}</button>
  </span>
</template>
