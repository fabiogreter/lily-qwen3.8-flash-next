<script setup lang="ts">
import { computed } from 'vue';
import EngineSchematic from '../components/EngineSchematic.vue';
import SectionHead from '../components/SectionHead.vue';
import { useDashboard } from '../composables/context';
import { useResource } from '../composables/useResource';
import type { Setting } from '../api/types';

const { backend, status } = useDashboard();
// A few settings describe the diagram (read once; they change only on restart).
const settings = useResource((signal) => backend.settings(signal));
const byName = computed(() => new Map<string, Setting>((settings.data.value?.settings ?? []).map((s) => [s.name, s])));
const num = (name: string): number | null => {
  const v = byName.value.get(name)?.value;
  return typeof v === 'number' ? v : null;
};
const str = (name: string): string | null => {
  const v = byName.value.get(name)?.value;
  return typeof v === 'string' ? v : null;
};
</script>

<template>
  <main>
    <section class="section first">
      <SectionHead title="Engine" :seed="9" />
      <div v-if="status.data.value" class="schematic-wrap">
        <EngineSchematic :st="status.data.value" :mtp="num('mtp_drafts')" :bind="str('bind')" :ngram="str('ngram_table')" />
      </div>
    </section>
  </main>
</template>

<style scoped>
.schematic-wrap {
  overflow-x: auto;
  margin-top: 8px;
}
</style>
