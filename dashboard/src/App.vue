<script setup lang="ts">
import { computed, provide, ref, watch, type Component } from 'vue';
import type { Backend } from './api/client';
import type { Range } from './api/types';
import AppHeader from './components/AppHeader.vue';
import InkFilters from './components/InkFilters.vue';
import PaperLayer from './components/PaperLayer.vue';
import Seal from './components/Seal.vue';
import StatusMark from './components/StatusMark.vue';
import { dashboardKey, STATUS_POLL_MS } from './composables/context';
import { useHashView } from './composables/useHashView';
import { useResource } from './composables/useResource';
import { duration } from './lib/format';
import { isStaleAge, statusAgeMs } from './lib/status';
import type { View } from './lib/view';
import EngineView from './views/EngineView.vue';
import OverviewView from './views/OverviewView.vue';
import RequestsView from './views/RequestsView.vue';
import SessionsView from './views/SessionsView.vue';
import SettingsView from './views/SettingsView.vue';

const props = defineProps<{ backend: Backend }>();

const { view, set: navigate } = useHashView();
const range = ref<Range>('1h');
const status = useResource((signal) => props.backend.status(signal), { intervalMs: STATUS_POLL_MS });
provide(dashboardKey, { backend: props.backend, status, range });

// How old the server's clock was when its last answer arrived. The page also
// says so when answers keep coming but are old (a stuck status thread).
const ageMs = ref(0);
watch(
  status.data,
  (s) => {
    if (s) ageMs.value = statusAgeMs(s, Date.now());
  },
  { immediate: true },
);
const stale = computed(() => !!status.data.value && isStaleAge(ageMs.value, STATUS_POLL_MS));
const notice = computed(() => {
  if (!status.data.value) return '';
  if (status.error.value) return `${status.error.value}. Showing the last values.`;
  if (stale.value) return `The last status is ${duration(ageMs.value / 1000)} old. Showing the last values.`;
  return '';
});

const views: Record<View, Component> = {
  overview: OverviewView,
  sessions: SessionsView,
  requests: RequestsView,
  engine: EngineView,
  settings: SettingsView,
};
const current = computed(() => views[view.value]);
const recorder = computed(() => status.data.value?.recorder ?? null);
const recorderNote = computed(() => {
  const r = recorder.value;
  if (!r) return '';
  if (!r.ok) return 'recorder unavailable, history is not being kept';
  if (r.dropped_rows > 0) return `recorder dropped ${r.dropped_rows} request row${r.dropped_rows === 1 ? '' : 's'}`;
  return '';
});
</script>

<template>
  <PaperLayer />
  <InkFilters />
  <div class="wrap">
    <AppHeader :view="view" :mock="backend.kind === 'mock'" @navigate="navigate" />
    <p v-if="notice" class="inline-error" role="alert">{{ notice }}</p>
    <component :is="current" />
    <footer class="colophon">
      <Seal :size="22" />
      <StatusMark v-if="recorderNote" level="warning" :text="recorderNote" />
    </footer>
  </div>
</template>

<style scoped>
.wrap {
  position: relative;
  z-index: 1;
}
.colophon {
  margin-top: 80px;
  display: flex;
  gap: 14px;
  align-items: center;
  font-size: 12px;
}
</style>
