<script setup lang="ts">
import { THEMES, usePrefs } from '../composables/usePrefs';
import { VIEWS, type View } from '../lib/view';
import Seal from './Seal.vue';

defineProps<{ mock: boolean; view: View }>();
defineEmits<{ navigate: [view: View] }>();
const { theme } = usePrefs();
</script>

<template>
  <header class="top">
    <div class="brand">
      <Seal :size="50" />
      <div>
        <h1>lily</h1>
        <div class="sub">local inference</div>
      </div>
    </div>
    <nav class="views" aria-label="Views">
      <button v-for="v in VIEWS" :key="v" type="button" :aria-current="view === v ? 'page' : undefined" @click="$emit('navigate', v)">{{ v }}</button>
    </nav>
    <div class="tools">
      <span v-if="mock" class="mock" title="All numbers on this page are invented">mock data</span>
      <span class="seg small" role="group" aria-label="Colour mode">
        <button v-for="t in THEMES" :key="t" type="button" :aria-pressed="theme === t" @click="theme = t">{{ t }}</button>
      </span>
    </div>
  </header>
</template>

<style scoped>
.top {
  display: flex;
  align-items: center;
  gap: 22px;
  padding: 24px 0 18px;
  flex-wrap: wrap;
}
.brand {
  display: flex;
  align-items: center;
  gap: 14px;
}
.brand h1 {
  font-size: 30px;
  line-height: 1;
}
.brand .sub {
  color: var(--ink-2);
  font-size: 13px;
}
nav.views {
  display: flex;
  flex-wrap: wrap;
  gap: 0 4px;
  margin-left: 28px;
  min-width: 0;
}
nav.views button {
  background: none;
  border: 0;
  padding: 6px 12px;
  cursor: pointer;
  color: var(--ink-2);
  border-radius: 2px;
  position: relative;
  font-family: var(--f-head);
  font-size: 16px;
}
nav.views button[aria-current='page'] {
  color: var(--ink);
}
nav.views button[aria-current='page']::after {
  content: '';
  position: absolute;
  left: 10px;
  right: 10px;
  bottom: 0;
  height: 2px;
  background: var(--ai);
  border-radius: 2px;
}
nav.views button:hover {
  background: var(--ai-wash);
}
.tools {
  margin-left: auto;
  display: flex;
  align-items: center;
  gap: 14px;
  font-size: 12px;
}
.mock {
  font-size: 12px;
  color: var(--ai);
  border: 1px solid var(--ai);
  padding: 1px 8px;
  border-radius: 2px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}
@media (max-width: 980px) {
  nav.views {
    margin-left: 0;
  }
}
@media (max-width: 560px) {
  .top {
    gap: 12px 16px;
  }
  nav.views {
    order: 3;
    flex-basis: 100%;
  }
  .tools {
    margin-left: auto;
  }
}
</style>
