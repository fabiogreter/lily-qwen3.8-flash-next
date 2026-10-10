<script setup lang="ts">
import { computed, ref, watch } from 'vue';
import type { Setting, SettingValue } from '../api/types';
import SectionHead from '../components/SectionHead.vue';
import { useDashboard } from '../composables/context';
import { usePrefs } from '../composables/usePrefs';
import { useResource } from '../composables/useResource';
import { useSettingsForm } from '../composables/useSettingsForm';
import { bounds, dateTime, label, show } from '../lib/format';
import { parseDurationSecs } from '../lib/duration';

const { backend, status } = useDashboard();
const { paper, papers } = usePrefs();

// Settings are read on entry and after each write, never polled.
const settingsRes = useResource((signal) => backend.settings(signal));
const logRes = useResource((signal) => backend.settingsLog(signal));
const settings = ref<Setting[] | null>(null);
watch(settingsRes.data, (v) => (settings.value = v ? v.settings : null), { immediate: true });

/**
 * An idle unload no longer than the time the engine has been idle unloads it
 * at once (the delay counts from the last request): say so before sending.
 */
function confirmChange(s: Setting, value: SettingValue): boolean {
  const st = status.data.value;
  if (s.name !== 'idle_unload' || typeof value !== 'string' || !st || st.state !== 'ready') return true;
  let secs: number;
  try {
    secs = parseDurationSecs(value);
  } catch {
    return true; // the server refuses it with its own message
  }
  if (secs === 0 || secs > st.idle_for_s) return true;
  return window.confirm(`lily will unload now: it has been idle for ${st.idle_for_s} s, longer than ${value}. Change it?`);
}

const form = useSettingsForm(backend, settings, () => void logRes.refresh(), {
  // A 5xx may or may not have changed something: reload what the server says.
  onUncertain: () => {
    void settingsRes.refresh();
    void logRes.refresh();
  },
  confirm: confirmChange,
});
const { drafts, errors, busy } = form;
const licensesUrl = `${import.meta.env.BASE_URL}assets/licenses.txt`;

// Sections come from the data: any group with an editable setting gets one,
// the usual ones first, in a stable order.
const USUAL = ['Sampling', 'Thinking', 'Lifecycle', 'Caches', 'Engine'];
const runtimeGroups = computed(() => {
  const editable = (settings.value ?? []).filter((s) => s.editable);
  const names = [...new Set(editable.map((s) => s.group))];
  const rank = (n: string): number => (USUAL.includes(n) ? USUAL.indexOf(n) : USUAL.length);
  names.sort((a, b) => rank(a) - rank(b));
  return names.map((name) => ({ name, items: editable.filter((s) => s.group === name) }));
});
const readOnly = computed(() => (settings.value ?? []).filter((s) => !s.editable));

/** What the field shows: the draft while typing, else the stored value. */
const shown = (s: Setting): string => drafts[s.name] ?? (s.value == null ? '' : String(s.value));

/** Row tooltip: the definition, then when a change takes effect. */
const tip = (s: Setting): string | undefined =>
  [s.help ? s.help.replace(/\.?$/, '.') : null, s.applies ? `Applies: ${s.applies}` : null].filter(Boolean).join(' ') || undefined;

function onInput(s: Setting, ev: Event): void {
  drafts[s.name] = (ev.target as HTMLInputElement).value;
  delete errors[s.name]; // editing again: the old refusal no longer applies
}
function onBlur(s: Setting): void {
  // A refusal already on screen is not asked again until the field is edited.
  if (drafts[s.name] !== undefined && !errors[s.name]) void form.commit(s, drafts[s.name] as string);
}
// A control that was refused (or is busy) goes back to the stored value: the
// DOM must not keep showing a state the server did not accept.
async function onToggle(s: Setting, ev: Event): Promise<void> {
  const el = ev.target as HTMLInputElement;
  if (!(await form.commit(s, el.checked))) el.checked = s.value === true;
}
async function onSelect(s: Setting, ev: Event): Promise<void> {
  const el = ev.target as HTMLSelectElement;
  if (!(await form.commit(s, el.value))) el.value = s.value == null ? '' : String(s.value);
}
function onEnter(s: Setting): void {
  if (drafts[s.name] !== undefined) void form.commit(s, drafts[s.name] as string);
}
</script>

<template>
  <main>
    <section class="section first">
      <SectionHead title="Dashboard (this browser)" :seed="4" />
      <div class="set-group">
        <div class="set-row">
          <div class="nm" title="Paper texture behind the page">paper texture</div>
          <div>
            <span class="seg" role="group" aria-label="Paper texture">
              <button v-for="p in papers" :key="p" type="button" :aria-pressed="paper === p" @click="paper = p">{{ p }}</button>
            </span>
          </div>
        </div>
      </div>
    </section>

    <p v-if="settingsRes.error.value" class="inline-error" role="alert">{{ settingsRes.error.value }}</p>

    <section v-for="g in runtimeGroups" :key="g.name" class="section">
      <SectionHead :title="g.name" :seed="g.name.length" />
      <div class="set-group">
        <div v-for="s in g.items" :key="s.name" class="set-row">
          <div>
            <div class="nm" :title="tip(s)">{{ label(s.name) }}</div>
            <div class="flagname">{{ s.flag }}</div>
          </div>
          <div>
            <label v-if="s.type === 'bool'" class="toggle">
              <input type="checkbox" :name="s.name" :checked="s.value === true" :aria-busy="busy[s.name] ? 'true' : undefined" @change="onToggle(s, $event)" />
              {{ s.value ? 'on' : 'off' }}
            </label>
            <select v-else-if="s.type === 'enum'" :name="s.name" :value="s.value == null ? '' : String(s.value)" :aria-busy="busy[s.name] ? 'true' : undefined" :aria-label="label(s.name)" @change="onSelect(s, $event)">
              <!-- Unset (the flag not given, no default): shown as such, not as the first choice. -->
              <option v-if="s.value == null" value="" disabled>unset</option>
              <option v-for="o in s.options" :key="o" :value="o">{{ o }}</option>
            </select>
            <input
              v-else
              :name="s.name"
              :type="s.type === 'number' || s.type === 'int' ? 'number' : 'text'"
              :step="s.step ?? (s.type === 'int' ? 1 : 'any')"
              :value="shown(s)"
              :aria-label="label(s.name)"
              :aria-invalid="errors[s.name] ? 'true' : undefined"
              :readonly="busy[s.name]"
              :aria-busy="busy[s.name] ? 'true' : undefined"
              @input="onInput(s, $event)"
              @keydown.enter="onEnter(s)"
              @blur="onBlur(s)"
            />
            <div v-if="errors[s.name]" class="err" role="alert">{{ errors[s.name] }}</div>
          </div>
          <div class="meta">
            <span :class="['src', s.source]">{{ s.source }}</span>
            <span v-if="s.source !== 'default'"> · default {{ show(s.default) }}</span>
            <span v-if="s.flag_value != null && s.source === 'database'"> · flag {{ show(s.flag_value) }}</span>
            <span v-if="bounds(s.min, s.max)"> · {{ bounds(s.min, s.max) }}</span>
          </div>
          <div><button v-if="s.source === 'database'" type="button" class="btn" :aria-busy="busy[s.name] ? 'true' : undefined" @click="form.reset(s)">reset</button></div>
        </div>
      </div>
    </section>

    <section v-if="readOnly.length" class="section">
      <SectionHead title="Restart required" :seed="12" aside="read-only" hint="Set in the launchd plist; a change needs a restart, which unloads the model" />
      <div class="set-group">
        <div v-for="s in readOnly" :key="s.name" class="set-row ro">
          <div>
            <div class="nm" :title="s.help ?? undefined">{{ label(s.name) }}</div>
            <div class="flagname">{{ s.flag }}</div>
          </div>
          <div class="ro-val">{{ show(s.value) }}</div>
          <div class="meta">
            <span :class="['src', s.source]">{{ s.source }}</span> · {{ s.group }}
          </div>
          <div></div>
        </div>
      </div>
    </section>

    <section v-if="logRes.data.value || logRes.error.value" class="section">
      <SectionHead title="Change log" :seed="1" hint="Kept 7 days" />
      <!-- No database (--stats-db off, or not open): the server says why. -->
      <p v-if="logRes.error.value && !logRes.data.value" class="muted" role="status">{{ logRes.error.value }}</p>
      <ul v-if="logRes.data.value" class="log">
        <li v-for="(e, i) in logRes.data.value.entries" :key="i">
          <span class="muted">{{ dateTime(e.t) }}</span>
          <span>{{ label(e.name) }}</span>
          <span class="num">{{ show(e.old) }} → {{ show(e.new) }}</span>
          <span class="muted">{{ e.source }}</span>
        </li>
      </ul>
    </section>
    <p class="legal"><a :href="licensesUrl" target="_blank" rel="noopener">Licences</a></p>
  </main>
</template>

<style scoped>
.set-group {
  margin-bottom: 8px;
}
.set-row {
  display: grid;
  grid-template-columns: 250px 280px 1fr auto;
  gap: 18px;
  align-items: start;
  padding: 12px 0;
  border-bottom: 1px solid var(--hair);
}
.nm {
  font-family: var(--f-head);
  font-size: 16px;
}
.flagname {
  color: var(--ink-2);
  font-size: 12px;
  font-family: ui-monospace, 'SF Mono', Menlo, monospace;
}
.meta {
  color: var(--ink-2);
  font-size: 12.5px;
}
.err {
  color: var(--crit);
  font-size: 12.5px;
  margin-top: 4px;
}
.set-row input[type='text'],
.set-row input[type='number'],
.set-row select {
  width: 100%;
  background: transparent;
  border: 0;
  border-bottom: 1px solid var(--base);
  padding: 4px 2px;
  font-family: var(--f-num);
  font-variant-numeric: tabular-nums;
}
.set-row input:focus,
.set-row select:focus {
  outline: none;
  border-bottom: 2px solid var(--ai);
}
.set-row input[aria-invalid='true'] {
  border-bottom-color: var(--crit);
}
.set-row select option {
  background: var(--paper);
}
.src {
  font-size: 11.5px;
  padding: 0 6px;
  border-radius: 2px;
  border: 1px solid var(--base);
  color: var(--ink-2);
}
.src.database {
  border-color: var(--ai);
  color: var(--ai);
}
.ro .nm {
  color: var(--ink-2);
}
.ro-val {
  font-family: var(--f-num);
  font-variant-numeric: tabular-nums;
  color: var(--ink-2);
  overflow-wrap: anywhere;
}
.toggle {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
}
.toggle input {
  accent-color: var(--ai);
  width: 16px;
  height: 16px;
}
.legal {
  margin: 48px 0 0;
  font-size: 12.5px;
}
.legal a {
  color: var(--ink-2);
}
.log {
  list-style: none;
  padding: 0;
  margin: 0;
}
.log li {
  display: grid;
  grid-template-columns: 160px 260px 1fr 100px;
  gap: 16px;
  padding: 8px 0;
  border-bottom: 1px solid var(--hair);
  font-variant-numeric: tabular-nums;
}

@media (max-width: 980px) {
  .set-row {
    grid-template-columns: 1fr 1fr;
  }
  .log li {
    grid-template-columns: 130px 1fr;
  }
}
</style>
