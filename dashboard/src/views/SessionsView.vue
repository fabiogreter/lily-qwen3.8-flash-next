<script setup lang="ts">
import { computed, ref } from 'vue';
import { avgMaxSeries } from '../components/chartSeries';
import LineChart from '../components/LineChart.vue';
import RangeSwitch from '../components/RangeSwitch.vue';
import SectionHead from '../components/SectionHead.vue';
import { HISTORY_POLL_MS, useDashboard } from '../composables/context';
import { useResource } from '../composables/useResource';
import { agoSince, bytes, count, pct, rate, tokens } from '../lib/format';

const { backend, range, status } = useDashboard();

const list = useResource((signal) => backend.sessions(signal), { intervalMs: HISTORY_POLL_MS });
/** Most recently used first. */
const sessions = computed(() => [...(list.data.value?.sessions ?? [])].sort((a, b) => b.last_used_t - a.last_used_t));
/** The server's clock, for "ago" columns. */
const now = computed(() => status.data.value?.now ?? Date.now());

const picked = ref<number | null>(null);
/** The picked entry, or the most recently used one that has an id while nothing (or something that is gone) is picked. */
const selected = computed(
  () => sessions.value.find((s) => s.lineage != null && s.lineage === picked.value) ?? sessions.value.find((s) => s.lineage != null) ?? null,
);
const lineage = computed(() => selected.value?.lineage ?? null);

// Switching entry drops the old history at once: another entry's numbers are
// never shown under this one's title while the new ones load.
const hist = useResource(
  (signal) => (lineage.value == null ? Promise.resolve(null) : backend.sessionHistory(lineage.value, range.value, signal)),
  { intervalMs: HISTORY_POLL_MS, sources: [range, lineage], resetOn: [lineage] },
);
const sh = hist.data;

const decodeSeries = avgMaxSeries('decode_tps_avg', 'decode_tps_max');
const prefillSeries = avgMaxSeries('prefill_tps_avg', 'prefill_tps_max');
</script>

<template>
  <main>
    <section class="section first">
      <SectionHead title="Cache entries" :seed="8" :aside="count(sessions.length, 'entry', 'entries')" />
      <p v-if="list.error.value" class="inline-error" role="alert">{{ list.error.value }}</p>
      <div class="tablewrap">
        <table class="ink">
          <thead>
            <tr>
              <th class="l" title="Lineage id of the cache entry">lineage</th>
              <th class="l">tier</th>
              <th>tokens</th>
              <th>size</th>
              <th>last used</th>
              <th>requests</th>
              <th>decode avg</th>
              <th>decode max</th>
              <th>prefill avg</th>
              <th>cache hit</th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="(s, i) in sessions"
              :key="`${s.lineage ?? 'none'}-${i}`"
              :aria-current="selected && selected.lineage === s.lineage && s.lineage != null ? 'true' : undefined"
              :class="{ pickable: s.lineage != null }"
              @click="s.lineage != null && (picked = s.lineage)"
            >
              <td class="l">
                <span v-if="s.active" class="livedot" title="running now"></span>
                <button v-if="s.lineage != null" type="button" class="rowbtn" :aria-pressed="selected?.lineage === s.lineage" @click.stop="picked = s.lineage">#{{ s.lineage }}</button>
                <span v-else title="Written before lineage ids existed">–</span>
              </td>
              <td class="l">{{ s.tier === 'gpu' ? 'GPU' : 'disk' }}<span v-if="s.durable" class="tag ai">durable</span></td>
              <td>{{ tokens(s.tokens) }}</td>
              <td>{{ bytes(s.bytes) }}</td>
              <td>{{ s.active ? 'now' : agoSince(s.last_used_t, now) }}</td>
              <td>{{ s.requests }}</td>
              <td>{{ rate(s.decode_tps_avg) }}</td>
              <td>{{ rate(s.decode_tps_max) }}</td>
              <td>{{ rate(s.prefill_tps_avg) }}</td>
              <td>{{ pct(s.cache_hit) }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <section v-if="selected" class="section">
      <SectionHead :title="`#${selected.lineage}`" :seed="2"><RangeSwitch v-model="range" /></SectionHead>
      <p v-if="hist.error.value" class="inline-error" role="alert">{{ hist.error.value }}</p>
      <template v-if="sh">
        <div class="summary">
          <div>
            <div class="k">tier</div>
            <div class="v">{{ selected.tier === 'gpu' ? 'GPU' : 'disk' }}<small v-if="selected.durable">durable</small></div>
          </div>
          <div>
            <div class="k">decode avg</div>
            <div class="v">{{ rate(sh.summary.decode.avg) }}<small>tok/s</small></div>
          </div>
          <div>
            <div class="k">decode max</div>
            <div class="v">{{ rate(sh.summary.decode.max) }}<small>tok/s</small></div>
          </div>
          <div>
            <div class="k">prefill avg</div>
            <div class="v">{{ rate(sh.summary.prefill.avg) }}<small>tok/s</small></div>
          </div>
          <div>
            <div class="k">generated</div>
            <div class="v">{{ tokens(sh.summary.tokens_generated) }}</div>
          </div>
          <div>
            <div class="k">acceptance</div>
            <div class="v">{{ pct(selected.acceptance) }}</div>
          </div>
          <div>
            <div class="k">created</div>
            <div class="v">{{ agoSince(selected.created_t, now) }}</div>
          </div>
        </div>
        <div class="cols2" style="margin-top: 24px">
          <LineChart title="Decode" unit="tok/s" :points="sh.points" :range="sh.range" :series="decodeSeries" />
          <LineChart title="Prefill" unit="tok/s" :points="sh.points" :range="sh.range" :series="prefillSeries" />
        </div>
      </template>
    </section>
  </main>
</template>

<style scoped>
.rowbtn {
  background: none;
  border: 0;
  padding: 0;
  cursor: pointer;
  color: var(--ink);
  font-family: inherit;
  font-size: inherit;
  text-decoration: underline transparent;
  text-underline-offset: 3px;
}
.rowbtn:hover,
.rowbtn[aria-pressed='true'] {
  text-decoration-color: var(--ai);
}
tr[aria-current='true'] td {
  background: var(--ai-wash);
}
tr[aria-current='true'] td:first-child {
  box-shadow: inset 2px 0 0 var(--ai);
}
tr.pickable {
  cursor: pointer;
}
tr.pickable:hover td {
  background: var(--ai-wash);
}
</style>
