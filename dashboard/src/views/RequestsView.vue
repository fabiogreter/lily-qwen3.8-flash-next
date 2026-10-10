<script setup lang="ts">
import SectionHead from '../components/SectionHead.vue';
import { HISTORY_POLL_MS, useDashboard } from '../composables/context';
import { useResource } from '../composables/useResource';
import type { RequestRow } from '../api/types';
import { clock, pct, rate, tokens } from '../lib/format';

const { backend } = useDashboard();
const LIMIT = 40;

/** Accepted over drafted; null when the draft head was off (or drafted nothing). */
function acceptance(r: RequestRow): number | null {
  return r.drafted_tokens && r.accepted_tokens != null ? r.accepted_tokens / r.drafted_tokens : null;
}

const res = useResource((signal) => backend.requests(LIMIT, signal), { intervalMs: HISTORY_POLL_MS });
</script>

<template>
  <main>
    <section class="section first">
      <SectionHead title="Recent requests" :seed="5" :aside="res.data.value ? `last ${res.data.value.requests.length}` : ''" />
      <p v-if="res.error.value" class="inline-error" role="alert">{{ res.error.value }}</p>
      <div v-if="res.data.value" class="tablewrap">
        <table class="ink">
          <thead>
            <tr>
              <th class="l">time</th>
              <th class="l">lineage</th>
              <th>prompt</th>
              <th>cached</th>
              <th title="Tokens computed: prompt minus cached">prefilled</th>
              <th>prefill tok/s</th>
              <th>generated</th>
              <th>decode tok/s</th>
              <th title="Draft tokens accepted over drafted">accept</th>
              <th>queue ms</th>
              <th title="Batch size while the request ran">batch</th>
              <th class="l">notes</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(r, i) in res.data.value.requests" :key="`${r.t}-${i}`">
              <td class="l">{{ clock(r.t) }}</td>
              <td class="l">{{ r.lineage != null ? '#' + r.lineage : '–' }}</td>
              <td>{{ tokens(r.prompt_tokens) }}</td>
              <td>
                {{ tokens(r.cached_tokens) }} <span class="muted">{{ pct(r.prompt_tokens ? r.cached_tokens / r.prompt_tokens : null) }}</span>
              </td>
              <td>{{ tokens(r.prefill_tokens) }}</td>
              <td>{{ rate(r.prefill_tps) }}</td>
              <td>{{ tokens(r.generated_tokens) }}</td>
              <td>{{ rate(r.decode_tps) }}</td>
              <td>{{ pct(acceptance(r)) }}</td>
              <td :class="{ muted: r.queue_ms < 10 }">{{ r.queue_ms }}</td>
              <td>{{ r.batch }}</td>
              <td class="l">
                <span v-if="r.failed" class="tag">failed</span
                ><span v-if="r.cancelled_by" class="tag">cancelled by {{ r.cancelled_by }}</span
                ><span v-if="r.thinking_forced" class="tag ai">thinking closed</span>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>
  </main>
</template>
