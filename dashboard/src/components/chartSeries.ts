import type { SeriesDef, SeriesKey } from './LineChart.vue';

/** The usual pair: the average as ink with an area, the maximum as a lighter indigo line. */
export function avgMaxSeries(avg: SeriesKey, max: SeriesKey): SeriesDef[] {
  return [
    { key: avg, label: 'average', color: 'var(--s1)', area: true },
    { key: max, label: 'max', color: 'var(--s2)', width: 1.5 },
  ];
}
