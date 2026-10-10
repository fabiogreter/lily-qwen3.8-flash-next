import { flushPromises, mount } from '@vue/test-utils';
import { describe, expect, it } from 'vitest';
import { ref } from 'vue';
import { dashboardKey } from '../composables/context';
import type { History, Range } from '../api/types';
import { MockBackend } from '../mock/backend';
import SessionsView from './SessionsView.vue';

/** Session histories that answer when the test says so; the generated-token count is the lineage, so a page can show whose data it is. */
class Gated extends MockBackend {
  pending: Record<number, (h: History) => void> = {};
  asked: number[] = [];
  async sessionHistory(lineage: number, range: Range): Promise<History> {
    this.asked.push(lineage);
    const base = await super.sessionHistory(lineage, range);
    return new Promise((resolve) => {
      this.pending[lineage] = () => resolve({ ...base, summary: { ...base.summary, tokens_generated: lineage } });
    });
  }
}

async function setup() {
  const backend = new Gated({ clock: () => Date.UTC(2026, 9, 10, 14, 20) });
  const wrapper = mount(SessionsView, {
    global: {
      provide: {
        [dashboardKey as symbol]: {
          backend,
          range: ref('1h'),
          status: { data: ref({ now: Date.UTC(2026, 9, 10, 14, 20) }), error: ref(null), refresh: async () => {} },
        },
      },
    },
  });
  await flushPromises();
  return { wrapper, backend };
}

const buttons = (w: Awaited<ReturnType<typeof setup>>['wrapper']) => w.findAll('button.rowbtn');

describe('SessionsView', () => {
  it('lists the entries most recently used first; one without a lineage shows a dash and cannot be picked', async () => {
    const { wrapper } = await setup();
    const rows = wrapper.findAll('tbody tr');
    expect(rows).toHaveLength(11);
    const noId = rows.filter((r) => !r.find('button').exists());
    expect(noId).toHaveLength(1);
    expect(noId[0]!.find('td').text()).toBe('–');
    await noId[0]!.trigger('click');
    await flushPromises();
    expect(noId[0]!.attributes('aria-current')).toBeUndefined();
    const ago = rows.map((r) => r.findAll('td')[4]!.text());
    expect(ago[0]).toBe('now');
  });

  it('rows are operated with real buttons, and the current one is marked', async () => {
    const { wrapper } = await setup();
    const b = buttons(wrapper);
    expect(b.length).toBe(10);
    expect(wrapper.findAll('tr[tabindex]')).toHaveLength(0);
    await b[3]!.trigger('click');
    expect(b[3]!.attributes('aria-pressed')).toBe('true');
    expect(wrapper.find('tr[aria-current="true"]').text()).toContain(b[3]!.text());
  });

  it('never shows another entry\'s history under the selected title', async () => {
    const { wrapper, backend } = await setup();
    const b = buttons(wrapper);
    const idA = Number(b[0]!.text().slice(1));
    const idB = Number(b[1]!.text().slice(1));
    const idC = Number(b[2]!.text().slice(1));
    // the first entry is selected by default and asks for its history
    expect(backend.asked).toEqual([idA]);
    backend.pending[idA]!({} as History);
    await flushPromises();
    expect(wrapper.find('.summary').exists()).toBe(true);

    // two quick picks: A's data is gone at once, B's request is superseded by C's
    await b[1]!.trigger('click');
    await b[2]!.trigger('click');
    await flushPromises();
    expect(backend.asked).toEqual([idA, idB, idC]);
    expect(wrapper.findAll('h2')[1]!.text()).toBe(`#${idC}`);
    expect(wrapper.find('.summary').exists()).toBe(false);

    // late answers for A and B must not appear
    backend.pending[idB]!({} as History);
    backend.pending[idA]!({} as History);
    await flushPromises();
    expect(wrapper.find('.summary').exists()).toBe(false);

    backend.pending[idC]!({} as History);
    await flushPromises();
    const shown = wrapper.find('.summary').text();
    expect(shown).toContain(idC >= 1000 ? `${(idC / 1000).toFixed(1)}K` : String(idC));
  });
});
