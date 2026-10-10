import { mount } from '@vue/test-utils';
import { afterEach, describe, expect, it } from 'vitest';
import { defineComponent, h } from 'vue';
import { useHashView } from './useHashView';

function host() {
  let api!: ReturnType<typeof useHashView>;
  const wrapper = mount(
    defineComponent({
      setup() {
        api = useHashView();
        return () => h('div');
      },
    }),
  );
  return { api, wrapper };
}

afterEach(() => history.replaceState(null, '', '#'));

describe('useHashView', () => {
  it('an empty or unknown hash is replaced by the canonical one on start', () => {
    history.replaceState(null, '', '#/nonsense');
    const { api, wrapper } = host();
    expect(api.view.value).toBe('overview');
    expect(location.hash).toBe('#/overview');
    wrapper.unmount();
    history.replaceState(null, '', '#');
    const second = host();
    expect(location.hash).toBe('#/overview');
    second.wrapper.unmount();
  });

  it('a known hash is kept, set() changes view and hash, and a later bad hash is normalised', async () => {
    history.replaceState(null, '', '#/sessions');
    const { api, wrapper } = host();
    expect(api.view.value).toBe('sessions');
    api.set('engine');
    expect(location.hash).toBe('#/engine');
    expect(api.view.value).toBe('engine');
    history.replaceState(null, '', '#/bogus');
    window.dispatchEvent(new HashChangeEvent('hashchange'));
    expect(api.view.value).toBe('overview');
    expect(location.hash).toBe('#/overview');
    wrapper.unmount();
  });
});
