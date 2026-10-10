import { onBeforeUnmount, ref } from 'vue';
import { hashFor, parseHash, type View } from '../lib/view';

/**
 * The current view, mirrored in `location.hash`. An unknown or empty hash is
 * replaced by the canonical one of the view it falls back to, so the address
 * bar always names what is shown.
 */
export function useHashView() {
  const view = ref<View>(parseHash(location.hash));

  function normalise(): void {
    const want = hashFor(view.value);
    if (location.hash !== want) history.replaceState(null, '', want);
  }
  function fromHash(): void {
    view.value = parseHash(location.hash);
    normalise();
  }
  window.addEventListener('hashchange', fromHash);
  onBeforeUnmount(() => window.removeEventListener('hashchange', fromHash));
  normalise();

  // Navigation by the nav buttons: a new history entry, like a link.
  const set = (v: View): void => {
    view.value = v;
    if (location.hash !== hashFor(v)) location.hash = hashFor(v);
  };
  return { view, set };
}
