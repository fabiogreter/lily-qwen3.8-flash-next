import { createApp } from 'vue';
import { createBackend } from './api';
import App from './App.vue';
import './composables/usePrefs';
import './styles/fonts.css';
import './styles/tokens.css';
import './styles/base.css';

/** Startup failed (e.g. the mock chunk did not load): say so on the page, as plain text. */
function showFatal(e: unknown): void {
  const el = document.getElementById('app');
  if (!el) return;
  const p = document.createElement('p');
  p.className = 'fatal';
  p.setAttribute('role', 'alert');
  p.textContent = `The dashboard could not start: ${e instanceof Error ? e.message : String(e)}`;
  el.replaceChildren(p);
}

createBackend()
  .then((backend) => {
    createApp(App, { backend }).mount('#app');
  })
  .catch(showFatal);
