import { readFileSync } from 'node:fs';
import vue from '@vitejs/plugin-vue';
import type { Plugin } from 'vite';
import { defineConfig } from 'vitest/config';

const read = (path: string): string => readFileSync(new URL(path, import.meta.url), 'utf8').trim();

/**
 * Emits `assets/licenses.txt` (an unhashed name, fixed content: the build stays
 * reproducible): the notices the bundled third-party code and fonts require.
 * The Settings view links to it.
 */
function licenses(): Plugin {
  return {
    name: 'lily-licenses',
    generateBundle() {
      const parts = [
        'Third-party code and fonts in this dashboard.',
        ['Vue (MIT)', read('./node_modules/vue/LICENSE')],
        ['Shippori Mincho (SIL Open Font License 1.1)', read('./licenses/shippori-mincho-OFL.txt')],
        ['IBM Plex Sans (SIL Open Font License 1.1)', read('./licenses/ibm-plex-sans-OFL.txt')],
      ].map((x) => (typeof x === 'string' ? x : `${'='.repeat(72)}\n${x[0]}\n${'='.repeat(72)}\n\n${x[1]}`));
      this.emitFile({ type: 'asset', fileName: 'assets/licenses.txt', source: parts.join('\n\n') + '\n' });
    },
  };
}

// `vite --mode mock` (npm run dev:mock, build:mock) swaps the HTTP backend for
// the in-browser mock. The choice is a build-time constant, so the HTTP build
// carries no mock code. (No .env files: the constant is defined right here.)
export default defineConfig(({ mode }) => ({
  base: '/',
  plugins: [vue(), licenses()],
  define: {
    'import.meta.env.VITE_BACKEND': JSON.stringify(mode === 'mock' ? 'mock' : 'http'),
  },
  server: {
    // `npm run dev` talks to a lily on its default port.
    proxy: { '/v1': 'http://127.0.0.1:8000' },
  },
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
    emptyOutDir: true,
    sourcemap: false,
  },
  test: {
    environment: 'jsdom',
    globalSetup: ['./tests/global-setup.ts'],
    include: ['src/**/*.test.ts', 'tests/**/*.test.ts'],
  },
}));
