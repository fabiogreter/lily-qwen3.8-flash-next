/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** 'mock' (vite --mode mock) or 'http'; defined in vite.config.ts. */
  readonly VITE_BACKEND: 'mock' | 'http';
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

declare module '*.vue' {
  import type { DefineComponent } from 'vue';
  const component: DefineComponent<object, object, unknown>;
  export default component;
}
