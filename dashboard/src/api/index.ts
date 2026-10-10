import type { Backend } from './client';
import { HttpBackend } from './http';

export type { Backend } from './client';
export { ApiError, errorMessage } from './client';
export * from './types';

/** The backend chosen at build time (`vite --mode mock` selects the mock). */
export async function createBackend(): Promise<Backend> {
  if (import.meta.env.VITE_BACKEND === 'mock') {
    const { MockBackend } = await import('../mock/backend');
    return new MockBackend();
  }
  return new HttpBackend();
}
