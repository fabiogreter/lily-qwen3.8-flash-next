import { inject, type InjectionKey, type Ref } from 'vue';
import type { Backend } from '../api/client';
import type { Range, Status } from '../api/types';
import type { Resource } from './useResource';

/** What App provides to the views. */
export interface DashboardContext {
  backend: Backend;
  /** Live status, polled every 2 s while the tab is visible. */
  status: Resource<Status>;
  /** The chart range, shared by the Overview and the Sessions view. */
  range: Ref<Range>;
}

export const dashboardKey: InjectionKey<DashboardContext> = Symbol('dashboard');

export function useDashboard(): DashboardContext {
  const ctx = inject(dashboardKey);
  if (!ctx) throw new Error('useDashboard: no dashboard context provided');
  return ctx;
}

/** Refresh intervals (ms). */
export const STATUS_POLL_MS = 2000;
export const HISTORY_POLL_MS = 60_000;
