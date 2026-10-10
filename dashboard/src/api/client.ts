import type {
  History,
  Range,
  RequestsResponse,
  SessionsResponse,
  SettingResetResponse,
  SettingsLogResponse,
  SettingsPatch,
  SettingsResponse,
  SettingsUpdateResponse,
  Status,
} from './types';

/** What the dashboard needs from lily. Two implementations: HttpBackend (the real one) and MockBackend. */
export interface Backend {
  /** 'mock' shows the "mock data" badge. */
  readonly kind: 'http' | 'mock';
  status(signal?: AbortSignal): Promise<Status>;
  history(range: Range, signal?: AbortSignal): Promise<History>;
  sessions(signal?: AbortSignal): Promise<SessionsResponse>;
  sessionHistory(lineage: number, range: Range, signal?: AbortSignal): Promise<History>;
  requests(limit: number, signal?: AbortSignal): Promise<RequestsResponse>;
  settings(signal?: AbortSignal): Promise<SettingsResponse>;
  settingsLog(signal?: AbortSignal): Promise<SettingsLogResponse>;
  /** Rejects with ApiError (400 invalid value, 404 unknown, 409 restart-only) and the server's message. */
  updateSettings(patch: SettingsPatch): Promise<SettingsUpdateResponse>;
  resetSetting(name: string): Promise<SettingResetResponse>;
}

/** An answer the dashboard shows as written: `message` comes from the server's error body. */
export class ApiError extends Error {
  /** HTTP status; 0 when no answer came (unreachable, timed out). */
  readonly status: number;
  /** The setting the error is about, when the server named one. */
  readonly setting: string | null;
  constructor(status: number, message: string, setting: string | null = null) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.setting = setting;
  }
}

export function errorMessage(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

/** True for an error of the given `name` (DOMException or Error; matched by name so it works across realms). */
export function isNamed(e: unknown, name: string): boolean {
  return typeof e === 'object' && e !== null && (e as { name?: unknown }).name === name;
}
