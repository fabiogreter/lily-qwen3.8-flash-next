import { ApiError, isNamed, type Backend } from './client';
import type {
  ErrorBody,
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

const BASE = '/v1/lily';

type FetchLike = (input: string, init?: RequestInit) => Promise<Response>;

export interface HttpBackendOptions {
  /** Injected in tests; defaults to the global fetch. */
  fetch?: FetchLike;
  /** Prefix for the routes; empty when the page is served by lily itself. */
  baseUrl?: string;
}

function isErrorBody(x: unknown): x is ErrorBody {
  if (typeof x !== 'object' || x === null || !('error' in x)) return false;
  const err = (x as { error: unknown }).error;
  return typeof err === 'object' && err !== null && typeof (err as { message?: unknown }).message === 'string';
}

/** The request's own abort (a newer request replaced it); the caller drops the answer. */
function isAbort(e: unknown): boolean {
  return isNamed(e, 'AbortError');
}

function isTimeout(e: unknown): boolean {
  return isNamed(e, 'TimeoutError');
}

/** The real backend: lily's `/v1/lily/*` routes. Reads only, except the settings writes. */
export class HttpBackend implements Backend {
  readonly kind = 'http';
  private readonly doFetch: FetchLike;
  private readonly baseUrl: string;

  constructor(opts: HttpBackendOptions = {}) {
    this.doFetch = opts.fetch ?? ((input, init) => globalThis.fetch(input, init));
    this.baseUrl = opts.baseUrl ?? '';
  }

  status(signal?: AbortSignal): Promise<Status> {
    return this.get('/status', signal);
  }

  history(range: Range, signal?: AbortSignal): Promise<History> {
    return this.get(`/history?range=${encodeURIComponent(range)}`, signal);
  }

  sessions(signal?: AbortSignal): Promise<SessionsResponse> {
    return this.get('/sessions', signal);
  }

  sessionHistory(lineage: number, range: Range, signal?: AbortSignal): Promise<History> {
    return this.get(`/sessions/${encodeURIComponent(String(lineage))}/history?range=${encodeURIComponent(range)}`, signal);
  }

  requests(limit: number, signal?: AbortSignal): Promise<RequestsResponse> {
    return this.get(`/requests?limit=${encodeURIComponent(String(limit))}`, signal);
  }

  settings(signal?: AbortSignal): Promise<SettingsResponse> {
    return this.get('/settings', signal);
  }

  settingsLog(signal?: AbortSignal): Promise<SettingsLogResponse> {
    return this.get('/settings/log', signal);
  }

  updateSettings(patch: SettingsPatch): Promise<SettingsUpdateResponse> {
    return this.send('PATCH', '/settings', JSON.stringify(patch));
  }

  resetSetting(name: string): Promise<SettingResetResponse> {
    return this.send('DELETE', `/settings/${encodeURIComponent(name)}`);
  }

  private get<T>(path: string, signal?: AbortSignal): Promise<T> {
    return this.request<T>(path, { method: 'GET', headers: { Accept: 'application/json' }, signal });
  }

  /**
   * Writes always carry `Content-Type: application/json`, DELETE without a body
   * included: the server refuses a write without it (cross-site write guard,
   * docs/dashboard.md § 3.2).
   */
  private send<T>(method: 'PATCH' | 'DELETE', path: string, body?: string): Promise<T> {
    return this.request<T>(path, {
      method,
      headers: { Accept: 'application/json', 'Content-Type': 'application/json' },
      body,
    });
  }

  private async request<T>(path: string, init: RequestInit): Promise<T> {
    let res: Response;
    try {
      res = await this.doFetch(this.baseUrl + BASE + path, { ...init, cache: 'no-store' });
    } catch (e) {
      if (isAbort(e)) throw e;
      if (isTimeout(e)) throw new ApiError(0, 'lily did not answer in time');
      throw new ApiError(0, 'lily is not reachable');
    }
    if (!res.ok) {
      let message = `HTTP ${res.status}`;
      let setting: string | null = null;
      try {
        const body: unknown = await res.json();
        if (isErrorBody(body)) {
          message = body.error.message;
          setting = body.error.setting ?? null;
        }
      } catch {
        // not JSON: keep the status line
      }
      throw new ApiError(res.status, message, setting);
    }
    try {
      return (await res.json()) as T;
    } catch (e) {
      if (isAbort(e)) throw e;
      if (isTimeout(e)) throw new ApiError(0, 'lily did not answer in time');
      throw new ApiError(res.status, 'lily sent an answer the dashboard cannot read');
    }
  }
}
