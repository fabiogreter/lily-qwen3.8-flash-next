import { describe, expect, it, vi } from 'vitest';
import { ApiError } from './client';
import { HttpBackend } from './http';

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
}

function setup(res: Response | (() => Response) = jsonResponse({})) {
  const fetch = vi.fn(async () => (typeof res === 'function' ? res() : res.clone()));
  const backend = new HttpBackend({ fetch });
  const call = (): { url: string; init: RequestInit } => {
    const [url, init] = fetch.mock.calls[0] as unknown as [string, RequestInit];
    return { url, init };
  };
  return { backend, fetch, call };
}

const headers = (init: RequestInit): Record<string, string> => init.headers as Record<string, string>;

describe('HttpBackend requests', () => {
  it('status: GET /v1/lily/status, no body, no content type', async () => {
    const { backend, call } = setup();
    await backend.status();
    const { url, init } = call();
    expect(url).toBe('/v1/lily/status');
    expect(init.method).toBe('GET');
    expect(init.body).toBeUndefined();
    expect(headers(init)['Content-Type']).toBeUndefined();
    expect(init.cache).toBe('no-store');
  });

  it('history: range goes in the query', async () => {
    const { backend, call } = setup();
    await backend.history('24h');
    expect(call().url).toBe('/v1/lily/history?range=24h');
  });

  it('sessions, session history and requests', async () => {
    const a = setup();
    await a.backend.sessions();
    expect(a.call().url).toBe('/v1/lily/sessions');
    const b = setup();
    await b.backend.sessionHistory(1222, '7d');
    expect(b.call().url).toBe('/v1/lily/sessions/1222/history?range=7d');
    const c = setup();
    await c.backend.requests(40);
    expect(c.call().url).toBe('/v1/lily/requests?limit=40');
  });

  it('settings and the change log are plain GETs', async () => {
    const a = setup();
    await a.backend.settings();
    expect(a.call().url).toBe('/v1/lily/settings');
    expect(a.call().init.method).toBe('GET');
    const b = setup();
    await b.backend.settingsLog();
    expect(b.call().url).toBe('/v1/lily/settings/log');
  });

  it('PATCH settings: JSON body {name: value} with Content-Type application/json', async () => {
    const { backend, call } = setup(jsonResponse({ updated: [] }));
    await backend.updateSettings({ top_p: 0.9 });
    const { url, init } = call();
    expect(url).toBe('/v1/lily/settings');
    expect(init.method).toBe('PATCH');
    expect(headers(init)['Content-Type']).toBe('application/json');
    expect(init.body).toBe('{"top_p":0.9}');
  });

  it('DELETE resets one setting and still sends Content-Type (write guard), without a body', async () => {
    const { backend, call } = setup(jsonResponse({ setting: {} }));
    await backend.resetSetting('top_p');
    const { url, init } = call();
    expect(url).toBe('/v1/lily/settings/top_p');
    expect(init.method).toBe('DELETE');
    expect(headers(init)['Content-Type']).toBe('application/json');
    expect(init.body).toBeUndefined();
  });

  it('passes the abort signal to fetch', async () => {
    const { backend, call } = setup();
    const ctl = new AbortController();
    await backend.status(ctl.signal);
    expect(call().init.signal).toBe(ctl.signal);
  });

  it('returns the parsed JSON body', async () => {
    const { backend } = setup(jsonResponse({ state: 'busy' }));
    await expect(backend.status()).resolves.toEqual({ state: 'busy' });
  });
});

describe('HttpBackend errors', () => {
  it('a 4xx with an error body surfaces the server message as written', async () => {
    const { backend } = setup(jsonResponse({ error: { message: 'top_p must be between 0 and 1' } }, 400));
    const e = await backend.updateSettings({ top_p: 5 }).catch((x: unknown) => x);
    expect(e).toBeInstanceOf(ApiError);
    expect((e as ApiError).message).toBe('top_p must be between 0 and 1');
    expect((e as ApiError).status).toBe(400);
  });

  it('carries the setting named in the error body', async () => {
    const { backend } = setup(jsonResponse({ error: { message: 'needs a restart', setting: 'max_seq' } }, 409));
    await expect(backend.updateSettings({ max_seq: 1 })).rejects.toMatchObject({ status: 409, setting: 'max_seq', message: 'needs a restart' });
  });

  it('a 2xx answer that is not JSON is a clean ApiError, not a SyntaxError', async () => {
    const { backend } = setup(new Response('<html>proxy page</html>', { status: 200 }));
    const e = await backend.status().catch((x: unknown) => x);
    expect(e).toBeInstanceOf(ApiError);
    expect((e as ApiError).message).toBe('lily sent an answer the dashboard cannot read');
  });

  it('a timeout of the request is reported as lily not answering', async () => {
    const backend = new HttpBackend({ fetch: async () => Promise.reject(new DOMException('signal timed out', 'TimeoutError')) });
    await expect(backend.status()).rejects.toMatchObject({ name: 'ApiError', status: 0, message: 'lily did not answer in time' });
  });

  it('falls back to the status line when the body is not an error body', async () => {
    const { backend } = setup(new Response('<html>nope</html>', { status: 403 }));
    await expect(backend.status()).rejects.toMatchObject({ message: 'HTTP 403', status: 403 });
    const { backend: b2 } = setup(jsonResponse({ unexpected: true }, 500));
    await expect(b2.status()).rejects.toMatchObject({ message: 'HTTP 500' });
  });

  it('a network failure is an ApiError with status 0', async () => {
    const backend = new HttpBackend({ fetch: async () => Promise.reject(new TypeError('Failed to fetch')) });
    await expect(backend.status()).rejects.toMatchObject({ name: 'ApiError', status: 0, message: 'lily is not reachable' });
  });

  it('an abort is rethrown as is, not reported as unreachable', async () => {
    const abort = new DOMException('aborted', 'AbortError');
    const backend = new HttpBackend({ fetch: async () => Promise.reject(abort) });
    await expect(backend.status()).rejects.toBe(abort);
  });
});
