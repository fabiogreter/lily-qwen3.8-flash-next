import { ApiError, type Backend } from '../api/client';
import type {
  History,
  Range,
  RequestsResponse,
  Session,
  SessionsResponse,
  Setting,
  SettingResetResponse,
  SettingsLogEntry,
  SettingsLogResponse,
  SettingsPatch,
  SettingsResponse,
  SettingsUpdateResponse,
  SettingValue,
  Status,
} from '../api/types';
import {
  buildHistory,
  buildRequests,
  buildSessionHistory,
  buildSessions,
  buildSettings,
  buildSettingsLog,
  buildStatus,
} from './data';
import { DurationError, parseDurationSecs } from '../lib/duration';

/**
 * Validates one value the way the server will: a refusal is a 400 with a
 * message the page shows inline. Returns the normalised value (a duration's
 * trimmed text, a number typed as text as a number).
 */
export function validateSetting(x: Setting, value: SettingValue): SettingValue {
  const name = x.name;
  const refuse = (message: string): never => {
    throw new ApiError(400, message, name);
  };
  switch (x.type) {
    case 'number':
    case 'int': {
      const v = typeof value === 'number' ? value : typeof value === 'string' && value.trim() !== '' ? Number(value) : NaN;
      if (!Number.isFinite(v)) return refuse(`${name} must be a number`);
      if ((x.min != null && v < x.min) || (x.max != null && v > x.max)) {
        return refuse(`${name} must be between ${x.min} and ${x.max}`);
      }
      if (x.type === 'int' && !Number.isInteger(v)) return refuse(`${name} must be a whole number`);
      return v;
    }
    case 'bool':
      if (typeof value !== 'boolean') return refuse(`${name} must be true or false`);
      return value;
    case 'enum':
      if (typeof value !== 'string' || !x.options?.includes(value)) {
        return refuse(`${name} must be one of ${(x.options ?? []).join(', ')}`);
      }
      return value;
    case 'duration': {
      if (typeof value !== 'string') return refuse(`${name} must be a duration such as 30m, 1h or 90`);
      try {
        parseDurationSecs(value);
      } catch (e) {
        if (e instanceof DurationError) return refuse(e.message);
        throw e;
      }
      return value.trim();
    }
    case 'string':
      if (typeof value !== 'string' || value.trim() === '') return refuse(`${name} must not be empty`);
      return value.trim();
  }
}

/** Sets the value and keeps `value_s` in step for durations. */
function assign(x: Setting, value: SettingValue | null): void {
  x.value = value;
  x.value_s = x.type === 'duration' && value != null ? parseDurationSecs(String(value)) : null;
}

export interface MockOptions {
  /** Wall clock (unix ms); injected in tests. */
  clock?: () => number;
}

/**
 * The mock backend: the generators of ./data behind the Backend interface, so
 * the app runs without lily. Deterministic given the clock: the n-th status
 * call returns the same status, histories depend only on the bucket times, and
 * the settings behave like the real ones (invalid values are refused with the
 * server's messages, changes are logged).
 */
export class MockBackend implements Backend {
  readonly kind = 'mock';
  private tick = 0;
  private readonly clock: () => number;
  private readonly sessionList: Session[];
  private readonly settingList: Setting[] = buildSettings();
  private readonly log: SettingsLogEntry[];

  constructor(opts: MockOptions = {}) {
    this.clock = opts.clock ?? Date.now;
    this.sessionList = buildSessions(this.clock());
    this.log = buildSettingsLog(this.clock());
  }

  async status(): Promise<Status> {
    return buildStatus(this.tick++, this.sessionList, this.clock());
  }

  async history(range: Range): Promise<History> {
    return buildHistory(range, this.clock());
  }

  async sessions(): Promise<SessionsResponse> {
    return { sessions: this.sessionList.map((x) => ({ ...x })) };
  }

  async sessionHistory(lineage: number, range: Range): Promise<History> {
    const h = buildSessionHistory(this.sessionList, lineage, range, this.clock());
    if (!h) throw new ApiError(404, `no cache entry with lineage ${lineage}`);
    return h;
  }

  async requests(limit: number): Promise<RequestsResponse> {
    return { requests: buildRequests(this.sessionList, this.clock()).slice(0, limit) };
  }

  async settings(): Promise<SettingsResponse> {
    return { settings: this.settingList.map((x) => ({ ...x })) };
  }

  async settingsLog(): Promise<SettingsLogResponse> {
    return { entries: this.log.slice() };
  }

  async updateSettings(patch: SettingsPatch): Promise<SettingsUpdateResponse> {
    // Validate everything first: a patch is applied all or none.
    const checked: [Setting, SettingValue][] = [];
    for (const [name, raw] of Object.entries(patch)) {
      const x = this.find(name);
      if (!x.editable) throw new ApiError(409, `${name} needs a restart and cannot be changed here`, name);
      checked.push([x, validateSetting(x, raw)]);
    }
    const updated: Setting[] = [];
    for (const [x, value] of checked) {
      if (value === x.value && x.source === 'database') continue;
      this.log.unshift({ t: this.clock(), name: x.name, old: x.value, new: value, source: 'dashboard' });
      assign(x, value);
      x.source = 'database';
      updated.push({ ...x });
    }
    return { updated };
  }

  async resetSetting(name: string): Promise<SettingResetResponse> {
    const x = this.find(name);
    if (!x.editable) throw new ApiError(409, `${name} needs a restart and cannot be changed here`, name);
    const back = x.flag_value ?? x.default;
    this.log.unshift({ t: this.clock(), name, old: x.value, new: back, source: 'dashboard' });
    assign(x, back);
    x.source = x.flag_value != null ? 'flag' : 'default';
    return { setting: { ...x } };
  }

  private find(name: string): Setting {
    const x = this.settingList.find((y) => y.name === name);
    if (!x) throw new ApiError(404, `unknown setting: ${name}`, name);
    return x;
  }
}
