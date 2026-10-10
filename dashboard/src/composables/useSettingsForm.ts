import { reactive, type Ref } from 'vue';
import { errorMessage, type Backend } from '../api/client';
import type { Setting, SettingValue } from '../api/types';

/** A draft typed into a field, or a value from a toggle or select. */
export type RawValue = string | boolean;

export type Coerced = { ok: true; value: SettingValue } | { ok: false; message: string };

/**
 * Field input to the JSON value the server expects. Only the type is checked
 * here; ranges and formats are the server's call, and its message is shown as
 * it comes (the server validates with the same parsers as the command line).
 */
export function coerce(setting: Pick<Setting, 'type'>, raw: RawValue): Coerced {
  switch (setting.type) {
    case 'number':
    case 'int': {
      const text = String(raw).trim();
      const v = Number(text);
      if (text === '' || !Number.isFinite(v)) return { ok: false, message: 'Enter a number.' };
      return { ok: true, value: v };
    }
    case 'bool':
      return { ok: true, value: raw === true || raw === 'true' };
    default:
      return { ok: true, value: String(raw) };
  }
}

/**
 * Editing state for the settings view: drafts per field, the server's refusal
 * per field, and the writes. `settings` is replaced with what the server
 * answers; `onChanged` runs after a successful write (the log reloads).
 */
export function useSettingsForm(backend: Backend, settings: Ref<Setting[] | null>, onChanged: () => void) {
  const drafts = reactive<Record<string, string>>({});
  const errors = reactive<Record<string, string>>({});
  const busy = reactive<Record<string, boolean>>({});

  function replace(updated: Setting[]): void {
    if (!settings.value) return;
    const byName = new Map(updated.map((s) => [s.name, s]));
    settings.value = settings.value.map((s) => byName.get(s.name) ?? s);
  }

  function clear(name: string): void {
    delete drafts[name];
    delete errors[name];
  }

  /**
   * Returns true when the server accepted the value (or it equals the current
   * one). A second commit of the same setting while one is in flight is
   * refused (false) without a request.
   */
  async function commit(setting: Setting, raw: RawValue): Promise<boolean> {
    if (busy[setting.name]) return false;
    const c = coerce(setting, raw);
    if (!c.ok) {
      errors[setting.name] = c.message;
      return false;
    }
    if (c.value === setting.value) {
      clear(setting.name);
      return true;
    }
    busy[setting.name] = true;
    try {
      const res = await backend.updateSettings({ [setting.name]: c.value });
      replace(res.updated);
      clear(setting.name);
      onChanged();
      return true;
    } catch (e) {
      errors[setting.name] = errorMessage(e);
      return false;
    } finally {
      busy[setting.name] = false;
    }
  }

  /** Back to the flag or default value (deletes the database override). */
  async function reset(setting: Setting): Promise<boolean> {
    if (busy[setting.name]) return false;
    busy[setting.name] = true;
    try {
      const res = await backend.resetSetting(setting.name);
      replace([res.setting]);
      clear(setting.name);
      onChanged();
      return true;
    } catch (e) {
      errors[setting.name] = errorMessage(e);
      return false;
    } finally {
      busy[setting.name] = false;
    }
  }

  function discard(name: string): void {
    clear(name);
  }

  return { drafts, errors, busy, commit, reset, discard };
}
