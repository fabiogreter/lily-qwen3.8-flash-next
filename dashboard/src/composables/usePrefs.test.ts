import { afterEach, describe, expect, it, vi } from 'vitest';

afterEach(() => {
  vi.resetModules();
  vi.unstubAllGlobals();
  localStorage.clear();
  delete document.documentElement.dataset.paper;
  delete document.documentElement.dataset.theme;
});

describe('usePrefs', () => {
  it('defaults to charcoal paper and the system theme', async () => {
    const m = await import('./usePrefs');
    expect(m.paper.value).toBe('charcoal');
    expect(m.theme.value).toBe('auto');
    expect(document.documentElement.dataset.paper).toBe('charcoal');
    expect(document.documentElement.dataset.theme).toBeUndefined();
  });

  it('restores stored choices and ignores junk', async () => {
    localStorage.setItem('lily-dashboard-paper', 'grain');
    localStorage.setItem('lily-dashboard-theme', 'purple');
    const m = await import('./usePrefs');
    expect(m.paper.value).toBe('grain');
    expect(m.theme.value).toBe('auto');
  });

  it('applies and stores a change', async () => {
    const m = await import('./usePrefs');
    m.paper.value = 'smooth';
    m.theme.value = 'dark';
    await Promise.resolve();
    expect(document.documentElement.dataset.paper).toBe('smooth');
    expect(document.documentElement.dataset.theme).toBe('dark');
    expect(localStorage.getItem('lily-dashboard-paper')).toBe('smooth');
    expect(localStorage.getItem('lily-dashboard-theme')).toBe('dark');
    m.theme.value = 'auto';
    await Promise.resolve();
    expect(document.documentElement.dataset.theme).toBeUndefined();
  });

  it('works when storage throws', async () => {
    vi.stubGlobal('localStorage', {
      getItem: () => {
        throw new Error('blocked');
      },
      setItem: () => {
        throw new Error('blocked');
      },
    });
    const m = await import('./usePrefs');
    expect(m.paper.value).toBe('charcoal');
    m.paper.value = 'grain';
    await Promise.resolve();
    expect(document.documentElement.dataset.paper).toBe('grain');
  });
});
