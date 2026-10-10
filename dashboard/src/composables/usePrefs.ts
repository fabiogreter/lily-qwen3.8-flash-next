import { ref, watch } from 'vue';

// Preferences of this browser only; they never reach lily. Stored in
// localStorage, which can be blocked or throw: every access is guarded.

export const THEMES = ['auto', 'light', 'dark'] as const;
export const PAPERS = ['smooth', 'grain', 'charcoal'] as const;
export type Theme = (typeof THEMES)[number];
export type Paper = (typeof PAPERS)[number];

export const THEME_KEY = 'lily-dashboard-theme';
export const PAPER_KEY = 'lily-dashboard-paper';

function read<T extends string>(key: string, allowed: readonly T[], fallback: T): T {
  try {
    const v = localStorage.getItem(key);
    return (allowed as readonly string[]).includes(v ?? '') ? (v as T) : fallback;
  } catch {
    return fallback;
  }
}

function write(key: string, value: string): void {
  try {
    localStorage.setItem(key, value);
  } catch {
    // storage blocked: the choice lasts for this page view only
  }
}

export function applyTheme(theme: Theme, root: HTMLElement = document.documentElement): void {
  if (theme === 'auto') delete root.dataset.theme;
  else root.dataset.theme = theme;
}

export function applyPaper(paper: Paper, root: HTMLElement = document.documentElement): void {
  root.dataset.paper = paper;
}

export const theme = ref<Theme>(read(THEME_KEY, THEMES, 'auto'));
/** Charcoal is the default texture. */
export const paper = ref<Paper>(read(PAPER_KEY, PAPERS, 'charcoal'));

applyTheme(theme.value);
applyPaper(paper.value);

watch(theme, (v) => {
  applyTheme(v);
  write(THEME_KEY, v);
});
watch(paper, (v) => {
  applyPaper(v);
  write(PAPER_KEY, v);
});

export function usePrefs() {
  return { theme, paper, themes: THEMES, papers: PAPERS };
}
