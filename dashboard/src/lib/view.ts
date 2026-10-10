// In-app views, kept in `location.hash` so a reload stays on the same view.

export const VIEWS = ['overview', 'sessions', 'requests', 'engine', 'settings'] as const;
export type View = (typeof VIEWS)[number];

export const DEFAULT_VIEW: View = 'overview';

/** "#/sessions" (or "#sessions") -> 'sessions'; anything unknown -> the default. */
export function parseHash(hash: string): View {
  const name = hash.replace(/^#\/?/, '').split(/[/?]/)[0] ?? '';
  return (VIEWS as readonly string[]).includes(name) ? (name as View) : DEFAULT_VIEW;
}

export function hashFor(view: View): string {
  return `#/${view}`;
}
