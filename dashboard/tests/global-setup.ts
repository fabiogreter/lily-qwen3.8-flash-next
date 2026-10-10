// Time formatting uses the browser's local time zone; pin it so the tests are
// the same on every machine.
export function setup(): void {
  process.env.TZ = 'UTC';
}
