// SelTranslate: automatic update checks once a week instead of on every
// start. Handy's UpdateChecker calls `startWeeklyChecks` where it used to
// check right away; a manual check (footer, tray) is not affected. The time
// of the last automatic check is a per-device convenience, so it lives in
// localStorage: losing it only means one check sooner.

const LAST_KEY = "seltranslate.updates.lastAutoCheck";
const PENDING_KEY = "seltranslate.updates.pending";
export const WEEK_MS = 7 * 24 * 60 * 60 * 1000;
/** How often a long-running app looks whether the week is over. */
const TICK_MS = 60 * 60 * 1000;

const read = (key: string): string | null => {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
};

const write = (key: string, value: string | null) => {
  try {
    if (value === null) localStorage.removeItem(key);
    else localStorage.setItem(key, value);
  } catch {
    // Storage blocked: checks just happen on every start, as in Handy.
  }
};

/** Whether an automatic check is due at `now`. A last check in the future
 * (the clock was moved back) counts as due. */
export const isDue = (last: number | null, now: number): boolean =>
  last === null ||
  !Number.isFinite(last) ||
  last > now ||
  now - last >= WEEK_MS;

/** Runs `check` now if a week has passed since the last automatic check (or
 * an update was found then), and then hourly whenever the week is over.
 * Returns the cleanup. */
export const startWeeklyChecks = (check: () => void): (() => void) => {
  const tick = (atStart: boolean) => {
    const raw = read(LAST_KEY);
    const last = raw === null ? null : Number(raw);
    const pending = atStart && read(PENDING_KEY) === "1";
    if (!pending && !isDue(last, Date.now())) return;
    write(LAST_KEY, String(Date.now()));
    write(PENDING_KEY, null);
    check();
  };
  tick(true);
  const id = setInterval(() => tick(false), TICK_MS);
  return () => clearInterval(id);
};

/** An update was found: show it again on the next start, without waiting a
 * week, until it is installed. */
export const rememberPendingUpdate = () => write(PENDING_KEY, "1");
