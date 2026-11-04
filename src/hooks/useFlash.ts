import { useCallback, useEffect, useRef, useState } from "react";

/**
 * A short-lived "done" mark for a button (Copied, Saved), the way History
 * shows a copy: Handy's notices are for problems, not for successes.
 */
export const useFlash = (ms = 2000) => {
  const [flashed, setFlashed] = useState<string | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(
    () => () => {
      if (timer.current) clearTimeout(timer.current);
    },
    [],
  );

  const flash = useCallback(
    (key: string) => {
      if (timer.current) clearTimeout(timer.current);
      setFlashed(key);
      timer.current = setTimeout(() => setFlashed(null), ms);
    },
    [ms],
  );

  return [flashed, flash] as const;
};
