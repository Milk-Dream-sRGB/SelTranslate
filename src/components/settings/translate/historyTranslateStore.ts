// SelTranslate: "Translate" on a History entry. The backend runs the same job
// as Translate file (one at a time, shared with it) and tags every event with
// the entry's id; this store keeps the latest translation of each entry until
// it is closed or the window reloads. Nothing is written to Handy's history.
import { create } from "zustand";
import { listen } from "@tauri-apps/api/event";
import { commands } from "@/bindings";

export type EntryTranslationStatus =
  | "starting_model"
  | "translating"
  | "done"
  | "error";

export interface EntryTranslation {
  status: EntryTranslationStatus;
  done: number;
  total: number;
  etaSecs: number | null;
  text: string | null;
  src: string | null;
  tgt: string | null;
  untranslated: number;
  error: string | null;
}

interface Tagged {
  entry_id: number | null;
}

interface ProgressPayload extends Tagged {
  phase: "starting_model" | "translating";
  done: number;
  total: number;
  eta_secs: number | null;
}

interface DonePayload extends Tagged {
  text: string;
  src: string;
  tgt: string;
  untranslated: number;
}

interface HistoryTranslateState {
  byEntry: Record<number, EntryTranslation>;
  /** The entry being translated now; `null` when none is. */
  running: number | null;
  init: () => void;
  start: (entryId: number) => Promise<void>;
  cancel: () => Promise<void>;
  close: (entryId: number) => void;
}

const fresh: EntryTranslation = {
  status: "starting_model",
  done: 0,
  total: 0,
  etaSecs: null,
  text: null,
  src: null,
  tgt: null,
  untranslated: 0,
  error: null,
};

let listening = false;

export const useHistoryTranslateStore = create<HistoryTranslateState>()((
  set,
  get,
) => {
  const update = (id: number, patch: Partial<EntryTranslation>) =>
    set(({ byEntry }) => ({
      byEntry: { ...byEntry, [id]: { ...(byEntry[id] ?? fresh), ...patch } },
    }));
  const finish = (id: number) =>
    set(({ running }) => ({ running: running === id ? null : running }));

  return {
    byEntry: {},
    running: null,

    init: () => {
      if (listening) return;
      listening = true;
      listen<ProgressPayload>("translate-file-progress", ({ payload }) => {
        if (payload.entry_id === null) return;
        update(payload.entry_id, {
          status: payload.phase,
          done: payload.done,
          total: payload.total,
          etaSecs: payload.eta_secs,
        });
      });
      listen<DonePayload>("translate-file-done", ({ payload }) => {
        if (payload.entry_id === null) return;
        update(payload.entry_id, {
          status: "done",
          text: payload.text,
          src: payload.src,
          tgt: payload.tgt,
          untranslated: payload.untranslated,
          etaSecs: null,
        });
        finish(payload.entry_id);
      });
      listen<Tagged & { message: string }>(
        "translate-file-error",
        ({ payload }) => {
          if (payload.entry_id === null) return;
          update(payload.entry_id, {
            status: "error",
            error: payload.message,
            etaSecs: null,
          });
          finish(payload.entry_id);
        },
      );
      listen<Tagged>("translate-file-cancelled", ({ payload }) => {
        if (payload.entry_id === null) return;
        get().close(payload.entry_id);
        finish(payload.entry_id);
      });
    },

    start: async (entryId) => {
      set(({ byEntry }) => ({
        running: entryId,
        byEntry: { ...byEntry, [entryId]: fresh },
      }));
      const result = await commands.translateHistoryEntryStart(entryId);
      if (result.status === "error") {
        update(entryId, { status: "error", error: result.error });
        finish(entryId);
      }
    },

    cancel: async () => {
      await commands.translateFileCancel();
    },

    close: (entryId) =>
      set(({ byEntry }) => {
        const rest = { ...byEntry };
        delete rest[entryId];
        return { byEntry: rest };
      }),
  };
});
