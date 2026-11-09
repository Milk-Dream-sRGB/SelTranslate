// SelTranslate: "Speakers" on a History entry (src-tauri/src/speakers). The
// backend runs one job at a time and tags every event with the entry's id;
// this store keeps each entry's panel (options, progress or result) until it
// is closed or the window reloads. Nothing is written to Handy's history.
import { create } from "zustand";
import { listen } from "@tauri-apps/api/event";
import { commands, type SpeakersOptions, type SpeakersState } from "@/bindings";

/** One line of the result, as `speakers-done` sends it (TurnText in Rust). */
export interface TurnText {
  start_ms: number;
  speaker: number;
  text: string;
}

export type EntrySpeakersStatus =
  | "options"
  | "downloading"
  | "detecting"
  | "transcribing"
  | "done"
  | "error";

export interface EntrySpeakers {
  status: EntrySpeakersStatus;
  done: number;
  total: number;
  turns: TurnText[];
  speakers: number;
  error: string | null;
}

interface ProgressPayload {
  entry_id: number;
  phase: "downloading" | "detecting" | "transcribing";
  done: number;
  total: number;
}

interface DonePayload {
  entry_id: number;
  turns: TurnText[];
  speakers: number;
}

interface HistorySpeakersState {
  byEntry: Record<number, EntrySpeakers>;
  /** The entry being worked on now; `null` when none is. */
  running: number | null;
  /** Options and the downloaded variants; `null` until loaded. */
  setup: SpeakersState | null;
  init: () => void;
  refresh: () => Promise<void>;
  open: (entryId: number) => void;
  /** Back to the options, for another run. */
  reset: (entryId: number) => void;
  setOptions: (patch: Partial<SpeakersOptions>) => Promise<void>;
  deleteVariant: (id: string) => Promise<string | null>;
  start: (entryId: number) => Promise<void>;
  cancel: () => Promise<void>;
  close: (entryId: number) => void;
}

const fresh: EntrySpeakers = {
  status: "options",
  done: 0,
  total: 0,
  turns: [],
  speakers: 0,
  error: null,
};

let listening = false;

export const useHistorySpeakersStore = create<HistorySpeakersState>()((
  set,
  get,
) => {
  const update = (id: number, patch: Partial<EntrySpeakers>) =>
    set(({ byEntry }) => ({
      byEntry: { ...byEntry, [id]: { ...(byEntry[id] ?? fresh), ...patch } },
    }));
  const finish = (id: number) => {
    set(({ running }) => ({ running: running === id ? null : running }));
    // A first run downloads its model: the list shows it as downloaded now.
    void get().refresh();
  };

  return {
    byEntry: {},
    running: null,
    setup: null,

    init: () => {
      if (listening) return;
      listening = true;
      void get().refresh();
      listen<ProgressPayload>("speakers-progress", ({ payload }) => {
        update(payload.entry_id, {
          status: payload.phase,
          done: payload.done,
          total: payload.total,
        });
      });
      listen<DonePayload>("speakers-done", ({ payload }) => {
        update(payload.entry_id, {
          status: "done",
          turns: payload.turns,
          speakers: payload.speakers,
        });
        finish(payload.entry_id);
      });
      listen<{ entry_id: number; message: string }>(
        "speakers-error",
        ({ payload }) => {
          update(payload.entry_id, { status: "error", error: payload.message });
          finish(payload.entry_id);
        },
      );
      listen<{ entry_id: number }>("speakers-cancelled", ({ payload }) => {
        update(payload.entry_id, { ...fresh });
        finish(payload.entry_id);
      });
    },

    refresh: async () => {
      const result = await commands.speakersState();
      if (result.status === "ok") {
        set({ setup: result.data });
        if (result.data.running !== null) set({ running: result.data.running });
      }
    },

    open: (entryId) => {
      if (get().byEntry[entryId]) return;
      update(entryId, fresh);
    },

    reset: (entryId) => update(entryId, fresh),

    setOptions: async (patch) => {
      const setup = get().setup;
      if (!setup) return;
      const options = { ...setup.options, ...patch };
      set({ setup: { ...setup, options } });
      await commands.speakersSetOptions(options);
    },

    deleteVariant: async (id) => {
      const result = await commands.speakersDeleteVariant(id);
      await get().refresh();
      return result.status === "error" ? result.error : null;
    },

    start: async (entryId) => {
      set({ running: entryId });
      update(entryId, { ...fresh, status: "detecting" });
      const result = await commands.speakersStart(entryId);
      if (result.status === "error") {
        update(entryId, { status: "error", error: result.error });
        set(({ running }) => ({
          running: running === entryId ? null : running,
        }));
      }
    },

    cancel: async () => {
      await commands.speakersCancel();
    },

    close: (entryId) =>
      set(({ byEntry }) => {
        const rest = { ...byEntry };
        delete rest[entryId];
        return { byEntry: rest };
      }),
  };
});
