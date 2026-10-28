// SelTranslate: state of the "Set up translation" card (DESIGN F.5). The setup
// itself lives in Rust (the card may be reopened mid-download); this store
// keeps the last plan, the progress events and the user's answers outside the
// component so they survive switching sidebar sections.
import { create } from "zustand";
import { listen } from "@tauri-apps/api/event";
import {
  commands,
  type ErrorKind,
  type SetupPlan,
  type SetupProgress,
  type SetupStateView,
} from "@/bindings";

export type SetupNotice = "no_gpu" | "cuda_failed";

interface DonePayload {
  installed: string[];
  skipped: string[];
  notices: SetupNotice[];
}

interface ErrorPayload {
  item: string;
  kind: ErrorKind;
  message: string;
}

interface TranslateSetupState {
  plan: SetupPlan | null;
  view: SetupStateView;
  /** What the last run reported: the engine could not use the card, etc. */
  notices: SetupNotice[];
  skipped: string[];
  /** Items of the last Download click: Resume and Retry go on with them. */
  lastItems: string[];
  /** The Gemma Terms box; not saved, the setup is rare (F.5). */
  accepted: boolean;
  /** A refusal of `translate_setup_start` (no space, terms not accepted). */
  startError: string | null;
  init: () => void;
  refresh: () => Promise<void>;
  setAccepted: (accepted: boolean) => void;
  start: (items: string[]) => Promise<boolean>;
  cancel: () => Promise<void>;
  discard: () => Promise<void>;
  dismissNotices: () => void;
  dismissError: () => Promise<void>;
}

export const isSetupRunning = (view: SetupStateView) =>
  view.state === "running";

let listening = false;

export const useTranslateSetupStore = create<TranslateSetupState>()(
  (set, get) => ({
    plan: null,
    view: { state: "idle" },
    notices: [],
    skipped: [],
    lastItems: [],
    accepted: false,
    startError: null,

    init: () => {
      if (listening) {
        void get().refresh();
        return;
      }
      listening = true;
      void listen<SetupProgress>("translate-setup-progress", (event) =>
        set({ view: { state: "running", progress: event.payload } }),
      );
      void listen<DonePayload>("translate-setup-done", (event) => {
        set({
          view: { state: "idle" },
          notices: event.payload.notices,
          skipped: event.payload.skipped,
        });
        void get().refresh();
      });
      void listen<ErrorPayload>("translate-setup-error", (event) => {
        set({
          view: {
            state: "error",
            item: event.payload.item,
            kind: event.payload.kind,
            message: event.payload.message,
          },
        });
        void get().refresh();
      });
      void listen("translate-setup-cancelled", () => {
        set({ view: { state: "idle" } });
        void get().refresh();
      });
      void get().refresh();
    },

    refresh: async () => {
      const result = await commands.translateSetupPlan();
      if (result.status === "ok") {
        set({ plan: result.data, view: result.data.state });
      }
    },

    setAccepted: (accepted) => set({ accepted }),

    start: async (items) => {
      set({ startError: null });
      const result = await commands.translateSetupStart(items, get().accepted);
      if (result.status !== "ok") {
        set({ startError: result.error });
        return false;
      }
      set({ lastItems: items, notices: [], skipped: [] });
      return true;
    },

    cancel: async () => {
      await commands.translateSetupCancel();
    },

    discard: async () => {
      const result = await commands.translateSetupDiscard();
      if (result.status !== "ok") set({ startError: result.error });
      await get().refresh();
    },

    dismissNotices: () => set({ notices: [], skipped: [] }),

    dismissError: async () => {
      await commands.translateSetupDismissError();
      await get().refresh();
    },
  }),
);
