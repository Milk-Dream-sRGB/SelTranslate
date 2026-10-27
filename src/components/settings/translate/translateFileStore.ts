// SelTranslate: state of the "Translate file" block. Lives outside the
// component so a running job survives switching sidebar sections.
import { create } from "zustand";
import { listen } from "@tauri-apps/api/event";
import { commands, type FileProbe } from "@/bindings";

export type TranslateFileStatus =
  | "idle"
  | "starting_model"
  | "translating"
  | "done"
  | "cancelled"
  | "error";

interface ProgressPayload {
  phase: "starting_model" | "translating";
  done: number;
  total: number;
  done_chars: number;
  total_chars: number;
  elapsed_ms: number;
  eta_secs: number | null;
}

export interface TranslateFileResult {
  /** Where the result was saved; null when `save_error` says why not. */
  output_path: string | null;
  save_error: string | null;
  text: string;
  pieces: number;
  untranslated: number;
  elapsed_secs: number;
  src: string;
  tgt: string;
}

interface TranslateFileState {
  path: string | null;
  probe: FileProbe | null;
  status: TranslateFileStatus;
  done: number;
  total: number;
  etaSecs: number | null;
  result: TranslateFileResult | null;
  error: string | null;
  init: () => void;
  selectFile: (path: string) => Promise<void>;
  start: () => Promise<void>;
  cancel: () => Promise<void>;
}

export const isRunning = (status: TranslateFileStatus) =>
  status === "starting_model" || status === "translating";

let listening = false;

export const useTranslateFileStore = create<TranslateFileState>()(
  (set, get) => ({
    path: null,
    probe: null,
    status: "idle",
    done: 0,
    total: 0,
    etaSecs: null,
    result: null,
    error: null,

    init: () => {
      if (listening) return;
      listening = true;
      listen<ProgressPayload>("translate-file-progress", ({ payload }) => {
        set({
          status: payload.phase,
          done: payload.done,
          total: payload.total,
          etaSecs: payload.eta_secs,
        });
      });
      listen<TranslateFileResult>("translate-file-done", ({ payload }) => {
        set({ status: "done", result: payload, etaSecs: null });
      });
      listen<{ message: string }>("translate-file-error", ({ payload }) => {
        set({ status: "error", error: payload.message, etaSecs: null });
      });
      listen("translate-file-cancelled", () => {
        set({ status: "cancelled", etaSecs: null });
      });
    },

    selectFile: async (path) => {
      set({
        path,
        probe: null,
        status: "idle",
        done: 0,
        total: 0,
        etaSecs: null,
        result: null,
        error: null,
      });
      const result = await commands.translateFileProbe(path);
      if (get().path !== path) return;
      if (result.status === "ok") {
        set({ probe: result.data, total: result.data.pieces });
      } else {
        set({ status: "error", error: result.error });
      }
    },

    start: async () => {
      const { path, probe } = get();
      if (!path || !probe) return;
      set({
        status: "starting_model",
        done: 0,
        total: probe.pieces,
        etaSecs: null,
        result: null,
        error: null,
      });
      const result = await commands.translateFileStart(path);
      if (result.status === "error") {
        set({ status: "error", error: result.error });
      }
    },

    cancel: async () => {
      await commands.translateFileCancel();
    },
  }),
);
