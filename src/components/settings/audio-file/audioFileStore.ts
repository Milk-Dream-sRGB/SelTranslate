// SelTranslate: state of the Audio file tab. Lives outside the component so
// a running job survives switching sidebar sections.
import { create } from "zustand";
import { listen } from "@tauri-apps/api/event";
import { commands, type AudioFileInfo } from "@/bindings";
import { useModelStore } from "../../../stores/modelStore";

// "" = same model as dictation. Kept in the browser, not in Handy settings.
const MODEL_KEY = "seltranslate.audioFile.model";

const loadModelChoice = (): string => {
  try {
    return localStorage.getItem(MODEL_KEY) ?? "";
  } catch {
    return "";
  }
};

export type AudioFileStatus =
  | "idle"
  | "loading_model"
  | "transcribing"
  | "done"
  | "cancelled"
  | "error";

interface ProgressPayload {
  phase: "loading_model" | "transcribing";
  processed_secs: number;
  total_secs: number | null;
  chunk_index: number;
  chunk_text: string;
}

interface DonePayload {
  text: string;
  elapsed_secs: number;
  audio_secs: number;
  /** File name in the transcript library, when it was saved there. */
  saved_as: string | null;
  save_error: string | null;
}

interface ErrorPayload {
  message: string;
}

interface AudioFileState {
  path: string | null;
  info: AudioFileInfo | null;
  /** Model picked for files; "" = same as dictation. */
  modelId: string;
  status: AudioFileStatus;
  processedSecs: number;
  totalSecs: number | null;
  text: string;
  /** When the transcribing phase began (ms), for the ETA. */
  transcribeStartedAt: number | null;
  elapsedSecs: number | null;
  error: string | null;
  /** Library result of the last finished job (stage H). */
  savedAs: string | null;
  saveError: string | null;
  init: () => void;
  selectFile: (path: string) => Promise<void>;
  setModel: (modelId: string) => void;
  start: () => Promise<void>;
  cancel: () => Promise<void>;
}

export const isRunning = (status: AudioFileStatus) =>
  status === "loading_model" || status === "transcribing";

let listening = false;

export const useAudioFileStore = create<AudioFileState>()((set, get) => ({
  path: null,
  info: null,
  modelId: loadModelChoice(),
  status: "idle",
  processedSecs: 0,
  totalSecs: null,
  text: "",
  transcribeStartedAt: null,
  elapsedSecs: null,
  error: null,
  savedAs: null,
  saveError: null,

  init: () => {
    if (listening) return;
    listening = true;
    listen<ProgressPayload>("audio-file-progress", ({ payload }) => {
      set((s) => ({
        status: payload.phase,
        processedSecs: payload.processed_secs,
        totalSecs: payload.total_secs,
        transcribeStartedAt:
          payload.phase === "transcribing"
            ? (s.transcribeStartedAt ?? Date.now())
            : s.transcribeStartedAt,
        text: payload.chunk_text
          ? s.text
            ? `${s.text} ${payload.chunk_text}`
            : payload.chunk_text
          : s.text,
      }));
    });
    listen<DonePayload>("audio-file-done", ({ payload }) => {
      set({
        status: "done",
        text: payload.text,
        processedSecs: payload.audio_secs,
        elapsedSecs: payload.elapsed_secs,
        savedAs: payload.saved_as,
        saveError: payload.save_error,
      });
    });
    listen<ErrorPayload>("audio-file-error", ({ payload }) => {
      set({ status: "error", error: payload.message });
    });
    listen("audio-file-cancelled", () => {
      set({ status: "cancelled" });
    });
  },

  selectFile: async (path) => {
    set({
      path,
      info: null,
      status: "idle",
      error: null,
      text: "",
      processedSecs: 0,
      totalSecs: null,
      elapsedSecs: null,
      savedAs: null,
      saveError: null,
    });
    const result = await commands.probeAudioFile(path);
    if (get().path !== path) return;
    if (result.status === "ok") {
      set({ info: result.data, totalSecs: result.data.duration_secs });
    } else {
      set({ status: "error", error: result.error });
    }
  },

  setModel: (modelId) => {
    set({ modelId });
    try {
      localStorage.setItem(MODEL_KEY, modelId);
    } catch {
      // storage unavailable: the choice just won't survive a restart
    }
  },

  start: async () => {
    const { path, info, modelId } = get();
    if (!path) return;
    // A deleted model falls back to the dictation model.
    const usable = useModelStore
      .getState()
      .models.some((m) => m.id === modelId && m.is_downloaded);
    set({
      status: "loading_model",
      text: "",
      error: null,
      processedSecs: 0,
      totalSecs: info?.duration_secs ?? null,
      transcribeStartedAt: null,
      elapsedSecs: null,
      savedAs: null,
      saveError: null,
    });
    const result = await commands.transcribeAudioFile(
      path,
      usable ? modelId : null,
    );
    if (result.status === "error") {
      set({ status: "error", error: result.error });
    }
  },

  cancel: async () => {
    await commands.cancelAudioFileTranscription();
  },
}));
